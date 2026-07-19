#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Mapping, Sequence

TRACE_METADATA_KEYS = [
    "repo",
    "base_commit",
    "version",
    "created_at",
    "hints_text",
    "test_patch",
    "environment_setup_commit",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
]

PROMPT_STYLE_CHOICES = [
    "style-2",
    "style-3",
    "full_file_gen",
    "style-2-edits-only",
]


def load_json_payload(path: str | Path) -> List[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [payload]


def read_instance_ids(path: str | Path | None) -> List[str] | None:
    if path is None:
        return None
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_instance_records(
    records: Sequence[Mapping[str, object]],
    *,
    instance_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> List[dict]:
    selected = list(records)
    if instance_ids is not None:
        wanted = set(instance_ids)
        selected = [dict(record) for record in selected if str(record["instance_id"]) in wanted]
    else:
        selected = [dict(record) for record in selected]
    selected.sort(key=lambda record: str(record["instance_id"]))
    if limit is not None:
        selected = selected[:limit]
    return selected


def processed_to_traced_instance(processed: Mapping[str, object]) -> dict:
    file_contents = dict(processed.get("file_contents", {}))
    if not file_contents:
        raise ValueError(
            f"processed instance {processed.get('instance_id')!r} is missing file_contents"
        )
    readmes = dict(processed.get("readmes", {}))
    metadata = {
        key: processed[key]
        for key in TRACE_METADATA_KEYS
        if key in processed and processed[key] not in {None, ""}
    }
    if "text_inputs" in processed:
        metadata["source_text_inputs_present"] = processed["text_inputs"] is not None
    return {
        "instance_id": str(processed["instance_id"]),
        "problem_statement": str(processed["problem_statement"]),
        "file_contents": {str(k): str(v) for k, v in file_contents.items()},
        "readmes": {str(k): str(v) for k, v in readmes.items()},
        "metadata": metadata,
    }


def write_traced_instances(
    processed_instances: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for processed in processed_instances:
            traced = processed_to_traced_instance(processed)
            handle.write(json.dumps(traced) + "\n")
            count += 1
    return count


def convert_processed_file(
    processed_instances_path: str | Path,
    output_path: str | Path,
    *,
    instance_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> int:
    processed_instances = load_json_payload(processed_instances_path)
    selected = select_instance_records(
        processed_instances,
        instance_ids=instance_ids,
        limit=limit,
    )
    return write_traced_instances(selected, output_path)


def load_split_records(dataset_name_or_path: str, split: str) -> List[dict]:
    from datasets import load_dataset, load_from_disk

    dataset = (
        load_from_disk(dataset_name_or_path)
        if Path(dataset_name_or_path).exists()
        else load_dataset(dataset_name_or_path)
    )
    return [dict(record) for record in dataset[split]]


def build_from_dataset(
    *,
    dataset_name_or_path: str,
    split: str,
    output_path: str | Path,
    prompt_style: str,
    file_source: str,
    retrieval_file: str | None,
    k: int | None,
    max_context_len: int | None,
    tokenizer_name: str | None,
    instance_ids: Sequence[str] | None,
    limit: int | None,
) -> int:
    from swebench.inference.make_datasets.create_instance import add_text_inputs

    if max_context_len is not None and tokenizer_name is None:
        raise ValueError("tokenizer_name is required when max_context_len is set")

    split_records = load_split_records(dataset_name_or_path, split)
    selected_records = select_instance_records(
        split_records,
        instance_ids=instance_ids,
        limit=limit,
    )
    instances = {record["instance_id"]: record for record in selected_records}

    if not instances:
        raise ValueError("No instances selected")

    with NamedTemporaryFile(
        mode="w",
        suffix=".progress.jsonl",
        delete=False,
        dir="/tmp",
        encoding="utf-8",
    ) as temp_progress:
        progress_path = temp_progress.name

    try:
        add_text_inputs(
            instances,
            retrieval_file=retrieval_file,
            k=k,
            prompt_style=prompt_style,
            file_source=file_source,
            max_context_len=max_context_len,
            tokenizer_name=tokenizer_name,
            progress_file=progress_path,
        )
        return convert_processed_file(
            progress_path,
            output_path,
            instance_ids=None,
            limit=None,
        )
    finally:
        if os.path.exists(progress_path):
            os.remove(progress_path)


def parse_args():
    parser = ArgumentParser(
        description="Prepare traced-agent input JSONL files for run_traced_api_agent."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Destination JSONL path for traced-agent input instances.",
    )
    parser.add_argument(
        "--processed_instances_path",
        type=str,
        help="Existing JSONL/JSON file containing processed instances with file_contents/readmes.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Hugging Face dataset name or load_from_disk path. Used when processed_instances_path is not provided.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use with dataset_name_or_path.",
    )
    parser.add_argument(
        "--instance_ids_path",
        type=str,
        help="Optional newline-delimited file of instance IDs to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of instances to include after filtering.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="style-3",
        choices=PROMPT_STYLE_CHOICES,
        help="Prompt style passed through to add_text_inputs when building from a dataset.",
    )
    parser.add_argument(
        "--file_source",
        type=str,
        default="oracle",
        choices=["oracle", "bm25", "all", "none"],
        help="File source passed through to add_text_inputs when building from a dataset.",
    )
    parser.add_argument(
        "--retrieval_file",
        type=str,
        help="Retrieval JSONL file required for file_source=bm25.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Maximum number of retrieved files when file_source=bm25.",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=None,
        help="Optional context limit passed through to add_text_inputs.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer required when max_context_len is set.",
    )
    args = parser.parse_args()
    if not args.processed_instances_path and not args.dataset_name_or_path:
        parser.error("Provide either --processed_instances_path or --dataset_name_or_path.")
    return args


def main():
    args = parse_args()
    instance_ids = read_instance_ids(args.instance_ids_path)

    if args.processed_instances_path:
        count = convert_processed_file(
            args.processed_instances_path,
            args.output_path,
            instance_ids=instance_ids,
            limit=args.limit,
        )
    else:
        count = build_from_dataset(
            dataset_name_or_path=args.dataset_name_or_path,
            split=args.split,
            output_path=args.output_path,
            prompt_style=args.prompt_style,
            file_source=args.file_source,
            retrieval_file=args.retrieval_file,
            k=args.k,
            max_context_len=args.max_context_len,
            tokenizer_name=args.tokenizer_name,
            instance_ids=instance_ids,
            limit=args.limit,
        )

    print(f"Wrote {count} traced-agent instances to {args.output_path}")


if __name__ == "__main__":
    main()
