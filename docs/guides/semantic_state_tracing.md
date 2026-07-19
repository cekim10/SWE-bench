# Semantic State Tracing

`SWE-bench` does not ship a multi-step agent runner, but it can ship the trace format and emitter that external runners use to instrument their own control loops.

The `swebench.inference.trace.TraceLogger` class writes JSONL events that match the semantic lifecycle schema used by offline replay experiments:

- `CREATE` / `DERIVE` for state creation
- `READ` for prompt assembly
- `SUPERSEDE` for semantic death caused by a newer version
- `RELEASE` for branch or task completion
- `MATERIALIZE` / `EVICT` / `RELOAD` for runtime placement decisions

## Install Surface

```python
from swebench.inference.trace import TraceLogger
```

The bundled schema lives at:

```python
from swebench.inference.trace import get_schema_path

schema_path = get_schema_path()
```

## Recommended Hook Points

For agent runners such as SWE-agent, OpenHands, or AutoCodeRover, the minimum useful hooks are:

1. State creation point:
   `plan`, `patch`, `tool output`, `test result`, `retrieval`
2. Prompt assembly point:
   log a `READ` for every state that actually enters `build_prompt` or `messages_for_llm`
3. Version transition point:
   emit `SUPERSEDE` on `replan`, `repatch`, `retest`, then `RELEASE` when the old branch or task context is no longer live

The key invariant is to preserve `logical_key + version + supersedes`.

## Example

```python
from pathlib import Path

from swebench.inference.trace import TraceLogger

trace_path = Path("logs") / "semantic_state_trace.jsonl"

with TraceLogger(
    trace_path,
    workflow_id="sympy__sympy-20590",
    tenant_id="local-eval",
) as trace:
    plan_v1 = trace.create_state(
        state_id="plan_v1",
        logical_key="planner/plan",
        state_type="plan",
        size_bytes=512,
        token_count=128,
        producer="planner",
        materialization="HBM",
    )

    retrieved_doc = trace.create_state(
        state_id="doc_17",
        logical_key="retrieval/sympy/core/add.py",
        state_type="retrieved_document",
        size_bytes=4096,
        token_count=880,
        producer="retriever",
        materialization="CPU",
    )

    trace.log_prompt_assembly(
        consumer="coder",
        state_ids=[plan_v1.state_id, retrieved_doc.state_id],
        prompt_id="coder-step-1",
        metadata={"hook": "messages_for_llm"},
    )

    patch_v1 = trace.create_state(
        state_id="patch_v1",
        logical_key="coder/patch",
        state_type="generated_artifact",
        size_bytes=1024,
        token_count=220,
        producer="coder",
        parent_state_ids=[plan_v1.state_id, retrieved_doc.state_id],
        materialization="HBM",
    )

    test_v1 = trace.create_state(
        state_id="test_v1",
        logical_key="tester/result",
        state_type="verification_result",
        size_bytes=768,
        token_count=140,
        producer="tester",
        parent_state_ids=[patch_v1.state_id],
        materialization="HBM",
    )

    plan_v2 = trace.create_state(
        state_id="plan_v2",
        logical_key="planner/plan",
        state_type="plan",
        size_bytes=640,
        token_count=156,
        producer="planner",
        parent_state_ids=[plan_v1.state_id, test_v1.state_id],
        materialization="HBM",
    )
    trace.supersede_state(plan_v1.state_id, plan_v2.state_id)
    trace.release_state(plan_v1.state_id, consumer="planner")
```

This produces a replayable semantic state trace without requiring any changes to the core SWE-bench harness.

## CPU/API Smoke Run

For emitter correctness, you do not need a GPU server.

The repository now includes a lightweight traced multi-step runner:

```bash
python -m swebench.inference.run_traced_api_agent \
  --demo_instance \
  --provider stub \
  --trace_dir /tmp/semantic-traces \
  --output_path /tmp/semantic-run.jsonl \
  --max_iterations 2
```

This path uses a built-in demo task plus a stub model backend, so it exercises:

- `retrieved_document` creation
- prompt-time `READ` events
- `plan` and `generated_artifact` versioning
- `SUPERSEDE` on replan / repatch
- final `RELEASE`

For a real small-sample run with an API model, prepare a JSONL file with:

- `instance_id`
- `problem_statement`
- `file_contents`
- optional `readmes`

Then run:

```bash
python -m swebench.inference.run_traced_api_agent \
  --instances_path /path/to/instances.jsonl \
  --provider openai \
  --model gpt-4.1-mini \
  --trace_dir ./logs/semantic-traces \
  --output_path ./logs/semantic-run.jsonl \
  --max_iterations 2 \
  --max_files 5
```

For zero-credit local replication on Apple Silicon or CPU, use Ollama:

```bash
python -m swebench.inference.run_traced_api_agent \
  --instances_path /path/to/instances.jsonl \
  --provider ollama \
  --model qwen2.5-coder \
  --trace_dir ./logs/semantic-traces-ollama \
  --output_path ./logs/semantic-run-ollama.jsonl \
  --max_iterations 2 \
  --max_files 5
```

For a free-tier external API path, use Groq with `GROQ_API_KEY` set:

```bash
python -m swebench.inference.run_traced_api_agent \
  --instances_path /path/to/instances.jsonl \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --trace_dir ./logs/semantic-traces-groq \
  --output_path ./logs/semantic-run-groq.jsonl \
  --max_iterations 2 \
  --max_files 5
```

That is the intended first step before moving the same hook contract into full agent runners such as SWE-agent, OpenHands, or AutoCodeRover.

## First 10 Tasks

To prepare the first `10` traced-runner inputs from a SWE-bench split, use:

```bash
python3 -m swebench.inference.make_traced_agent_inputs \
  --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
  --split test \
  --file_source oracle \
  --limit 10 \
  --output_path ./logs/traced-inputs/verified-test-10.jsonl
```

That produces the minimal JSONL expected by `run_traced_api_agent`.

Then run the traced loop itself:

```bash
python3 -m swebench.inference.run_traced_api_agent \
  --instances_path ./logs/traced-inputs/verified-test-10.jsonl \
  --provider openai \
  --model gpt-4.1-mini \
  --trace_dir ./logs/semantic-traces \
  --output_path ./logs/semantic-run-verified-10.jsonl \
  --max_iterations 2 \
  --max_files 5
```

If you want an immediate `stub / ollama / groq` comparison pass with one command, use:

```bash
python3 -m swebench.inference.run_traced_agent_matrix \
  --instances_path ./logs/traced-inputs/verified-test-10.jsonl \
  --provider_spec stub \
  --provider_spec ollama:qwen2.5-coder \
  --provider_spec groq:llama-3.3-70b-versatile \
  --output_dir ./logs/provider-matrix \
  --request_timeout 300 \
  --max_iterations 2 \
  --max_files 5
```

This produces:

- `./logs/provider-matrix/runs/*.jsonl`
- `./logs/provider-matrix/traces/<provider>/*.jsonl`
- `./logs/provider-matrix/combined.jsonl`
- `./logs/provider-matrix/analysis.json`
- `./logs/provider-matrix/analysis.md`

If you already have a processed progress file with `file_contents` and `readmes`, convert it directly:

```bash
python3 -m swebench.inference.make_traced_agent_inputs \
  --processed_instances_path /path/to/test.progress.jsonl \
  --limit 10 \
  --output_path ./logs/traced-inputs/from-progress-10.jsonl
```

For BM25-backed context instead of oracle files:

```bash
python3 -m swebench.inference.make_traced_agent_inputs \
  --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
  --split test \
  --file_source bm25 \
  --retrieval_file ./retrieval_results/verified-test.jsonl \
  --k 5 \
  --limit 10 \
  --output_path ./logs/traced-inputs/verified-bm25-10.jsonl
```

## Phase 1-3 Analysis

Once the traces are collected, run the analyzer:

```bash
python3 -m swebench.inference.analyze_traces \
  --run_output_path ./logs/semantic-run-verified-10.jsonl \
  --output_path ./logs/analysis/verified-10.json \
  --markdown_path ./logs/analysis/verified-10.md
```

This emits three sections aligned with the characterization-first paper structure:

- `Prompt Composition`
- `Lifecycle Characterization`
- `Abstraction Mismatch`

The key mismatch outputs are:

- `mixed_lifecycle_prompt_rate`
- `monolithic_invalidation_events`
- `stale_bytes_before_next_prompt`
- `reusable_live_bytes`
- `avg_lifetime_spread`

That is the first pass for answering whether a monolithic materialization unit is already a bad abstraction before any runtime optimization claims.
