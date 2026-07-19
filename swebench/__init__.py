from __future__ import annotations

from importlib import import_module

__version__ = "4.1.0"

_LAZY_IMPORTS = {
    "build_dataset": ("swebench.collect.build_dataset", "main"),
    "get_tasks_pipeline": ("swebench.collect.get_tasks_pipeline", "main"),
    "print_pulls": ("swebench.collect.print_pulls", "main"),
    "KEY_INSTANCE_ID": ("swebench.harness.constants", "KEY_INSTANCE_ID"),
    "KEY_MODEL": ("swebench.harness.constants", "KEY_MODEL"),
    "KEY_PREDICTION": ("swebench.harness.constants", "KEY_PREDICTION"),
    "MAP_REPO_VERSION_TO_SPECS": ("swebench.harness.constants", "MAP_REPO_VERSION_TO_SPECS"),
    "build_image": ("swebench.harness.docker_build", "build_image"),
    "build_base_images": ("swebench.harness.docker_build", "build_base_images"),
    "build_env_images": ("swebench.harness.docker_build", "build_env_images"),
    "build_instance_images": ("swebench.harness.docker_build", "build_instance_images"),
    "build_instance_image": ("swebench.harness.docker_build", "build_instance_image"),
    "close_logger": ("swebench.harness.docker_build", "close_logger"),
    "setup_logger": ("swebench.harness.docker_build", "setup_logger"),
    "cleanup_container": ("swebench.harness.docker_utils", "cleanup_container"),
    "remove_image": ("swebench.harness.docker_utils", "remove_image"),
    "copy_to_container": ("swebench.harness.docker_utils", "copy_to_container"),
    "exec_run_with_timeout": ("swebench.harness.docker_utils", "exec_run_with_timeout"),
    "list_images": ("swebench.harness.docker_utils", "list_images"),
    "compute_fail_to_pass": ("swebench.harness.grading", "compute_fail_to_pass"),
    "compute_pass_to_pass": ("swebench.harness.grading", "compute_pass_to_pass"),
    "get_logs_eval": ("swebench.harness.grading", "get_logs_eval"),
    "get_eval_report": ("swebench.harness.grading", "get_eval_report"),
    "get_resolution_status": ("swebench.harness.grading", "get_resolution_status"),
    "ResolvedStatus": ("swebench.harness.grading", "ResolvedStatus"),
    "TestStatus": ("swebench.harness.grading", "TestStatus"),
    "MAP_REPO_TO_PARSER": ("swebench.harness.log_parsers", "MAP_REPO_TO_PARSER"),
    "run_evaluation": ("swebench.harness.run_evaluation", "main"),
    "run_threadpool": ("swebench.harness.utils", "run_threadpool"),
    "MAP_REPO_TO_VERSION_PATHS": ("swebench.versioning.constants", "MAP_REPO_TO_VERSION_PATHS"),
    "MAP_REPO_TO_VERSION_PATTERNS": ("swebench.versioning.constants", "MAP_REPO_TO_VERSION_PATTERNS"),
    "get_version": ("swebench.versioning.get_versions", "get_version"),
    "get_versions_from_build": ("swebench.versioning.get_versions", "get_versions_from_build"),
    "get_versions_from_web": ("swebench.versioning.get_versions", "get_versions_from_web"),
    "map_version_to_task_instances": ("swebench.versioning.get_versions", "map_version_to_task_instances"),
    "split_instances": ("swebench.versioning.utils", "split_instances"),
}

__all__ = ["__version__", *_LAZY_IMPORTS.keys()]


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
