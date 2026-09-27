from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .common import load_yaml, object_sha256, to_jsonable

KNOWN_TARGETS = ("k_min", "k_max", "sigma", "beta")
STAGE_ORDER = (
    "validate_data",
    "grid_search",
    "select_candidates",
    "retrain_candidates",
    "select_final",
    "train_ensemble",
    "evaluate_ensemble",
    "interpretability",
    "report",
)
STAGE_DEPENDENCIES = {
    "validate_data": (),
    "grid_search": ("validate_data",),
    "select_candidates": ("grid_search",),
    "retrain_candidates": ("select_candidates", "validate_data"),
    "select_final": ("retrain_candidates",),
    "train_ensemble": ("select_final",),
    "evaluate_ensemble": ("train_ensemble",),
    "interpretability": ("train_ensemble",),
    "report": ("evaluate_ensemble",),
}

DEFAULTS: dict[str, Any] = {}


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base




def _load_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic YAML extends chain detected at {path}")
    seen.add(path)
    raw = load_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"YAML root must be a mapping: {path}")
    raw_dict = copy.deepcopy(dict(raw))
    parent_value = raw_dict.pop("extends", None)
    if parent_value is None:
        return raw_dict
    parent = Path(os.path.expandvars(os.path.expanduser(str(parent_value))))
    if not parent.is_absolute():
        parent = path.parent / parent
    inherited = _load_with_extends(parent, seen)
    return dict(_deep_merge(inherited, raw_dict))

def _expand_path(value: str, config_dir: Path, project_root: Path | None = None) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        anchor = project_root if project_root is not None else config_dir
        path = anchor / path
    return str(path.resolve())


def _as_nonempty_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty YAML list.")
    return value


def _validate_positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    """Validate an integer without silently truncating floats or accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, received {value!r}.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def _validate_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, received a boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, received {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, received {value!r}.")
    return number


def _validate_yaml_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a YAML boolean.")
    return value


def _validate_probability_percentiles(lower: Any, upper: Any) -> tuple[float, float]:
    lower_f = _validate_finite_float(lower, "data.normalization.lower_percentile")
    upper_f = _validate_finite_float(upper, "data.normalization.upper_percentile")
    if not (0.0 <= lower_f < upper_f <= 100.0):
        raise ValueError("Normalization percentiles must satisfy 0 <= lower < upper <= 100.")
    return lower_f, upper_f


def _validate_selection(selection: Mapping[str, Any], task_targets: Sequence[str], name: str) -> None:
    required = {
        "metric", "target", "space", "mse_weight", "r2_weight", "scaling",
        "group_by", "top_per_group", "top_k", "require_all",
    }
    missing = sorted(required.difference(selection))
    if missing:
        raise KeyError(f"{name} is missing required fields: {missing}")
    metric = selection.get("metric")
    if metric not in {"rmse", "mse", "r2", "mse_r2", "composite"}:
        raise ValueError(f"{name}.metric is unsupported: {metric!r}")
    target = selection.get("target")
    if target != "overall" and target not in task_targets:
        raise ValueError(f"{name}.target must be 'overall' or one of {list(task_targets)}.")
    if selection.get("space") not in {"normalized", "physical"}:
        raise ValueError(f"{name}.space must be normalized or physical.")
    if target == "overall" and metric in {"rmse", "mse", "mse_r2", "composite"} and selection.get("space") != "normalized":
        raise ValueError(f"{name}: overall error-based selection must use normalized space.")
    if selection.get("scaling") not in {"raw", "minmax"}:
        raise ValueError(f"{name}.scaling must be raw or minmax.")
    mse_weight = _validate_finite_float(selection["mse_weight"], f"{name}.mse_weight")
    r2_weight = _validate_finite_float(selection["r2_weight"], f"{name}.r2_weight")
    if mse_weight < 0.0 or r2_weight < 0.0:
        raise ValueError(f"{name}.mse_weight and r2_weight must be non-negative.")
    if metric in {"mse_r2", "composite"} and mse_weight == 0.0 and r2_weight == 0.0:
        raise ValueError(f"{name}: at least one composite weight must be positive.")
    group_by = selection.get("group_by")
    allowed_groups = {None, "architecture", "in_channels", "pretrained", "dropout", "batch_size", "learning_rate", "weight_decay", "loss", "smooth_l1_beta"}
    if group_by not in allowed_groups:
        raise ValueError(f"{name}.group_by is unsupported: {group_by!r}")
    _validate_positive_int(selection["top_per_group"], f"{name}.top_per_group")
    _validate_positive_int(selection["top_k"], f"{name}.top_k")
    _validate_yaml_bool(selection["require_all"], f"{name}.require_all")


def load_and_resolve_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    raw = _load_with_extends(path)
    config = _deep_merge(copy.deepcopy(DEFAULTS), raw)
    config["_meta"] = {"source_config": str(path), "source_directory": str(path.parent)}

    schema_version = _validate_positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("Only schema_version: 1 is supported.")
    config["schema_version"] = schema_version

    required_sections = {
        "experiment", "task", "data", "model", "search", "candidate_training",
        "ensemble", "training", "evaluation", "runtime", "checkpointing",
        "storage", "interpretability", "stages", "logging",
    }
    missing_sections = sorted(required_sections.difference(config))
    if missing_sections:
        raise KeyError(
            "Configuration is missing required top-level sections: "
            f"{missing_sections}. Define them in the YAML config/base config; "
            "scientific settings are not silently supplied by Python defaults."
        )

    experiment = config["experiment"]
    if not isinstance(experiment, MutableMapping):
        raise TypeError("experiment must be a YAML mapping.")
    _validate_yaml_bool(experiment.get("allow_config_change"), "experiment.allow_config_change")
    project_root = Path(_expand_path(str(experiment["project_root"]), path.parent))
    experiment["project_root"] = str(project_root)
    experiment["output_root"] = _expand_path(str(experiment["output_root"]), path.parent, project_root)
    for key in ("name", "version"):
        value = str(experiment.get(key, "")).strip()
        if not value or any(ch in value for ch in "/\\\0"):
            raise ValueError(f"experiment.{key} must be a safe, non-empty path component.")
        experiment[key] = value

    data = config["data"]
    data["small_root"] = _expand_path(str(data["small_root"]), path.parent, project_root)
    data["large_root"] = _expand_path(str(data["large_root"]), path.parent, project_root)
    split_directories = data.get("split_directories")
    if not isinstance(split_directories, Mapping) or set(split_directories) != {"train", "validation", "test"}:
        raise ValueError("data.split_directories must define exactly train, validation, and test.")
    for split, folder in split_directories.items():
        folder_text = str(folder).strip()
        if not folder_text or Path(folder_text).is_absolute() or ".." in Path(folder_text).parts:
            raise ValueError(f"data.split_directories.{split} must be a safe relative folder name.")
        split_directories[split] = folder_text
    _validate_yaml_bool(data.get("require_square_for_augmentation"), "data.require_square_for_augmentation")
    _validate_yaml_bool(data.get("enforce_disjoint_files"), "data.enforce_disjoint_files")
    normalization = data["normalization"]
    normalization["lower_percentile"], normalization["upper_percentile"] = _validate_probability_percentiles(
        normalization["lower_percentile"], normalization["upper_percentile"]
    )
    normalization["max_images"] = _validate_positive_int(normalization["max_images"], "data.normalization.max_images")
    normalization["max_pixels"] = _validate_positive_int(normalization["max_pixels"], "data.normalization.max_pixels")
    normalization["seed"] = _validate_positive_int(normalization["seed"], "data.normalization.seed", allow_zero=True)
    normalization["epsilon"] = _validate_finite_float(normalization["epsilon"], "data.normalization.epsilon")
    if normalization["epsilon"] <= 0:
        raise ValueError("data.normalization.epsilon must be positive.")

    task = config["task"]
    mode = str(task["mode"])
    if mode not in {"multitask", "single_target"}:
        raise ValueError("task.mode must be multitask or single_target.")
    targets = [str(t) for t in _as_nonempty_list(task["targets"], "task.targets")]
    if len(set(targets)) != len(targets):
        raise ValueError("task.targets contains duplicates.")
    unknown = sorted(set(targets).difference(KNOWN_TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Known targets are {list(KNOWN_TARGETS)}.")
    if mode == "single_target" and len(targets) != 1:
        raise ValueError("single_target mode requires exactly one target.")
    if mode == "multitask" and len(targets) < 2:
        raise ValueError("multitask mode requires at least two targets.")
    task["targets"] = targets
    ranges = task["target_ranges"]
    if not isinstance(ranges, Mapping):
        raise TypeError("task.target_ranges must be a mapping.")
    normalized_ranges: dict[str, list[float]] = {}
    for target in targets:
        if target not in ranges:
            raise KeyError(f"Missing task.target_ranges entry for {target}.")
        pair = ranges[target]
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"Range for {target} must contain [minimum, maximum].")
        low = _validate_finite_float(pair[0], f"task.target_ranges.{target}[0]")
        high = _validate_finite_float(pair[1], f"task.target_ranges.{target}[1]")
        if not low < high:
            raise ValueError(f"Invalid range for {target}: {pair}")
        normalized_ranges[target] = [low, high]
    task["target_ranges"] = normalized_ranges

    model = config["model"]
    architectures = [str(x) for x in _as_nonempty_list(model["architectures"], "model.architectures")]
    unsupported_arch = sorted(set(architectures).difference({"resnet18", "resnet34", "resnet50"}))
    if unsupported_arch:
        raise ValueError(f"Unsupported architectures: {unsupported_arch}")
    if len(set(architectures)) != len(architectures):
        raise ValueError("model.architectures contains duplicate entries.")
    model["architectures"] = architectures
    channels = [_validate_positive_int(x, "model.in_channels") for x in _as_nonempty_list(model["in_channels"], "model.in_channels")]
    if any(x not in {1, 3} for x in channels):
        raise ValueError("model.in_channels values must be 1 or 3.")
    if len(set(channels)) != len(channels):
        raise ValueError("model.in_channels contains duplicate entries.")
    model["in_channels"] = channels
    pretrained = _as_nonempty_list(model["pretrained"], "model.pretrained")
    if any(not isinstance(x, bool) for x in pretrained):
        raise TypeError("model.pretrained entries must be YAML booleans.")
    if len(set(pretrained)) != len(pretrained):
        raise ValueError("model.pretrained contains duplicate entries.")
    model["pretrained"] = list(pretrained)
    pretrained_weights = model.get("pretrained_weights")
    if not isinstance(pretrained_weights, Mapping):
        raise TypeError("model.pretrained_weights must be a YAML mapping.")
    expected_weight_keys = set(architectures)
    missing_weight_keys = sorted(expected_weight_keys.difference(pretrained_weights))
    if missing_weight_keys:
        raise ValueError(
            "model.pretrained_weights is missing configured architectures: "
            f"{missing_weight_keys}."
        )
    # YAML inheritance deep-merges mappings. A child configuration that narrows the
    # architecture list may therefore inherit unused weight entries from the base
    # configuration. Keep only recipes for active architectures so the resolved
    # configuration reflects the actual model search space and hashes cleanly.
    pretrained_weights = {architecture: pretrained_weights[architecture] for architecture in architectures}
    model["pretrained_weights"] = pretrained_weights
    supported_weights = {
        "resnet18": {"IMAGENET1K_V1"},
        "resnet34": {"IMAGENET1K_V1"},
        "resnet50": {"IMAGENET1K_V1", "IMAGENET1K_V2"},
    }
    for architecture, weight_name in pretrained_weights.items():
        weight_name = str(weight_name)
        if weight_name not in supported_weights[str(architecture)]:
            raise ValueError(
                f"Unsupported pretrained weight {weight_name!r} for {architecture}. "
                f"Supported: {sorted(supported_weights[str(architecture)])}"
            )
        pretrained_weights[architecture] = weight_name
    if model["input_standardization"] not in {"imagenet", "none"}:
        raise ValueError("model.input_standardization must be imagenet or none.")
    shared_dimensions = [
        _validate_positive_int(x, f"model.shared_dimensions[{index}]")
        for index, x in enumerate(_as_nonempty_list(model["shared_dimensions"], "model.shared_dimensions"))
    ]
    model["shared_dimensions"] = shared_dimensions
    model["head_hidden_dimension"] = _validate_positive_int(model["head_hidden_dimension"], "model.head_hidden_dimension")
    _validate_yaml_bool(model.get("preserve_resolution"), "model.preserve_resolution")
    dropout_policy = model.get("dropout_policy")
    if not isinstance(dropout_policy, Mapping):
        raise TypeError("model.dropout_policy must be a mapping.")
    for key in ("shared_increment_per_layer", "shared_max", "head_multiplier", "head_max"):
        if key not in dropout_policy:
            raise KeyError(f"model.dropout_policy is missing {key}.")
        dropout_policy[key] = _validate_finite_float(dropout_policy[key], f"model.dropout_policy.{key}")
    if dropout_policy["shared_increment_per_layer"] < 0:
        raise ValueError("model.dropout_policy.shared_increment_per_layer must be non-negative.")
    if not 0.0 <= dropout_policy["shared_max"] < 1.0:
        raise ValueError("model.dropout_policy.shared_max must lie in [0,1).")
    if dropout_policy["head_multiplier"] < 0:
        raise ValueError("model.dropout_policy.head_multiplier must be non-negative.")
    if not 0.0 <= dropout_policy["head_max"] < 1.0:
        raise ValueError("model.dropout_policy.head_max must lie in [0,1).")

    search = config["search"]
    repetitions = _validate_positive_int(search["repetitions_per_configuration"], "search.repetitions_per_configuration")
    search["repetitions_per_configuration"] = repetitions
    search_seeds = [
        _validate_positive_int(x, f"search.seeds[{index}]", allow_zero=True)
        for index, x in enumerate(_as_nonempty_list(search["seeds"], "search.seeds"))
    ]
    if len(search_seeds) < repetitions:
        raise ValueError("search.seeds must provide at least one seed per repetition.")
    if len(set(search_seeds[:repetitions])) != repetitions:
        raise ValueError("search seeds used for repetitions must be unique.")
    search["seeds"] = search_seeds
    space = search["space"]
    if not isinstance(space, Mapping) or not space:
        raise ValueError("search.space must be a non-empty mapping of lists.")
    required_space = {"dropout", "learning_rate", "batch_size", "weight_decay", "loss", "smooth_l1_beta"}
    missing_space = sorted(required_space.difference(space))
    extra_space = sorted(set(space).difference(required_space))
    if missing_space or extra_space:
        raise ValueError(f"search.space keys must be exactly {sorted(required_space)}; missing={missing_space}, extra={extra_space}")
    for key, values in space.items():
        values = _as_nonempty_list(values, f"search.space.{key}")
        if len({repr(value) for value in values}) != len(values):
            raise ValueError(f"search.space.{key} contains duplicate levels.")
    space["dropout"] = [_validate_finite_float(x, "search.space.dropout") for x in space["dropout"]]
    if any(not 0.0 <= x < 1.0 for x in space["dropout"]):
        raise ValueError("search.space.dropout values must lie in [0,1).")
    space["learning_rate"] = [_validate_finite_float(x, "search.space.learning_rate") for x in space["learning_rate"]]
    if any(x <= 0.0 for x in space["learning_rate"]):
        raise ValueError("search.space.learning_rate values must be positive.")
    space["batch_size"] = [_validate_positive_int(x, "search.space.batch_size") for x in space["batch_size"]]
    space["weight_decay"] = [_validate_finite_float(x, "search.space.weight_decay") for x in space["weight_decay"]]
    if any(x < 0.0 for x in space["weight_decay"]):
        raise ValueError("search.space.weight_decay values must be non-negative.")
    space["loss"] = [str(x) for x in space["loss"]]
    if any(x != "smooth_l1" for x in space["loss"]):
        raise ValueError("search.space.loss currently supports only smooth_l1.")
    space["smooth_l1_beta"] = [_validate_finite_float(x, "search.space.smooth_l1_beta") for x in space["smooth_l1_beta"]]
    if any(x <= 0.0 for x in space["smooth_l1_beta"]):
        raise ValueError("search.space.smooth_l1_beta values must be positive.")
    _validate_selection(search["selection"], targets, "search.selection")

    candidate = config["candidate_training"]
    candidate_reps = _validate_positive_int(candidate["repetitions_per_configuration"], "candidate_training.repetitions_per_configuration")
    candidate["repetitions_per_configuration"] = candidate_reps
    candidate_seeds = [
        _validate_positive_int(x, f"candidate_training.seeds[{index}]", allow_zero=True)
        for index, x in enumerate(_as_nonempty_list(candidate["seeds"], "candidate_training.seeds"))
    ]
    if len(candidate_seeds) < candidate_reps:
        raise ValueError("candidate_training.seeds must provide at least one seed per repetition.")
    if len(set(candidate_seeds[:candidate_reps])) != candidate_reps:
        raise ValueError("candidate-training seeds used for repetitions must be unique.")
    candidate["seeds"] = candidate_seeds
    _validate_selection(candidate["selection"], targets, "candidate_training.selection")
    if candidate["selection"]["group_by"] is not None or int(candidate["selection"]["top_per_group"]) != 1:
        raise ValueError(
            "candidate_training.selection must use group_by: null and top_per_group: 1 "
            "because the final ensemble trains exactly one selected configuration."
        )

    ensemble = config["ensemble"]
    members = _validate_positive_int(ensemble["members"], "ensemble.members")
    if members < 2:
        raise ValueError("ensemble.members must be at least 2.")
    seeds = [
        _validate_positive_int(x, f"ensemble.seeds[{index}]", allow_zero=True)
        for index, x in enumerate(_as_nonempty_list(ensemble["seeds"], "ensemble.seeds"))
    ]
    if len(seeds) < members:
        raise ValueError("ensemble.seeds must contain at least ensemble.members values.")
    if len(set(seeds[:members])) != members:
        raise ValueError("The ensemble seeds used by the configured members must be unique.")
    ensemble["members"] = members
    ensemble["seeds"] = seeds
    ensemble["normalization_seed"] = _validate_positive_int(ensemble["normalization_seed"], "ensemble.normalization_seed", allow_zero=True)
    levels = [
        _validate_finite_float(x, f"ensemble.interval_levels[{index}]")
        for index, x in enumerate(_as_nonempty_list(ensemble["interval_levels"], "ensemble.interval_levels"))
    ]
    if any(not 0.0 < x < 1.0 for x in levels):
        raise ValueError("ensemble.interval_levels must lie in (0,1).")
    ensemble["interval_levels"] = sorted(set(levels))
    ensemble["bootstrap_repetitions"] = _validate_positive_int(ensemble["bootstrap_repetitions"], "ensemble.bootstrap_repetitions")

    training = config["training"]
    training["epochs"] = _validate_positive_int(training["epochs"], "training.epochs")
    training["num_workers"] = _validate_positive_int(training["num_workers"], "training.num_workers", allow_zero=True)
    if training["num_workers"] > 0:
        training["prefetch_factor"] = _validate_positive_int(training["prefetch_factor"], "training.prefetch_factor")
    for key in ("augmentation", "amp", "deterministic", "cudnn_benchmark", "pin_memory", "persistent_workers"):
        _validate_yaml_bool(training.get(key), f"training.{key}")
    if training["deterministic"] and training["cudnn_benchmark"]:
        training["cudnn_benchmark"] = False

    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise TypeError("training.optimizer must be a mapping.")
    if str(optimizer.get("name")) != "adamw":
        raise ValueError("training.optimizer.name currently supports only adamw.")
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("training.optimizer.betas must contain exactly two values.")
    beta1 = _validate_finite_float(betas[0], "training.optimizer.betas[0]")
    beta2 = _validate_finite_float(betas[1], "training.optimizer.betas[1]")
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("training.optimizer.betas values must lie in [0,1).")
    optimizer["betas"] = [beta1, beta2]
    optimizer["eps"] = _validate_finite_float(optimizer.get("eps"), "training.optimizer.eps")
    if optimizer["eps"] <= 0.0:
        raise ValueError("training.optimizer.eps must be positive.")
    _validate_yaml_bool(optimizer.get("amsgrad"), "training.optimizer.amsgrad")
    training["gradient_clip_norm"] = _validate_finite_float(training["gradient_clip_norm"], "training.gradient_clip_norm")
    if training["gradient_clip_norm"] < 0.0:
        raise ValueError("training.gradient_clip_norm must be non-negative.")
    scheduler = training["scheduler"]
    if str(scheduler.get("name")) != "reduce_on_plateau":
        raise ValueError("training.scheduler.name currently supports only reduce_on_plateau.")
    scheduler["factor"] = _validate_finite_float(scheduler["factor"], "training.scheduler.factor")
    if not 0.0 < scheduler["factor"] < 1.0:
        raise ValueError("training.scheduler.factor must lie in (0,1).")
    scheduler["patience"] = _validate_positive_int(scheduler["patience"], "training.scheduler.patience", allow_zero=True)
    scheduler["min_lr"] = _validate_finite_float(scheduler["min_lr"], "training.scheduler.min_lr")
    scheduler["threshold"] = _validate_finite_float(scheduler.get("threshold"), "training.scheduler.threshold")
    scheduler["cooldown"] = _validate_positive_int(scheduler.get("cooldown"), "training.scheduler.cooldown", allow_zero=True)
    scheduler["eps"] = _validate_finite_float(scheduler.get("eps"), "training.scheduler.eps")
    if scheduler["min_lr"] < 0.0 or scheduler["threshold"] < 0.0 or scheduler["eps"] < 0.0:
        raise ValueError("training.scheduler min_lr/threshold/eps must be non-negative.")
    if scheduler.get("threshold_mode") not in {"rel", "abs"}:
        raise ValueError("training.scheduler.threshold_mode must be rel or abs.")
    early = training["early_stopping"]
    if early.get("monitor") != "validation_loss" or early.get("mode") != "min":
        raise ValueError("training.early_stopping currently supports monitor=validation_loss and mode=min.")
    early["patience"] = _validate_positive_int(early["patience"], "training.early_stopping.patience", allow_zero=True)
    early["min_delta"] = _validate_finite_float(early["min_delta"], "training.early_stopping.min_delta")
    if early["min_delta"] < 0.0:
        raise ValueError("training.early_stopping.min_delta must be non-negative.")

    evaluation = config["evaluation"]
    for profile in ("grid", "candidate", "final"):
        if profile not in evaluation or not isinstance(evaluation[profile], Mapping):
            raise ValueError(f"evaluation.{profile} must be a mapping.")
        for key in ("evaluate_validation", "evaluate_test"):
            if not isinstance(evaluation[profile].get(key), bool):
                raise TypeError(f"evaluation.{profile}.{key} must be a YAML boolean.")
        if not evaluation[profile]["evaluate_validation"]:
            raise ValueError(f"evaluation.{profile}.evaluate_validation must be true for validation-based selection/checkpointing.")
    if evaluation["grid"]["evaluate_test"] or evaluation["candidate"]["evaluate_test"]:
        raise ValueError(
            "evaluation.grid.evaluate_test and evaluation.candidate.evaluate_test must be false "
            "to preserve held-out test isolation during hyperparameter selection."
        )
    if not evaluation["final"]["evaluate_test"]:
        raise ValueError("evaluation.final.evaluate_test must be true for final ensemble evaluation.")

    runtime = config["runtime"]
    gpus = [
        _validate_positive_int(x, f"runtime.gpus[{index}]", allow_zero=True)
        for index, x in enumerate(_as_nonempty_list(runtime["gpus"], "runtime.gpus"))
    ]
    if any(x < 0 for x in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("runtime.gpus must contain unique non-negative indices.")
    runtime["gpus"] = gpus
    runtime["max_parallel_jobs"] = _validate_positive_int(runtime["max_parallel_jobs"], "runtime.max_parallel_jobs")
    runtime["poll_seconds"] = _validate_positive_int(runtime["poll_seconds"], "runtime.poll_seconds")
    for key in ("fail_fast_grid", "fail_fast_other_stages"):
        _validate_yaml_bool(runtime.get(key), f"runtime.{key}")
    python_executable = str(runtime.get("python_executable", "")).strip()
    if not python_executable:
        raise ValueError("runtime.python_executable must be 'current' or a non-empty executable/path.")
    runtime["python_executable"] = python_executable
    environment = runtime.get("environment")
    if not isinstance(environment, Mapping):
        raise TypeError("runtime.environment must be a YAML mapping.")
    normalized_environment: dict[str, str] = {}
    for key, value in environment.items():
        key_text = str(key).strip()
        if not key_text or "=" in key_text or "\x00" in key_text:
            raise ValueError(f"Invalid runtime.environment key: {key!r}")
        normalized_environment[key_text] = str(value)
    runtime["environment"] = normalized_environment
    if runtime["device"] not in {"cuda", "cpu", "auto"}:
        raise ValueError("runtime.device must be cuda, cpu, or auto.")
    gpu_wait = runtime.get("gpu_wait")
    if not isinstance(gpu_wait, MutableMapping):
        raise TypeError("runtime.gpu_wait must be a YAML mapping.")
    required_gpu_wait = {
        "enabled", "poll_seconds", "stable_seconds", "report_every_seconds",
        "timeout_seconds", "nvidia_smi",
    }
    missing_gpu_wait = sorted(required_gpu_wait.difference(gpu_wait))
    if missing_gpu_wait:
        raise KeyError(f"runtime.gpu_wait is missing required fields: {missing_gpu_wait}")
    _validate_yaml_bool(gpu_wait["enabled"], "runtime.gpu_wait.enabled")
    gpu_wait["poll_seconds"] = _validate_positive_int(gpu_wait["poll_seconds"], "runtime.gpu_wait.poll_seconds")
    gpu_wait["stable_seconds"] = _validate_positive_int(gpu_wait["stable_seconds"], "runtime.gpu_wait.stable_seconds", allow_zero=True)
    gpu_wait["report_every_seconds"] = _validate_positive_int(gpu_wait["report_every_seconds"], "runtime.gpu_wait.report_every_seconds")
    gpu_wait["timeout_seconds"] = _validate_positive_int(gpu_wait["timeout_seconds"], "runtime.gpu_wait.timeout_seconds", allow_zero=True)
    if not str(gpu_wait["nvidia_smi"]).strip():
        raise ValueError("runtime.gpu_wait.nvidia_smi must not be empty.")

    checkpointing = config["checkpointing"]
    if not isinstance(checkpointing.get("resume"), bool) or not isinstance(checkpointing.get("verify_after_write"), bool):
        raise TypeError("checkpointing.resume and checkpointing.verify_after_write must be YAML booleans.")
    checkpointing["every_n_epochs"] = _validate_positive_int(checkpointing["every_n_epochs"], "checkpointing.every_n_epochs")

    storage = config["storage"]
    for profile in ("grid", "candidate", "final"):
        if profile not in storage or not isinstance(storage[profile], Mapping):
            raise ValueError(f"storage.{profile} must be a mapping.")
        for key in ("save_predictions", "save_plots", "save_environment", "keep_best_checkpoint", "keep_latest_checkpoint"):
            if not isinstance(storage[profile].get(key), bool):
                raise TypeError(f"storage.{profile}.{key} must be a YAML boolean.")
    if not isinstance(storage.get("write_per_image_csv"), bool):
        raise TypeError("storage.write_per_image_csv must be a YAML boolean.")
    if not storage["final"]["keep_best_checkpoint"]:
        raise ValueError("storage.final.keep_best_checkpoint must be true for inference and ensemble evaluation.")
    if not storage["final"]["save_predictions"]:
        raise ValueError("storage.final.save_predictions must be true for ensemble evaluation and comparison.")

    interpretability = config["interpretability"]
    if not isinstance(interpretability.get("enabled"), bool):
        raise TypeError("interpretability.enabled must be a YAML boolean.")
    layers = [str(x) for x in _as_nonempty_list(interpretability["embedding_layers"], "interpretability.embedding_layers")]
    if any(layer not in {"shared", "backbone"} for layer in layers):
        raise ValueError("interpretability.embedding_layers supports shared and backbone.")
    interpretability["embedding_layers"] = list(dict.fromkeys(layers))
    if interpretability["fit_split"] != "validation":
        raise ValueError("interpretability.fit_split must be validation to avoid fitting latent axes on test data.")
    if interpretability["plot_split"] not in {"validation", "test"}:
        raise ValueError("interpretability.plot_split must be validation or test.")
    interpretability["batch_size"] = _validate_positive_int(interpretability["batch_size"], "interpretability.batch_size")
    for key in (
        "pca_components", "pc_correlation_components", "cca_components",
        "cca_pca_components", "max_plot_points",
    ):
        interpretability[key] = _validate_positive_int(interpretability[key], f"interpretability.{key}")
    for key in ("max_analysis_samples", "pca_fit_max_samples", "cca_fit_max_samples"):
        interpretability[key] = _validate_positive_int(interpretability[key], f"interpretability.{key}", allow_zero=True)
    if str(interpretability["pca_backend"]) not in {"auto", "torch", "incremental"}:
        raise ValueError("interpretability.pca_backend must be auto, torch, or incremental.")
    if not str(interpretability["pca_device"]).strip():
        raise ValueError("interpretability.pca_device must not be empty.")
    for key in ("pca_oversamples", "pca_power_iterations", "pca_cpu_batch_size", "pca_transform_batch_size", "linear_algebra_threads", "cca_max_iter"):
        interpretability[key] = _validate_positive_int(interpretability[key], f"interpretability.{key}")
    interpretability["cca_tol"] = _validate_finite_float(interpretability["cca_tol"], "interpretability.cca_tol")
    if interpretability["cca_tol"] <= 0.0:
        raise ValueError("interpretability.cca_tol must be positive.")
    interpretability["cca_pca_variance"] = _validate_finite_float(interpretability["cca_pca_variance"], "interpretability.cca_pca_variance")
    if not 0.0 < interpretability["cca_pca_variance"] <= 1.0:
        raise ValueError("interpretability.cca_pca_variance must lie in (0,1].")
    umap = interpretability["umap"]
    # PyYAML follows YAML 1.1 boolean spellings, where an unquoted `off` can
    # become False. Accept that common spelling and normalize it explicitly.
    if umap.get("policy") is False:
        umap["policy"] = "off"
    else:
        umap["policy"] = str(umap.get("policy", "optional")).lower()
    if umap["policy"] not in {"off", "optional", "require"}:
        raise ValueError("interpretability.umap.policy must be off, optional, or require.")
    umap["neighbors"] = _validate_positive_int(umap["neighbors"], "interpretability.umap.neighbors")
    umap["min_dist"] = _validate_finite_float(umap["min_dist"], "interpretability.umap.min_dist")
    if umap["min_dist"] < 0.0:
        raise ValueError("interpretability.umap.min_dist must be non-negative.")
    umap["fit_max_samples"] = _validate_positive_int(umap["fit_max_samples"], "interpretability.umap.fit_max_samples", allow_zero=True)
    umap["transform_batch_size"] = _validate_positive_int(umap["transform_batch_size"], "interpretability.umap.transform_batch_size")
    if umap["representations"] not in {"none", "consensus", "all"}:
        raise ValueError("interpretability.umap.representations must be none, consensus, or all.")
    if not isinstance(interpretability.get("save_embeddings"), bool):
        raise TypeError("interpretability.save_embeddings must be a YAML boolean.")

    logging_config = config["logging"]
    if "level" not in logging_config:
        raise KeyError("logging.level is required.")
    level = str(logging_config["level"]).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("logging.level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    logging_config["level"] = level
    if not isinstance(logging_config.get("retain_individual_run_logs"), bool):
        raise TypeError("logging.retain_individual_run_logs must be a YAML boolean.")

    stages = config["stages"]
    enabled = [str(x) for x in _as_nonempty_list(stages["enabled"], "stages.enabled")]
    unknown_stages = sorted(set(enabled).difference(STAGE_ORDER))
    if unknown_stages:
        raise ValueError(f"Unknown stages: {unknown_stages}. Valid stages: {list(STAGE_ORDER)}")
    if len(set(enabled)) != len(enabled):
        raise ValueError("stages.enabled contains duplicate stage names.")
    stages["enabled"] = [stage for stage in STAGE_ORDER if stage in enabled]
    _validate_yaml_bool(stages.get("skip_valid_completed"), "stages.skip_valid_completed")
    if not config["interpretability"]["enabled"] and "interpretability" in stages["enabled"]:
        stages["enabled"].remove("interpretability")

    run_root = Path(experiment["output_root"]) / experiment["name"] / experiment["version"]
    config["paths"] = {
        "run_root": str(run_root),
        "stage_state": str(run_root / "pipeline_state"),
        "logs": str(run_root / "logs"),
        "grid_root": str(run_root / "grid_search"),
        "candidate_root": str(run_root / "candidate_retraining"),
        "ensemble_root": str(run_root / "ensemble"),
        "report_root": str(run_root / "report"),
    }

    config_for_hash = copy.deepcopy(config)
    config_for_hash.pop("_meta", None)
    config["_meta"]["resolved_config_hash"] = object_sha256(config_for_hash)
    config["_meta"]["run_root"] = str(run_root)
    return to_jsonable(config)


def selection_group_count(config: Mapping[str, Any], selection: Mapping[str, Any]) -> int:
    """Return the number of distinct groups implied by a validated selection rule."""
    group_by = selection.get("group_by")
    if group_by is None:
        return 1
    model_key_map = {
        "architecture": "architectures",
        "in_channels": "in_channels",
        "pretrained": "pretrained",
    }
    if group_by in model_key_map:
        return len(config["model"][model_key_map[str(group_by)]])
    if group_by in config["search"]["space"]:
        return len(config["search"]["space"][group_by])
    raise ValueError(f"Cannot determine selection groups for {group_by!r}.")


def configuration_count(config: Mapping[str, Any]) -> int:
    model = config["model"]
    count = len(model["architectures"]) * len(model["in_channels"]) * len(model["pretrained"])
    for values in config["search"]["space"].values():
        count *= len(values)
    count *= int(config["search"]["repetitions_per_configuration"])
    return int(count)


def selected_stage_window(
    config: Mapping[str, Any],
    requested_stages: Sequence[str] | None,
    from_stage: str | None,
    to_stage: str | None,
) -> list[str]:
    enabled = list(config["stages"]["enabled"])
    if requested_stages:
        requested = [str(x) for x in requested_stages]
        unknown = sorted(set(requested).difference(STAGE_ORDER))
        if unknown:
            raise ValueError(f"Unknown requested stages: {unknown}")
        selected = [stage for stage in STAGE_ORDER if stage in requested]
    else:
        selected = enabled
    if from_stage:
        if from_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown --from-stage: {from_stage}")
        selected = [s for s in selected if STAGE_ORDER.index(s) >= STAGE_ORDER.index(from_stage)]
    if to_stage:
        if to_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown --to-stage: {to_stage}")
        selected = [s for s in selected if STAGE_ORDER.index(s) <= STAGE_ORDER.index(to_stage)]
    if not selected:
        raise ValueError("No stages remain after applying the requested stage filters.")
    return selected
