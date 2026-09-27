from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .common import utc_now


@dataclass(frozen=True)
class GPUProcess:
    pid: str
    process_name: str
    used_memory: str | None = None


@dataclass
class GPUAvailabilityState:
    free_since: dict[str, float] = field(default_factory=dict)
    last_report: dict[str, float] = field(default_factory=dict)
    last_signature: dict[str, str] = field(default_factory=dict)


def _guard_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("runtime", {}).get("gpu_wait", {})


def guard_enabled(config: Mapping[str, Any]) -> bool:
    if str(config.get("runtime", {}).get("device", "cuda")) == "cpu":
        return False
    return bool(_guard_config(config).get("enabled", False))


def _nvidia_smi(config: Mapping[str, Any]) -> str:
    executable = str(_guard_config(config).get("nvidia_smi", "nvidia-smi"))
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    # Allow an absolute executable path supplied in YAML.
    if executable.startswith("/"):
        return executable
    raise FileNotFoundError(
        f"GPU guard is enabled but NVIDIA SMI executable was not found: {executable!r}"
    )


def query_gpu_processes(config: Mapping[str, Any], gpu_token: str) -> list[GPUProcess]:
    """Return compute applications reported by NVIDIA SMI for one GPU token.

    This intentionally relies on the NVIDIA driver rather than `ps`, because the
    server can hide other users' processes from /proc while nvidia-smi can still
    report the GPU workload.
    """
    executable = _nvidia_smi(config)
    base = [executable, "-i", str(gpu_token)]
    queries = [
        "pid,process_name,used_gpu_memory",
        "pid,process_name,used_memory",
        "pid,process_name",
    ]
    last_error = ""
    for query in queries:
        command = base + [f"--query-compute-apps={query}", "--format=csv,noheader,nounits"]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            last_error = completed.stderr.strip() or completed.stdout.strip()
            continue
        rows: list[GPUProcess] = []
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if not parts or not parts[0] or parts[0].lower() in {"n/a", "[not supported]"}:
                continue
            rows.append(
                GPUProcess(
                    pid=parts[0],
                    process_name=parts[1] if len(parts) > 1 else "unknown",
                    used_memory=parts[2] if len(parts) > 2 else None,
                )
            )
        return rows
    raise RuntimeError(
        f"nvidia-smi could not query compute applications for GPU {gpu_token}. "
        f"Last error: {last_error or 'unknown error'}"
    )


def describe_processes(processes: Sequence[GPUProcess]) -> str:
    if not processes:
        return "none"
    chunks = []
    for item in processes:
        text = f"pid={item.pid} {item.process_name}"
        if item.used_memory not in (None, "", "N/A"):
            text += f" memory={item.used_memory}MiB"
        chunks.append(text)
    return "; ".join(chunks)


def ready_gpu_tokens(
    config: Mapping[str, Any],
    tokens: Sequence[str],
    state: GPUAvailabilityState,
) -> list[str]:
    """Return tokens that have remained process-free for the configured grace time."""
    if not guard_enabled(config):
        return list(tokens)

    settings = _guard_config(config)
    stable_seconds = max(0, int(settings.get("stable_seconds", 60)))
    report_every = max(1, int(settings.get("report_every_seconds", 60)))
    now = time.monotonic()
    ready: list[str] = []

    for token in tokens:
        processes = query_gpu_processes(config, str(token))
        signature = describe_processes(processes)
        last_signature = state.last_signature.get(str(token))
        last_report = state.last_report.get(str(token), 0.0)
        should_report = signature != last_signature or (now - last_report) >= report_every

        if processes:
            state.free_since.pop(str(token), None)
            if should_report:
                print(
                    f"[{utc_now()}] [GPU GUARD] GPU {token} BUSY: {signature}",
                    flush=True,
                )
        else:
            free_since = state.free_since.setdefault(str(token), now)
            free_for = max(0.0, now - free_since)
            if free_for >= stable_seconds:
                ready.append(str(token))
                if should_report:
                    print(
                        f"[{utc_now()}] [GPU GUARD] GPU {token} FREE "
                        f"for {free_for:.0f}s (required {stable_seconds}s).",
                        flush=True,
                    )
            elif should_report:
                print(
                    f"[{utc_now()}] [GPU GUARD] GPU {token} currently free; "
                    f"stability {free_for:.0f}/{stable_seconds}s.",
                    flush=True,
                )

        state.last_signature[str(token)] = signature
        if should_report:
            state.last_report[str(token)] = now

    return ready


def wait_for_gpu_tokens(
    config: Mapping[str, Any],
    tokens: Sequence[str],
    required_count: int = 1,
) -> list[str]:
    """Wait until at least required_count requested GPUs are stably free.

    timeout_seconds=0 means wait indefinitely. This is useful on shared workstations
    where GPU jobs may also be launched outside this pipeline.
    """
    if required_count < 1:
        return []
    if required_count > len(tokens):
        raise ValueError(
            f"Requested {required_count} free GPUs but only {len(tokens)} tokens are configured: {list(tokens)}"
        )
    if not guard_enabled(config):
        return list(tokens[:required_count])

    settings = _guard_config(config)
    poll_seconds = max(1, int(settings.get("poll_seconds", 30)))
    timeout_seconds = max(0, int(settings.get("timeout_seconds", 0)))
    state = GPUAvailabilityState()
    started = time.monotonic()
    print(
        f"[{utc_now()}] [GPU GUARD] Waiting for {required_count} free GPU(s) among {list(tokens)}. "
        f"poll={poll_seconds}s stable={int(settings.get('stable_seconds', 60))}s "
        f"timeout={'infinite' if timeout_seconds == 0 else str(timeout_seconds) + 's'}",
        flush=True,
    )

    while True:
        ready = ready_gpu_tokens(config, tokens, state)
        if len(ready) >= required_count:
            selected = ready[:required_count]
            print(
                f"[{utc_now()}] [GPU GUARD] Selected free GPU token(s): {selected}",
                flush=True,
            )
            return selected
        if timeout_seconds and (time.monotonic() - started) >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for {required_count} free GPU(s) among {list(tokens)}."
            )
        time.sleep(poll_seconds)
