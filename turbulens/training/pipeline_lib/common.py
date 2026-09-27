from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_text_write(text: str, path: Path) -> None:
    _atomic_write_bytes(text.encode("utf-8"), path)


def atomic_json_dump(data: Any, path: Path, indent: int = 2) -> None:
    text = json.dumps(
        to_jsonable(data), indent=indent, sort_keys=True, allow_nan=False
    ) + "\n"
    atomic_text_write(text, path)


def atomic_yaml_dump(data: Any, path: Path) -> None:
    text = yaml.safe_dump(to_jsonable(data), sort_keys=False, allow_unicode=True)
    atomic_text_write(text, path)


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def atomic_csv_write(rows: Sequence[Mapping[str, Any]], path: Path, fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: to_jsonable(v) for k, v in row.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_empty_or_matching_directory(path: Path, config_hash: str, overwrite: bool = False) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".run_config_hash"
    if marker.is_file():
        saved = marker.read_text(encoding="utf-8").strip()
        if saved != config_hash:
            raise ValueError(
                f"Output directory already belongs to a different configuration: {path}\n"
                f"saved={saved}\ncurrent={config_hash}\n"
                "Use a new experiment version or explicitly request overwrite."
            )
    elif any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is non-empty but has no configuration marker: {path}. "
            "Refusing to adopt or overwrite it automatically."
        )
    atomic_text_write(config_hash + "\n", marker)


class ExclusiveFileLock:
    """Simple process lock suitable for one shared Linux filesystem."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd: int | None = None
        for attempt in range(2):
            try:
                fd = os.open(self.path, flags, 0o644)
                break
            except FileExistsError as exc:
                holder_text = "unknown"
                holder: Mapping[str, Any] | None = None
                with contextlib.suppress(Exception):
                    holder_text = self.path.read_text(encoding="utf-8").strip()
                    parsed = json.loads(holder_text)
                    if isinstance(parsed, Mapping):
                        holder = parsed
                stale = False
                if holder is not None and str(holder.get("host")) == socket.gethostname():
                    with contextlib.suppress(Exception):
                        pid = int(holder["pid"])
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            stale = True
                        except PermissionError:
                            stale = False
                if stale and attempt == 0:
                    with contextlib.suppress(FileNotFoundError):
                        self.path.unlink()
                    continue
                raise RuntimeError(
                    f"Another pipeline process appears to hold the lock {self.path}: {holder_text}"
                ) from exc
        if fd is None:
            raise RuntimeError(f"Could not acquire pipeline lock: {self.path}")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now()}))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
            self.acquired = False

    def __enter__(self) -> "ExclusiveFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def valid_status_marker(path: Path, expected_status: str = "completed") -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    return isinstance(payload, Mapping) and payload.get("status") == expected_status


def command_string(command: Sequence[str]) -> str:
    import shlex
    return " ".join(shlex.quote(str(item)) for item in command)


def run_checked(
    command: Sequence[str],
    log_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return_code: int | None = None
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] COMMAND: {command_string(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait()
        except BaseException:
            with contextlib.suppress(Exception):
                process.terminate()
            try:
                process.wait(timeout=30)
            except Exception:
                with contextlib.suppress(Exception):
                    process.kill()
                with contextlib.suppress(Exception):
                    process.wait(timeout=5)
            log.write(f"[{utc_now()}] INTERRUPTED_EXIT_CODE: {process.returncode}\n")
            log.flush()
            raise
        log.write(f"[{utc_now()}] EXIT_CODE: {return_code}\n")
        log.flush()
    if return_code is None:
        raise RuntimeError("Subprocess finished without a return code.")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def capture_environment(include_pip_freeze: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONNOUSERSITE"
            )
        },
    }
    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            info["torch"]["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    if include_pip_freeze:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                check=False, timeout=120
            )
            info["pip_freeze"] = result.stdout.splitlines() if result.returncode == 0 else []
            if result.returncode != 0:
                info["pip_freeze_error"] = result.stderr.strip()
        except Exception as exc:
            info["pip_freeze_error"] = f"{type(exc).__name__}: {exc}"
    return info


def remove_if_exists(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        with contextlib.suppress(OSError):
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
    return total


def format_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
