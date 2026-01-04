import collections
import inspect
import logging
import os
import re
import shutil
import sys
import typing
import warnings
from contextlib import contextmanager, suppress
from importlib.util import find_spec
from pathlib import Path

import omegaconf.base
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import whisperx
from lightning.pytorch import __version__ as pl_version
from lightning.pytorch.utilities.migration import migrate_checkpoint, pl_legacy_patch
from packaging.version import Version

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio.system_info import (
    print_acceleration_info,
    setup_torch_optimization,
)


def configure_warning_filters(log_level: str) -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if log_level == "debug":
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Module 'speechbrain\.pretrained' was deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
        category=UserWarning,
    )


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    prefixes = (
        "huggingface_hub",
        "lightning",
        "pytorch_lightning",
        "pyannote",
        "pyannote.audio",
        "speechbrain",
        "torchaudio",
        "whisperx",
    )

    def tune_logger(logger: logging.Logger) -> None:
        logger.setLevel(target_level)
        logger.propagate = True
        if logger.handlers:
            logger.handlers.clear()

    for logger_name in prefixes:
        tune_logger(logging.getLogger(logger_name))

    for logger_name, logger in logging.root.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        for prefix in prefixes:
            if logger_name == prefix or logger_name.startswith(f"{prefix}."):
                tune_logger(logger)
                break


def setup_environment(model_cache_dir: str, log_level: str) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HOME", model_cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", model_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", model_cache_dir)
    configure_warning_filters(log_level)
    configure_third_party_logging(log_level)


def configure_torch_serialization(torch) -> None:
    # Allowlist safe types used by OmegaConf/typing in PyTorch weights-only loads.
    torch.serialization.add_safe_globals(
        [
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
            typing.Any,
            list,
            dict,
            tuple,
            set,
            collections.defaultdict,
            collections.OrderedDict,
        ]
    )
    original_torch_load = torch.load

    def torch_load_unrestricted(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_unrestricted
    torch.serialization.load = torch_load_unrestricted


def maybe_upgrade_whisperx_checkpoint(torch) -> None:
    try:
        spec = find_spec("whisperx")
        if not spec or not spec.origin:
            return
        checkpoint_path = (
            Path(spec.origin).resolve().parent / "assets" / "pytorch_model.bin"
        )
        if not checkpoint_path.is_file():
            return

        with pl_legacy_patch():
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        ckpt_version = checkpoint.get("pytorch-lightning_version")
        if not ckpt_version:
            return
        if Version(ckpt_version) >= Version(pl_version):
            return

        backup_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.bak")
        if not backup_path.exists():
            shutil.copyfile(checkpoint_path, backup_path)

        migrate_checkpoint(checkpoint)
        torch.save(checkpoint, checkpoint_path)
    except Exception as error:
        logging.debug("WhisperX checkpoint upgrade skipped: %s", error)


DEFAULT_MODEL_NAME = "large-v3"
DEFAULT_BATCH_SIZE = 4
DEFAULT_LANGUAGE = None


_PROGRESS_PATTERN = re.compile(r"^Progress:\s+([0-9]+(?:\.[0-9]+)?)%")


class ProgressTracker:
    def __init__(self, min_delta: float = 0.01) -> None:
        self._min_delta = min_delta
        self._last = -1.0

    def report_fraction(self, progress: float) -> None:
        progress = max(0.0, min(progress, 1.0))
        if progress <= self._last:
            return
        if (
            self._last >= 0.0
            and progress < self._last + self._min_delta
            and progress < 1.0
        ):
            return
        self._last = progress
        send_message({"type": "progress", "progress": progress})

    def report_percent(self, percent: float) -> None:
        self.report_fraction(percent / 100.0)

    def ensure_minimum(self, progress: float) -> None:
        if progress > self._last:
            self.report_fraction(progress)

    def finalize(self) -> None:
        self.report_fraction(1.0)


class ProgressLineInterceptor:
    def __init__(self, stream, tracker: ProgressTracker) -> None:
        self._stream = stream
        self._tracker = tracker
        self._buffer = ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _maybe_report_progress(self, line: str) -> bool:
        match = _PROGRESS_PATTERN.match(line)
        if not match:
            return False
        try:
            percent = float(match.group(1))
        except ValueError:
            return False
        self._tracker.report_percent(percent)
        return True

    def write(self, message: str | bytes) -> int:
        if not message:
            return 0
        if isinstance(message, bytes):
            message = message.decode(self.encoding, errors=self.errors)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if self._maybe_report_progress(line):
                continue
            self._stream.write(line + "\n")
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line and not self._maybe_report_progress(line):
                self._stream.write(line)
            self._buffer = ""
        with suppress(Exception):
            self._stream.flush()

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        try:
            return self._stream.fileno()
        except Exception:
            return -1

    def writable(self) -> bool:
        return True


@contextmanager
def intercept_progress_lines(tracker: ProgressTracker):
    original_stdout = sys.stdout
    wrapper = ProgressLineInterceptor(original_stdout, tracker)
    sys.stdout = wrapper
    try:
        yield
    finally:
        sys.stdout = original_stdout
        wrapper.flush()


def ensure_model_cache_dir() -> str:
    envs.models_dir.mkdir(parents=True, exist_ok=True)
    return str(envs.models_dir)


def call_with_model_cache_dir(func, model_cache_dir: str, *args, **kwargs):
    params = inspect.signature(func).parameters
    if "download_root" in params:
        kwargs["download_root"] = model_cache_dir
    elif "model_dir" in params:
        kwargs["model_dir"] = model_cache_dir
    elif "cache_dir" in params:
        kwargs["cache_dir"] = model_cache_dir
    return func(*args, **kwargs)


def transcribe_with_whisperx(audio_path: str, log_level: str = "info"):
    model_cache_dir = ensure_model_cache_dir()
    setup_environment(model_cache_dir, log_level)

    configure_torch_serialization(torch)
    configure_third_party_logging(log_level)
    maybe_upgrade_whisperx_checkpoint(torch)
    print_acceleration_info()
    setup_torch_optimization()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = call_with_model_cache_dir(
        whisperx.load_model,
        model_cache_dir,
        DEFAULT_MODEL_NAME,
        device,
        compute_type=compute_type,
    )
    audio = whisperx.load_audio(audio_path)
    progress_tracker = ProgressTracker()
    progress_tracker.report_fraction(0.0)
    with intercept_progress_lines(progress_tracker):
        result = model.transcribe(
            audio,
            batch_size=DEFAULT_BATCH_SIZE,
            language=DEFAULT_LANGUAGE,
            print_progress=True,
            combined_progress=True,
        )
        progress_tracker.ensure_minimum(0.5)

    segments = result.get("segments", [])
    detected_language = result.get("language")

    try:
        align_model, metadata = call_with_model_cache_dir(
            whisperx.load_align_model,
            model_cache_dir,
            language_code=detected_language,
            device=device,
        )
        with intercept_progress_lines(progress_tracker):
            aligned = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
                print_progress=True,
                combined_progress=True,
            )
        segments = aligned.get("segments", segments)
        detected_language = aligned.get("language", detected_language)
    except Exception as align_error:
        logging.warning("Alignment skipped: %s", align_error)
    finally:
        progress_tracker.finalize()

    return segments, detected_language
