import logging
import subprocess
from collections.abc import Iterable, Sequence

error_keywords: tuple[str, ...] = (
    "error",
    "fail",
    "invalid",
    "unable",
    "could not",
    "not found",
    "illegal",
)
warning_keywords: tuple[str, ...] = ("warn", "deprecated", "non monotone")


def _iter_lines(output: str) -> Iterable[str]:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line:
            yield line


def _log_ffmpeg_output(stderr_text: str) -> tuple[str | None, str | None]:
    logger = logging.getLogger(__name__)
    last_error: str | None = None
    last_warning: str | None = None

    for line in _iter_lines(stderr_text):
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in error_keywords):
            logger.error("ffmpeg: %s", line)
            last_error = line
        elif any(keyword in lower_line for keyword in warning_keywords):
            logger.warning("ffmpeg: %s", line)
            last_warning = line
        else:
            logger.debug("ffmpeg: %s", line)

    return last_error, last_warning


def run_ffmpeg(
    command: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    capture_stdout: bool = False,
    context: str,
) -> bytes:
    process = subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    stderr_bytes = process.stderr or b""
    stderr_text = stderr_bytes.decode("utf-8", errors="ignore")
    last_error, last_warning = _log_ffmpeg_output(stderr_text)

    if process.returncode != 0:
        summary = last_error or last_warning
        if not summary:
            summary = f"Process exited with code {process.returncode}"
        raise RuntimeError(f"{context}: {summary}")

    return process.stdout if capture_stdout else b""
