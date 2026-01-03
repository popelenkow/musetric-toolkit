import logging
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from musetric_toolkit.common import envs

CHUNK_SIZE = 1024 * 1024
PROGRESS_WIDTH = 40


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, *, use_color: bool = True) -> None:
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        original = record.levelname
        color = self.COLORS.get(record.levelno)
        if color:
            record.levelname = f"{color}{original}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def download_file(url: str, destination: Path, force: bool, label: str) -> None:
    if destination.exists() and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url) as response, destination.open("wb") as target:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                target.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress = downloaded / total
                    filled = int(PROGRESS_WIDTH * progress)
                    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
                    message = (
                        f"{label}: [{bar}] {progress * 100:5.1f}% "
                        f"({format_bytes(downloaded)}/{format_bytes(total)})"
                    )
                else:
                    message = f"{label}: {format_bytes(downloaded)} downloaded"
                sys.stdout.write("\r" + message)
                sys.stdout.flush()
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Failed to download {label}: {error}") from error
    finally:
        sys.stdout.write("\n")


def ensure_model_files() -> None:
    envs.models_dir.mkdir(parents=True, exist_ok=True)

    for url, path, label in (
        (envs.model_checkpoint_url, envs.model_checkpoint_path, "Model checkpoint"),
        (envs.model_config_url, envs.model_config_path, "Model configuration"),
    ):
        if path.exists():
            logging.info("%s file already exists at %s", label, path)
            continue

        download_file(url, path, False, label)
        logging.info("%s file downloaded to %s", label, path)


def main() -> None:
    handler = logging.StreamHandler()
    use_color = getattr(handler.stream, "isatty", lambda: False)()
    handler.setFormatter(
        ColorFormatter("%(levelname)s %(message)s", use_color=use_color)
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
    )

    try:
        ensure_model_files()
    except Exception as error:
        logging.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
