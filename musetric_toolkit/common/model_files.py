import logging
import sys
from pathlib import Path


def ensureModelFiles(
    model_checkpoint_path: Path,
    config_path: Path,
) -> None:
    missing = []
    if not model_checkpoint_path.exists():
        missing.append(("Model checkpoint", model_checkpoint_path))
    if not config_path.exists():
        missing.append(("Model configuration", config_path))

    if missing:
        for label, path in missing:
            logging.error(
                "%s not found at %s. Run musetric-download-models first.",
                label,
                path,
            )
        sys.exit(1)
