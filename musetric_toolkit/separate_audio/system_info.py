# ruff: noqa: S603
import logging
import shutil
import subprocess
from contextlib import suppress

import torch


def ensure_ffmpeg() -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.error(
            "FFmpeg not found in PATH. Please install and add ...\\ffmpeg\\bin to PATH."
        )
        raise FileNotFoundError("ffmpeg not found in PATH")
    try:
        subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
    except Exception:
        logging.error(
            "FFmpeg not found in PATH. Please install and add ...\\ffmpeg\\bin to PATH."
        )
        raise


def print_acceleration_info() -> None:
    logging.debug("=== Acceleration info ===")
    logging.debug(f"PyTorch: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    logging.debug(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.debug(f"CUDA device name: {device_name}")
    else:
        logging.warning("PyTorch built without CUDA or drivers/CUDA incompatible.")

    logging.debug("=========================")


def setup_torch_optimization() -> None:
    with suppress(Exception):
        torch.set_float32_matmul_precision("high")

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        logging.debug(f"Flash SDP enabled: {flash_enabled}")
    except Exception as error:
        logging.warning(f"Could not enable Flash SDP: {error}")
