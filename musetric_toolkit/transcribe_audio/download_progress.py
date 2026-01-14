from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from huggingface_hub import file_download
from huggingface_hub import utils as hf_utils
from tqdm.auto import tqdm as base_tqdm

from musetric_toolkit.common.logger import send_message

DOWNLOAD_REPORT_BYTES = 1024 * 1024


class DownloadProgressTqdm:
    def __init__(
        self,
        *args,
        label: str,
        report_delta_bytes: int = DOWNLOAD_REPORT_BYTES,
        activity: dict[str, bool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("name", None)
        self._tqdm = base_tqdm(*args, **kwargs)
        self._label = label
        self._file = kwargs.get("desc") or getattr(self._tqdm, "desc", None)
        self._total = kwargs.get("total")
        if self._total is None:
            self._total = getattr(self._tqdm, "total", None)
        if self._total is not None and self._total <= 0:
            self._total = None
        self._downloaded = int(getattr(self._tqdm, "n", 0))
        self._raw_downloaded = self._downloaded
        self._last_reported = -1
        self._report_delta = max(int(report_delta_bytes), 1)
        self._activity = activity
        unit = kwargs.get("unit") or getattr(self._tqdm, "unit", None)
        self._should_report = unit == "B"
        self._closed = False
        if self._should_report:
            self._emit(status="processing", force=True)

    def _emit(self, status: str | None = None, force: bool = False) -> None:
        if not self._should_report:
            return
        if (
            not force
            and self._last_reported >= 0
            and self._downloaded - self._last_reported < self._report_delta
            and status != "done"
        ):
            return
        if status is None:
            if self._total is not None and self._downloaded >= self._total:
                status = "done"
            else:
                status = "processing"
        self._last_reported = self._downloaded
        payload = {
            "type": "download",
            "label": self._label,
            "downloaded": self._downloaded,
            "status": status,
        }
        if self._file:
            payload["file"] = self._file
        if self._total is not None:
            payload["total"] = self._total
        if self._activity is not None:
            self._activity["emitted"] = True
        send_message(payload)

    def update(self, n: int = 1) -> None:
        self._tqdm.update(n)
        delta = 0
        try:
            delta = int(n)
        except (TypeError, ValueError):
            delta = 0
        if delta:
            self._raw_downloaded += delta
        current = int(getattr(self._tqdm, "n", self._downloaded))
        self._downloaded = max(self._raw_downloaded, current)
        self._emit()

    def __iter__(self):
        for item in self._tqdm:
            yield item
            current = int(getattr(self._tqdm, "n", self._downloaded))
            self._downloaded = max(self._raw_downloaded, current)
            self._emit()
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tqdm.close()
        self._emit(status="done", force=True)

    def __enter__(self):
        self._tqdm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._tqdm.__exit__(exc_type, exc, tb)
        finally:
            self.close()

    def __getattr__(self, name):
        return getattr(self._tqdm, name)


@contextmanager
def intercept_hf_downloads(label: str) -> Iterator[None]:
    activity = {"emitted": False}
    error_raised = False
    hf_tqdm_module = None
    original_utils_tqdm = None
    original_module_tqdm = None
    original_file_tqdm = None
    patched = False

    try:
        hf_tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")
        original_utils_tqdm = getattr(hf_utils, "tqdm", None)
        original_module_tqdm = getattr(hf_tqdm_module, "tqdm", None)
        original_file_tqdm = getattr(file_download, "tqdm", None)
        if original_utils_tqdm is not None and original_module_tqdm is not None:

            def _tqdm(*args, **kwargs):
                name = kwargs.get("name")
                if name is not None and hf_tqdm_module.are_progress_bars_disabled(name):
                    kwargs["disable"] = True
                return DownloadProgressTqdm(
                    *args,
                    label=label,
                    activity=activity,
                    **kwargs,
                )

            hf_utils.tqdm = _tqdm
            hf_tqdm_module.tqdm = _tqdm
            if original_file_tqdm is not None:
                file_download.tqdm = _tqdm
            patched = True
    except Exception:
        hf_tqdm_module = None
        patched = False

    try:
        yield
    except Exception:
        error_raised = True
        raise
    finally:
        if patched and hf_tqdm_module is not None:
            hf_utils.tqdm = original_utils_tqdm
            hf_tqdm_module.tqdm = original_module_tqdm
            if original_file_tqdm is not None:
                file_download.tqdm = original_file_tqdm
        if not activity["emitted"] and not error_raised:
            send_message(
                {
                    "type": "download",
                    "label": label,
                    "downloaded": 0,
                    "status": "cached",
                }
            )
