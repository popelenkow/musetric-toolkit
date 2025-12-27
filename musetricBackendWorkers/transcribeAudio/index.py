import json
import logging
import re
import sys
from pathlib import Path

from musetricBackendWorkers.common.logger import redirectStdStreams, setupLogging
from musetricBackendWorkers.transcribeAudio.cli import parse_arguments
from musetricBackendWorkers.transcribeAudio.response_builder import (
    build_payload_segments,
)
from musetricBackendWorkers.transcribeAudio.whisperx_runner import (
    transcribe_with_whisperx,
)


def build_stream_suppression_patterns(
    log_level: str,
) -> list[re.Pattern[str]]:
    if log_level == "debug":
        return []
    return [
        re.compile(r"\bwhisperx\.[\w\.]+ - INFO - "),
        re.compile(r"^Model was trained with "),
    ]


def main() -> None:
    args = parse_arguments()
    setupLogging(args.log_level)
    redirectStdStreams(
        suppress_patterns=build_stream_suppression_patterns(args.log_level)
    )

    try:
        segments, language = transcribe_with_whisperx(args.audio_path, args.log_level)
        payload_segments = build_payload_segments(segments)
        payload = {
            "type": "result",
            "language": language,
            "segments": payload_segments,
            "lines": payload_segments,
        }
        output_path = Path(args.result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as result_file:
            json.dump(payload, result_file)
    except Exception as error:
        logging.error("Audio transcription failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
