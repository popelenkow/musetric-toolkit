import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe vocals with WhisperX")
    parser.add_argument("--audio-path", required=True, help="Path to vocal audio file")
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path to write transcription JSON result",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    return parser.parse_args()
