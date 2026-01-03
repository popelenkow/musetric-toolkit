import json
import sys


def report_progress(progress: float) -> None:
    data = {"type": "progress", "progress": progress}
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()
