import json
import sys


def reportProgress(progress: float) -> None:
    data = {"type": "progress", "progress": progress}
    stream = sys.__stdout__ if getattr(sys, "__stdout__", None) else sys.stdout
    stream.write(json.dumps(data) + "\n")
    stream.flush()
