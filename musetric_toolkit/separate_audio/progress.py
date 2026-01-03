import json


def report_progress(progress: float) -> None:
    data = {"type": "progress", "progress": progress}
    print(json.dumps(data), flush=True)
