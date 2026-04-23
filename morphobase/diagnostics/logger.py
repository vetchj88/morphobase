import json
from pathlib import Path
class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def log(self, record: dict) -> None:
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
