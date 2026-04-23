from pathlib import Path
import json, numpy as np
from morphobase.organism.state import OrganismState

def save_snapshot(state: OrganismState, path: str | Path) -> None:
    payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.__dict__.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
