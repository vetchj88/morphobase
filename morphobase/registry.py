import csv
from pathlib import Path

FIELDS = ['date','run_name','assay','seed','verdict','mean_energy','mean_stress','mean_z_alignment','notes']

def append_run_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in FIELDS})
