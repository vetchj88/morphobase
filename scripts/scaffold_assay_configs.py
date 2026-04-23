import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from morphobase.assays.registry import ASSAYS


ASSAY_OVERRIDES = {
    "smoke": {"runtime": {"total_steps": 96}, "body": {"num_cells": 25}, "assay": {"noise_scale": 0.02, "target_value": 0.75}},
    "identity": {"runtime": {"total_steps": 128}, "body": {"num_cells": 25}, "assay": {"noise_scale": 0.01, "target_value": 0.8}},
    "wound_closure": {"runtime": {"total_steps": 160}, "body": {"num_cells": 36}, "assay": {"noise_scale": 0.015, "target_value": 0.78}},
    "stress_recruitment": {"runtime": {"total_steps": 160}, "body": {"num_cells": 36}, "assay": {"noise_scale": 0.02, "target_value": 0.8}},
    "growth_usefulness": {"runtime": {"total_steps": 192}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.02, "target_value": 0.82}},
    "compensation_block": {"runtime": {"total_steps": 192}, "body": {"num_cells": 36}, "assay": {"noise_scale": 0.015, "target_value": 0.8}},
    "sequential_rules": {"runtime": {"total_steps": 224}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.02, "target_value": 0.83}},
    "plasticity_stress": {"runtime": {"total_steps": 224}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.03, "target_value": 0.8}},
    "setpoint_rewrite": {"runtime": {"total_steps": 256}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.02, "target_value": 0.84}},
    "lightcone": {"runtime": {"total_steps": 224}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.02, "target_value": 0.8}},
    "lesion_battery": {"runtime": {"total_steps": 256}, "body": {"num_cells": 64}, "assay": {"noise_scale": 0.025, "target_value": 0.84}},
    "mnist_sanity": {"runtime": {"total_steps": 64}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.01, "target_value": 0.75}},
    "split_mnist": {"runtime": {"total_steps": 64}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.01, "target_value": 0.75}},
    "port_remap": {"runtime": {"total_steps": 224}, "body": {"num_cells": 49}, "assay": {"noise_scale": 0.02, "target_value": 0.82}},
}


def merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_assay_config(defaults: dict, assay_name: str) -> dict:
    config = merge_dicts(defaults, ASSAY_OVERRIDES.get(assay_name, {}))
    config.setdefault("run", {})
    config.setdefault("assay", {})
    config["run"]["name"] = f"{assay_name}_assay"
    config["assay"]["name"] = assay_name
    return config


def scaffold_configs(destination: Path, defaults_path: Path, overwrite: bool = False) -> list[Path]:
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    destination.mkdir(parents=True, exist_ok=True)

    written = []
    for assay_name in sorted(ASSAYS):
        target_path = destination / f"{assay_name}.yaml"
        if target_path.exists() and not overwrite:
            continue
        payload = build_assay_config(defaults, assay_name)
        target_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        written.append(target_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", default="configs/defaults.yaml")
    parser.add_argument("--destination", default="configs/assay")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = scaffold_configs(
        destination=Path(args.destination),
        defaults_path=Path(args.defaults),
        overwrite=args.overwrite,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
