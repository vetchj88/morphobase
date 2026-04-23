from dataclasses import asdict
from pathlib import Path
import yaml
from morphobase.config.schema import ExperimentConfig, RunConfig, RuntimeConfig, BodyConfig, AssayConfig, LoggingConfig

def _merge(cls, payload):
    return cls(**(payload or {}))

def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    cfg = ExperimentConfig(
        run=_merge(RunConfig, raw.get('run')),
        runtime=_merge(RuntimeConfig, raw.get('runtime')),
        body=_merge(BodyConfig, raw.get('body')),
        assay=_merge(AssayConfig, raw.get('assay')),
        logging=_merge(LoggingConfig, raw.get('logging')),
    )
    validate_config(cfg)
    return cfg

def validate_config(cfg: ExperimentConfig) -> None:
    assert cfg.body.num_cells > 0
    assert cfg.body.hidden_dim > 0
    assert cfg.runtime.total_steps > 0
    assert cfg.runtime.log_every > 0
    assert cfg.assay.name

def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)
