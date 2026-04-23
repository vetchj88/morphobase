from dataclasses import dataclass, field

@dataclass(slots=True)
class RunConfig:
    name: str = 'morphobase_run'
    output_dir: str = 'artifacts'
    save_plots: bool = True
    summary_name: str = 'summary.md'
    registry_name: str = 'run_registry.csv'
    seed: int = 42

@dataclass(slots=True)
class RuntimeConfig:
    total_steps: int = 64
    dt: float = 1.0
    log_every: int = 8

@dataclass(slots=True)
class BodyConfig:
    num_cells: int = 16
    hidden_dim: int = 8
    energy_init: float = 1.0
    stress_init: float = 0.0
    plasticity_init: float = 0.5
    z_alignment_init: float = 0.0

@dataclass(slots=True)
class AssayConfig:
    name: str = 'smoke'
    noise_scale: float = 0.01
    target_value: float = 0.75

@dataclass(slots=True)
class LoggingConfig:
    level: str = 'INFO'
    event_log_name: str = 'events.jsonl'

@dataclass(slots=True)
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    assay: AssayConfig = field(default_factory=AssayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
