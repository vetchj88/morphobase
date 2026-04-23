import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from morphobase.assays.registry import ASSAYS
from morphobase.config.validate import config_to_dict, load_config
from morphobase.diagnostics.alerts import classify_run, hard_fail_alerts
from morphobase.diagnostics.logger import JsonlLogger
from morphobase.diagnostics.plots import plot_scalar_history, plot_stage_occupancy
from morphobase.diagnostics.summaries import build_markdown_summary, write_summary
from morphobase.registry import append_run_row
from morphobase.seeds import set_seed


def classify_verdict(summary: dict) -> str:
    return classify_run(summary).value


def execute_assay(cfg) -> dict:
    set_seed(cfg.run.seed)
    out_dir = Path(cfg.run.output_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config_to_dict(cfg)
    (out_dir / 'resolved_config.yaml').write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False),
        encoding='utf-8',
    )

    logger = JsonlLogger(out_dir / cfg.logging.event_log_name)
    result = ASSAYS[cfg.assay.name]().run(cfg)
    for item in result.history:
        logger.log(item)

    if cfg.run.save_plots and result.history:
        plot_scalar_history(result.history, 'mean_energy', out_dir / 'mean_energy.png')
        plot_scalar_history(result.history, 'mean_stress', out_dir / 'mean_stress.png')
        plot_scalar_history(result.history, 'mean_z_alignment', out_dir / 'mean_z_alignment.png')
        plot_scalar_history(result.history, 'dormant_fraction', out_dir / 'plasticity_dormant_fraction.png')
        plot_stage_occupancy(result.history, out_dir / 'stage_occupancy.png')

    alerts = hard_fail_alerts(result.final_metrics)
    triggered = [alert.name for alert in alerts if alert.triggered]
    notes = result.notes
    if triggered:
        notes = f"{notes} Alerts triggered: {', '.join(triggered)}.".strip()

    write_summary(
        out_dir / cfg.run.summary_name,
        build_markdown_summary(cfg_dict, result.final_metrics, notes),
    )
    (out_dir / 'final_metrics.json').write_text(
        json.dumps(result.final_metrics, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    (out_dir / 'alerts.json').write_text(
        json.dumps([asdict(alert) for alert in alerts], indent=2),
        encoding='utf-8',
    )

    verdict = classify_verdict(result.final_metrics)
    append_run_row(
        Path(cfg.run.output_dir) / cfg.run.registry_name,
        {
            'date': datetime.now(UTC).isoformat(),
            'run_name': cfg.run.name,
            'assay': cfg.assay.name,
            'seed': cfg.run.seed,
            'verdict': verdict,
            'mean_energy': result.final_metrics.get('mean_energy', ''),
            'mean_stress': result.final_metrics.get('mean_stress', ''),
            'mean_z_alignment': result.final_metrics.get('mean_z_alignment', ''),
            'notes': notes,
        },
    )

    return {
        'config': cfg_dict,
        'config_obj': cfg,
        'result': result,
        'alerts': [asdict(alert) for alert in alerts],
        'verdict': verdict,
        'artifacts_dir': out_dir,
        'notes': notes,
    }


def run_config_path(config_path: str | Path) -> dict:
    return execute_assay(load_config(config_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    execution = run_config_path(args.config)
    cfg = execution['config_obj']
    print(f'Finished run: {cfg.run.name}')
    print(f"Artifacts: {execution['artifacts_dir']}")


if __name__ == '__main__':
    main()
