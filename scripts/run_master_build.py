import argparse
import json
from pathlib import Path

import yaml

from scripts.report_benchmark_robustness import (
    build_benchmark_robustness_report,
    write_benchmark_robustness_outputs,
)
from scripts.report_stack_d_robustness import (
    build_stack_d_robustness_report,
    write_stack_d_robustness_outputs,
)
from scripts.run_assay import run_config_path
from scripts.scaffold_assay_configs import scaffold_configs


DEFAULT_LADDER = Path('configs/build/master_ladder.yaml')


def _metric_passes(actual, minimum=None, maximum=None) -> bool:
    if minimum is not None and actual < minimum:
        return False
    if maximum is not None and actual > maximum:
        return False
    return True


def evaluate_gates(metrics: dict, gates: dict) -> tuple[bool, list[str]]:
    failures = []
    for metric_name, bounds in gates.items():
        actual = float(metrics.get(metric_name, 0.0))
        minimum = bounds.get('min')
        maximum = bounds.get('max')
        if not _metric_passes(actual, minimum=minimum, maximum=maximum):
            failures.append(
                f"{metric_name}={actual:.4f} outside [{minimum if minimum is not None else '-inf'}, {maximum if maximum is not None else 'inf'}]"
            )
    return (not failures, failures)


def load_ladder(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def run_ladder(ladder: dict, fail_fast: bool = True, scaffold_missing: bool = False) -> dict:
    if scaffold_missing:
        scaffold_configs(Path('configs/assay'), Path('configs/defaults.yaml'), overwrite=False)

    phase_results = []
    for phase in ladder.get('phases', []):
        phase_name = phase['name']
        phase_ok = True
        assay_results = []
        for step in phase.get('steps', []):
            config_path = Path(step['config'])
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config for {step['assay']}: {config_path}")

            execution = run_config_path(config_path)
            passed, failures = evaluate_gates(
                execution['result'].final_metrics,
                step.get('gates', {}),
            )
            if execution['verdict'] != 'pass':
                passed = False
                failures.append(f"verdict={execution['verdict']}")

            assay_results.append(
                {
                    'assay': step['assay'],
                    'config': str(config_path),
                    'passed': passed,
                    'verdict': execution['verdict'],
                    'alerts': execution['alerts'],
                    'failures': failures,
                    'metrics': execution['result'].final_metrics,
                    'artifacts_dir': str(execution['artifacts_dir']),
                }
            )
            phase_ok = phase_ok and passed
            if fail_fast and not phase_ok:
                break

        phase_results.append(
            {
                'name': phase_name,
                'description': phase.get('description', ''),
                'passed': phase_ok,
                'steps': assay_results,
            }
        )
        if fail_fast and not phase_ok:
            break

    overall = all(phase['passed'] for phase in phase_results) if phase_results else False
    return {'passed': overall, 'phases': phase_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ladder', default=str(DEFAULT_LADDER))
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--scaffold-missing-configs', action='store_true')
    parser.add_argument('--report', default='artifacts/master_build_report.json')
    args = parser.parse_args()

    ladder = load_ladder(Path(args.ladder))
    report = run_ladder(
        ladder=ladder,
        fail_fast=args.fail_fast,
        scaffold_missing=args.scaffold_missing_configs,
    )

    report_path = Path(args.report)
    artifacts_root = report_path.parent
    benchmark_robustness = build_benchmark_robustness_report(artifacts_root)
    robustness_json_path, robustness_markdown_path = write_benchmark_robustness_outputs(
        benchmark_robustness,
        artifacts_root,
    )
    stack_d_robustness = build_stack_d_robustness_report(artifacts_root)
    stack_d_json_path, stack_d_markdown_path = write_stack_d_robustness_outputs(
        stack_d_robustness,
        artifacts_root,
    )
    report['benchmark_robustness'] = benchmark_robustness
    report['benchmark_robustness_artifacts'] = {
        'json': str(robustness_json_path),
        'markdown': str(robustness_markdown_path),
    }
    report['stack_d_robustness'] = stack_d_robustness
    report['stack_d_robustness_artifacts'] = {
        'json': str(stack_d_json_path),
        'markdown': str(stack_d_markdown_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report_path)
    print('PASS' if report['passed'] else 'FAIL')


if __name__ == '__main__':
    main()
