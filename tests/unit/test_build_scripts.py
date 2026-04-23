from pathlib import Path

import yaml

from morphobase.assays.lesion_battery import LesionBatteryAssay
from morphobase.assays.lesion_gridworld_remap import LesionGridworldRemapAssay
from morphobase.assays.lesion_preserves_competence import LesionPreservesCompetenceAssay
from morphobase.assays.lesion_sequential_rules import LesionSequentialRulesAssay
from morphobase.assays.lesion_split_mnist import LesionSplitMNISTAssay
from morphobase.assays.lightcone import LightconeAssay
from morphobase.assays.mnist_sanity import MNISTSanityAssay
from morphobase.assays.oscillatory_coupling_probe import OscillatoryCouplingProbeAssay
from morphobase.assays.permuted_fashion_mnist import PermutedFashionMNISTAssay
from morphobase.assays.permuted_mnist import PermutedMNISTAssay
from morphobase.assays.port_remap import PortRemapAssay
from morphobase.assays.predictive_coding_probe import PredictiveCodingProbeAssay
from morphobase.assays.reaction_diffusion_probe import ReactionDiffusionProbeAssay
from morphobase.assays.sequential_rules import SequentialRulesAssay
from morphobase.assays.stigmergic_highway_probe import StigmergicHighwayProbeAssay
from morphobase.assays.setpoint_rewrite import SetpointRewriteAssay
from morphobase.assays.split_fashion_mnist import SplitFashionMNISTAssay
from morphobase.assays.split_mnist import SplitMNISTAssay
from morphobase.assays.tissue_field_probe import TissueFieldProbeAssay
from morphobase.assays.growth_usefulness import GrowthUsefulnessAssay
from morphobase.assays.gridworld_remap import GridworldRemapAssay
from morphobase.config.validate import load_config
from morphobase.diagnostics.alerts import classify_run
from morphobase.seeds import set_seed
from scripts.export_summary import build_combined_summary
from scripts.report_benchmark_robustness import (
    build_benchmark_robustness_report,
    build_benchmark_robustness_markdown,
)
from scripts.report_stack_d_robustness import (
    build_stack_d_robustness_markdown,
    build_stack_d_robustness_report,
)
from scripts.rank_split_mnist_sweeps import build_markdown_report, rank_runs
from scripts.run_gridworld_remap_ablations import summarize_ablation_runs as summarize_gridworld_ablation_runs
from scripts.run_gridworld_remap_seed_robustness import summarize_seed_runs as summarize_gridworld_seed_runs
from scripts.run_permuted_mnist_ablation_seed_robustness import summarize_seed_reports as summarize_permuted_growth_probe_seed_reports
from scripts.run_permuted_mnist_ablations import summarize_ablation_runs as summarize_permuted_ablation_runs
from scripts.run_permuted_fashion_mnist_ablations import summarize_ablation_runs as summarize_permuted_fashion_ablation_runs
from scripts.run_permuted_fashion_mnist_seed_robustness import summarize_seed_runs as summarize_permuted_fashion_seed_runs
from scripts.run_sequential_rules_ablations import summarize_ablation_runs as summarize_sequential_rules_ablation_runs
from scripts.run_sequential_rules_seed_robustness import summarize_seed_runs as summarize_sequential_rules_seed_runs
from scripts.run_split_fashion_mnist_ablations import summarize_ablation_runs as summarize_split_fashion_ablation_runs
from scripts.run_split_fashion_mnist_seed_robustness import summarize_seed_runs as summarize_split_fashion_seed_runs
from scripts.run_split_mnist_ablations import summarize_ablation_runs
from scripts.run_split_mnist_seed_robustness import summarize_seed_runs
from scripts.run_master_build import evaluate_gates, load_ladder
from scripts.scaffold_assay_configs import build_assay_config, scaffold_configs


def test_build_assay_config_sets_name_and_run_name():
    defaults = yaml.safe_load(Path('configs/defaults.yaml').read_text(encoding='utf-8'))
    payload = build_assay_config(defaults, 'lightcone')
    assert payload['assay']['name'] == 'lightcone'
    assert payload['run']['name'] == 'lightcone_assay'


def test_scaffold_configs_writes_missing_files(tmp_path):
    defaults_path = tmp_path / 'defaults.yaml'
    defaults_path.write_text(Path('configs/defaults.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    written = scaffold_configs(tmp_path / 'assay', defaults_path, overwrite=False)
    assert written
    assert (tmp_path / 'assay' / 'smoke.yaml').exists()
    assert (tmp_path / 'assay' / 'setpoint_rewrite.yaml').exists()


def test_evaluate_gates_reports_failure():
    passed, failures = evaluate_gates({'mean_energy': 0.1}, {'mean_energy': {'min': 0.2}})
    assert not passed
    assert failures


def test_load_ladder_has_phases():
    ladder = load_ladder(Path('configs/build/master_ladder.yaml'))
    assert ladder['phases']


def test_build_combined_summary_collects_sections(tmp_path):
    run_dir = tmp_path / 'run_a'
    run_dir.mkdir()
    (run_dir / 'summary.md').write_text('# Summary\n', encoding='utf-8')
    combined = build_combined_summary(tmp_path)
    assert 'Combined Run Summaries' in combined
    assert 'run_a' in combined


def test_build_benchmark_robustness_report_summarizes_stack_c(tmp_path):
    (tmp_path / 'split_mnist_seed_robustness.json').write_text(
        '{"summary":{"seed_count":5,"final_accuracy":{"mean":0.36,"std":0.04,"min":0.30},"forgetting":{"max":0.18},"bwt":{"mean":-0.13},"primary_score":{"mean":0.37},"stable_across_seeds":true}}',
        encoding='utf-8',
    )
    (tmp_path / 'split_mnist_ablations.json').write_text(
        '{"summary":{"mechanism_dependency_supported_count":3.0,"mechanism_dependency_supported_fraction":1.0}}',
        encoding='utf-8',
    )
    (tmp_path / 'permuted_mnist_seed_robustness.json').write_text(
        '{"summary":{"seed_count":5,"final_accuracy":{"mean":0.26,"std":0.03,"min":0.21},"forgetting":{"max":0.06},"bwt":{"mean":-0.05},"primary_score":{"mean":0.30},"stable_across_seeds":false}}',
        encoding='utf-8',
    )
    (tmp_path / 'permuted_mnist_ablations.json').write_text(
        '{"summary":{"mechanism_dependency_supported_count":3.0,"mechanism_dependency_supported_fraction":1.0}}',
        encoding='utf-8',
    )
    (tmp_path / 'permuted_mnist_growth_probe_seed_robustness.json').write_text(
        '{"summary":{"stable_across_seeds":true,"all_mechanisms_supported_fraction":1.0}}',
        encoding='utf-8',
    )
    (tmp_path / 'split_fashion_mnist_seed_robustness.json').write_text(
        '{"summary":{"seed_count":5,"final_accuracy":{"mean":0.47,"std":0.05,"min":0.42},"forgetting":{"max":0.21},"bwt":{"mean":-0.14},"primary_score":{"mean":0.53},"stable_across_seeds":true}}',
        encoding='utf-8',
    )
    (tmp_path / 'split_fashion_mnist_ablations.json').write_text(
        '{"summary":{"mechanism_dependency_supported_count":3.0,"mechanism_dependency_supported_fraction":1.0}}',
        encoding='utf-8',
    )
    (tmp_path / 'permuted_fashion_mnist_seed_robustness.json').write_text(
        '{"summary":{"seed_count":5,"final_accuracy":{"mean":0.27,"std":0.02,"min":0.24},"forgetting":{"max":0.04},"bwt":{"mean":0.02},"primary_score":{"mean":0.37},"stable_across_seeds":true}}',
        encoding='utf-8',
    )
    (tmp_path / 'permuted_fashion_mnist_ablations.json').write_text(
        '{"summary":{"mechanism_dependency_supported_count":3.0,"mechanism_dependency_supported_fraction":1.0}}',
        encoding='utf-8',
    )

    report = build_benchmark_robustness_report(tmp_path)
    assert report['summary']['benchmark_count'] == 4.0
    assert report['summary']['stable_seed_benchmark_count'] == 3.0
    assert report['summary']['full_mechanism_support_count'] == 4.0
    assert report['summary']['ready_benchmark_count'] == 3.0
    assert report['summary']['attention_required_count'] == 1.0
    permuted_mnist = next(item for item in report['benchmarks'] if item['assay'] == 'permuted_mnist')
    assert permuted_mnist['status'] == 'causal_chamber_only'
    assert permuted_mnist['mechanism_chamber_seed_stable'] is True
    markdown = build_benchmark_robustness_markdown(report)
    assert 'Stack C Benchmark Robustness' in markdown
    assert 'Permuted-MNIST' in markdown


def test_build_stack_d_robustness_report_summarizes_stack_d(tmp_path):
    (tmp_path / 'gridworld_remap_seed_robustness.json').write_text(
        '{"summary":{"seed_count":5,"final_success":{"mean":0.86,"std":0.04,"min":0.79},"forgetting":{"max":0.06},"stable_across_seeds":true}}',
        encoding='utf-8',
    )
    (tmp_path / 'gridworld_remap_ablations.json').write_text(
        '{"summary":{"mechanism_dependency_supported_count":2.0,"mechanism_dependency_supported_fraction":1.0}}',
        encoding='utf-8',
    )
    (tmp_path / 'sequential_rules_seed_robustness.json').write_text(
        '{"summary":{"seed_count":5,"final_accuracy":{"mean":0.79,"std":0.03,"min":0.74},"forgetting":{"max":0.07},"stable_across_seeds":true}}',
        encoding='utf-8',
    )
    (tmp_path / 'sequential_rules_ablations.json').write_text(
        '{"summary":{"mechanism_dependency_supported_count":2.0,"mechanism_dependency_supported_fraction":1.0}}',
        encoding='utf-8',
    )

    report = build_stack_d_robustness_report(tmp_path)
    assert report['summary']['assay_count'] == 2.0
    assert report['summary']['stable_seed_assay_count'] == 2.0
    assert report['summary']['ready_assay_count'] == 2.0
    markdown = build_stack_d_robustness_markdown(report)
    assert 'Stack D Robustness' in markdown
    assert 'Gridworld Remap' in markdown


def test_rank_split_mnist_sweeps_orders_by_primary_metrics(tmp_path):
    better = tmp_path / 'split_mnist_good'
    better.mkdir()
    (better / 'resolved_config.yaml').write_text(
        'run:\n  name: split_mnist_good\n  seed: 1\nassay:\n  name: split_mnist\n',
        encoding='utf-8',
    )
    (better / 'final_metrics.json').write_text(
        '{"split_mnist_final_accuracy_mean": 0.45, "split_mnist_peak_accuracy_mean": 0.50, "split_mnist_mean_forgetting": 0.10, "split_mnist_bwt": -0.05, "split_mnist_mean_margin": 0.08, "mean_energy": 0.9, "mean_stress": 0.05, "mean_plasticity": 0.5, "mean_z_alignment": 0.2}',
        encoding='utf-8',
    )

    worse = tmp_path / 'split_mnist_bad'
    worse.mkdir()
    (worse / 'resolved_config.yaml').write_text(
        'run:\n  name: split_mnist_bad\n  seed: 2\nassay:\n  name: split_mnist\n',
        encoding='utf-8',
    )
    (worse / 'final_metrics.json').write_text(
        '{"split_mnist_final_accuracy_mean": 0.30, "split_mnist_peak_accuracy_mean": 0.40, "split_mnist_mean_forgetting": 0.25, "split_mnist_bwt": -0.20, "split_mnist_mean_margin": 0.04, "mean_energy": 0.95, "mean_stress": 0.04, "mean_plasticity": 0.6, "mean_z_alignment": 0.3}',
        encoding='utf-8',
    )

    ranked = rank_runs(tmp_path)
    assert ranked[0]['run_name'] == 'split_mnist_good'
    report = build_markdown_report(ranked)
    assert 'Split-MNIST Sweep Ranking' in report
    assert 'split_mnist_good' in report


def test_summarize_seed_runs_marks_stable_window():
    seed_runs = [
        {
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.38,
                'split_mnist_mean_forgetting': 0.14,
                'split_mnist_bwt': -0.14,
                'split_mnist_mean_margin': 0.08,
            },
            'primary_score': 0.40,
            'biomarker_score': 0.45,
        },
        {
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.35,
                'split_mnist_mean_forgetting': 0.16,
                'split_mnist_bwt': -0.16,
                'split_mnist_mean_margin': 0.07,
            },
            'primary_score': 0.36,
            'biomarker_score': 0.43,
        },
        {
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.33,
                'split_mnist_mean_forgetting': 0.18,
                'split_mnist_bwt': -0.18,
                'split_mnist_mean_margin': 0.06,
            },
            'primary_score': 0.34,
            'biomarker_score': 0.41,
        },
    ]
    summary = summarize_seed_runs(seed_runs)
    assert summary['seed_count'] == 3.0
    assert summary['final_accuracy']['min'] >= 0.33
    assert summary['forgetting']['max'] <= 0.18
    assert summary['stable_across_seeds']


def test_summarize_sequential_rules_seed_runs_marks_stability():
    seed_runs = [
        {
            'metrics': {
                'sequential_rules_final_accuracy_mean': 0.82,
                'sequential_rules_mean_forgetting': 0.03,
                'sequential_rules_bwt': -0.02,
                'sequential_rules_mean_margin': 0.07,
            },
            'primary_score': 0.80,
            'biomarker_score': 0.47,
        },
        {
            'metrics': {
                'sequential_rules_final_accuracy_mean': 0.77,
                'sequential_rules_mean_forgetting': 0.05,
                'sequential_rules_bwt': -0.04,
                'sequential_rules_mean_margin': 0.06,
            },
            'primary_score': 0.74,
            'biomarker_score': 0.45,
        },
        {
            'metrics': {
                'sequential_rules_final_accuracy_mean': 0.72,
                'sequential_rules_mean_forgetting': 0.08,
                'sequential_rules_bwt': -0.06,
                'sequential_rules_mean_margin': 0.05,
            },
            'primary_score': 0.67,
            'biomarker_score': 0.42,
        },
    ]
    summary = summarize_sequential_rules_seed_runs(seed_runs)
    assert summary['final_accuracy']['min'] >= 0.72
    assert summary['forgetting']['max'] <= 0.08
    assert summary['stable_across_seeds']


def test_summarize_gridworld_seed_runs_marks_stability():
    seed_runs = [
        {
            'metrics': {
                'gridworld_remap_final_success_mean': 0.90,
                'gridworld_remap_mean_forgetting': 0.02,
                'gridworld_remap_bwt': 0.08,
                'gridworld_remap_efficiency_mean': 0.61,
                'gridworld_remap_mean_margin': 0.04,
            },
            'primary_score': 1.03,
            'biomarker_score': 0.46,
        },
        {
            'metrics': {
                'gridworld_remap_final_success_mean': 0.84,
                'gridworld_remap_mean_forgetting': 0.05,
                'gridworld_remap_bwt': 0.04,
                'gridworld_remap_efficiency_mean': 0.58,
                'gridworld_remap_mean_margin': 0.03,
            },
            'primary_score': 0.95,
            'biomarker_score': 0.44,
        },
        {
            'metrics': {
                'gridworld_remap_final_success_mean': 0.79,
                'gridworld_remap_mean_forgetting': 0.09,
                'gridworld_remap_bwt': 0.01,
                'gridworld_remap_efficiency_mean': 0.55,
                'gridworld_remap_mean_margin': 0.02,
            },
            'primary_score': 0.86,
            'biomarker_score': 0.41,
        },
    ]
    summary = summarize_gridworld_seed_runs(seed_runs)
    assert summary['final_success']['min'] >= 0.79
    assert summary['forgetting']['max'] <= 0.09
    assert summary['stable_across_seeds']


def test_summarize_ablation_runs_marks_dependency_deltas():
    condition_runs = [
        {
            'condition': 'baseline',
            'run_name': 'split_mnist_baseline',
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.40,
                'split_mnist_peak_accuracy_mean': 0.48,
                'split_mnist_mean_forgetting': 0.10,
                'split_mnist_bwt': -0.10,
                'split_mnist_mean_margin': 0.08,
                'mean_energy': 0.90,
                'mean_stress': 0.05,
                'mean_z_alignment': 0.20,
            },
            'primary_score': 0.50,
            'biomarker_score': 0.45,
        },
        {
            'condition': 'no_growth',
            'run_name': 'split_mnist_no_growth',
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.34,
                'split_mnist_peak_accuracy_mean': 0.42,
                'split_mnist_mean_forgetting': 0.16,
                'split_mnist_bwt': -0.16,
                'split_mnist_mean_margin': 0.05,
                'mean_energy': 0.86,
                'mean_stress': 0.07,
                'mean_z_alignment': 0.18,
            },
            'primary_score': 0.35,
            'biomarker_score': 0.40,
        },
        {
            'condition': 'no_stress',
            'run_name': 'split_mnist_no_stress',
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.36,
                'split_mnist_peak_accuracy_mean': 0.43,
                'split_mnist_mean_forgetting': 0.15,
                'split_mnist_bwt': -0.15,
                'split_mnist_mean_margin': 0.06,
                'mean_energy': 0.88,
                'mean_stress': 0.06,
                'mean_z_alignment': 0.19,
            },
            'primary_score': 0.39,
            'biomarker_score': 0.41,
        },
        {
            'condition': 'no_z_field',
            'run_name': 'split_mnist_no_z_field',
            'metrics': {
                'split_mnist_final_accuracy_mean': 0.31,
                'split_mnist_peak_accuracy_mean': 0.39,
                'split_mnist_mean_forgetting': 0.18,
                'split_mnist_bwt': -0.18,
                'split_mnist_mean_margin': 0.04,
                'mean_energy': 0.84,
                'mean_stress': 0.08,
                'mean_z_alignment': 0.00,
            },
            'primary_score': 0.28,
            'biomarker_score': 0.33,
        },
    ]
    summary = summarize_ablation_runs(condition_runs)
    assert summary['baseline_run'] == 'split_mnist_baseline'
    assert summary['no_growth_final_accuracy_drop'] >= 0.05
    assert summary['no_stress_dependency_supported']
    assert summary['no_z_field_z_alignment_drop'] >= 0.15
    assert summary['mechanism_dependency_supported_count'] >= 3.0


def test_summarize_sequential_rules_ablations_counts_dependency():
    condition_runs = [
        {
            'condition': 'baseline',
            'run_name': 'sequential_rules_baseline',
            'metrics': {
                'sequential_rules_final_accuracy_mean': 0.80,
                'sequential_rules_peak_accuracy_mean': 0.84,
                'sequential_rules_mean_forgetting': 0.04,
                'sequential_rules_bwt': -0.03,
                'sequential_rules_mean_margin': 0.07,
                'mean_energy': 0.94,
                'mean_stress': 0.05,
                'mean_z_alignment': 0.16,
                'mean_growth_pressure': 0.22,
                'recent_growth_energy_transferred': 0.012,
            },
            'primary_score': 0.79,
            'biomarker_score': 0.47,
        },
        {
            'condition': 'no_growth',
            'run_name': 'sequential_rules_no_growth',
            'metrics': {
                'sequential_rules_final_accuracy_mean': 0.74,
                'sequential_rules_peak_accuracy_mean': 0.79,
                'sequential_rules_mean_forgetting': 0.06,
                'sequential_rules_bwt': -0.05,
                'sequential_rules_mean_margin': 0.05,
                'mean_energy': 0.92,
                'mean_stress': 0.06,
                'mean_z_alignment': 0.14,
                'mean_growth_pressure': 0.0,
                'recent_growth_energy_transferred': 0.0,
            },
            'primary_score': 0.71,
            'biomarker_score': 0.43,
        },
        {
            'condition': 'no_z_field',
            'run_name': 'sequential_rules_no_z_field',
            'metrics': {
                'sequential_rules_final_accuracy_mean': 0.68,
                'sequential_rules_peak_accuracy_mean': 0.75,
                'sequential_rules_mean_forgetting': 0.09,
                'sequential_rules_bwt': -0.07,
                'sequential_rules_mean_margin': 0.04,
                'mean_energy': 0.93,
                'mean_stress': 0.06,
                'mean_z_alignment': 0.0,
                'mean_growth_pressure': 0.18,
                'recent_growth_energy_transferred': 0.008,
            },
            'primary_score': 0.64,
            'biomarker_score': 0.40,
        },
    ]
    summary = summarize_sequential_rules_ablation_runs(condition_runs)
    assert summary['no_growth_growth_pressure_drop'] >= 0.20
    assert summary['no_growth_growth_energy_transfer_drop'] >= 0.01
    assert summary['no_growth_dependency_supported']
    assert summary['no_z_field_z_alignment_magnitude_drop'] >= 0.10
    assert summary['mechanism_dependency_supported_count'] >= 2.0


def test_summarize_gridworld_ablations_counts_dependency():
    condition_runs = [
        {
            'condition': 'baseline',
            'run_name': 'gridworld_baseline',
            'metrics': {
                'gridworld_remap_final_success_mean': 0.88,
                'gridworld_remap_peak_success_mean': 0.90,
                'gridworld_remap_mean_forgetting': 0.03,
                'gridworld_remap_bwt': 0.06,
                'gridworld_remap_efficiency_mean': 0.60,
                'gridworld_remap_mean_margin': 0.04,
                'mean_energy': 0.98,
                'mean_stress': 0.22,
                'mean_z_alignment': 0.12,
                'mean_growth_pressure': 0.19,
                'recent_growth_energy_transferred': 0.010,
            },
            'primary_score': 0.99,
            'biomarker_score': 0.46,
        },
        {
            'condition': 'no_growth',
            'run_name': 'gridworld_no_growth',
            'metrics': {
                'gridworld_remap_final_success_mean': 0.80,
                'gridworld_remap_peak_success_mean': 0.84,
                'gridworld_remap_mean_forgetting': 0.06,
                'gridworld_remap_bwt': 0.02,
                'gridworld_remap_efficiency_mean': 0.53,
                'gridworld_remap_mean_margin': 0.03,
                'mean_energy': 0.97,
                'mean_stress': 0.23,
                'mean_z_alignment': 0.10,
                'mean_growth_pressure': 0.0,
                'recent_growth_energy_transferred': 0.0,
            },
            'primary_score': 0.88,
            'biomarker_score': 0.43,
        },
        {
            'condition': 'no_z_field',
            'run_name': 'gridworld_no_z_field',
            'metrics': {
                'gridworld_remap_final_success_mean': 0.70,
                'gridworld_remap_peak_success_mean': 0.79,
                'gridworld_remap_mean_forgetting': 0.08,
                'gridworld_remap_bwt': -0.01,
                'gridworld_remap_efficiency_mean': 0.45,
                'gridworld_remap_mean_margin': 0.02,
                'mean_energy': 0.97,
                'mean_stress': 0.24,
                'mean_z_alignment': 0.0,
                'mean_growth_pressure': 0.15,
                'recent_growth_energy_transferred': 0.006,
            },
            'primary_score': 0.74,
            'biomarker_score': 0.39,
        },
    ]
    summary = summarize_gridworld_ablation_runs(condition_runs)
    assert summary['no_growth_growth_pressure_drop'] >= 0.15
    assert summary['no_growth_growth_energy_transfer_drop'] >= 0.01
    assert summary['no_growth_dependency_supported']
    assert summary['no_z_field_z_alignment_magnitude_drop'] >= 0.10
    assert summary['no_z_field_dependency_supported']
    assert summary['mechanism_dependency_supported_count'] >= 2.0


def test_split_fashion_seed_summary_marks_stable_window():
    seed_runs = [
        {
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.43,
                'split_fashion_mnist_mean_forgetting': 0.18,
                'split_fashion_mnist_bwt': -0.18,
                'split_fashion_mnist_mean_margin': 0.10,
            },
            'primary_score': 0.45,
            'biomarker_score': 0.46,
        },
        {
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.39,
                'split_fashion_mnist_mean_forgetting': 0.20,
                'split_fashion_mnist_bwt': -0.20,
                'split_fashion_mnist_mean_margin': 0.09,
            },
            'primary_score': 0.40,
            'biomarker_score': 0.44,
        },
        {
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.34,
                'split_fashion_mnist_mean_forgetting': 0.24,
                'split_fashion_mnist_bwt': -0.23,
                'split_fashion_mnist_mean_margin': 0.08,
            },
            'primary_score': 0.34,
            'biomarker_score': 0.42,
        },
    ]
    summary = summarize_split_fashion_seed_runs(seed_runs)
    assert summary['seed_count'] == 3.0
    assert summary['final_accuracy']['min'] >= 0.34
    assert summary['forgetting']['max'] <= 0.24
    assert summary['stable_across_seeds']


def test_split_fashion_ablation_summary_marks_dependency_deltas():
    condition_runs = [
        {
            'condition': 'baseline',
            'run_name': 'split_fashion_mnist_baseline',
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.43,
                'split_fashion_mnist_peak_accuracy_mean': 0.58,
                'split_fashion_mnist_mean_forgetting': 0.18,
                'split_fashion_mnist_bwt': -0.18,
                'split_fashion_mnist_mean_margin': 0.10,
                'mean_energy': 0.95,
                'mean_stress': 0.04,
                'mean_z_alignment': -0.05,
                'split_fashion_mnist_peak_growth_pressure_mean': 0.23,
                'split_fashion_mnist_growth_trigger_crossed_fraction': 0.06,
            },
            'primary_score': 0.46,
            'biomarker_score': 0.45,
        },
        {
            'condition': 'no_growth',
            'run_name': 'split_fashion_mnist_no_growth',
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.38,
                'split_fashion_mnist_peak_accuracy_mean': 0.52,
                'split_fashion_mnist_mean_forgetting': 0.22,
                'split_fashion_mnist_bwt': -0.22,
                'split_fashion_mnist_mean_margin': 0.08,
                'mean_energy': 0.93,
                'mean_stress': 0.05,
                'mean_z_alignment': -0.04,
                'split_fashion_mnist_peak_growth_pressure_mean': 0.12,
                'split_fashion_mnist_growth_trigger_crossed_fraction': 0.0,
            },
            'primary_score': 0.38,
            'biomarker_score': 0.43,
        },
        {
            'condition': 'no_stress',
            'run_name': 'split_fashion_mnist_no_stress',
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.36,
                'split_fashion_mnist_peak_accuracy_mean': 0.50,
                'split_fashion_mnist_mean_forgetting': 0.24,
                'split_fashion_mnist_bwt': -0.24,
                'split_fashion_mnist_mean_margin': 0.07,
                'mean_energy': 0.91,
                'mean_stress': 0.07,
                'mean_z_alignment': -0.03,
                'split_fashion_mnist_peak_growth_pressure_mean': 0.21,
                'split_fashion_mnist_growth_trigger_crossed_fraction': 0.04,
            },
            'primary_score': 0.35,
            'biomarker_score': 0.41,
        },
        {
            'condition': 'no_z_field',
            'run_name': 'split_fashion_mnist_no_z_field',
            'metrics': {
                'split_fashion_mnist_final_accuracy_mean': 0.37,
                'split_fashion_mnist_peak_accuracy_mean': 0.51,
                'split_fashion_mnist_mean_forgetting': 0.23,
                'split_fashion_mnist_bwt': -0.22,
                'split_fashion_mnist_mean_margin': 0.08,
                'mean_energy': 0.94,
                'mean_stress': 0.05,
                'mean_z_alignment': 0.0,
                'split_fashion_mnist_peak_growth_pressure_mean': 0.22,
                'split_fashion_mnist_growth_trigger_crossed_fraction': 0.05,
            },
            'primary_score': 0.37,
            'biomarker_score': 0.40,
        },
    ]
    summary = summarize_split_fashion_ablation_runs(condition_runs)
    assert summary['baseline_run'] == 'split_fashion_mnist_baseline'
    assert summary['no_growth_growth_peak_pressure_drop'] >= 0.10
    assert summary['no_growth_growth_trigger_crossing_drop'] >= 0.05
    assert summary['no_growth_dependency_supported']
    assert summary['no_stress_dependency_supported']
    assert summary['no_z_field_z_alignment_magnitude_drop'] >= 0.03
    assert summary['mechanism_dependency_supported_count'] >= 3.0


def test_growth_usefulness_reports_advantage_metrics():
    cfg = load_config('configs/assay/growth_usefulness.yaml')
    set_seed(cfg.run.seed)
    result = GrowthUsefulnessAssay().run(cfg)
    assert 'energy_advantage' in result.final_metrics
    assert 'z_alignment_advantage' in result.final_metrics
    assert result.final_metrics['growth_utility_gain'] >= 0.01
    assert result.final_metrics['growth_efficiency_advantage'] >= 0.005
    assert result.final_metrics['mean_growth_decorative_fraction'] <= 0.05
    assert result.final_metrics['late_growth_event_fraction_mean'] <= 0.05


def test_tissue_field_probe_reports_localized_regional_response():
    cfg = load_config('configs/assay/tissue_field_probe.yaml')
    set_seed(cfg.run.seed)
    result = TissueFieldProbeAssay().run(cfg)
    assert abs(result.final_metrics['mean_tissue_field']) <= 1.0
    assert result.final_metrics['tissue_field_peak_magnitude'] <= 1.0
    assert result.final_metrics['tissue_field_local_response'] > 0.0
    assert result.final_metrics['tissue_field_localization_ratio'] >= 1.10
    assert result.final_metrics['tissue_field_regionality_gain'] >= 0.0
    assert result.final_metrics['tissue_field_bounded'] == 1.0


def test_oscillatory_coupling_probe_reports_localized_entrainment():
    cfg = load_config('configs/assay/oscillatory_coupling_probe.yaml')
    set_seed(cfg.run.seed)
    result = OscillatoryCouplingProbeAssay().run(cfg)
    assert 0.0 <= result.final_metrics['mean_oscillation_amplitude'] <= 1.0
    assert 0.0 <= result.final_metrics['oscillation_phase_coherence'] <= 1.0
    assert result.final_metrics['oscillation_local_amplitude_gain'] > 0.0
    assert result.final_metrics['oscillation_localization_ratio'] >= 1.10
    assert result.final_metrics['oscillation_coherence_advantage'] >= 0.0
    assert result.final_metrics['oscillation_bounded'] == 1.0


def test_reaction_diffusion_probe_reports_localized_patterning():
    cfg = load_config('configs/assay/reaction_diffusion_probe.yaml')
    set_seed(cfg.run.seed)
    result = ReactionDiffusionProbeAssay().run(cfg)
    assert 0.0 <= result.final_metrics['mean_morphogen_activator'] <= 1.0
    assert 0.0 <= result.final_metrics['mean_morphogen_inhibitor'] <= 1.0
    assert result.final_metrics['reaction_diffusion_pattern_strength'] > 0.0
    assert result.final_metrics['reaction_diffusion_local_response'] > 0.0
    assert result.final_metrics['reaction_diffusion_localization_ratio'] >= 1.10
    assert result.final_metrics['reaction_diffusion_pattern_gain'] >= 0.0
    assert result.final_metrics['reaction_diffusion_bounded'] == 1.0


def test_stigmergic_highway_probe_reports_localized_persistent_trails():
    cfg = load_config('configs/assay/stigmergic_highway_probe.yaml')
    set_seed(cfg.run.seed)
    result = StigmergicHighwayProbeAssay().run(cfg)
    assert 0.0 <= result.final_metrics['mean_highway_trace'] <= 1.0
    assert 0.0 <= result.final_metrics['mean_highway_flux'] <= 1.0
    assert result.final_metrics['stigmergic_highway_strength'] > 0.0
    assert result.final_metrics['stigmergic_local_response'] > 0.0
    assert result.final_metrics['stigmergic_localization_ratio'] >= 1.10
    assert result.final_metrics['stigmergic_persistence_retention'] > 0.0
    assert result.final_metrics['stigmergic_bounded'] == 1.0


def test_predictive_coding_probe_reports_local_error_and_recovery():
    cfg = load_config('configs/assay/predictive_coding_probe.yaml')
    set_seed(cfg.run.seed)
    result = PredictiveCodingProbeAssay().run(cfg)
    assert abs(result.final_metrics['mean_predictive_prediction']) <= 1.0
    assert 0.0 <= result.final_metrics['mean_predictive_error'] <= 1.0
    assert 0.0 <= result.final_metrics['mean_predictive_precision'] <= 1.0
    assert result.final_metrics['predictive_local_error_response'] > 0.0
    assert result.final_metrics['predictive_localization_ratio'] >= 1.10
    assert result.final_metrics['predictive_error_recovery_gain'] >= 0.0
    assert result.final_metrics['predictive_coding_bounded'] == 1.0


def test_classify_run_detects_pseudo_maturity():
    verdict = classify_run({
        'mean_energy': 0.5,
        'mean_commitment': 0.7,
        'z_field_drift': 0.05,
        'dormant_fraction': 0.5,
        'active_fraction': 0.2,
        'plasticity_loss_index': 0.4,
        'pseudo_maturity_index': 0.5,
        'low_energy_fraction': 0.0,
        'conductance_entropy': 1.0,
        'mean_growth_pressure': 0.1,
        'stress_variance': 0.01,
    })
    assert verdict.value == 'pseudo_maturity'


def test_setpoint_rewrite_exposes_persistence_metric():
    cfg = load_config('configs/assay/setpoint_rewrite.yaml')
    result = SetpointRewriteAssay().run(cfg)
    assert 'rewrite_persistence' in result.final_metrics
    assert 'cryptic_shift' in result.final_metrics
    assert 'rewrite_mode_supported_count' in result.final_metrics
    assert 'strong_cryptic_mode_count' in result.final_metrics
    assert result.final_metrics['z_bias_rewrite_persistence_min'] >= 0.18
    assert result.final_metrics['conductance_bias_rewrite_persistence_min'] >= 0.18
    assert result.final_metrics['stress_bias_rewrite_persistence_min'] >= 0.18
    assert result.final_metrics['rewrite_persistence'] >= 0.0


def test_lightcone_assay_measures_radius_and_duration():
    cfg = load_config('configs/assay/lightcone.yaml')
    result = LightconeAssay().run(cfg)
    assert result.final_metrics['lightcone_radius'] > 0.0
    assert result.final_metrics['lightcone_duration'] > 0.0
    assert result.final_metrics['lightcone_spread_radius_mean'] > 0.0
    assert result.final_metrics['lightcone_spread_duration'] > 0.0
    assert result.final_metrics['lightcone_port_duration'] > 0.0
    assert result.final_metrics['lightcone_strong_port_onset'] > 0.0
    assert 'lightcone_ablation_supported_count' in result.final_metrics
    assert 'stress_sharing_off_spread_radius_mean_delta' in result.final_metrics
    assert 'stress_sharing_off_port_duration_delta' in result.final_metrics
    assert 'stress_sharing_off_strong_port_onset_delay' in result.final_metrics
    assert result.final_metrics['stress_sharing_off_ablation_support_score'] >= 7.0
    assert 'conductance_ablated_spread_radius_mean_delta' in result.final_metrics
    assert 'conductance_ablated_port_balance_shift' in result.final_metrics
    assert result.final_metrics['conductance_ablated_ablation_support_score'] >= 7.0
    assert 'z_memory_ablated_lightcone_left_strong_port_onset' in result.final_metrics
    assert 'z_memory_ablated_lightcone_right_strong_port_onset' in result.final_metrics
    assert 'z_memory_ablated_spread_duration_delta' in result.final_metrics
    assert result.final_metrics['z_memory_ablated_ablation_support_score'] >= 7.0


def test_lesion_battery_reports_recovery_summary():
    cfg = load_config('configs/assay/lesion_battery.yaml')
    result = LesionBatteryAssay().run(cfg)
    assert 'lesion_recovery_mean' in result.final_metrics
    assert 'lesion_success_fraction' in result.final_metrics
    assert 'lesion_recovery_probability_mean' in result.final_metrics
    assert 'lesion_recovery_steps_mean' in result.final_metrics
    assert 'lesion_energy_cost_mean' in result.final_metrics
    assert 'no_gradient_recovery_mean' in result.final_metrics
    assert 'recovery_retention_without_gradients' in result.final_metrics
    assert 'retraining_recovery_mean' in result.final_metrics
    assert 'organismal_recovery_vs_retraining_ratio' in result.final_metrics
    assert result.final_metrics['retraining_recovery_probability_mean'] >= 0.8
    assert result.final_metrics['organismal_recovery_vs_retraining_ratio'] >= 0.75
    assert 'repeated_injury_no_gradient_recovery_mean' in result.final_metrics
    assert 'repeated_injury_vs_retraining_ratio' in result.final_metrics
    assert result.final_metrics['repeated_injury_vs_retraining_ratio'] >= 0.6
    assert 'parameter_corruption_recovery' in result.final_metrics
    assert 'parameter_corruption_recovery_probability' in result.final_metrics
    assert result.final_metrics['cell_ablation_recovery'] >= 0.9
    assert result.final_metrics['parameter_corruption_recovery'] >= 0.45
    assert result.final_metrics['parameter_corruption_no_gradient_recovery'] >= 0.25
    assert 'parameter_corruption_retraining_recovery' in result.final_metrics
    assert 'targeted_tissue_ablation_recovery_probability' in result.final_metrics
    assert 'port_disruption_recovery_probability' in result.final_metrics
    assert result.final_metrics['port_disruption_recovery'] >= 0.5
    assert result.final_metrics['port_disruption_no_gradient_recovery_probability'] >= 1.0
    assert 'port_disruption_distal_sparing_fraction' in result.final_metrics
    assert 'port_disruption_boundary_locality_ratio' in result.final_metrics
    assert 'whole_body_port_disruption_distal_core_peak_disturbance' in result.final_metrics
    assert result.final_metrics['port_localization_advantage'] > 0.0
    assert 'repeated_cell_ablation_recovery_probability' in result.final_metrics
    assert 'repeated_port_disruption_recovery_probability' in result.final_metrics
    assert 'repeated_cell_ablation_retraining_recovery' in result.final_metrics
    assert 'repeated_port_disruption_retraining_recovery' in result.final_metrics
    assert result.final_metrics['repeated_port_disruption_recovery'] >= 0.5
    assert 'cell_ablation_recovery_onset_gap' in result.final_metrics
    assert 'cell_ablation_recovery_onset_skew' in result.final_metrics
    assert 'z_field_corruption_recovery_onset_gap' in result.final_metrics
    assert 'z_field_corruption_preferred_recovery_side' in result.final_metrics


def test_lesion_preserves_competence_reports_task_retention():
    cfg = load_config('configs/assay/lesion_preserves_competence.yaml')
    result = LesionPreservesCompetenceAssay().run(cfg)
    assert result.final_metrics['pre_lesion_competence'] >= 0.72
    assert result.final_metrics['post_recovery_competence'] >= 0.72
    assert result.final_metrics['competence_retention_ratio'] >= 0.95
    assert result.final_metrics['organismal_competence_advantage_vs_no_gradient'] >= 0.03
    assert result.final_metrics['organismal_competence_vs_retraining_ratio'] >= 0.95
    assert result.final_metrics['competence_supported_lesion_count'] >= 4.0
    assert result.final_metrics['competence_supported_task_count'] >= 2.0
    assert result.final_metrics['relay_tracking_post_recovery_competence'] >= 0.72
    assert result.final_metrics['relay_tracking_competence_retention_ratio'] >= 0.95
    assert result.final_metrics['relay_tracking_supported_lesion_count'] >= 2.0
    assert result.final_metrics['inverted_remap_post_recovery_competence'] >= 0.72
    assert result.final_metrics['inverted_remap_competence_retention_ratio'] >= 0.95
    assert result.final_metrics['inverted_remap_supported_lesion_count'] >= 2.0


def test_port_remap_reports_localized_competence_preservation():
    cfg = load_config('configs/assay/port_remap.yaml')
    result = PortRemapAssay().run(cfg)
    assert result.final_metrics['pre_remap_competence'] >= 0.70
    assert result.final_metrics['post_recovery_competence'] >= 0.76
    assert result.final_metrics['competence_retention_ratio'] >= 0.98
    assert result.final_metrics['local_vs_whole_body_competence_advantage'] >= -0.03
    assert result.final_metrics['boundary_locality_ratio'] >= 1.60
    assert result.final_metrics['port_localization_advantage'] >= 0.50
    assert result.final_metrics['port_remap_mode_supported_count'] >= 3.0
    assert result.final_metrics['supported_port_family_count'] >= 2.0
    assert result.final_metrics['rule_post_recovery_competence'] >= 0.70
    assert result.final_metrics['rule_boundary_locality_ratio'] >= 1.35
    assert result.final_metrics['rule_supported_mode_count'] >= 1.0
    assert result.final_metrics['pattern_post_recovery_competence'] >= 0.84
    assert result.final_metrics['pattern_boundary_locality_ratio'] >= 1.85
    assert result.final_metrics['pattern_supported_mode_count'] >= 2.0
    assert result.final_metrics['cross_port_competence_gap'] <= 0.20


def test_mnist_sanity_reports_accuracy_above_chance():
    cfg = load_config('configs/assay/mnist_sanity.yaml')
    set_seed(cfg.run.seed)
    result = MNISTSanityAssay().run(cfg)
    assert result.final_metrics['mnist_eval_accuracy'] >= 0.30
    assert result.final_metrics['mnist_accuracy_advantage'] >= 0.10
    assert result.final_metrics['mnist_mean_margin'] >= 0.0
    assert result.final_metrics['mnist_class_count'] == 5.0
    assert result.final_metrics['mnist_support_count'] == 30.0
    assert result.final_metrics['mnist_eval_count'] == 30.0


def test_split_mnist_reports_forgetting_metrics():
    cfg = load_config('configs/assay/split_mnist.yaml')
    set_seed(cfg.run.seed)
    result = SplitMNISTAssay().run(cfg)
    assert result.final_metrics['split_mnist_task_count'] == 5.0
    assert result.final_metrics['split_mnist_final_accuracy_mean'] >= 0.35
    assert result.final_metrics['split_mnist_peak_accuracy_mean'] >= result.final_metrics['split_mnist_final_accuracy_mean']
    assert result.final_metrics['split_mnist_mean_forgetting'] <= 0.20
    assert result.final_metrics['split_mnist_bwt'] >= -0.20
    assert result.final_metrics['split_mnist_mean_margin'] >= 0.05
    assert result.final_metrics['split_mnist_first_task_final_accuracy'] >= 0.70
    assert result.final_metrics['split_mnist_last_task_accuracy'] >= 0.20
    assert result.final_metrics['split_mnist_support_count'] == 100.0
    assert result.final_metrics['split_mnist_eval_count'] == 60.0


def test_lesion_split_mnist_reports_visual_lesion_bridge_metrics():
    cfg = load_config('configs/assay/lesion_split_mnist.yaml')
    set_seed(cfg.run.seed)
    result = LesionSplitMNISTAssay().run_condition(cfg, 'no_z_field')
    assert result.final_metrics['lesion_split_mnist_lesion_active'] == 1.0
    assert result.final_metrics['lesion_split_mnist_condition_no_z_field'] == 1.0
    assert result.final_metrics['lesion_split_mnist_lesion_row_count'] == 8.0
    assert result.final_metrics['lesion_split_mnist_final_accuracy_mean'] >= 0.15
    assert result.final_metrics['lesion_split_mnist_peak_accuracy_mean'] >= result.final_metrics['lesion_split_mnist_final_accuracy_mean']
    assert result.final_metrics['lesion_split_mnist_growth_trigger_crossed_fraction'] >= 0.0


def test_split_fashion_mnist_reports_family_shift_metrics():
    cfg = load_config('configs/assay/split_fashion_mnist.yaml')
    set_seed(cfg.run.seed)
    result = SplitFashionMNISTAssay().run(cfg)
    assert result.final_metrics['split_fashion_mnist_task_count'] == 5.0
    assert result.final_metrics['split_fashion_mnist_final_accuracy_mean'] >= 0.18
    assert (
        result.final_metrics['split_fashion_mnist_peak_accuracy_mean']
        >= result.final_metrics['split_fashion_mnist_final_accuracy_mean']
    )
    assert result.final_metrics['split_fashion_mnist_mean_forgetting'] <= 0.35
    assert result.final_metrics['split_fashion_mnist_bwt'] >= -0.35
    assert result.final_metrics['split_fashion_mnist_mean_margin'] >= 0.01
    assert result.final_metrics['split_fashion_mnist_first_task_final_accuracy'] >= 0.30
    assert result.final_metrics['split_fashion_mnist_last_task_accuracy'] >= 0.10
    assert result.final_metrics['split_fashion_mnist_support_count'] == 100.0
    assert result.final_metrics['split_fashion_mnist_eval_count'] == 60.0
    assert (
        result.final_metrics['split_fashion_mnist_dataset_source_fashion_mnist']
        + result.final_metrics['split_fashion_mnist_dataset_source_kmnist']
        + result.final_metrics['split_fashion_mnist_dataset_source_mnist_fallback']
        == 1.0
    )


def test_sequential_rules_reports_nonvisual_bridge_metrics():
    cfg = load_config('configs/assay/sequential_rules.yaml')
    set_seed(cfg.run.seed)
    result = SequentialRulesAssay().run(cfg)
    assert result.final_metrics['sequential_rules_task_count'] == 5.0
    assert result.final_metrics['sequential_rules_final_accuracy_mean'] >= 0.45
    assert result.final_metrics['sequential_rules_peak_accuracy_mean'] >= result.final_metrics['sequential_rules_final_accuracy_mean']
    assert result.final_metrics['sequential_rules_mean_forgetting'] <= 0.15
    assert result.final_metrics['sequential_rules_bwt'] >= -0.10
    assert result.final_metrics['sequential_rules_mean_margin'] >= 0.05
    assert result.final_metrics['sequential_rules_first_task_final_accuracy'] >= 0.70
    assert result.final_metrics['sequential_rules_last_task_accuracy'] >= 0.30
    assert result.final_metrics['sequential_rules_support_count'] == 100.0
    assert result.final_metrics['sequential_rules_eval_count'] == 60.0
    assert result.final_metrics['sequential_rules_port_family_rule'] == 1.0


def test_lesion_sequential_rules_reports_conditioned_nonvisual_bridge_metrics():
    cfg = load_config('configs/assay/lesion_sequential_rules.yaml')
    set_seed(cfg.run.seed)
    result = LesionSequentialRulesAssay().run_condition(cfg, 'no_growth')
    assert result.final_metrics['lesion_sequential_rules_lesion_active'] == 1.0
    assert result.final_metrics['lesion_sequential_rules_condition_no_growth'] == 1.0
    assert result.final_metrics['lesion_sequential_rules_lesion_window_length'] == 4.0
    assert result.final_metrics['lesion_sequential_rules_final_accuracy_mean'] >= 0.35
    assert result.final_metrics['lesion_sequential_rules_mean_forgetting'] <= 0.20


def test_gridworld_remap_reports_control_bridge_metrics():
    cfg = load_config('configs/assay/gridworld_remap.yaml')
    set_seed(cfg.run.seed)
    result = GridworldRemapAssay().run(cfg)
    assert result.final_metrics['gridworld_remap_task_count'] == 5.0
    assert result.final_metrics['gridworld_remap_final_success_mean'] >= 0.75
    assert result.final_metrics['gridworld_remap_peak_success_mean'] >= result.final_metrics['gridworld_remap_final_success_mean']
    assert result.final_metrics['gridworld_remap_mean_forgetting'] <= 0.10
    assert result.final_metrics['gridworld_remap_bwt'] >= 0.0
    assert result.final_metrics['gridworld_remap_mean_margin'] >= 0.05
    assert result.final_metrics['gridworld_remap_efficiency_mean'] >= 0.45
    assert result.final_metrics['gridworld_remap_first_task_final_success'] >= 0.75
    assert result.final_metrics['gridworld_remap_last_task_success'] >= 0.45
    assert result.final_metrics['gridworld_remap_support_count'] >= 150.0
    assert result.final_metrics['gridworld_remap_eval_count'] == 60.0
    assert result.final_metrics['gridworld_remap_port_family_control'] == 1.0
    assert result.final_metrics['mean_energy'] >= 0.95
    assert result.final_metrics['mean_stress'] <= 0.30


def test_lesion_gridworld_remap_reports_conditioned_control_bridge_metrics():
    cfg = load_config('configs/assay/lesion_gridworld_remap.yaml')
    set_seed(cfg.run.seed)
    result = LesionGridworldRemapAssay().run_condition(cfg, 'no_z_field')
    assert result.final_metrics['lesion_gridworld_remap_lesion_active'] == 1.0
    assert result.final_metrics['lesion_gridworld_remap_observation_patch_lesion'] == 1.0
    assert result.final_metrics['lesion_gridworld_remap_condition_no_z_field'] == 1.0
    assert result.final_metrics['lesion_gridworld_remap_final_success_mean'] >= 0.50
    assert result.final_metrics['lesion_gridworld_remap_efficiency_mean'] >= 0.20


def test_permuted_mnist_reports_shift_metrics():
    cfg = load_config('configs/assay/permuted_mnist.yaml')
    set_seed(cfg.run.seed)
    result = PermutedMNISTAssay().run(cfg)
    assert result.final_metrics['permuted_mnist_task_count'] == 5.0
    assert result.final_metrics['permuted_mnist_final_accuracy_mean'] >= 0.28
    assert result.final_metrics['permuted_mnist_peak_accuracy_mean'] >= result.final_metrics['permuted_mnist_final_accuracy_mean']
    assert result.final_metrics['permuted_mnist_mean_forgetting'] <= 0.10
    assert result.final_metrics['permuted_mnist_bwt'] >= -0.10
    assert result.final_metrics['permuted_mnist_mean_margin'] >= 0.02
    assert result.final_metrics['permuted_mnist_first_task_final_accuracy'] >= 0.30
    assert result.final_metrics['permuted_mnist_last_task_accuracy'] >= 0.25
    assert result.final_metrics['permuted_mnist_support_count'] == 400.0
    assert result.final_metrics['permuted_mnist_eval_count'] == 250.0


def test_permuted_mnist_growth_probe_crosses_growth_trigger():
    cfg = load_config('configs/assay/permuted_mnist.yaml')
    set_seed(cfg.run.seed)
    assay = PermutedMNISTAssay()
    assay.challenge_variant = 'growth_probe'
    result = assay.run_condition(cfg, 'baseline')
    assert result.final_metrics['permuted_mnist_peak_growth_pressure_max'] >= 0.30
    assert result.final_metrics['permuted_mnist_growth_trigger_crossed_fraction'] > 0.0
    assert abs(result.final_metrics['mean_z_alignment']) > 0.01
    assert result.final_metrics['z_memory_alignment_gap'] > 0.0


def test_permuted_ablation_summary_counts_growth_and_z_dependence():
    condition_runs = [
        {
            'condition': 'baseline',
            'run_name': 'permuted_mnist_baseline',
            'metrics': {
                'permuted_mnist_final_accuracy_mean': 0.32,
                'permuted_mnist_peak_accuracy_mean': 0.36,
                'permuted_mnist_mean_forgetting': 0.08,
                'permuted_mnist_bwt': -0.07,
                'permuted_mnist_mean_margin': 0.03,
                'mean_energy': 0.94,
                'mean_stress': 0.04,
                'mean_z_alignment': -0.12,
                'z_memory_alignment_gap': 0.02,
                'permuted_mnist_peak_growth_pressure_mean': 0.31,
                'permuted_mnist_growth_trigger_crossed_fraction': 0.02,
            },
            'primary_score': 0.36,
            'biomarker_score': 0.46,
        },
        {
            'condition': 'no_growth',
            'run_name': 'permuted_mnist_no_growth',
            'metrics': {
                'permuted_mnist_final_accuracy_mean': 0.29,
                'permuted_mnist_peak_accuracy_mean': 0.34,
                'permuted_mnist_mean_forgetting': 0.09,
                'permuted_mnist_bwt': -0.08,
                'permuted_mnist_mean_margin': 0.027,
                'mean_energy': 0.95,
                'mean_stress': 0.04,
                'mean_z_alignment': -0.11,
                'z_memory_alignment_gap': 0.018,
                'permuted_mnist_peak_growth_pressure_mean': 0.20,
                'permuted_mnist_growth_trigger_crossed_fraction': 0.0,
            },
            'primary_score': 0.32,
            'biomarker_score': 0.45,
        },
        {
            'condition': 'no_stress',
            'run_name': 'permuted_mnist_no_stress',
            'metrics': {
                'permuted_mnist_final_accuracy_mean': 0.27,
                'permuted_mnist_peak_accuracy_mean': 0.31,
                'permuted_mnist_mean_forgetting': 0.11,
                'permuted_mnist_bwt': -0.10,
                'permuted_mnist_mean_margin': 0.021,
                'mean_energy': 0.91,
                'mean_stress': 0.07,
                'mean_z_alignment': -0.10,
                'z_memory_alignment_gap': 0.015,
                'permuted_mnist_peak_growth_pressure_mean': 0.29,
                'permuted_mnist_growth_trigger_crossed_fraction': 0.01,
            },
            'primary_score': 0.29,
            'biomarker_score': 0.42,
        },
        {
            'condition': 'no_z_field',
            'run_name': 'permuted_mnist_no_z_field',
            'metrics': {
                'permuted_mnist_final_accuracy_mean': 0.28,
                'permuted_mnist_peak_accuracy_mean': 0.33,
                'permuted_mnist_mean_forgetting': 0.10,
                'permuted_mnist_bwt': -0.08,
                'permuted_mnist_mean_margin': 0.025,
                'mean_energy': 0.93,
                'mean_stress': 0.05,
                'mean_z_alignment': 0.00,
                'z_memory_alignment_gap': 0.0,
                'permuted_mnist_peak_growth_pressure_mean': 0.30,
                'permuted_mnist_growth_trigger_crossed_fraction': 0.015,
            },
            'primary_score': 0.31,
            'biomarker_score': 0.40,
        },
    ]
    summary = summarize_permuted_ablation_runs(condition_runs)
    assert summary['no_growth_growth_peak_pressure_drop'] >= 0.10
    assert summary['no_growth_growth_trigger_crossing_drop'] >= 0.01
    assert summary['no_growth_dependency_supported']
    assert summary['no_z_field_z_alignment_magnitude_drop'] >= 0.10
    assert summary['no_z_field_z_memory_gap_drop'] >= 0.01
    assert summary['no_z_field_dependency_supported']
    assert summary['mechanism_dependency_supported_count'] >= 3.0


def test_permuted_growth_probe_seed_summary_marks_stability():
    seed_reports = [
        {
            'summary': {
                'baseline_final_accuracy_mean': 0.28,
                'baseline_mean_forgetting': 0.07,
                'baseline_bwt': -0.06,
                'baseline_peak_growth_pressure_mean': 0.29,
                'baseline_growth_trigger_crossed_fraction': 0.30,
                'mechanism_dependency_supported_count': 3.0,
                'no_growth_dependency_supported': True,
                'no_stress_dependency_supported': True,
                'no_z_field_dependency_supported': True,
            }
        },
        {
            'summary': {
                'baseline_final_accuracy_mean': 0.27,
                'baseline_mean_forgetting': 0.08,
                'baseline_bwt': -0.07,
                'baseline_peak_growth_pressure_mean': 0.31,
                'baseline_growth_trigger_crossed_fraction': 0.25,
                'mechanism_dependency_supported_count': 3.0,
                'no_growth_dependency_supported': True,
                'no_stress_dependency_supported': True,
                'no_z_field_dependency_supported': True,
            }
        },
        {
            'summary': {
                'baseline_final_accuracy_mean': 0.26,
                'baseline_mean_forgetting': 0.09,
                'baseline_bwt': -0.08,
                'baseline_peak_growth_pressure_mean': 0.28,
                'baseline_growth_trigger_crossed_fraction': 0.18,
                'mechanism_dependency_supported_count': 2.0,
                'no_growth_dependency_supported': True,
                'no_stress_dependency_supported': True,
                'no_z_field_dependency_supported': False,
            }
        },
    ]
    summary = summarize_permuted_growth_probe_seed_reports(seed_reports)
    assert summary['baseline_growth_trigger_crossed_fraction']['min'] >= 0.18
    assert summary['mechanism_dependency_supported_count']['min'] >= 2.0
    assert summary['no_growth_supported_fraction'] >= 1.0
    assert summary['all_mechanisms_supported_fraction'] >= 0.66
    assert summary['stable_across_seeds']


def test_permuted_fashion_seed_summary_marks_stability():
    seed_runs = [
        {
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.28,
                'permuted_fashion_mnist_mean_forgetting': 0.09,
                'permuted_fashion_mnist_bwt': -0.08,
                'permuted_fashion_mnist_mean_margin': 0.03,
            },
            'primary_score': 0.30,
            'biomarker_score': 0.44,
        },
        {
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.24,
                'permuted_fashion_mnist_mean_forgetting': 0.12,
                'permuted_fashion_mnist_bwt': -0.11,
                'permuted_fashion_mnist_mean_margin': 0.025,
            },
            'primary_score': 0.23,
            'biomarker_score': 0.42,
        },
        {
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.22,
                'permuted_fashion_mnist_mean_forgetting': 0.15,
                'permuted_fashion_mnist_bwt': -0.14,
                'permuted_fashion_mnist_mean_margin': 0.022,
            },
            'primary_score': 0.19,
            'biomarker_score': 0.41,
        },
    ]
    summary = summarize_permuted_fashion_seed_runs(seed_runs)
    assert summary['final_accuracy']['min'] >= 0.22
    assert summary['forgetting']['max'] <= 0.15
    assert summary['stable_across_seeds']


def test_permuted_fashion_ablation_summary_counts_dependency():
    condition_runs = [
        {
            'condition': 'baseline',
            'run_name': 'permuted_fashion_baseline',
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.27,
                'permuted_fashion_mnist_peak_accuracy_mean': 0.31,
                'permuted_fashion_mnist_mean_forgetting': 0.10,
                'permuted_fashion_mnist_bwt': -0.09,
                'permuted_fashion_mnist_mean_margin': 0.03,
                'mean_energy': 0.86,
                'mean_stress': 0.05,
                'mean_z_alignment': -0.10,
                'z_memory_alignment_gap': 0.02,
                'permuted_fashion_mnist_peak_growth_pressure_mean': 0.32,
                'permuted_fashion_mnist_growth_trigger_crossed_fraction': 0.03,
            },
            'primary_score': 0.28,
            'biomarker_score': 0.40,
        },
        {
            'condition': 'no_growth',
            'run_name': 'permuted_fashion_no_growth',
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.24,
                'permuted_fashion_mnist_peak_accuracy_mean': 0.28,
                'permuted_fashion_mnist_mean_forgetting': 0.12,
                'permuted_fashion_mnist_bwt': -0.11,
                'permuted_fashion_mnist_mean_margin': 0.025,
                'mean_energy': 0.87,
                'mean_stress': 0.05,
                'mean_z_alignment': -0.09,
                'z_memory_alignment_gap': 0.018,
                'permuted_fashion_mnist_peak_growth_pressure_mean': 0.18,
                'permuted_fashion_mnist_growth_trigger_crossed_fraction': 0.0,
            },
            'primary_score': 0.22,
            'biomarker_score': 0.41,
        },
        {
            'condition': 'no_stress',
            'run_name': 'permuted_fashion_no_stress',
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.23,
                'permuted_fashion_mnist_peak_accuracy_mean': 0.27,
                'permuted_fashion_mnist_mean_forgetting': 0.13,
                'permuted_fashion_mnist_bwt': -0.12,
                'permuted_fashion_mnist_mean_margin': 0.023,
                'mean_energy': 0.84,
                'mean_stress': 0.07,
                'mean_z_alignment': -0.08,
                'z_memory_alignment_gap': 0.015,
                'permuted_fashion_mnist_peak_growth_pressure_mean': 0.29,
                'permuted_fashion_mnist_growth_trigger_crossed_fraction': 0.02,
            },
            'primary_score': 0.20,
            'biomarker_score': 0.38,
        },
        {
            'condition': 'no_z_field',
            'run_name': 'permuted_fashion_no_z_field',
            'metrics': {
                'permuted_fashion_mnist_final_accuracy_mean': 0.24,
                'permuted_fashion_mnist_peak_accuracy_mean': 0.29,
                'permuted_fashion_mnist_mean_forgetting': 0.12,
                'permuted_fashion_mnist_bwt': -0.10,
                'permuted_fashion_mnist_mean_margin': 0.024,
                'mean_energy': 0.85,
                'mean_stress': 0.05,
                'mean_z_alignment': 0.0,
                'z_memory_alignment_gap': 0.0,
                'permuted_fashion_mnist_peak_growth_pressure_mean': 0.31,
                'permuted_fashion_mnist_growth_trigger_crossed_fraction': 0.025,
            },
            'primary_score': 0.21,
            'biomarker_score': 0.37,
        },
    ]
    summary = summarize_permuted_fashion_ablation_runs(condition_runs)
    assert summary['no_growth_growth_peak_pressure_drop'] >= 0.10
    assert summary['no_growth_growth_trigger_crossing_drop'] >= 0.03
    assert summary['no_z_field_z_alignment_magnitude_drop'] >= 0.10
    assert summary['no_z_field_z_memory_gap_drop'] >= 0.01
    assert summary['mechanism_dependency_supported_count'] >= 3.0


def test_permuted_fashion_mnist_reports_shift_metrics():
    cfg = load_config('configs/assay/permuted_fashion_mnist.yaml')
    set_seed(cfg.run.seed)
    result = PermutedFashionMNISTAssay().run(cfg)
    assert result.final_metrics['permuted_fashion_mnist_task_count'] == 5.0
    assert result.final_metrics['permuted_fashion_mnist_final_accuracy_mean'] >= 0.24
    assert (
        result.final_metrics['permuted_fashion_mnist_peak_accuracy_mean']
        >= result.final_metrics['permuted_fashion_mnist_final_accuracy_mean']
    )
    assert result.final_metrics['permuted_fashion_mnist_peak_accuracy_mean'] >= 0.26
    assert result.final_metrics['permuted_fashion_mnist_mean_forgetting'] <= 0.05
    assert result.final_metrics['permuted_fashion_mnist_bwt'] >= 0.0
    assert result.final_metrics['permuted_fashion_mnist_mean_margin'] >= 0.05
    assert result.final_metrics['permuted_fashion_mnist_first_task_final_accuracy'] >= 0.28
    assert result.final_metrics['permuted_fashion_mnist_last_task_accuracy'] >= 0.24
    assert result.final_metrics['permuted_fashion_mnist_support_count'] == 400.0
    assert result.final_metrics['permuted_fashion_mnist_eval_count'] == 250.0
    assert result.final_metrics['mean_energy'] >= 0.95
    assert result.final_metrics['mean_stress'] <= 0.05
    assert (
        result.final_metrics['permuted_fashion_mnist_dataset_source_fashion_mnist']
        + result.final_metrics['permuted_fashion_mnist_dataset_source_kmnist']
        + result.final_metrics['permuted_fashion_mnist_dataset_source_mnist_fallback']
        == 1.0
    )
