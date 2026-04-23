from morphobase.types import Alert, RunVerdict


def collect_alerts(summary: dict) -> list[Alert]:
    mean_energy = float(summary.get('mean_energy', 1.0))
    mean_commitment = float(summary.get('mean_commitment', 0.0))
    z_field_drift = float(summary.get('z_field_drift', 0.0))
    dormant_fraction = float(summary.get('dormant_fraction', 0.0))
    active_fraction = float(summary.get('active_fraction', 1.0))
    plasticity_loss_index = float(summary.get('plasticity_loss_index', 0.0))
    pseudo_maturity_index = float(summary.get('pseudo_maturity_index', 0.0))
    low_energy_fraction = float(summary.get('low_energy_fraction', 0.0))
    conductance_entropy = float(summary.get('conductance_entropy', 1.0))
    mean_growth_pressure = float(summary.get('mean_growth_pressure', 0.0))
    recent_growth_decorative_fraction = float(summary.get('recent_growth_decorative_fraction', 0.0))
    recent_structural_churn = float(summary.get('recent_structural_churn', 0.0))
    stress_variance = float(summary.get('stress_variance', 0.0))

    return [
        Alert('energy_floor', mean_energy <= 0.02, 'Energy exhausted.'),
        Alert('energy_reserve_low', mean_energy < 0.08 or low_energy_fraction > 0.85, 'Energy reserve is critically low.'),
        Alert('dead_field', z_field_drift < 0.02 and mean_commitment > 0.1, 'Z-field is effectively flat under ongoing adaptation.'),
        Alert('plasticity_loss', dormant_fraction > 0.75 or active_fraction < 0.1 or plasticity_loss_index > 0.65, 'Plasticity has largely collapsed.'),
        Alert('pseudo_maturity', pseudo_maturity_index > 0.35 and mean_commitment > 0.45, 'The body looks stable by over-commitment rather than healthy learning readiness.'),
        Alert('degenerate_lock', conductance_entropy < 0.25 or (z_field_drift < 0.02 and stress_variance < 1e-5 and mean_commitment > 0.2), 'The organism has fallen into a low-mobility locked state.'),
        Alert(
            'chronic_growth',
            (mean_growth_pressure > 0.55 and mean_energy < 0.15)
            or (recent_growth_decorative_fraction > 0.35 and recent_structural_churn > 0.08),
            'Growth remains active after mismatch resolution or is decoratively churning without utility.',
        ),
    ]


def hard_fail_alerts(summary: dict):
    return [alert for alert in collect_alerts(summary) if alert.triggered]


def classify_run(summary: dict) -> RunVerdict:
    alerts = {alert.name: alert for alert in collect_alerts(summary)}
    if alerts['energy_floor'].triggered:
        return RunVerdict.UNSTABLE
    if alerts['chronic_growth'].triggered:
        return RunVerdict.CHRONIC_GROWTH
    if alerts['pseudo_maturity'].triggered:
        return RunVerdict.PSEUDO_MATURITY
    if alerts['plasticity_loss'].triggered:
        return RunVerdict.PLASTICITY_LOSS
    if alerts['dead_field'].triggered:
        return RunVerdict.DEAD_FIELD
    if alerts['degenerate_lock'].triggered:
        return RunVerdict.DEGENERATE_LOCK
    return RunVerdict.PASS
