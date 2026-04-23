from morphobase.types import Alert

def audit_state(mean_stress: float, dormant_fraction: float, z_drift: float):
    return [Alert("high_stress", mean_stress > 1.0, "Mean stress above band."), Alert("plasticity_collapse", dormant_fraction > 0.7, "Dormant fraction too high."), Alert("dead_z_field", z_drift == 0.0, "Z-field flatlined.")]
