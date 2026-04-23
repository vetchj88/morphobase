def allow_dedifferentiation(mean_stress: float, gap_closeable: bool) -> bool:
    return mean_stress > 0.8 and not gap_closeable
