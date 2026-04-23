from morphobase.assays.smoke import SmokeAssay
from morphobase.config.validate import load_config

def test_smoke_assay_runs():
    cfg = load_config('configs/assay/smoke.yaml')
    result = SmokeAssay().run(cfg)
    assert result.history
    assert 'mean_energy' in result.final_metrics
