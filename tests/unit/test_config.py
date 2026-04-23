from morphobase.config.validate import load_config

def test_load_config_smoke():
    cfg = load_config('configs/assay/smoke.yaml')
    assert cfg.assay.name == 'smoke'
    assert cfg.body.num_cells > 0
