import argparse, subprocess, yaml
from pathlib import Path

def merge(a,b):
    out=dict(a)
    for k,v in b.items(): out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    sw=Path(args.config); payload=yaml.safe_load(sw.read_text(encoding='utf-8'))
    base_path=Path(payload['base_config']); base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    for i,var in enumerate(payload.get('variations',[]), start=1):
        cfg=merge(base,var); tmp=sw.parent/f'_tmp_sweep_{i}.yaml'; tmp.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        subprocess.run(['python','scripts/run_assay.py','--config',str(tmp)], check=True); tmp.unlink(missing_ok=True)
if __name__ == '__main__': main()
