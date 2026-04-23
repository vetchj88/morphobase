from pathlib import Path
import matplotlib.pyplot as plt

def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_scalar_history(history, key: str, path, title=None):
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(range(len(history)), [float(h.get(key,0.0)) for h in history]); ax.set_title(title or key); ax.set_xlabel('log index'); ax.set_ylabel(key); _save(fig, path)

def plot_stage_occupancy(history, path):
    stage_keys = sorted({k for item in history for k in item if k.startswith('stage_')})
    fig, ax = plt.subplots(figsize=(7,4))
    if stage_keys:
        ys = [[float(item.get(k,0.0)) for item in history] for k in stage_keys]
        ax.stackplot(range(len(history)), ys, labels=stage_keys); ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Stage Occupancy'); ax.set_xlabel('log index'); ax.set_ylabel('fraction'); _save(fig, path)
