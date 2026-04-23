from pathlib import Path


def build_markdown_summary(config: dict, final_metrics: dict, notes: str = '') -> str:
    lines = [
        f"# Run Summary: {config['run']['name']}",
        '',
        '## Assay',
        f"- name: {config['assay']['name']}",
        f"- seed: {config['run']['seed']}",
        '',
        '## Final metrics',
    ]
    for k, v in sorted(final_metrics.items()):
        lines.append(f"- {k}: {v}")
    if notes:
        lines += ['', '## Notes', notes]
    return '\n'.join(lines) + '\n'


def write_summary(path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
