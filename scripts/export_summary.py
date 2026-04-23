import argparse
from pathlib import Path


def build_combined_summary(root: Path) -> str:
    sections = ["# Combined Run Summaries", ""]
    for summary_path in sorted(root.glob("*/summary.md")):
        sections.append(f"## {summary_path.parent.name}")
        sections.append("")
        sections.append(summary_path.read_text(encoding="utf-8").rstrip())
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    root = Path(args.artifacts)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "combined_summary.md"
    out_path.write_text(build_combined_summary(root), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
