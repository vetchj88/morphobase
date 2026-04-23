# MorphoBASE v1.3: Organism-First Adaptive Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**MorphoBASE** is a computational substrate that inverts the standard machine learning design sequence. Instead of building a task optimizer and retrofitting adaptation, we build a synthetic multicellular body with explicit physiology—metabolism, stress, repair, growth control, and setpoint memory—and attach task competence through boundary ports.

This project is inspired by research in developmental biology and bioelectric signaling (notably the work of Michael Levin and colleagues), exploring how multi-scale competency and homeostatic maintenance can drive robust AI adaptation.

## 🔑 Key Features

- **Physiological Substrate**: Multicellular organisms with three-clock dynamics (fast cell updates, medium communication, slow developmental repair).
- **Setpoint Memory (Z-Field)**: Hidden anatomical scaffolds that guide morphological repair after severe injury.
- **Lesion Recovery**: Demonstrated 85% mean morphological recovery across diverse lesion types without gradient-based retraining.
- **Competence Preservation**: 99% task competence retention after injury, verified across multiple benchmark families.
- **Selective Growth**: Repair-oriented growth control that responds to physiological need rather than decorative expansion.
- **Task-Agnostic Core**: A body that maintains itself independently of the tasks (visual, symbolic, or control) attached to its boundary ports.

## 📖 Whitepaper

A detailed analysis of the architecture, methodology, and experimental results can be found in:
[**MorphoBASE v1.3 Whitepaper**](docs/MorphoBASE_v1.3_Whitepaper.md)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/justin/morphobase.git
cd morphobase

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running an Assay

To run a basic survival "smoke" assay:
```bash
python scripts/run_assay.py --config configs/assay/smoke.yaml
```

To run the full lesion recovery battery:
```bash
python scripts/run_assay.py --config configs/assay/lesion_battery.yaml
```

### Running Tests
```bash
pytest
```

## 📊 Experimental Results

MorphoBASE v1.3 has been validated across 25+ assays with matched ablation controls. Highlights include:

| Metric | Result |
|--------|--------|
| Mean Morphological Recovery | 84.95% |
| Competence Retention after Injury | 98.94% |
| Recovery without Gradients Ratio | 1.17 |
| Boundary Locality Ratio | 1.69 |

*Note: Visual benchmark performance (MNIST/FashionMNIST) ranges from 27-47%. The system is optimized for robustness and repair, not raw benchmark scores.*

## 📂 Project Structure

- `morphobase/`: Core package including cells, communication, metabolism, and organism logic.
- `configs/`: YAML definitions for assays, sweeps, and builds.
- `scripts/`: Research runners and figure generation scripts.
- `docs/`: Whitepaper, architecture specifications, and figures.
- `tests/`: Unit and integration test suite.

## 📝 Citation

If you use MorphoBASE in your research, please cite the whitepaper:

```bibtex
@article{morphobase2026,
  title={MorphoBASE v1.3: A Computational Substrate for Organism-First Adaptive Intelligence},
  author={Justin},
  year={2026},
  journal={Internal Report / GitHub Repository}
}
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
