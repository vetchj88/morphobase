from __future__ import annotations

from pathlib import Path

from morphobase.assays.common import AssayResult
from morphobase.assays.split_mnist import SplitMNISTAssay


class SplitFashionMNISTAssay(SplitMNISTAssay):
    METRIC_PREFIX = "split_fashion_mnist"

    def _load_dataset(self, root: Path):
        try:
            from torchvision.datasets import FashionMNIST
            from torchvision.transforms import ToTensor

            train_set = FashionMNIST(root=str(root), train=True, download=True, transform=ToTensor())
            test_set = FashionMNIST(root=str(root), train=False, download=True, transform=ToTensor())
            train_images = train_set.data.numpy().astype("float32") / 255.0
            train_labels = train_set.targets.numpy()
            test_images = test_set.data.numpy().astype("float32") / 255.0
            test_labels = test_set.targets.numpy()
            return train_images, train_labels, test_images, test_labels, "fashion_mnist"
        except Exception:
            try:
                from torchvision.datasets import KMNIST
                from torchvision.transforms import ToTensor

                train_set = KMNIST(root=str(root), train=True, download=True, transform=ToTensor())
                test_set = KMNIST(root=str(root), train=False, download=True, transform=ToTensor())
                train_images = train_set.data.numpy().astype("float32") / 255.0
                train_labels = train_set.targets.numpy()
                test_images = test_set.data.numpy().astype("float32") / 255.0
                test_labels = test_set.targets.numpy()
                return train_images, train_labels, test_images, test_labels, "kmnist"
            except Exception:
                return super()._load_dataset(root)

    def _run_condition(self, cfg, *, condition_name: str) -> AssayResult:
        result = super()._run_condition(cfg, condition_name=condition_name)
        remapped_metrics = {}
        dataset_source = "unknown"

        for key, value in result.final_metrics.items():
            if key.startswith("split_mnist_"):
                remapped_metrics[key.replace("split_mnist_", f"{self.METRIC_PREFIX}_", 1)] = value
            else:
                remapped_metrics[key] = value

        if remapped_metrics.get(f"{self.METRIC_PREFIX}_dataset_source_mnist", 0.0) == 1.0:
            dataset_source = "mnist"
        remapped_metrics.pop(f"{self.METRIC_PREFIX}_dataset_source_mnist", None)

        note_parts = []
        if "source=" in result.notes:
            dataset_source = result.notes.split("source=", 1)[1].split(";", 1)[0].strip()
        remapped_metrics[f"{self.METRIC_PREFIX}_dataset_source_fashion_mnist"] = 1.0 if dataset_source == "fashion_mnist" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_dataset_source_kmnist"] = 1.0 if dataset_source == "kmnist" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_dataset_source_mnist_fallback"] = 1.0 if dataset_source == "mnist" else 0.0

        for fragment in (
            result.notes.replace("Split-MNIST", "Split-FashionMNIST").replace("split_mnist", "split_fashion_mnist"),
            f"dataset_family={dataset_source}",
        ):
            if fragment:
                note_parts.append(fragment)

        return AssayResult(
            history=result.history,
            final_metrics=remapped_metrics,
            notes=" ".join(note_parts),
        )
