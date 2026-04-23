from __future__ import annotations

from pathlib import Path

import numpy as np

from morphobase.assays.common import AssayResult
from morphobase.assays.permuted_mnist import PermutedMNISTAssay
from morphobase.training.trainer import SequentialLinearTrainer


class PermutedFashionMNISTAssay(PermutedMNISTAssay):
    METRIC_PREFIX = "permuted_fashion_mnist"

    def __init__(self) -> None:
        super().__init__()
        self._permutation_signature = np.zeros(12, dtype=float)

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

    @staticmethod
    def _build_permutation_signature(permutation: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        rows, cols = image_shape
        remapped_rows = (permutation // cols).reshape(rows, cols).astype(np.float32)
        remapped_cols = (permutation % cols).reshape(rows, cols).astype(np.float32)
        base_rows = np.repeat(np.arange(rows, dtype=np.float32)[:, None], cols, axis=1)
        base_cols = np.repeat(np.arange(cols, dtype=np.float32)[None, :], rows, axis=0)

        row_shift = (remapped_rows - base_rows) / max(rows - 1, 1)
        col_shift = (remapped_cols - base_cols) / max(cols - 1, 1)
        row_profile = row_shift.mean(axis=1)
        col_profile = col_shift.mean(axis=0)
        folded_profile = 0.5 * row_profile + 0.5 * np.interp(
            np.linspace(0.0, cols - 1, rows),
            np.arange(cols),
            col_profile,
        )

        signature = np.array(
            [
                float(row_shift.mean()),
                float(row_shift.std()),
                float(col_shift.mean()),
                float(col_shift.std()),
                float(np.mean(np.abs(row_shift))),
                float(np.mean(np.abs(col_shift))),
                float(row_profile[: rows // 2].mean() - row_profile[rows // 2 :].mean()),
                float(col_profile[: cols // 2].mean() - col_profile[cols // 2 :].mean()),
                float(folded_profile.mean()),
                float(folded_profile.std()),
                float(np.max(folded_profile)),
                float(np.min(folded_profile)),
            ],
            dtype=float,
        )
        return np.clip(signature, -1.0, 1.0)

    def _set_task_context(self, permutation: np.ndarray, image_shape: tuple[int, int]) -> None:
        super()._set_task_context(permutation, image_shape)
        self._permutation_signature = self._build_permutation_signature(permutation, image_shape)

    def _augment_embedding(self, embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=float)
        signature = np.asarray(self._permutation_signature, dtype=float)
        return np.concatenate([embedding, 0.35 * signature], axis=0)

    def run_condition(self, cfg, condition_name: str = "baseline"):
        if condition_name not in self.ABLATION_CONDITIONS:
            raise ValueError(f"Unknown Permuted-FashionMNIST condition '{condition_name}'.")

        dataset_root = Path(cfg.run.output_dir) / "datasets"
        train_images, train_labels, test_images, test_labels, source = self._load_dataset(dataset_root)
        support_images, support_labels = self._balanced_select(
            train_images,
            train_labels,
            classes=self.CLASS_IDS,
            per_class=self._support_per_class(),
            seed=cfg.run.seed,
        )
        eval_images, eval_labels = self._balanced_select(
            test_images,
            test_labels,
            classes=self.CLASS_IDS,
            per_class=self._eval_per_class(),
            seed=cfg.run.seed + 17,
        )

        permutations = self._task_permutations(cfg.run.seed, support_images[0].shape)
        task_support_embeddings: list[np.ndarray] = []
        task_eval_embeddings: list[np.ndarray] = []
        representative_history = None
        representative_metrics = None
        growth_pressure_peaks: list[float] = []
        growth_trigger_crossings: list[float] = []

        for permutation in permutations:
            self._set_task_context(permutation, support_images[0].shape)
            support_embeddings = []
            eval_embeddings = []

            for image in support_images:
                rollout = self._rollout_image(
                    cfg,
                    self._apply_permutation(image, permutation),
                    condition_name=condition_name,
                )
                support_embeddings.append(self._augment_embedding(rollout["embedding"]))
                growth_pressure_peaks.append(float(rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0)))
                growth_trigger_crossings.append(
                    1.0
                    if rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0) >= self.GROWTH_TRIGGER_THRESHOLD
                    else 0.0
                )
                if representative_history is None:
                    representative_history = rollout["history"]
                    representative_metrics = rollout["final_metrics"].copy()

            for image in eval_images:
                rollout = self._rollout_image(
                    cfg,
                    self._apply_permutation(image, permutation),
                    condition_name=condition_name,
                )
                eval_embeddings.append(self._augment_embedding(rollout["embedding"]))
                growth_pressure_peaks.append(float(rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0)))
                growth_trigger_crossings.append(
                    1.0
                    if rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0) >= self.GROWTH_TRIGGER_THRESHOLD
                    else 0.0
                )

            task_support_embeddings.append(np.stack(support_embeddings, axis=0))
            task_eval_embeddings.append(np.stack(eval_embeddings, axis=0))

        task_peak_accuracies: dict[int, float] = {}
        task_initial_accuracies: dict[int, float] = {}
        task_final_accuracies: dict[int, float] = {}
        evaluation_grid: list[list[float]] = []
        feature_dim = task_support_embeddings[0].shape[1]
        trainer = SequentialLinearTrainer(np.array(self.CLASS_IDS, dtype=int), feature_dim, seed=cfg.run.seed + 911)
        final_model = None
        seen_train_embeddings: list[np.ndarray] = []
        seen_train_labels: list[np.ndarray] = []

        for task_index, task_embeddings in enumerate(task_support_embeddings):
            seen_train_embeddings.append(task_embeddings)
            seen_train_labels.append(support_labels)
            train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
            train_labels_all = np.concatenate(seen_train_labels, axis=0)

            final_model = trainer.train_task(
                train_embeddings,
                train_labels_all,
                epochs=96,
                learning_rate=0.14,
                l2=3e-4,
            )

            row = []
            for seen_task_index, eval_embeddings in enumerate(task_eval_embeddings[: task_index + 1]):
                accuracy = final_model.score(eval_embeddings, eval_labels)
                row.append(accuracy)
                task_peak_accuracies[seen_task_index] = max(task_peak_accuracies.get(seen_task_index, 0.0), accuracy)
                if seen_task_index == task_index:
                    task_initial_accuracies[seen_task_index] = accuracy
                task_final_accuracies[seen_task_index] = accuracy
            evaluation_grid.append(row)

        forgetting_values = [
            task_peak_accuracies[index] - task_final_accuracies.get(index, 0.0)
            for index in range(self.TASK_COUNT - 1)
        ]
        bwt_values = [
            task_final_accuracies.get(index, 0.0) - task_initial_accuracies.get(index, 0.0)
            for index in range(self.TASK_COUNT - 1)
        ]
        final_row = evaluation_grid[-1] if evaluation_grid else []
        final_accuracy_mean = float(np.mean(final_row)) if final_row else 0.0
        peak_accuracy_mean = float(np.mean(list(task_peak_accuracies.values()))) if task_peak_accuracies else 0.0
        mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        bwt = float(np.mean(bwt_values)) if bwt_values else 0.0
        mean_margin = final_model.mean_margin(task_eval_embeddings[-1]) if final_model is not None else 0.0

        remapped_metrics = representative_metrics or {}
        remapped_metrics[f"{self.METRIC_PREFIX}_task_count"] = float(self.TASK_COUNT)
        remapped_metrics[f"{self.METRIC_PREFIX}_final_accuracy_mean"] = final_accuracy_mean
        remapped_metrics[f"{self.METRIC_PREFIX}_peak_accuracy_mean"] = peak_accuracy_mean
        remapped_metrics[f"{self.METRIC_PREFIX}_mean_forgetting"] = mean_forgetting
        remapped_metrics[f"{self.METRIC_PREFIX}_bwt"] = bwt
        remapped_metrics[f"{self.METRIC_PREFIX}_mean_margin"] = float(mean_margin)
        remapped_metrics[f"{self.METRIC_PREFIX}_support_count"] = float(len(support_labels) * self.TASK_COUNT)
        remapped_metrics[f"{self.METRIC_PREFIX}_eval_count"] = float(len(eval_labels) * self.TASK_COUNT)
        remapped_metrics[f"{self.METRIC_PREFIX}_first_task_final_accuracy"] = task_final_accuracies.get(0, 0.0)
        remapped_metrics[f"{self.METRIC_PREFIX}_last_task_accuracy"] = task_final_accuracies.get(self.TASK_COUNT - 1, 0.0)
        remapped_metrics[f"{self.METRIC_PREFIX}_peak_growth_pressure_mean"] = float(np.mean(growth_pressure_peaks)) if growth_pressure_peaks else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_peak_growth_pressure_max"] = float(np.max(growth_pressure_peaks)) if growth_pressure_peaks else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_growth_trigger_threshold"] = self.GROWTH_TRIGGER_THRESHOLD
        remapped_metrics[f"{self.METRIC_PREFIX}_growth_trigger_crossed_fraction"] = (
            float(np.mean(growth_trigger_crossings)) if growth_trigger_crossings else 0.0
        )
        remapped_metrics[f"{self.METRIC_PREFIX}_condition_is_baseline"] = 1.0 if condition_name == "baseline" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_condition_no_growth"] = 1.0 if condition_name == "no_growth" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_condition_no_stress"] = 1.0 if condition_name == "no_stress" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_condition_no_z_field"] = 1.0 if condition_name == "no_z_field" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_dataset_source_fashion_mnist"] = 1.0 if source == "fashion_mnist" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_dataset_source_kmnist"] = 1.0 if source == "kmnist" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_dataset_source_mnist_fallback"] = 1.0 if source == "mnist" else 0.0
        remapped_metrics[f"{self.METRIC_PREFIX}_embedding_augmented_dim"] = float(feature_dim)

        for task_index, accuracy in task_final_accuracies.items():
            remapped_metrics[f"{self.METRIC_PREFIX}_task_{task_index}_final_accuracy"] = float(accuracy)
        for task_index, peak_accuracy in task_peak_accuracies.items():
            remapped_metrics[f"{self.METRIC_PREFIX}_task_{task_index}_peak_accuracy"] = float(peak_accuracy)

        notes = (
            "Permuted-FashionMNIST assay ran a controlled five-task permutation sequence on top of the explicit MNIST port. "
            f"condition={condition_name}; source={source}; final_mean={final_accuracy_mean:.4f}; "
            f"forgetting={mean_forgetting:.4f}; bwt={bwt:.4f}; challenge_variant={self.challenge_variant}; "
            "representation=permutation_signature+sequential_linear_readout."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=remapped_metrics,
            notes=notes,
        )

    def run(self, cfg):
        return self.run_condition(cfg, "baseline")
