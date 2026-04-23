from __future__ import annotations

from pathlib import Path

import numpy as np

from morphobase.assays.common import AssayResult
from morphobase.assays.mnist_sanity import MNISTSanityAssay
from morphobase.ports.mnist_port import MNISTPort
from morphobase.training.trainer import Trainer


class SplitMNISTAssay(MNISTSanityAssay):
    CLASS_IDS = tuple(range(10))
    TASK_SPLITS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    SUPPORT_PER_CLASS = 10
    EVAL_PER_CLASS = 6
    SETTLE_STEPS = 18
    ABLATION_CONDITIONS = ("baseline", "no_growth", "no_stress", "no_z_field")
    GROWTH_TRIGGER_THRESHOLD = 0.30
    CHALLENGE_PERIOD = 4
    SETTLE_CHALLENGE_PERIOD = 4

    def __init__(self) -> None:
        self.challenge_variant = "standard"

    @staticmethod
    def _focus_slice(signal: np.ndarray, num_cells: int, radius: int = 2) -> slice:
        weights = np.asarray(signal, dtype=float).reshape(-1)
        if not np.any(weights > 0.0):
            center = num_cells // 2
        else:
            positions = np.linspace(0.0, 1.0, weights.size)
            focus = float(np.sum(positions * weights) / max(np.sum(weights), 1e-8))
            center = int(round(focus * max(num_cells - 1, 1)))
        return slice(max(0, center - radius), min(num_cells, center + radius + 1))

    def _apply_row_load(self, body, port: MNISTPort, row: np.ndarray, row_index: int, condition_name: str) -> None:
        super()._apply_row_load(body, port, row, row_index, condition_name)
        if self.challenge_variant != "growth_probe":
            return

        row_signal = np.asarray(row, dtype=float).reshape(-1)
        row_gradient = float(np.mean(np.abs(np.diff(row_signal)))) if row_signal.size > 1 else 0.0
        row_intensity = float(np.mean(row_signal))
        challenge_load = 0.72 * row_gradient + 0.39 * row_intensity
        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        output_mask = port.support_mask("output")
        relay_slice = self._focus_slice(row_signal, body.state.hidden.shape[0], radius=2)

        body.state.stress[support_mask] = np.clip(
            body.state.stress[support_mask] + 0.22 * challenge_load,
            0.0,
            5.0,
        )
        body.state.energy[support_mask] = np.clip(
            body.state.energy[support_mask] - 0.10 * challenge_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] - 0.30 * challenge_load,
            0.0,
            1.0,
        )
        body.state.hidden[relay_slice] *= (1.0 - 0.10 * challenge_load)
        body.state.membrane[relay_slice] = np.clip(
            body.state.membrane[relay_slice] * (1.0 - 0.12 * challenge_load),
            -1.0,
            1.0,
        )
        body.state.stress[relay_slice] = np.clip(
            body.state.stress[relay_slice] + 0.18 * challenge_load,
            0.0,
            5.0,
        )
        body.state.energy[relay_slice] = np.clip(
            body.state.energy[relay_slice] - 0.08 * challenge_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[relay_slice] = np.clip(
            body.state.field_alignment[relay_slice] - 0.20 * challenge_load,
            0.0,
            1.0,
        )

        if row_index % self.CHALLENGE_PERIOD == 0:
            body.state.hidden[output_mask] *= (1.0 - 0.10 * challenge_load)
            body.state.membrane[output_mask] = np.clip(
                body.state.membrane[output_mask] * (1.0 - 0.18 * challenge_load),
                -1.0,
                1.0,
            )
            body.state.energy[output_mask] = np.clip(
                body.state.energy[output_mask] - 0.16 * challenge_load,
                0.0,
                1.0,
            )
            body.state.stress[output_mask] = np.clip(
                body.state.stress[output_mask] + 0.30 * challenge_load,
                0.0,
                5.0,
            )
            body.state.field_alignment[output_mask] = np.clip(
                body.state.field_alignment[output_mask] - 0.34 * challenge_load,
                0.0,
                1.0,
            )
            body.state.z_alignment[output_mask] *= (1.0 - 0.10 * challenge_load)

    def _apply_settle_load(self, body, port: MNISTPort, image: np.ndarray, settle_index: int, condition_name: str) -> None:
        if self.challenge_variant != "growth_probe":
            return

        image_signal = np.asarray(image, dtype=float)
        image_intensity = float(np.mean(image_signal))
        image_contrast = float(np.std(image_signal))
        settle_load = 0.43 * image_intensity + 0.45 * image_contrast
        output_mask = port.support_mask("output")
        relay_profile = np.mean(image_signal, axis=0)
        relay_slice = self._focus_slice(relay_profile, body.state.hidden.shape[0], radius=1)

        body.state.stress[relay_slice] = np.clip(
            body.state.stress[relay_slice] + 0.12 * settle_load,
            0.0,
            5.0,
        )
        body.state.energy[relay_slice] = np.clip(
            body.state.energy[relay_slice] - 0.08 * settle_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[relay_slice] = np.clip(
            body.state.field_alignment[relay_slice] - 0.18 * settle_load,
            0.0,
            1.0,
        )

        if settle_index % self.SETTLE_CHALLENGE_PERIOD == 0:
            body.state.energy[output_mask] = np.clip(
                body.state.energy[output_mask] - 0.12 * settle_load,
                0.0,
                1.0,
            )
            body.state.stress[output_mask] = np.clip(
                body.state.stress[output_mask] + 0.24 * settle_load,
                0.0,
                5.0,
            )
            body.state.field_alignment[output_mask] = np.clip(
                body.state.field_alignment[output_mask] - 0.28 * settle_load,
                0.0,
                1.0,
            )
            body.state.hidden[output_mask] *= (1.0 - 0.10 * settle_load)
            body.state.z_alignment[output_mask] *= (1.0 - 0.08 * settle_load)

    def _run_condition(self, cfg, *, condition_name: str) -> AssayResult:
        dataset_root = Path(cfg.run.output_dir) / "datasets"
        train_images, train_labels, test_images, test_labels, source = self._load_dataset(dataset_root)
        support_images, support_labels = self._balanced_select(
            train_images,
            train_labels,
            classes=self.CLASS_IDS,
            per_class=self.SUPPORT_PER_CLASS,
            seed=cfg.run.seed,
        )
        eval_images, eval_labels = self._balanced_select(
            test_images,
            test_labels,
            classes=self.CLASS_IDS,
            per_class=self.EVAL_PER_CLASS,
            seed=cfg.run.seed + 17,
        )

        support_embeddings = []
        eval_embeddings = []
        representative_history = None
        representative_metrics = None
        growth_pressure_peaks: list[float] = []
        growth_trigger_crossings: list[float] = []

        for image in support_images:
            rollout = self._rollout_image(cfg, image, condition_name=condition_name)
            support_embeddings.append(rollout["embedding"])
            growth_pressure_peaks.append(float(rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0)))
            growth_trigger_crossings.append(
                1.0 if rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0) >= self.GROWTH_TRIGGER_THRESHOLD else 0.0
            )
            if representative_history is None:
                representative_history = rollout["history"]
                representative_metrics = rollout["final_metrics"].copy()

        for image in eval_images:
            rollout = self._rollout_image(cfg, image, condition_name=condition_name)
            eval_embeddings.append(rollout["embedding"])
            growth_pressure_peaks.append(float(rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0)))
            growth_trigger_crossings.append(
                1.0 if rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0) >= self.GROWTH_TRIGGER_THRESHOLD else 0.0
            )

        support_embeddings = np.stack(support_embeddings, axis=0)
        eval_embeddings = np.stack(eval_embeddings, axis=0)

        seen_classes: list[int] = []
        task_peak_accuracies: dict[int, float] = {}
        task_initial_accuracies: dict[int, float] = {}
        task_final_accuracies: dict[int, float] = {}
        evaluation_grid: list[list[float]] = []
        final_model = None
        seen_train_embeddings: list[np.ndarray] = []
        seen_train_labels: list[np.ndarray] = []

        for task_index, task_classes in enumerate(self.TASK_SPLITS):
            task_classes = tuple(int(label) for label in task_classes)
            train_mask = np.isin(support_labels, task_classes)
            task_embeddings = support_embeddings[train_mask]
            task_labels = support_labels[train_mask]

            seen_train_embeddings.append(task_embeddings)
            seen_train_labels.append(task_labels)
            train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
            train_labels = np.concatenate(seen_train_labels, axis=0)

            final_model = Trainer().train_step(train_embeddings, train_labels)

            seen_classes.extend(task_classes)
            allowed_classes = np.array(sorted(set(seen_classes)), dtype=int)
            row = []
            for seen_task_index, seen_task_classes in enumerate(self.TASK_SPLITS[: task_index + 1]):
                eval_mask = np.isin(eval_labels, seen_task_classes)
                accuracy = final_model.score(
                    eval_embeddings[eval_mask],
                    eval_labels[eval_mask],
                    allowed_classes=allowed_classes,
                )
                row.append(accuracy)
                task_peak_accuracies[seen_task_index] = max(task_peak_accuracies.get(seen_task_index, 0.0), accuracy)
                if seen_task_index == task_index:
                    task_initial_accuracies[seen_task_index] = accuracy
                task_final_accuracies[seen_task_index] = accuracy
            evaluation_grid.append(row)


        forgetting_values = [
            task_peak_accuracies[index] - task_final_accuracies.get(index, 0.0)
            for index in range(len(self.TASK_SPLITS) - 1)
        ]
        bwt_values = [
            task_final_accuracies.get(index, 0.0) - task_initial_accuracies.get(index, 0.0)
            for index in range(len(self.TASK_SPLITS) - 1)
        ]
        final_row = evaluation_grid[-1] if evaluation_grid else []
        final_accuracy_mean = float(np.mean(final_row)) if final_row else 0.0
        peak_accuracy_mean = float(np.mean(list(task_peak_accuracies.values()))) if task_peak_accuracies else 0.0
        mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        bwt = float(np.mean(bwt_values)) if bwt_values else 0.0
        mean_margin = (
            final_model.mean_margin(eval_embeddings)
            if final_model is not None
            else 0.0
        )

        final_metrics = representative_metrics or {}
        final_metrics["split_mnist_task_count"] = float(len(self.TASK_SPLITS))
        final_metrics["split_mnist_final_accuracy_mean"] = final_accuracy_mean
        final_metrics["split_mnist_peak_accuracy_mean"] = peak_accuracy_mean
        final_metrics["split_mnist_mean_forgetting"] = mean_forgetting
        final_metrics["split_mnist_bwt"] = bwt
        final_metrics["split_mnist_mean_margin"] = float(mean_margin)
        final_metrics["split_mnist_support_count"] = float(len(support_labels))
        final_metrics["split_mnist_eval_count"] = float(len(eval_labels))
        final_metrics["split_mnist_dataset_source_mnist"] = 1.0 if source == "mnist" else 0.0
        final_metrics["split_mnist_first_task_final_accuracy"] = task_final_accuracies.get(0, 0.0)
        final_metrics["split_mnist_last_task_accuracy"] = task_final_accuracies.get(len(self.TASK_SPLITS) - 1, 0.0)
        final_metrics["split_mnist_peak_growth_pressure_mean"] = float(np.mean(growth_pressure_peaks)) if growth_pressure_peaks else 0.0
        final_metrics["split_mnist_peak_growth_pressure_max"] = float(np.max(growth_pressure_peaks)) if growth_pressure_peaks else 0.0
        final_metrics["split_mnist_growth_trigger_threshold"] = self.GROWTH_TRIGGER_THRESHOLD
        final_metrics["split_mnist_growth_trigger_crossed_fraction"] = (
            float(np.mean(growth_trigger_crossings)) if growth_trigger_crossings else 0.0
        )
        final_metrics["split_mnist_condition_is_baseline"] = 1.0 if condition_name == "baseline" else 0.0
        final_metrics["split_mnist_condition_no_growth"] = 1.0 if condition_name == "no_growth" else 0.0
        final_metrics["split_mnist_condition_no_stress"] = 1.0 if condition_name == "no_stress" else 0.0
        final_metrics["split_mnist_condition_no_z_field"] = 1.0 if condition_name == "no_z_field" else 0.0

        for task_index, accuracy in task_final_accuracies.items():
            final_metrics[f"split_mnist_task_{task_index}_final_accuracy"] = float(accuracy)
        for task_index, peak_accuracy in task_peak_accuracies.items():
            final_metrics[f"split_mnist_task_{task_index}_peak_accuracy"] = float(peak_accuracy)

        notes = (
            "Split-MNIST assay ran a minimal sequential five-task bridge on top of the explicit MNIST port. "
            f"condition={condition_name}; source={source}; final_mean={final_accuracy_mean:.4f}; "
            f"forgetting={mean_forgetting:.4f}; bwt={bwt:.4f}."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )

    def run_condition(self, cfg, condition_name: str) -> AssayResult:
        if condition_name not in self.ABLATION_CONDITIONS:
            raise ValueError(f"Unknown Split-MNIST condition '{condition_name}'.")
        return self._run_condition(cfg, condition_name=condition_name)

    def run(self, cfg):
        return self.run_condition(cfg, "baseline")
