from __future__ import annotations

from pathlib import Path

import numpy as np

from morphobase.assays.common import AssayResult
from morphobase.assays.mnist_sanity import MNISTSanityAssay
from morphobase.ports.mnist_port import MNISTPort
from morphobase.training.trainer import Trainer


class PermutedMNISTAssay(MNISTSanityAssay):
    CLASS_IDS = tuple(range(10))
    TASK_COUNT = 5
    SUPPORT_PER_CLASS = 8
    EVAL_PER_CLASS = 5
    SETTLE_STEPS = 18
    ABLATION_CONDITIONS = ("baseline", "no_growth", "no_stress", "no_z_field")
    GROWTH_TRIGGER_THRESHOLD = 0.30
    CHALLENGE_PERIOD = 4
    SETTLE_CHALLENGE_PERIOD = 3
    SUPPORT_SELECTION_SEED = 42
    EVAL_SELECTION_SEED = 59
    PERMUTATION_SEED = 253

    def __init__(self) -> None:
        self.challenge_variant = "standard"
        self._task_context: dict[str, np.ndarray | float] | None = None
        self._permutation_signature = np.zeros(12, dtype=float)

    def _support_per_class(self) -> int:
        return self.SUPPORT_PER_CLASS

    def _eval_per_class(self) -> int:
        return self.EVAL_PER_CLASS

    def _task_permutations(self, seed: int, image_shape: tuple[int, int]) -> list[np.ndarray]:
        rng = np.random.default_rng(self.PERMUTATION_SEED)
        pixel_count = int(np.prod(image_shape))
        return [rng.permutation(pixel_count) for _ in range(self.TASK_COUNT)]

    @staticmethod
    def _apply_permutation(image: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        flat = np.asarray(image, dtype=np.float32).reshape(-1)
        remapped = flat[permutation]
        return remapped.reshape(image.shape)

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
        folded_profile = 0.55 * row_profile + 0.45 * np.interp(
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

    def _augment_embedding(self, embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=float)
        return np.concatenate([embedding, 0.30 * self._permutation_signature], axis=0)

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
        row_intensity = float(np.mean(np.abs(row_signal)))
        row_peaks = float(np.mean(np.partition(row_signal, -4)[-4:])) if row_signal.size >= 4 else row_intensity
        challenge_load = 0.76 * row_gradient + 0.44 * row_intensity + 0.22 * row_peaks

        input_mask = port.support_mask("input")
        output_mask = port.support_mask("output")
        support_mask = port.union_mask(input_mask, output_mask)
        relay_slice = self._focus_slice(row_signal, body.state.hidden.shape[0], radius=2)

        body.state.stress[support_mask] = np.clip(
            body.state.stress[support_mask] + 0.26 * challenge_load,
            0.0,
            5.0,
        )
        body.state.energy[support_mask] = np.clip(
            body.state.energy[support_mask] - 0.13 * challenge_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] - 0.34 * challenge_load,
            0.0,
            1.0,
        )
        body.state.hidden[relay_slice] *= (1.0 - 0.12 * challenge_load)
        body.state.membrane[relay_slice] = np.clip(
            body.state.membrane[relay_slice] * (1.0 - 0.16 * challenge_load),
            -1.0,
            1.0,
        )
        body.state.stress[relay_slice] = np.clip(
            body.state.stress[relay_slice] + 0.21 * challenge_load,
            0.0,
            5.0,
        )
        body.state.energy[relay_slice] = np.clip(
            body.state.energy[relay_slice] - 0.10 * challenge_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[relay_slice] = np.clip(
            body.state.field_alignment[relay_slice] - 0.25 * challenge_load,
            0.0,
            1.0,
        )

        if row_index % self.CHALLENGE_PERIOD == 0:
            body.state.hidden[output_mask] *= (1.0 - 0.14 * challenge_load)
            body.state.membrane[output_mask] = np.clip(
                body.state.membrane[output_mask] * (1.0 - 0.22 * challenge_load),
                -1.0,
                1.0,
            )
            body.state.energy[output_mask] = np.clip(
                body.state.energy[output_mask] - 0.20 * challenge_load,
                0.0,
                1.0,
            )
            body.state.stress[output_mask] = np.clip(
                body.state.stress[output_mask] + 0.34 * challenge_load,
                0.0,
                5.0,
            )
            body.state.field_alignment[output_mask] = np.clip(
                body.state.field_alignment[output_mask] - 0.40 * challenge_load,
                0.0,
                1.0,
            )
            body.state.z_alignment[output_mask] *= (1.0 - 0.14 * challenge_load)

    def _apply_settle_load(self, body, port: MNISTPort, image: np.ndarray, settle_index: int, condition_name: str) -> None:
        if self.challenge_variant != "growth_probe":
            return

        image_signal = np.asarray(image, dtype=float)
        image_intensity = float(np.mean(np.abs(image_signal)))
        image_contrast = float(np.std(image_signal))
        edge_load = float(
            np.mean(np.abs(np.diff(image_signal, axis=0))) + np.mean(np.abs(np.diff(image_signal, axis=1)))
        ) / 2.0
        settle_load = 0.48 * image_intensity + 0.52 * image_contrast + 0.28 * edge_load

        output_mask = port.support_mask("output")
        input_mask = port.support_mask("input")
        relay_profile = np.mean(np.abs(image_signal), axis=0)
        relay_slice = self._focus_slice(relay_profile, body.state.hidden.shape[0], radius=1)

        body.state.stress[relay_slice] = np.clip(
            body.state.stress[relay_slice] + 0.16 * settle_load,
            0.0,
            5.0,
        )
        body.state.energy[relay_slice] = np.clip(
            body.state.energy[relay_slice] - 0.11 * settle_load,
            0.0,
            1.0,
        )
        body.state.field_alignment[relay_slice] = np.clip(
            body.state.field_alignment[relay_slice] - 0.22 * settle_load,
            0.0,
            1.0,
        )

        if settle_index % self.SETTLE_CHALLENGE_PERIOD == 0:
            body.state.energy[output_mask] = np.clip(
                body.state.energy[output_mask] - 0.15 * settle_load,
                0.0,
                1.0,
            )
            body.state.stress[output_mask] = np.clip(
                body.state.stress[output_mask] + 0.28 * settle_load,
                0.0,
                5.0,
            )
            body.state.field_alignment[output_mask] = np.clip(
                body.state.field_alignment[output_mask] - 0.30 * settle_load,
                0.0,
                1.0,
            )
            body.state.hidden[output_mask] *= (1.0 - 0.11 * settle_load)
            body.state.z_alignment[output_mask] *= (1.0 - 0.09 * settle_load)
            body.state.energy[input_mask] = np.clip(
                body.state.energy[input_mask] - 0.08 * settle_load,
                0.0,
                1.0,
            )
            body.state.stress[input_mask] = np.clip(
                body.state.stress[input_mask] + 0.12 * settle_load,
                0.0,
                5.0,
            )

    def _set_task_context(self, permutation: np.ndarray, image_shape: tuple[int, int]) -> None:
        self._permutation_signature = self._build_permutation_signature(permutation, image_shape)
        if self.challenge_variant != "growth_probe":
            self._task_context = None
            return

        rows, cols = image_shape
        remapped_rows = (permutation // cols).reshape(rows, cols).astype(np.float32)
        remapped_cols = (permutation % cols).reshape(rows, cols).astype(np.float32)
        row_profile = (remapped_rows.mean(axis=1) / max(rows - 1, 1)) * 2.0 - 1.0
        col_profile = (remapped_cols.mean(axis=0) / max(cols - 1, 1)) * 2.0 - 1.0
        folded_profile = 0.58 * row_profile + 0.42 * np.interp(
            np.linspace(0.0, cols - 1, rows),
            np.arange(cols),
            col_profile,
        )
        context_gain = float(np.clip(np.std(remapped_rows) / max(rows, 1) + np.std(remapped_cols) / max(cols, 1), 0.12, 0.42))
        profile = np.clip(folded_profile * (0.55 + context_gain), -1.0, 1.0)
        self._task_context = {
            "profile": profile,
            "global_bias": float(np.clip(np.mean(profile), -0.35, 0.35)),
            "gain": context_gain,
        }

    def _apply_task_context(self, body, port: MNISTPort, row_index: int) -> None:
        if self.challenge_variant != "growth_probe" or not self._task_context:
            return

        profile = np.asarray(self._task_context["profile"], dtype=float)
        gain = float(self._task_context["gain"])
        global_bias = float(self._task_context["global_bias"])
        row_bias = float(profile[row_index % profile.size]) if profile.size else 0.0
        num_cells = body.state.hidden.shape[0]
        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        relay_slice = self._focus_slice(profile, num_cells, radius=2)
        body.state.z_memory[support_mask] = np.clip(
            body.state.z_memory[support_mask] * (1.0 - 0.10 * gain) + (0.24 + 0.18 * gain) * row_bias,
            -1.0,
            1.0,
        )
        body.state.z_alignment[support_mask] = np.clip(
            body.state.z_alignment[support_mask] * (1.0 - 0.08 * gain) + (0.16 + 0.12 * gain) * row_bias,
            -1.0,
            1.0,
        )
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] + 0.04 * gain * np.clip(1.0 - abs(row_bias), 0.0, 1.0),
            0.0,
            1.0,
        )
        body.state.hidden[relay_slice] = np.clip(
            body.state.hidden[relay_slice] + (0.06 + 0.05 * gain) * row_bias,
            -2.0,
            2.0,
        )
        body.state.z_memory[relay_slice] = np.clip(
            body.state.z_memory[relay_slice] + (0.12 + 0.10 * gain) * global_bias,
            -1.0,
            1.0,
        )
        body.state.z_alignment[relay_slice] = np.clip(
            body.state.z_alignment[relay_slice] + (0.10 + 0.06 * gain) * row_bias,
            -1.0,
            1.0,
        )

    def _apply_settle_context(self, body, port: MNISTPort, settle_index: int) -> None:
        if self.challenge_variant != "growth_probe" or not self._task_context:
            return

        profile = np.asarray(self._task_context["profile"], dtype=float)
        gain = float(self._task_context["gain"])
        global_bias = float(self._task_context["global_bias"])
        output_mask = port.support_mask("output")
        relay_slice = self._focus_slice(profile, body.state.hidden.shape[0], radius=1)
        phase_bias = float(profile[settle_index % profile.size]) if profile.size else 0.0
        body.state.z_memory[output_mask] = np.clip(
            body.state.z_memory[output_mask] + (0.10 + 0.08 * gain) * global_bias,
            -1.0,
            1.0,
        )
        body.state.z_alignment[output_mask] = np.clip(
            body.state.z_alignment[output_mask] + (0.08 + 0.05 * gain) * phase_bias,
            -1.0,
            1.0,
        )
        body.state.hidden[relay_slice] = np.clip(
            body.state.hidden[relay_slice] + (0.04 + 0.04 * gain) * global_bias,
            -2.0,
            2.0,
        )
        body.state.field_alignment[relay_slice] = np.clip(
            body.state.field_alignment[relay_slice] + 0.03 * gain,
            0.0,
            1.0,
        )

    def run_condition(self, cfg, condition_name: str = "baseline"):
        if condition_name not in self.ABLATION_CONDITIONS:
            raise ValueError(f"Unknown Permuted-MNIST condition '{condition_name}'.")
        dataset_root = Path(cfg.run.output_dir) / "datasets"
        train_images, train_labels, test_images, test_labels, source = self._load_dataset(dataset_root)
        support_images, support_labels = self._balanced_select(
            train_images,
            train_labels,
            classes=self.CLASS_IDS,
            per_class=self._support_per_class(),
            seed=self.SUPPORT_SELECTION_SEED,
        )
        eval_images, eval_labels = self._balanced_select(
            test_images,
            test_labels,
            classes=self.CLASS_IDS,
            per_class=self._eval_per_class(),
            seed=self.EVAL_SELECTION_SEED,
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
        final_model = None
        seen_train_embeddings: list[np.ndarray] = []
        seen_train_labels: list[np.ndarray] = []
        feature_dim = task_support_embeddings[0].shape[1]
        for task_index, task_embeddings in enumerate(task_support_embeddings):
            seen_train_embeddings.append(task_embeddings)
            seen_train_labels.append(support_labels)
            train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
            train_labels = np.concatenate(seen_train_labels, axis=0)

            final_model = Trainer().train_step(train_embeddings, train_labels)

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

        final_metrics = representative_metrics or {}
        final_metrics["permuted_mnist_task_count"] = float(self.TASK_COUNT)
        final_metrics["permuted_mnist_final_accuracy_mean"] = final_accuracy_mean
        final_metrics["permuted_mnist_peak_accuracy_mean"] = peak_accuracy_mean
        final_metrics["permuted_mnist_mean_forgetting"] = mean_forgetting
        final_metrics["permuted_mnist_bwt"] = bwt
        final_metrics["permuted_mnist_mean_margin"] = float(mean_margin)
        final_metrics["permuted_mnist_support_count"] = float(len(support_labels) * self.TASK_COUNT)
        final_metrics["permuted_mnist_eval_count"] = float(len(eval_labels) * self.TASK_COUNT)
        final_metrics["permuted_mnist_dataset_source_mnist"] = 1.0 if source == "mnist" else 0.0
        final_metrics["permuted_mnist_first_task_final_accuracy"] = task_final_accuracies.get(0, 0.0)
        final_metrics["permuted_mnist_last_task_accuracy"] = task_final_accuracies.get(self.TASK_COUNT - 1, 0.0)
        final_metrics["permuted_mnist_peak_growth_pressure_mean"] = float(np.mean(growth_pressure_peaks)) if growth_pressure_peaks else 0.0
        final_metrics["permuted_mnist_peak_growth_pressure_max"] = float(np.max(growth_pressure_peaks)) if growth_pressure_peaks else 0.0
        final_metrics["permuted_mnist_growth_trigger_threshold"] = self.GROWTH_TRIGGER_THRESHOLD
        final_metrics["permuted_mnist_growth_trigger_crossed_fraction"] = (
            float(np.mean(growth_trigger_crossings)) if growth_trigger_crossings else 0.0
        )
        final_metrics["permuted_mnist_embedding_augmented_dim"] = float(feature_dim)
        final_metrics["permuted_mnist_condition_is_baseline"] = 1.0 if condition_name == "baseline" else 0.0
        final_metrics["permuted_mnist_condition_no_growth"] = 1.0 if condition_name == "no_growth" else 0.0
        final_metrics["permuted_mnist_condition_no_stress"] = 1.0 if condition_name == "no_stress" else 0.0
        final_metrics["permuted_mnist_condition_no_z_field"] = 1.0 if condition_name == "no_z_field" else 0.0

        for task_index, accuracy in task_final_accuracies.items():
            final_metrics[f"permuted_mnist_task_{task_index}_final_accuracy"] = float(accuracy)
        for task_index, peak_accuracy in task_peak_accuracies.items():
            final_metrics[f"permuted_mnist_task_{task_index}_peak_accuracy"] = float(peak_accuracy)

        notes = (
            "Permuted-MNIST assay ran a controlled five-task permutation sequence on top of the explicit MNIST port. "
            f"condition={condition_name}; source={source}; final_mean={final_accuracy_mean:.4f}; "
            f"forgetting={mean_forgetting:.4f}; bwt={bwt:.4f}; challenge_variant={self.challenge_variant}; "
            "representation=permutation_signature+prototype_readout."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )

    def run(self, cfg):
        return self.run_condition(cfg, "baseline")
