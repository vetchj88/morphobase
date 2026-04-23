from __future__ import annotations

import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, build_synthetic_body
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.organism.scheduler import Scheduler
from morphobase.ports.toy_rule_port import ToyRulePort
from morphobase.training.trainer import SequentialLinearTrainer


class SequentialRulesAssay(AssayRunner):
    METRIC_PREFIX = "sequential_rules"
    TASK_LABEL = "Sequential-rules"
    CLASS_IDS = tuple(range(10))
    TASK_SPLITS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    SUPPORT_PER_CLASS = 10
    EVAL_PER_CLASS = 6
    SEQUENCE_LENGTH = 12
    SETTLE_STEPS = 4
    ABLATION_CONDITIONS = ("baseline", "no_growth", "no_stress", "no_z_field")
    GROWTH_TRIGGER_THRESHOLD = 0.18
    CHALLENGE_PERIOD = 3
    CONDITION_SPECS = {
        "baseline": {"allow_growth": True, "z_field": True},
        "no_growth": {"allow_growth": False, "z_field": True},
        "no_stress": {"allow_growth": True, "z_field": True},
        "no_z_field": {"allow_growth": True, "z_field": False},
    }

    def __init__(self) -> None:
        self.challenge_variant = "standard"

    @staticmethod
    def _balanced_labels(classes: tuple[int, ...], per_class: int) -> np.ndarray:
        return np.concatenate(
            [np.full(per_class, label, dtype=int) for label in classes],
            axis=0,
        )

    def _build_sequence(self, label: int, sample_index: int, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 97 * label + 13 * sample_index)
        phase = (sample_index % 3) * 0.03
        seq = np.full(self.SEQUENCE_LENGTH, 0.12, dtype=float)
        task_id = label // 2
        variant = label % 2

        if task_id == 0:
            pulse = slice(1, 4) if variant == 0 else slice(8, 11)
            seq[pulse] = 0.88
        elif task_id == 1:
            ramp = np.linspace(0.12, 0.88, self.SEQUENCE_LENGTH)
            seq = ramp if variant == 0 else ramp[::-1]
        elif task_id == 2:
            seq = np.full(self.SEQUENCE_LENGTH, 0.14, dtype=float)
            spike_positions = np.arange(0, self.SEQUENCE_LENGTH, 2) if variant == 0 else np.arange(1, self.SEQUENCE_LENGTH, 2)
            seq[spike_positions] = 0.86
        elif task_id == 3:
            seq = np.full(self.SEQUENCE_LENGTH, 0.14, dtype=float)
            if variant == 0:
                seq[4:8] = np.array([0.45, 0.82, 0.82, 0.45], dtype=float)
            else:
                seq[4:8] = np.array([0.72, 0.18, 0.18, 0.72], dtype=float)
        else:
            third = self.SEQUENCE_LENGTH // 3
            if variant == 0:
                seq[:third] = 0.82
                seq[third : 2 * third] = 0.18
                seq[2 * third :] = 0.82
            else:
                seq[:third] = 0.18
                seq[third : 2 * third] = 0.82
                seq[2 * third :] = 0.18

        seq = np.roll(seq, sample_index % 2)
        seq = np.clip(seq + phase + rng.normal(0.0, 0.025, size=self.SEQUENCE_LENGTH), 0.0, 1.0)
        return seq.astype(np.float32)

    def _build_dataset(self, *, classes: tuple[int, ...], per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        sequences = []
        labels = []
        for label in classes:
            for sample_index in range(per_class):
                sequences.append(self._build_sequence(label, sample_index, seed=seed))
                labels.append(label)
        return np.stack(sequences, axis=0), np.asarray(labels, dtype=int)

    @staticmethod
    def _sequence_signature(sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=float)
        positions = np.linspace(0.0, 1.0, seq.size)
        high_mask = seq >= 0.6
        center_of_mass = float(np.sum(positions * seq) / max(np.sum(seq), 1e-8))
        return np.array(
            [
                float(seq.mean()),
                float(seq.std()),
                float(seq[0]),
                float(seq[-1]),
                float(seq[-1] - seq[0]),
                float(np.mean(np.abs(np.diff(seq)))),
                float(np.mean(high_mask)),
                center_of_mass,
            ],
            dtype=float,
        )

    @staticmethod
    def _sequence_setpoint(sequence: np.ndarray) -> float:
        seq = np.asarray(sequence, dtype=float)
        prefix_mean = float(np.mean(seq[: max(seq.size // 3, 1)]))
        suffix_mean = float(np.mean(seq[-max(seq.size // 3, 1) :]))
        edge_contrast = float(seq[-1] - seq[0])
        local_variation = float(np.mean(np.abs(np.diff(seq)))) if seq.size > 1 else 0.0
        return float(np.clip(np.tanh(2.4 * (prefix_mean - suffix_mean) + 1.5 * edge_contrast + 0.8 * local_variation), -1.0, 1.0))

    @staticmethod
    def _apply_token_load(body, port: ToyRulePort, token: float, prev_token: float) -> None:
        support_mask = port.support_mask("input")
        transition = abs(token - prev_token)
        body.state.stress[support_mask] = np.clip(
            body.state.stress[support_mask] + 0.05 * transition + 0.02 * token,
            0.0,
            5.0,
        )
        body.state.energy[support_mask] = np.clip(
            body.state.energy[support_mask] - 0.03 * transition,
            0.0,
            1.0,
        )
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] - 0.04 * transition + 0.03 * token,
            0.0,
            1.0,
        )

    def _apply_probe_load(self, body, port: ToyRulePort, token: float, prev_token: float, token_index: int) -> None:
        if self.challenge_variant != "repair_probe":
            return
        transition = abs(token - prev_token)
        challenge_load = 0.58 * transition + 0.24 * token + 0.18 * abs(token - 0.5)
        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        output_mask = port.support_mask("output")
        num_cells = body.state.hidden.shape[0]
        relay_slice = slice(max(0, num_cells // 2 - 3), min(num_cells, num_cells // 2 + 3))

        body.state.stress[support_mask] = np.clip(body.state.stress[support_mask] + 0.18 * challenge_load, 0.0, 5.0)
        body.state.energy[support_mask] = np.clip(body.state.energy[support_mask] - 0.09 * challenge_load, 0.0, 1.0)
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] - 0.14 * challenge_load,
            0.0,
            1.0,
        )
        body.state.predictive_error[relay_slice] = np.clip(
            body.state.predictive_error[relay_slice] + 0.16 * challenge_load,
            0.0,
            1.0,
        )
        body.state.z_memory[output_mask] = np.clip(
            0.82 * body.state.z_memory[output_mask] + 0.18 * np.tanh(2.0 * (token - 0.5)),
            -1.0,
            1.0,
        )

        if token_index % self.CHALLENGE_PERIOD == 0:
            body.state.hidden[output_mask] *= (1.0 - 0.12 * challenge_load)
            body.state.membrane[output_mask] = np.clip(
                body.state.membrane[output_mask] * (1.0 - 0.18 * challenge_load),
                -1.0,
                1.0,
            )
            body.state.stress[output_mask] = np.clip(body.state.stress[output_mask] + 0.24 * challenge_load, 0.0, 5.0)
            body.state.energy[output_mask] = np.clip(body.state.energy[output_mask] - 0.10 * challenge_load, 0.0, 1.0)
            body.state.field_alignment[output_mask] = np.clip(
                body.state.field_alignment[output_mask] - 0.20 * challenge_load,
                0.0,
                1.0,
            )
            body.state.z_alignment[output_mask] *= (1.0 - 0.12 * challenge_load)

    def _apply_settle_load(self, body, port: ToyRulePort, sequence: np.ndarray, settle_index: int) -> None:
        if self.challenge_variant != "repair_probe":
            return
        tail_mean = float(np.mean(sequence[-3:]))
        tail_variation = float(np.mean(np.abs(np.diff(sequence[-4:])))) if sequence.size >= 4 else 0.0
        settle_load = 0.42 * tail_mean + 0.48 * tail_variation + 0.16 * abs(self._sequence_setpoint(sequence))
        output_mask = port.support_mask("output")
        relay_mask = port.union_mask(port.support_mask("input"), output_mask)

        body.state.stress[relay_mask] = np.clip(body.state.stress[relay_mask] + 0.16 * settle_load, 0.0, 5.0)
        body.state.energy[relay_mask] = np.clip(body.state.energy[relay_mask] - 0.08 * settle_load, 0.0, 1.0)
        body.state.field_alignment[relay_mask] = np.clip(
            body.state.field_alignment[relay_mask] - 0.16 * settle_load,
            0.0,
            1.0,
        )
        body.state.predictive_error[relay_mask] = np.clip(
            body.state.predictive_error[relay_mask] + 0.12 * settle_load,
            0.0,
            1.0,
        )
        if settle_index % self.CHALLENGE_PERIOD == 0:
            body.state.hidden[output_mask] *= (1.0 - 0.10 * settle_load)
            body.state.membrane[output_mask] = np.clip(
                body.state.membrane[output_mask] * (1.0 - 0.14 * settle_load),
                -1.0,
                1.0,
            )

    def _condition_spec(self, condition_name: str) -> dict[str, bool]:
        if condition_name not in self.CONDITION_SPECS:
            raise ValueError(f"Unknown SequentialRules condition '{condition_name}'.")
        return dict(self.CONDITION_SPECS[condition_name])

    def _apply_condition_post_step(self, body, port: ToyRulePort, condition_name: str) -> None:
        if condition_name == "baseline":
            return
        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        distal_mask = port.distal_mask()
        if condition_name == "no_stress":
            body.state.stress[distal_mask] *= 0.08
            body.state.stress[support_mask] = np.clip(body.state.stress[support_mask] + 0.03, 0.0, 5.0)
            body.state.hidden[distal_mask] *= 0.985
            body.state.membrane[distal_mask] *= 0.96
            body.state.field_alignment[distal_mask] *= 0.84
            support_indices = np.flatnonzero(support_mask)
            distal_indices = np.flatnonzero(distal_mask)
            if support_indices.size and distal_indices.size:
                body.state.conductance[np.ix_(distal_indices, support_indices)] *= 0.90
                body.state.conductance[np.ix_(support_indices, distal_indices)] *= 0.90
                diagonal = np.diag_indices_from(body.state.conductance)
                body.state.conductance = np.clip(body.state.conductance, 0.0, 2.0)
                body.state.conductance[diagonal] = 1.0
        if condition_name == "no_z_field":
            body.state.z_alignment.fill(0.0)
            body.state.z_memory.fill(0.0)

    def _maybe_apply_internal_lesion(
        self,
        body,
        port: ToyRulePort,
        *,
        sequence: np.ndarray,
        token_index: int,
        phase: str,
    ) -> None:
        return

    def _rollout_sequence(self, cfg, sequence: np.ndarray, *, condition_name: str = "baseline") -> dict:
        condition = self._condition_spec(condition_name)
        body = build_synthetic_body(cfg)
        port = ToyRulePort(cfg.body.num_cells)
        scheduler = Scheduler()
        history = []
        state_history = [body.state.copy()]
        z_history = [body.state.z_alignment.copy()]
        prev_token = float(sequence[0])

        for token_index, token in enumerate(sequence):
            token = float(token)
            port.apply_input(body, token)
            self._apply_token_load(body, port, token, prev_token)
            self._apply_probe_load(body, port, token, prev_token, token_index)
            self._maybe_apply_internal_lesion(
                body,
                port,
                sequence=sequence,
                token_index=token_index,
                phase="sequence",
            )
            self._apply_condition_post_step(body, port, condition_name)
            due = scheduler.due(body.state.step_count)
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                token,
                allow_growth=condition["allow_growth"],
            )
            self._apply_condition_post_step(body, port, condition_name)
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if body.state.step_count % cfg.runtime.log_every == 0:
                history.append(summarize_state(body.state, z_history=z_history))
            prev_token = token

        settle_target = float(np.mean(sequence[-3:]))
        for settle_index in range(self.SETTLE_STEPS):
            self._apply_settle_load(body, port, sequence, settle_index)
            self._maybe_apply_internal_lesion(
                body,
                port,
                sequence=sequence,
                token_index=settle_index,
                phase="settle",
            )
            self._apply_condition_post_step(body, port, condition_name)
            due = scheduler.due(body.state.step_count)
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                settle_target,
                allow_growth=condition["allow_growth"],
            )
            self._apply_condition_post_step(body, port, condition_name)
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if body.state.step_count % cfg.runtime.log_every == 0:
                history.append(summarize_state(body.state, z_history=z_history))

        final_metrics = summarize_state(body.state, z_history=z_history)
        final_metrics["lightcone_proxy"] = lightcone_proxy(state_history)
        boundary_state = port.capture_boundary_state(body, kind="output")
        hidden = np.asarray(boundary_state["hidden"], dtype=float)
        membrane = np.asarray(boundary_state["membrane"], dtype=float)
        field = np.asarray(boundary_state["field_alignment"], dtype=float)
        z_alignment = np.asarray(boundary_state["z_alignment"], dtype=float)
        z_memory = np.asarray(boundary_state["z_memory"], dtype=float)
        sequence_signature = self._sequence_signature(sequence)
        embedding = np.concatenate(
            [
                np.asarray(sequence, dtype=float),
                hidden.reshape(-1),
                membrane.reshape(-1),
                field.reshape(-1),
                z_alignment.reshape(-1),
                z_memory.reshape(-1),
                sequence_signature,
                np.array(
                    [
                        final_metrics["mean_energy"],
                        final_metrics["mean_stress"],
                        final_metrics["mean_plasticity"],
                        final_metrics["mean_z_alignment"],
                        final_metrics["mean_growth_pressure"],
                        final_metrics["lightcone_proxy"],
                    ],
                    dtype=float,
                ),
            ],
            axis=0,
        )
        return {
            "embedding": embedding,
            "history": history,
            "final_metrics": final_metrics,
        }

    def run_condition(self, cfg, condition_name: str = "baseline"):
        support_sequences, support_labels = self._build_dataset(
            classes=self.CLASS_IDS,
            per_class=self.SUPPORT_PER_CLASS,
            seed=cfg.run.seed,
        )
        eval_sequences, eval_labels = self._build_dataset(
            classes=self.CLASS_IDS,
            per_class=self.EVAL_PER_CLASS,
            seed=cfg.run.seed + 17,
        )

        support_embeddings_by_task: list[np.ndarray] = []
        eval_embeddings_by_task: list[np.ndarray] = []
        representative_history = None
        representative_metrics = None

        for task_classes in self.TASK_SPLITS:
            task_support = []
            task_eval = []
            task_mask_support = np.isin(support_labels, task_classes)
            task_mask_eval = np.isin(eval_labels, task_classes)
            for sequence in support_sequences[task_mask_support]:
                rollout = self._rollout_sequence(cfg, sequence, condition_name=condition_name)
                task_support.append(rollout["embedding"])
                if representative_history is None:
                    representative_history = rollout["history"]
                    representative_metrics = rollout["final_metrics"].copy()
            for sequence in eval_sequences[task_mask_eval]:
                rollout = self._rollout_sequence(cfg, sequence, condition_name=condition_name)
                task_eval.append(rollout["embedding"])
            support_embeddings_by_task.append(np.stack(task_support, axis=0))
            eval_embeddings_by_task.append(np.stack(task_eval, axis=0))

        trainer = SequentialLinearTrainer(np.array(self.CLASS_IDS, dtype=int), support_embeddings_by_task[0].shape[1], seed=cfg.run.seed + 503)
        seen_classes: list[int] = []
        seen_train_embeddings: list[np.ndarray] = []
        seen_train_labels: list[np.ndarray] = []
        task_peak_accuracies: dict[int, float] = {}
        task_initial_accuracies: dict[int, float] = {}
        task_final_accuracies: dict[int, float] = {}
        evaluation_grid: list[list[float]] = []
        final_model = None

        for task_index, task_classes in enumerate(self.TASK_SPLITS):
            task_support_mask = np.isin(support_labels, task_classes)
            seen_train_embeddings.append(support_embeddings_by_task[task_index])
            seen_train_labels.append(support_labels[task_support_mask])
            train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
            train_labels = np.concatenate(seen_train_labels, axis=0)

            final_model = trainer.train_task(
                train_embeddings,
                train_labels,
                epochs=88,
                learning_rate=0.16,
                l2=2e-4,
            )

            seen_classes.extend(int(label) for label in task_classes)
            allowed_classes = np.array(sorted(set(seen_classes)), dtype=int)
            row = []
            for seen_task_index, seen_task_classes in enumerate(self.TASK_SPLITS[: task_index + 1]):
                eval_mask = np.isin(eval_labels, seen_task_classes)
                accuracy = final_model.score(
                    eval_embeddings_by_task[seen_task_index],
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
        mean_margin = final_model.mean_margin(np.concatenate(eval_embeddings_by_task, axis=0)) if final_model is not None else 0.0

        final_metrics = representative_metrics or {}
        prefix = self.METRIC_PREFIX
        final_metrics[f"{prefix}_task_count"] = float(len(self.TASK_SPLITS))
        final_metrics[f"{prefix}_final_accuracy_mean"] = final_accuracy_mean
        final_metrics[f"{prefix}_peak_accuracy_mean"] = peak_accuracy_mean
        final_metrics[f"{prefix}_mean_forgetting"] = mean_forgetting
        final_metrics[f"{prefix}_bwt"] = bwt
        final_metrics[f"{prefix}_mean_margin"] = float(mean_margin)
        final_metrics[f"{prefix}_support_count"] = float(len(support_labels))
        final_metrics[f"{prefix}_eval_count"] = float(len(eval_labels))
        final_metrics[f"{prefix}_sequence_length"] = float(self.SEQUENCE_LENGTH)
        final_metrics[f"{prefix}_port_family_rule"] = 1.0
        final_metrics[f"{prefix}_first_task_final_accuracy"] = task_final_accuracies.get(0, 0.0)
        final_metrics[f"{prefix}_last_task_accuracy"] = task_final_accuracies.get(len(self.TASK_SPLITS) - 1, 0.0)
        final_metrics[f"{prefix}_condition_is_baseline"] = 1.0 if condition_name == "baseline" else 0.0
        final_metrics[f"{prefix}_condition_no_growth"] = 1.0 if condition_name == "no_growth" else 0.0
        final_metrics[f"{prefix}_condition_no_stress"] = 1.0 if condition_name == "no_stress" else 0.0
        final_metrics[f"{prefix}_condition_no_z_field"] = 1.0 if condition_name == "no_z_field" else 0.0
        final_metrics[f"{prefix}_peak_growth_pressure_mean"] = float(
            max(item.get("mean_growth_pressure", 0.0) for item in representative_history) if representative_history else final_metrics["mean_growth_pressure"]
        )
        final_metrics[f"{prefix}_growth_trigger_threshold"] = self.GROWTH_TRIGGER_THRESHOLD
        final_metrics[f"{prefix}_growth_trigger_crossed"] = 1.0 if final_metrics[f"{prefix}_peak_growth_pressure_mean"] >= self.GROWTH_TRIGGER_THRESHOLD else 0.0
        for task_index, accuracy in task_final_accuracies.items():
            final_metrics[f"{prefix}_task_{task_index}_final_accuracy"] = float(accuracy)
        for task_index, peak_accuracy in task_peak_accuracies.items():
            final_metrics[f"{prefix}_task_{task_index}_peak_accuracy"] = float(peak_accuracy)

        notes = (
            f"{self.TASK_LABEL} assay ran a five-task non-visual symbolic bridge on top of the toy rule port. "
            f"final_mean={final_accuracy_mean:.4f}; forgetting={mean_forgetting:.4f}; "
            f"bwt={bwt:.4f}; margin={mean_margin:.4f}; condition={condition_name}; challenge_variant={self.challenge_variant}."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )

    def run(self, cfg):
        return self.run_condition(cfg, "baseline")
