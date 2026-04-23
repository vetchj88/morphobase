from __future__ import annotations

import numpy as np

from morphobase.assays.common import (
    AssayResult,
    AssayRunner,
    apply_retraining_correction,
    build_synthetic_body,
    disrupt_port_region,
    lesion_segment,
)
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.organism.scheduler import Scheduler


class LesionPreservesCompetenceAssay(AssayRunner):
    SIGNAL_BLOCK = 24
    OUTPUT_LAG = 8
    PRE_WINDOW = 48
    DROP_WINDOW = 32
    RECOVERY_OFFSET = 64
    INPUT_HIGH = 0.85
    INPUT_LOW = 0.15
    OUTPUT_BLEND_FIELD = 0.6
    OUTPUT_BLEND_HIDDEN = 0.4
    TASKS = ("relay_tracking", "inverted_remap")
    LESIONS = ("cell_ablation", "port_disruption")
    MODES = ("organismal", "no_gradient", "retraining")

    @staticmethod
    def _signal_value(step: int) -> float:
        return (
            LesionPreservesCompetenceAssay.INPUT_HIGH
            if (step // LesionPreservesCompetenceAssay.SIGNAL_BLOCK) % 2 == 0
            else LesionPreservesCompetenceAssay.INPUT_LOW
        )

    @classmethod
    def _task_target(cls, task_name: str, step: int) -> float:
        signal = cls._signal_value(step)
        if task_name == "relay_tracking":
            return signal
        if task_name == "inverted_remap":
            return 1.0 - signal
        raise ValueError(f"Unknown task: {task_name}")

    @staticmethod
    def _decode_output(body, output_slice: slice) -> float:
        hidden_mean = float(body.state.hidden[output_slice].mean())
        field_mean = float(body.state.field_alignment[output_slice].mean())
        hidden_score = np.clip(hidden_mean / 1.5, 0.0, 1.0)
        return float(
            np.clip(
                LesionPreservesCompetenceAssay.OUTPUT_BLEND_FIELD * field_mean
                + LesionPreservesCompetenceAssay.OUTPUT_BLEND_HIDDEN * hidden_score,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _window_mean(trace: list[float], start: int, stop: int, *, offset: int = 0) -> float:
        values = [
            value
            for index, value in enumerate(trace)
            if start <= (index + offset) < stop
        ]
        return float(np.mean(values)) if values else 0.0

    def _lesion_bounds(self, cfg, lesion_name: str) -> tuple[int, int]:
        if lesion_name == "cell_ablation":
            start = max(1, cfg.body.num_cells // 3)
            stop = min(cfg.body.num_cells, start + max(2, cfg.body.num_cells // 5))
            return start, stop
        if lesion_name == "port_disruption":
            start = 2
            stop = min(cfg.body.num_cells - 2, start + max(4, cfg.body.num_cells // 8))
            return start, stop
        raise ValueError(f"Unknown lesion: {lesion_name}")

    def _apply_lesion(self, body, lesion_name: str, start: int, stop: int) -> None:
        if lesion_name == "cell_ablation":
            lesion_segment(body, start, stop)
            return
        if lesion_name == "port_disruption":
            disrupt_port_region(body, start, stop, attenuation=0.30)
            return
        raise ValueError(f"Unknown lesion: {lesion_name}")

    def _run_rollout(
        self,
        cfg,
        *,
        task_name: str,
        lesion_name: str | None,
        mode: str,
        reference_states: list | None = None,
    ) -> dict:
        seed = (
            cfg.run.seed
            + 1000 * (lesion_name is not None)
            + sum(ord(ch) for ch in (lesion_name or "control"))
            + 17 * sum(ord(ch) for ch in task_name)
        )
        np.random.seed(seed)
        body = build_synthetic_body(cfg)
        scheduler = Scheduler()
        history = []
        state_history = [body.state.copy()]
        z_history = [body.state.z_alignment.copy()]
        outputs: list[float] = []
        targets: list[float] = []

        output_slice = slice(cfg.body.num_cells - 4, cfg.body.num_cells)
        lesion_start, lesion_stop = self._lesion_bounds(cfg, lesion_name or "cell_ablation")
        lesion_step = cfg.runtime.total_steps // 2

        for step in range(cfg.runtime.total_steps):
            target_signal = self._task_target(task_name, step)
            if lesion_name is not None and step == lesion_step:
                self._apply_lesion(body, lesion_name, lesion_start, lesion_stop)

            due = scheduler.due(step)
            post_lesion_no_gradient = mode == "no_gradient" and lesion_name is not None and step >= lesion_step
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                target_signal,
                no_gradient=post_lesion_no_gradient,
            )
            if mode == "retraining" and lesion_name is not None and step >= lesion_step and reference_states is not None:
                reference_index = min(step + 1, len(reference_states) - 1)
                apply_retraining_correction(
                    body,
                    reference_states[reference_index],
                    lesion_start,
                    lesion_stop,
                )

            outputs.append(self._decode_output(body, output_slice))
            targets.append(target_signal)
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if step % cfg.runtime.log_every == 0 or step == cfg.runtime.total_steps - 1:
                history.append(summarize_state(body.state, z_history=z_history))

        competence_trace = [
            float(np.clip(1.0 - abs(outputs[idx] - targets[idx - self.OUTPUT_LAG]), 0.0, 1.0))
            for idx in range(self.OUTPUT_LAG, len(outputs))
        ]
        final_metrics = history[-1].copy()
        final_metrics["lightcone_proxy"] = lightcone_proxy(state_history)
        return {
            "history": history,
            "state_history": state_history,
            "competence_trace": competence_trace,
            "outputs": outputs,
            "targets": targets,
            "final_metrics": final_metrics,
            "lesion_step": lesion_step,
        }

    def _summarize_case(self, rollout: dict, control_rollout: dict) -> dict[str, float]:
        lesion_step = rollout["lesion_step"]
        pre_range = (lesion_step - self.PRE_WINDOW, lesion_step)
        drop_range = (lesion_step, lesion_step + self.DROP_WINDOW)
        recovery_range = (lesion_step + self.RECOVERY_OFFSET, len(rollout["outputs"]))

        pre_comp = self._window_mean(rollout["competence_trace"], *pre_range, offset=self.OUTPUT_LAG)
        drop_comp = self._window_mean(rollout["competence_trace"], *drop_range, offset=self.OUTPUT_LAG)
        post_comp = self._window_mean(rollout["competence_trace"], *recovery_range, offset=self.OUTPUT_LAG)
        control_post = self._window_mean(control_rollout["competence_trace"], *recovery_range, offset=self.OUTPUT_LAG)

        return {
            "pre_lesion_competence": pre_comp,
            "post_lesion_competence": drop_comp,
            "post_recovery_competence": post_comp,
            "control_post_recovery_competence": control_post,
            "competence_drop": float(max(pre_comp - drop_comp, 0.0)),
            "competence_retention_ratio": float(post_comp / max(pre_comp, 1e-8)),
            "control_relative_post_recovery": float(post_comp / max(control_post, 1e-8)),
            "competence_recovery_gain": float(post_comp - drop_comp),
        }

    def run(self, cfg):
        per_task_results: dict[str, dict[str, dict[str, dict]]] = {}
        representative_history = None
        representative_metrics = None

        for task_name in self.TASKS:
            task_results = {}
            for lesion_name in self.LESIONS:
                control_rollout = self._run_rollout(cfg, task_name=task_name, lesion_name=None, mode="organismal")
                lesion_results = {}
                for mode in self.MODES:
                    reference_states = control_rollout["state_history"] if mode == "retraining" else None
                    lesion_rollout = self._run_rollout(
                        cfg,
                        task_name=task_name,
                        lesion_name=lesion_name,
                        mode=mode,
                        reference_states=reference_states,
                    )
                    lesion_results[mode] = self._summarize_case(lesion_rollout, control_rollout)
                    if representative_history is None and mode == "organismal":
                        representative_history = lesion_rollout["history"]
                        representative_metrics = lesion_rollout["final_metrics"].copy()
                task_results[lesion_name] = lesion_results
            per_task_results[task_name] = task_results

        final_metrics = representative_metrics or {}
        organismal_posts = []
        organismal_pre = []
        no_gradient_posts = []
        retraining_posts = []
        supported_task_lesions = 0
        supported_tasks = 0

        for task_name, task_results in per_task_results.items():
            task_organismal_posts = []
            task_organismal_pre = []
            task_no_gradient_posts = []
            task_retraining_posts = []
            task_supported_lesions = 0

            for lesion_name, mode_results in task_results.items():
                organismal = mode_results["organismal"]
                no_gradient = mode_results["no_gradient"]
                retraining = mode_results["retraining"]

                organismal_posts.append(organismal["post_recovery_competence"])
                organismal_pre.append(organismal["pre_lesion_competence"])
                no_gradient_posts.append(no_gradient["post_recovery_competence"])
                retraining_posts.append(retraining["post_recovery_competence"])
                task_organismal_posts.append(organismal["post_recovery_competence"])
                task_organismal_pre.append(organismal["pre_lesion_competence"])
                task_no_gradient_posts.append(no_gradient["post_recovery_competence"])
                task_retraining_posts.append(retraining["post_recovery_competence"])

                if (
                    organismal["post_recovery_competence"] >= 0.70
                    and organismal["competence_retention_ratio"] >= 0.90
                ):
                    supported_task_lesions += 1
                    task_supported_lesions += 1

                for mode_name, metrics in mode_results.items():
                    prefix = f"{task_name}_{lesion_name}_{mode_name}"
                    for key, value in metrics.items():
                        final_metrics[f"{prefix}_{key}"] = value

                final_metrics[f"{task_name}_{lesion_name}_organismal_competence_advantage_vs_no_gradient"] = float(
                    organismal["post_recovery_competence"] - no_gradient["post_recovery_competence"]
                )
                final_metrics[f"{task_name}_{lesion_name}_organismal_competence_vs_retraining_ratio"] = float(
                    organismal["post_recovery_competence"] / max(retraining["post_recovery_competence"], 1e-8)
                )

            task_pre = float(np.mean(task_organismal_pre))
            task_post = float(np.mean(task_organismal_posts))
            task_no_gradient = float(np.mean(task_no_gradient_posts))
            task_retraining = float(np.mean(task_retraining_posts))
            task_retention = float(task_post / max(task_pre, 1e-8))
            final_metrics[f"{task_name}_pre_lesion_competence"] = task_pre
            final_metrics[f"{task_name}_post_recovery_competence"] = task_post
            final_metrics[f"{task_name}_competence_retention_ratio"] = task_retention
            final_metrics[f"{task_name}_organismal_competence_advantage_vs_no_gradient"] = float(
                task_post - task_no_gradient
            )
            final_metrics[f"{task_name}_organismal_competence_vs_retraining_ratio"] = float(
                task_post / max(task_retraining, 1e-8)
            )
            final_metrics[f"{task_name}_supported_lesion_count"] = float(task_supported_lesions)
            if task_supported_lesions == len(self.LESIONS):
                supported_tasks += 1

        final_metrics["pre_lesion_competence"] = float(np.mean(organismal_pre))
        final_metrics["post_recovery_competence"] = float(np.mean(organismal_posts))
        final_metrics["competence_retention_ratio"] = float(
            final_metrics["post_recovery_competence"] / max(final_metrics["pre_lesion_competence"], 1e-8)
        )
        final_metrics["no_gradient_post_recovery_competence"] = float(np.mean(no_gradient_posts))
        final_metrics["retraining_post_recovery_competence"] = float(np.mean(retraining_posts))
        final_metrics["organismal_competence_advantage_vs_no_gradient"] = float(
            final_metrics["post_recovery_competence"] - final_metrics["no_gradient_post_recovery_competence"]
        )
        final_metrics["organismal_competence_vs_retraining_ratio"] = float(
            final_metrics["post_recovery_competence"] / max(final_metrics["retraining_post_recovery_competence"], 1e-8)
        )
        final_metrics["competence_supported_lesion_count"] = float(supported_task_lesions)
        final_metrics["competence_supported_task_count"] = float(supported_tasks)

        notes = (
            "Lesion-preserves-competence assay ran relay and inverted-remap toy tasks with mid-run lesions. "
            f"Pre={final_metrics['pre_lesion_competence']:.4f}; "
            f"post={final_metrics['post_recovery_competence']:.4f}; "
            f"retention={final_metrics['competence_retention_ratio']:.4f}; "
            f"vs_no_gradient={final_metrics['organismal_competence_advantage_vs_no_gradient']:.4f}; "
            f"vs_retraining={final_metrics['organismal_competence_vs_retraining_ratio']:.4f}."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )
