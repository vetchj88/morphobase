from __future__ import annotations

import numpy as np

from morphobase.assays.common import (
    AssayResult,
    AssayRunner,
    bias_conductance_region,
    bias_stress_region,
    bias_z_field,
    build_synthetic_body,
    lesion_segment,
    rollout_body,
)


class SetpointRewriteAssay(AssayRunner):
    REWRITE_MODES = ("z_bias", "conductance_bias", "stress_bias")

    def _region_field_context(self, body, start: int, stop: int) -> np.ndarray:
        return np.clip(2.0 * body.state.field_alignment[start:stop] - 1.0, -1.0, 1.0)

    def _apply_mode_bias(self, mode: str, body, start: int, stop: int) -> None:
        if mode == "z_bias":
            bias_z_field(body, start, stop, 0.18)
        elif mode == "conductance_bias":
            bias_conductance_region(body, start, stop)
            body.state.z_alignment[start:stop] = np.clip(body.state.z_alignment[start:stop] + 0.04, -1.0, 1.0)
        elif mode == "stress_bias":
            bias_stress_region(body, start, stop)
            body.state.field_alignment[start:stop] = np.clip(body.state.field_alignment[start:stop] * 0.7, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown rewrite mode: {mode}")

    def _imprint_hidden_rewrite(self, mode: str, body, start: int, stop: int, preserve_hidden: bool) -> None:
        visible_context = self._region_field_context(body, start, stop)
        if preserve_hidden:
            imprint = {
                "z_bias": 0.35,
                "conductance_bias": 0.32,
                "stress_bias": 0.24,
            }[mode]
            body.state.z_memory[start:stop] = np.clip(body.state.z_memory[start:stop] + imprint, -1.0, 1.0)
            body.state.z_alignment[start:stop] = np.clip(
                0.75 * body.state.z_alignment[start:stop] + 0.25 * body.state.z_memory[start:stop],
                -1.0,
                1.0,
            )
        else:
            body.state.z_memory[start:stop] = visible_context
            body.state.z_alignment[start:stop] = visible_context

    def _restore_visible_state(
        self,
        mode: str,
        body,
        start: int,
        stop: int,
        *,
        baseline_conductance: np.ndarray,
        preserve_hidden: bool,
        target_value: float,
    ) -> None:
        body.state.hidden[start:stop] = target_value
        body.state.membrane[start:stop] = 0.0
        region_mean_stress = float(body.state.stress.mean())
        body.state.stress[start:stop] = region_mean_stress
        body.state.field_alignment[start:stop] = np.clip(body.state.field_alignment.mean(), 0.0, 1.0)
        body.state.conductance[start:stop, :] = baseline_conductance[start:stop, :]
        body.state.conductance[:, start:stop] = baseline_conductance[:, start:stop]
        self._imprint_hidden_rewrite(mode, body, start, stop, preserve_hidden)

    def _cloak_visible_state(self, body, start: int, stop: int) -> None:
        surrounding_mean = body.state.hidden.mean(axis=0, keepdims=True)
        body.state.hidden[start:stop] = np.clip(
            0.35 * body.state.hidden[start:stop] + 0.65 * surrounding_mean,
            -2.0,
            2.0,
        )
        body.state.membrane[start:stop] *= 0.25
        visible_context = self._region_field_context(body, start, stop)
        body.state.z_alignment[start:stop] = np.clip(
            0.55 * body.state.z_alignment[start:stop] + 0.45 * visible_context,
            -1.0,
            1.0,
        )

    def _run_condition(
        self,
        cfg,
        *,
        mode: str | None,
        preserve_hidden: bool,
    ) -> dict:
        body = build_synthetic_body(cfg)
        baseline_conductance = body.state.conductance.copy()
        lesion_start = max(1, cfg.body.num_cells // 3)
        lesion_stop = min(cfg.body.num_cells, lesion_start + max(2, cfg.body.num_cells // 5))
        rewrite_start = cfg.runtime.total_steps // 6
        rewrite_stop = rewrite_start + max(6, cfg.runtime.total_steps // 8)
        lesion_steps = [int(cfg.runtime.total_steps * frac) for frac in (0.62, 0.76, 0.90)]
        lesion_steps = sorted({min(cfg.runtime.total_steps - 3, step) for step in lesion_steps if step > rewrite_stop + 8})
        cycle_end_steps = {
            lesion_step: lesion_steps[index + 1] - 1 if index + 1 < len(lesion_steps) else cfg.runtime.total_steps - 1
            for index, lesion_step in enumerate(lesion_steps)
        }
        markers: dict[str, float] = {}

        def target_schedule(step: int, _body) -> float:
            if mode is not None and rewrite_start <= step < rewrite_stop:
                return min(1.0, cfg.assay.target_value + 0.20)
            return cfg.assay.target_value

        def step_hook(step: int, hooked_body) -> None:
            if mode is not None and rewrite_start <= step < rewrite_stop:
                self._apply_mode_bias(mode, hooked_body, lesion_start, lesion_stop)

            if mode is not None and step == rewrite_stop:
                self._restore_visible_state(
                    mode,
                    hooked_body,
                    lesion_start,
                    lesion_stop,
                    baseline_conductance=baseline_conductance,
                    preserve_hidden=preserve_hidden,
                    target_value=cfg.assay.target_value,
                )
                post_context = self._region_field_context(hooked_body, lesion_start, lesion_stop)
                markers["post_rewrite_region_z"] = float(hooked_body.state.z_alignment[lesion_start:lesion_stop].mean())
                markers["post_rewrite_region_hidden"] = float(hooked_body.state.hidden[lesion_start:lesion_stop].mean())
                markers["post_rewrite_field_context"] = float(post_context.mean())

            if mode is not None and step == lesion_steps[0] - 1:
                self._cloak_visible_state(hooked_body, lesion_start, lesion_stop)

            if step == lesion_steps[0] - 1:
                pre_context = self._region_field_context(hooked_body, lesion_start, lesion_stop)
                markers["pre_lesion_region_z"] = float(hooked_body.state.z_alignment[lesion_start:lesion_stop].mean())
                markers["pre_lesion_field_context"] = float(pre_context.mean())
                markers["pre_lesion_visible_z_context_gap"] = float(
                    np.mean(np.abs(hooked_body.state.z_alignment[lesion_start:lesion_stop] - pre_context))
                )
                markers["pre_lesion_hidden_z_memory_gap"] = float(
                    np.mean(
                        np.abs(
                            hooked_body.state.z_memory[lesion_start:lesion_stop]
                            - hooked_body.state.z_alignment[lesion_start:lesion_stop]
                        )
                    )
                )
                markers["pre_lesion_region_hidden"] = float(hooked_body.state.hidden[lesion_start:lesion_stop].mean())

            if step in lesion_steps:
                cycle_index = lesion_steps.index(step) + 1
                lesion_segment(hooked_body, lesion_start, lesion_stop)
                hooked_body.state.z_alignment[lesion_start:lesion_stop] = np.clip(
                    0.15 * hooked_body.state.z_alignment[lesion_start:lesion_stop]
                    + 0.85 * hooked_body.state.z_memory[lesion_start:lesion_stop],
                    -1.0,
                    1.0,
                )
                markers[f"cycle_{cycle_index}_post_lesion_hidden"] = float(
                    hooked_body.state.hidden[lesion_start:lesion_stop].mean()
                )

            for lesion_step, cycle_end in cycle_end_steps.items():
                cycle_index = lesion_steps.index(lesion_step) + 1
                if step == cycle_end:
                    markers[f"cycle_{cycle_index}_final_hidden"] = float(
                        hooked_body.state.hidden[lesion_start:lesion_stop].mean()
                    )
                    markers[f"cycle_{cycle_index}_final_z"] = float(
                        hooked_body.state.z_alignment[lesion_start:lesion_stop].mean()
                    )
                    markers[f"cycle_{cycle_index}_visible_z_context_gap"] = float(
                        np.mean(
                            np.abs(
                                hooked_body.state.z_alignment[lesion_start:lesion_stop]
                                - self._region_field_context(hooked_body, lesion_start, lesion_stop)
                            )
                        )
                    )

        np.random.seed(cfg.run.seed + 17)
        rollout = rollout_body(
            cfg,
            body=body,
            step_hook=step_hook,
            target_schedule=target_schedule,
        )
        final_state = rollout["body"].state
        markers["final_region_z"] = float(final_state.z_alignment[lesion_start:lesion_stop].mean())
        markers["final_region_hidden"] = float(final_state.hidden[lesion_start:lesion_stop].mean())
        markers["lesion_cycles"] = float(len(lesion_steps))
        return {"rollout": rollout, "markers": markers}

    def _summarize_mode(self, mode: str, control: dict, rewritten: dict, falsified: dict) -> dict[str, float]:
        control_markers = control["markers"]
        rewritten_markers = rewritten["markers"]
        falsified_markers = falsified["markers"]

        pre_hidden_diff = abs(
            rewritten_markers["pre_lesion_region_hidden"] - control_markers["pre_lesion_region_hidden"]
        )
        pre_visible_context_divergence = abs(
            rewritten_markers["pre_lesion_visible_z_context_gap"]
            - control_markers["pre_lesion_visible_z_context_gap"]
        )
        hidden_z_memory_gap_advantage = (
            rewritten_markers["pre_lesion_hidden_z_memory_gap"]
            - control_markers["pre_lesion_hidden_z_memory_gap"]
        )

        cycle_persistences = []
        cycle_falsification_margins = []
        cycle_supported = 0
        lesion_cycles = int(rewritten_markers["lesion_cycles"])

        for cycle_index in range(1, lesion_cycles + 1):
            rewritten_hidden = rewritten_markers[f"cycle_{cycle_index}_final_hidden"]
            control_hidden = control_markers[f"cycle_{cycle_index}_final_hidden"]
            falsified_hidden = falsified_markers[f"cycle_{cycle_index}_final_hidden"]
            persistence = abs(rewritten_hidden - control_hidden)
            falsification_margin = persistence - abs(falsified_hidden - control_hidden)
            cycle_persistences.append(persistence)
            cycle_falsification_margins.append(falsification_margin)
            if persistence > 0.08 and falsification_margin > 0.04:
                cycle_supported += 1

        mode_metrics = {
            f"{mode}_pre_hidden_diff": pre_hidden_diff,
            f"{mode}_pre_visible_context_divergence": pre_visible_context_divergence,
            f"{mode}_hidden_z_memory_gap_advantage": hidden_z_memory_gap_advantage,
            f"{mode}_rewrite_persistence_mean": float(np.mean(cycle_persistences)),
            f"{mode}_rewrite_persistence_min": float(np.min(cycle_persistences)),
            f"{mode}_falsification_margin_mean": float(np.mean(cycle_falsification_margins)),
            f"{mode}_falsification_margin_min": float(np.min(cycle_falsification_margins)),
            f"{mode}_cycle_supported_count": float(cycle_supported),
            f"{mode}_cycle_supported_fraction": float(cycle_supported / lesion_cycles),
        }

        for cycle_index, persistence in enumerate(cycle_persistences, start=1):
            mode_metrics[f"{mode}_cycle_{cycle_index}_persistence"] = persistence
            mode_metrics[f"{mode}_cycle_{cycle_index}_falsification_margin"] = cycle_falsification_margins[cycle_index - 1]

        return mode_metrics

    def run(self, cfg):
        control = self._run_condition(cfg, mode=None, preserve_hidden=False)
        per_mode_results = {}
        mode_metrics: dict[str, float] = {}

        for mode in self.REWRITE_MODES:
            rewritten = self._run_condition(cfg, mode=mode, preserve_hidden=True)
            falsified = self._run_condition(cfg, mode=mode, preserve_hidden=False)
            per_mode_results[mode] = {
                "rewritten": rewritten,
                "falsified": falsified,
            }
            mode_metrics.update(self._summarize_mode(mode, control, rewritten, falsified))

        supported_modes = [
            mode
            for mode in self.REWRITE_MODES
            if mode_metrics[f"{mode}_cycle_supported_count"] >= 2.0
            and mode_metrics[f"{mode}_falsification_margin_mean"] > 0.04
        ]
        strong_cryptic_modes = [
            mode
            for mode in supported_modes
            if mode_metrics[f"{mode}_pre_visible_context_divergence"] <= 0.25
            and mode_metrics[f"{mode}_hidden_z_memory_gap_advantage"] >= 0.10
        ]
        candidate_modes = strong_cryptic_modes or supported_modes or list(self.REWRITE_MODES)
        best_mode = max(
            candidate_modes,
            key=lambda mode: (
                mode_metrics[f"{mode}_rewrite_persistence_mean"]
                + mode_metrics[f"{mode}_falsification_margin_mean"]
                + mode_metrics[f"{mode}_hidden_z_memory_gap_advantage"]
                - mode_metrics[f"{mode}_pre_visible_context_divergence"]
            ),
        )

        best_rewritten = per_mode_results[best_mode]["rewritten"]
        final_metrics = best_rewritten["rollout"]["final_metrics"].copy()
        final_metrics.update(mode_metrics)

        # Preserve the existing top-level metrics, but now tie them to the strongest supported mode.
        final_metrics["best_rewrite_mode"] = best_mode
        final_metrics["rewrite_mode_supported_count"] = float(len(supported_modes))
        final_metrics["rewrite_mode_supported_fraction"] = float(len(supported_modes) / len(self.REWRITE_MODES))
        final_metrics["strong_cryptic_mode_count"] = float(len(strong_cryptic_modes))
        final_metrics["strong_cryptic_mode_fraction"] = float(len(strong_cryptic_modes) / len(self.REWRITE_MODES))
        final_metrics["rewrite_persistence"] = mode_metrics[f"{best_mode}_rewrite_persistence_mean"]
        final_metrics["cryptic_shift"] = (
            mode_metrics[f"{best_mode}_rewrite_persistence_mean"]
            - mode_metrics[f"{best_mode}_pre_hidden_diff"]
        )
        final_metrics["pre_lesion_visible_context_divergence"] = mode_metrics[
            f"{best_mode}_pre_visible_context_divergence"
        ]
        final_metrics["hidden_z_memory_gap_advantage"] = mode_metrics[
            f"{best_mode}_hidden_z_memory_gap_advantage"
        ]
        final_metrics["baseline_pre_lesion_region_hidden"] = control["markers"]["pre_lesion_region_hidden"]
        final_metrics["rewritten_pre_lesion_region_hidden"] = best_rewritten["markers"]["pre_lesion_region_hidden"]
        final_metrics["baseline_final_region_hidden"] = control["markers"]["final_region_hidden"]
        final_metrics["rewritten_final_region_hidden"] = best_rewritten["markers"]["final_region_hidden"]
        final_metrics["baseline_pre_lesion_visible_z_context_gap"] = control["markers"]["pre_lesion_visible_z_context_gap"]
        final_metrics["rewritten_pre_lesion_visible_z_context_gap"] = best_rewritten["markers"]["pre_lesion_visible_z_context_gap"]
        final_metrics["baseline_pre_lesion_hidden_z_memory_gap"] = control["markers"]["pre_lesion_hidden_z_memory_gap"]
        final_metrics["rewritten_pre_lesion_hidden_z_memory_gap"] = best_rewritten["markers"]["pre_lesion_hidden_z_memory_gap"]
        final_metrics["baseline_region_z"] = control["markers"]["final_region_z"]
        final_metrics["rewritten_region_z"] = best_rewritten["markers"]["final_region_z"]

        notes = (
            "Setpoint rewrite now tests three rewrite modes with matched falsification runs and repeated re-lesion cycles. "
            f"Best mode={best_mode}; supported_modes={len(supported_modes)}/{len(self.REWRITE_MODES)}; "
            f"rewrite_persistence={final_metrics['rewrite_persistence']:.4f}; "
            f"pre_visible_context_divergence={final_metrics['pre_lesion_visible_context_divergence']:.4f}; "
            f"hidden_z_memory_gap_advantage={final_metrics['hidden_z_memory_gap_advantage']:.4f}."
        )
        return AssayResult(
            history=best_rewritten["rollout"]["history"],
            final_metrics=final_metrics,
            notes=notes,
        )
