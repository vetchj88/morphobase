import numpy as np

from morphobase.assays.common import (
    AssayResult,
    AssayRunner,
    apply_retraining_correction,
    bias_z_field,
    build_synthetic_body,
    corrupt_parameters,
    corrupt_field_alignment,
    disrupt_global_port_state,
    disrupt_port_region,
    lesion_segment,
    rollout_body,
    sever_conductance,
    targeted_tissue_ablation,
)


class LesionBatteryAssay(AssayRunner):
    RECOVERY_SUSTAIN_STEPS = 2
    RECOVERY_THRESHOLD_FRACTION = 0.85
    NADIR_WINDOW = 24

    def _region_score(self, carrier, start: int, stop: int, target_value: float) -> float:
        if stop <= start:
            return 0.0
        state = carrier.state if hasattr(carrier, "state") else carrier
        local_hidden = state.hidden[start:stop].mean(axis=1)
        local_z = 0.5 * (state.z_alignment[start:stop] + 1.0)
        local_field = state.field_alignment[start:stop]
        local_role = state.role_logits[start:stop]
        hidden_score = np.clip(1.0 - np.mean(np.abs(local_hidden - target_value)), 0.0, 1.0)
        z_score = float(np.clip(local_z.mean(), 0.0, 1.0))
        field_score = float(np.clip(local_field.mean(), 0.0, 1.0))
        role_score = float(np.clip(1.0 - 0.45 * np.mean(np.std(local_role, axis=1)), 0.0, 1.0))
        return float((hidden_score + z_score + field_score + role_score) / 4.0)

    @staticmethod
    def _first_sustained_onset(trace: list[float], threshold: float, sustain: int, fallback: float) -> float:
        if len(trace) < sustain:
            return fallback
        for idx in range(len(trace) - sustain + 1):
            if all(value >= threshold for value in trace[idx : idx + sustain]):
                return float(idx + 1)
        return fallback

    @staticmethod
    def _preferred_side(left_onset: float, right_onset: float, left_final: float, right_final: float) -> float:
        if right_onset < left_onset:
            return 1.0
        if left_onset < right_onset:
            return -1.0
        if right_final > left_final:
            return 1.0
        if left_final > right_final:
            return -1.0
        return 0.0

    @staticmethod
    def _control_deficit_trace(case_trace: list[float], control_trace: list[float]) -> list[float]:
        return [float(max(control_value - case_value, 0.0)) for case_value, control_value in zip(case_trace, control_trace)]

    def _control_relative_recovery(self, case_trace: list[float], control_trace: list[float]) -> tuple[float, float]:
        deficit_trace = self._control_deficit_trace(case_trace, control_trace)
        if not deficit_trace:
            return 1.0, 1.0

        injury_deficit = float(max(deficit_trace))
        if injury_deficit <= 1e-8:
            return 1.0, 1.0

        peak_idx = int(np.argmax(deficit_trace))
        target_deficit = injury_deficit * (1.0 - self.RECOVERY_THRESHOLD_FRACTION)
        recovery_steps = float(len(deficit_trace))
        for idx in range(peak_idx, len(deficit_trace) - self.RECOVERY_SUSTAIN_STEPS + 1):
            if all(value <= target_deficit for value in deficit_trace[idx : idx + self.RECOVERY_SUSTAIN_STEPS]):
                recovery_steps = float(idx + 1)
                break

        final_deficit = deficit_trace[-1]
        recovery_fraction = float(np.clip(1.0 - final_deficit / max(injury_deficit, 1e-8), 0.0, 1.0))
        return recovery_fraction, recovery_steps

    def _paired_region_disturbance(self, case_state, control_state, start: int, stop: int) -> float:
        if stop <= start:
            return 0.0
        hidden_gap = float(np.mean(np.abs(case_state.hidden[start:stop] - control_state.hidden[start:stop])))
        membrane_gap = float(np.mean(np.abs(case_state.membrane[start:stop] - control_state.membrane[start:stop])))
        energy_gap = float(np.mean(np.abs(case_state.energy[start:stop] - control_state.energy[start:stop])))
        field_gap = float(np.mean(np.abs(case_state.field_alignment[start:stop] - control_state.field_alignment[start:stop])))
        z_gap = float(np.mean(np.abs(case_state.z_alignment[start:stop] - control_state.z_alignment[start:stop])))
        return float((hidden_gap + membrane_gap + energy_gap + field_gap + z_gap) / 5.0)

    @staticmethod
    def _region_windows(cfg, lesion_start: int, lesion_stop: int) -> dict[str, tuple[int, int]]:
        lesion_width = max(2, lesion_stop - lesion_start)
        flank_width = max(3, lesion_width // 2)
        lesion_mid = lesion_start + lesion_width // 2
        return {
            "left_support": (max(0, lesion_start - flank_width), lesion_start),
            "lesion_left": (lesion_start, lesion_mid),
            "lesion_right": (lesion_mid, lesion_stop),
            "right_support": (lesion_stop, min(cfg.body.num_cells, lesion_stop + flank_width)),
        }

    @staticmethod
    def _locality_windows(cfg, lesion_start: int, lesion_stop: int) -> dict[str, tuple[int, int]]:
        lesion_width = max(2, lesion_stop - lesion_start)
        distal_width = max(lesion_width, cfg.body.num_cells // 6)
        distal_start = max(0, (cfg.body.num_cells // 2) - distal_width // 2)
        distal_stop = min(cfg.body.num_cells, distal_start + distal_width)
        return {
            "boundary_port": (lesion_start, lesion_stop),
            "distal_core": (distal_start, distal_stop),
            "whole_body": (0, cfg.body.num_cells),
        }

    def _snapshot_regions(self, carrier, windows: dict[str, tuple[int, int]], target_value: float) -> dict[str, float]:
        return {
            region_name: self._region_score(carrier, start, stop, target_value)
            for region_name, (start, stop) in windows.items()
        }

    @staticmethod
    def _lesion_bounds(cfg, lesion_name: str) -> tuple[int, int]:
        base_width = max(2, cfg.body.num_cells // 5)
        if lesion_name == "targeted_tissue_ablation":
            start = max(2, cfg.body.num_cells // 4)
            stop = min(cfg.body.num_cells - 2, start + max(base_width + 4, cfg.body.num_cells // 3))
        elif lesion_name == "parameter_corruption":
            width = max(3, base_width // 2)
            start = max(2, cfg.body.num_cells // 3)
            stop = min(cfg.body.num_cells - 2, start + width)
        elif "port_disruption" in lesion_name:
            start = 2
            stop = min(cfg.body.num_cells - 2, start + max(4, cfg.body.num_cells // 8))
        else:
            start = max(1, cfg.body.num_cells // 3)
            stop = min(cfg.body.num_cells, start + base_width)
        return start, stop

    @staticmethod
    def _lesion_schedule(cfg, lesion_name: str, repeat_count: int) -> list[int]:
        if repeat_count <= 1:
            return [cfg.runtime.total_steps // 2]
        start = cfg.runtime.total_steps // 3
        end = (2 * cfg.runtime.total_steps) // 3
        return [int(step) for step in np.linspace(start, end, repeat_count)]

    def _run_case(
        self,
        cfg,
        lesion_name: str,
        lesion_fn,
        *,
        repeat_count: int = 1,
        no_gradient: bool = False,
        retraining: bool = False,
    ) -> dict:
        seed = cfg.run.seed + sum(ord(ch) for ch in lesion_name)
        np.random.seed(seed)
        control_rollout = rollout_body(cfg, body=build_synthetic_body(cfg), no_gradient=no_gradient)
        body = build_synthetic_body(cfg)
        lesion_start, lesion_stop = self._lesion_bounds(cfg, lesion_name)
        lesion_steps = self._lesion_schedule(cfg, lesion_name, repeat_count)
        lesion_step = lesion_steps[-1]
        region_windows = self._region_windows(cfg, lesion_start, lesion_stop)
        markers: dict[str, float] = {}
        energy_trace: list[float] = []

        def step_hook(step: int, hooked_body) -> None:
            if step == lesion_step - 1:
                markers["pre_score"] = self._region_score(
                    hooked_body, lesion_start, lesion_stop, cfg.assay.target_value
                )
                markers["pre_energy"] = float(np.mean(hooked_body.state.energy))
                for region_name, value in self._snapshot_regions(
                    hooked_body,
                    region_windows,
                    cfg.assay.target_value,
                ).items():
                    markers[f"pre_{region_name}_score"] = value
            if step in lesion_steps:
                lesion_fn(hooked_body, lesion_start, lesion_stop)
                if step == lesion_step:
                    markers["lesion_score"] = self._region_score(
                        hooked_body, lesion_start, lesion_stop, cfg.assay.target_value
                    )
                    for region_name, value in self._snapshot_regions(
                        hooked_body,
                        region_windows,
                        cfg.assay.target_value,
                    ).items():
                        markers[f"lesion_{region_name}_score"] = value
            if step >= lesion_step:
                energy_trace.append(float(np.mean(hooked_body.state.energy)))

        def after_step_hook(step: int, hooked_body) -> None:
            if not retraining or step < lesion_steps[0]:
                return
            reference_index = min(step + 1, len(control_rollout["state_history"]) - 1)
            apply_retraining_correction(
                hooked_body,
                control_rollout["state_history"][reference_index],
                lesion_start,
                lesion_stop,
            )

        np.random.seed(seed)
        rollout = rollout_body(
            cfg,
            body=body,
            step_hook=step_hook,
            after_step_hook=after_step_hook if retraining else None,
            no_gradient=no_gradient,
        )
        markers["final_score"] = self._region_score(
            rollout["body"], lesion_start, lesion_stop, cfg.assay.target_value
        )
        post_states = rollout["state_history"][lesion_step + 1 :]
        control_post_states = control_rollout["state_history"][lesion_step + 1 :]
        lesion_trace = [
            self._region_score(state, lesion_start, lesion_stop, cfg.assay.target_value)
            for state in post_states
        ]
        control_trace = [
            self._region_score(state, lesion_start, lesion_stop, cfg.assay.target_value)
            for state in control_post_states
        ]
        region_traces = {
            region_name: [
                self._region_score(state, start, stop, cfg.assay.target_value)
                for state in post_states
            ]
            for region_name, (start, stop) in region_windows.items()
        }
        control_region_traces = {
            region_name: [
                self._region_score(state, start, stop, cfg.assay.target_value)
                for state in control_post_states
            ]
            for region_name, (start, stop) in region_windows.items()
        }
        if "port_disruption" in lesion_name:
            locality_windows = self._locality_windows(cfg, lesion_start, lesion_stop)
            for region_name, (start, stop) in locality_windows.items():
                disturbance_trace = [
                    self._paired_region_disturbance(case_state, control_state, start, stop)
                    for case_state, control_state in zip(post_states, control_post_states)
                ]
                markers[f"{region_name}_peak_disturbance"] = float(max(disturbance_trace) if disturbance_trace else 0.0)
                markers[f"{region_name}_final_disturbance"] = float(disturbance_trace[-1] if disturbance_trace else 0.0)
            markers["distal_sparing_fraction"] = float(np.clip(
                1.0 - markers["distal_core_peak_disturbance"] / max(markers["boundary_port_peak_disturbance"], 1e-8),
                0.0,
                1.0,
            ))
            markers["boundary_locality_ratio"] = float(
                markers["boundary_port_peak_disturbance"] / max(markers["distal_core_peak_disturbance"], 1e-8)
            )
        deficit_trace = self._control_deficit_trace(lesion_trace, control_trace)
        peak_idx = int(np.argmax(deficit_trace)) if deficit_trace else 0
        markers["immediate_lesion_score"] = markers["lesion_score"]
        markers["lesion_score"] = float(min(markers["lesion_score"], min(lesion_trace) if lesion_trace else markers["lesion_score"]))
        markers["lesion_nadir_step"] = float(lesion_step + 1 + peak_idx)
        for region_name, trace in region_traces.items():
            markers[f"final_{region_name}_score"] = trace[-1] if trace else markers.get(
                f"lesion_{region_name}_score",
                markers.get(f"pre_{region_name}_score", 0.0),
            )
            markers[f"immediate_{region_name}_score"] = markers[f"lesion_{region_name}_score"]
            markers[f"lesion_{region_name}_score"] = float(
                min(markers[f"lesion_{region_name}_score"], min(trace) if trace else markers[f"lesion_{region_name}_score"])
            )

        markers["recovery_fraction"], markers["recovery_steps"] = self._control_relative_recovery(
            lesion_trace,
            control_trace,
        )

        markers["recovery_probability"] = (
            1.0
            if (
                markers["recovery_steps"] < float(len(lesion_trace))
                and markers["recovery_fraction"] > 0.25
            )
            else 0.0
        )
        if markers["recovery_probability"] == 0.0:
            markers["recovery_steps"] = float(cfg.runtime.total_steps)
        pre_energy = markers.get("pre_energy", float(np.mean(rollout["body"].state.energy)))
        markers["energy_cost"] = float(
            sum(max(pre_energy - energy_value, 0.0) for energy_value in energy_trace[: int(markers["recovery_steps"])]
                if np.isfinite(markers["recovery_steps"]))
            if markers["recovery_steps"] < float(cfg.runtime.total_steps)
            else sum(max(pre_energy - energy_value, 0.0) for energy_value in energy_trace)
        )

        for region_name in ("lesion_left", "lesion_right"):
            recovery, onset = self._control_relative_recovery(
                region_traces[region_name],
                control_region_traces[region_name],
            )
            markers[f"{region_name}_recovery"] = recovery
            markers[f"{region_name}_recovery_onset"] = onset

        left_support_trace = region_traces["left_support"]
        right_support_trace = region_traces["right_support"]
        left_support_total = float(sum(left_support_trace))
        right_support_total = float(sum(right_support_trace))
        lesion_left_total = float(sum(region_traces["lesion_left"]))
        lesion_right_total = float(sum(region_traces["lesion_right"]))
        markers["support_balance"] = float(
            (right_support_total - left_support_total)
            / max(right_support_total + left_support_total, 1e-8)
        )
        markers["lesion_half_balance"] = float(
            (lesion_right_total - lesion_left_total)
            / max(lesion_right_total + lesion_left_total, 1e-8)
        )
        markers["recovery_onset_gap"] = float(
            abs(markers["lesion_right_recovery_onset"] - markers["lesion_left_recovery_onset"])
        )
        markers["recovery_onset_skew"] = float(
            markers["lesion_right_recovery_onset"] - markers["lesion_left_recovery_onset"]
        )
        markers["preferred_recovery_side"] = self._preferred_side(
            markers["lesion_left_recovery_onset"],
            markers["lesion_right_recovery_onset"],
            markers["final_lesion_left_score"],
            markers["final_lesion_right_score"],
        )
        markers["injury_count"] = float(repeat_count)
        markers["no_gradient_mode"] = 1.0 if no_gradient else 0.0
        markers["retraining_mode"] = 1.0 if retraining else 0.0
        return {"rollout": rollout, "markers": markers}

    def run(self, cfg):
        lesions = {
            "cell_ablation": {"fn": lambda body, start, stop: lesion_segment(body, start, stop), "repeat_count": 1},
            "repeated_cell_ablation": {
                "fn": lambda body, start, stop: lesion_segment(body, start, stop, energy_scale=0.32, stress_boost=0.70),
                "repeat_count": 3,
            },
            "targeted_tissue_ablation": {
                "fn": lambda body, start, stop: targeted_tissue_ablation(body, start, stop),
                "repeat_count": 1,
            },
            "conductance_severance": {
                "fn": lambda body, start, stop: sever_conductance(body, start, stop),
                "repeat_count": 1,
            },
            "parameter_corruption": {
                "fn": lambda body, start, stop: corrupt_parameters(body, start, stop),
                "repeat_count": 1,
            },
            "field_corruption": {
                "fn": lambda body, start, stop: corrupt_field_alignment(body, start, stop),
                "repeat_count": 1,
            },
            "z_field_corruption": {
                "fn": lambda body, start, stop: bias_z_field(body, start, stop, -0.8),
                "repeat_count": 1,
            },
            "port_disruption": {
                "fn": lambda body, start, stop: disrupt_port_region(body, start, stop, attenuation=0.30),
                "repeat_count": 1,
            },
            "repeated_port_disruption": {
                "fn": lambda body, start, stop: disrupt_port_region(body, start, stop, attenuation=0.30),
                "repeat_count": 2,
            },
            "whole_body_port_disruption": {
                "fn": lambda body, start, stop: disrupt_global_port_state(body),
                "repeat_count": 1,
            },
        }

        results = {
            name: self._run_case(cfg, name, spec["fn"], repeat_count=spec["repeat_count"])
            for name, spec in lesions.items()
        }
        no_gradient_targets = {
            "cell_ablation",
            "parameter_corruption",
            "port_disruption",
            "repeated_cell_ablation",
            "repeated_port_disruption",
        }
        no_gradient_results = {
            name: self._run_case(
                cfg,
                name,
                lesions[name]["fn"],
                repeat_count=lesions[name]["repeat_count"],
                no_gradient=True,
            )
            for name in no_gradient_targets
        }
        retraining_targets = {
            "cell_ablation",
            "parameter_corruption",
            "port_disruption",
            "repeated_cell_ablation",
            "repeated_port_disruption",
        }
        retraining_results = {
            name: self._run_case(
                cfg,
                name,
                lesions[name]["fn"],
                repeat_count=lesions[name]["repeat_count"],
                retraining=True,
            )
            for name in retraining_targets
        }
        primary_names = [name for name in results if name != "whole_body_port_disruption"]
        recovery_values = [results[name]["markers"]["recovery_fraction"] for name in primary_names]
        recovery_probabilities = [results[name]["markers"]["recovery_probability"] for name in primary_names]
        recovery_steps = [results[name]["markers"]["recovery_steps"] for name in primary_names]
        energy_costs = [results[name]["markers"]["energy_cost"] for name in primary_names]
        no_gradient_recovery_values = [case["markers"]["recovery_fraction"] for case in no_gradient_results.values()]
        no_gradient_recovery_probabilities = [case["markers"]["recovery_probability"] for case in no_gradient_results.values()]
        no_gradient_recovery_steps = [case["markers"]["recovery_steps"] for case in no_gradient_results.values()]
        matched_full_recovery_values = [results[name]["markers"]["recovery_fraction"] for name in no_gradient_targets]
        retraining_recovery_values = [case["markers"]["recovery_fraction"] for case in retraining_results.values()]
        retraining_recovery_probabilities = [case["markers"]["recovery_probability"] for case in retraining_results.values()]
        matched_organismal_retraining_values = [results[name]["markers"]["recovery_fraction"] for name in retraining_targets]
        repeated_retraining_targets = {"repeated_cell_ablation", "repeated_port_disruption"}
        repeated_no_gradient_values = [no_gradient_results[name]["markers"]["recovery_fraction"] for name in repeated_retraining_targets]
        repeated_retraining_values = [retraining_results[name]["markers"]["recovery_fraction"] for name in repeated_retraining_targets]
        repeated_organismal_values = [results[name]["markers"]["recovery_fraction"] for name in repeated_retraining_targets]
        exemplar = results["cell_ablation"]["rollout"]
        final_metrics = exemplar["final_metrics"].copy()
        final_metrics["lesion_recovery_mean"] = float(np.mean(recovery_values))
        final_metrics["lesion_success_fraction"] = float(
            np.mean(np.asarray(recovery_values) > 0.25)
        )
        final_metrics["lesion_recovery_probability_mean"] = float(np.mean(recovery_probabilities))
        final_metrics["lesion_recovery_steps_mean"] = float(np.mean(recovery_steps))
        final_metrics["lesion_energy_cost_mean"] = float(np.mean(energy_costs))
        final_metrics["matched_gradient_recovery_mean"] = float(np.mean(matched_full_recovery_values))
        final_metrics["no_gradient_recovery_mean"] = float(np.mean(no_gradient_recovery_values))
        final_metrics["no_gradient_recovery_probability_mean"] = float(np.mean(no_gradient_recovery_probabilities))
        final_metrics["no_gradient_recovery_steps_mean"] = float(np.mean(no_gradient_recovery_steps))
        final_metrics["recovery_retention_without_gradients"] = float(
            final_metrics["no_gradient_recovery_mean"] / max(final_metrics["matched_gradient_recovery_mean"], 1e-8)
        )
        final_metrics["matched_organismal_retraining_probe_mean"] = float(np.mean(matched_organismal_retraining_values))
        final_metrics["retraining_recovery_mean"] = float(np.mean(retraining_recovery_values))
        final_metrics["retraining_recovery_probability_mean"] = float(np.mean(retraining_recovery_probabilities))
        final_metrics["organismal_recovery_vs_retraining_ratio"] = float(
            final_metrics["matched_organismal_retraining_probe_mean"] / max(final_metrics["retraining_recovery_mean"], 1e-8)
        )
        final_metrics["repeated_injury_organismal_recovery_mean"] = float(np.mean(repeated_organismal_values))
        final_metrics["repeated_injury_no_gradient_recovery_mean"] = float(np.mean(repeated_no_gradient_values))
        final_metrics["repeated_injury_retraining_recovery_mean"] = float(np.mean(repeated_retraining_values))
        final_metrics["repeated_injury_vs_retraining_ratio"] = float(
            final_metrics["repeated_injury_organismal_recovery_mean"]
            / max(final_metrics["repeated_injury_retraining_recovery_mean"], 1e-8)
        )
        final_metrics["repeated_injury_retention_without_gradients"] = float(
            final_metrics["repeated_injury_no_gradient_recovery_mean"]
            / max(final_metrics["repeated_injury_organismal_recovery_mean"], 1e-8)
        )
        for lesion_name, case in results.items():
            final_metrics[f"{lesion_name}_injury_count"] = case["markers"]["injury_count"]
            final_metrics[f"{lesion_name}_recovery"] = case["markers"]["recovery_fraction"]
            final_metrics[f"{lesion_name}_recovery_probability"] = case["markers"]["recovery_probability"]
            final_metrics[f"{lesion_name}_recovery_steps"] = case["markers"]["recovery_steps"]
            final_metrics[f"{lesion_name}_energy_cost"] = case["markers"]["energy_cost"]
            final_metrics[f"{lesion_name}_left_half_recovery"] = case["markers"]["lesion_left_recovery"]
            final_metrics[f"{lesion_name}_right_half_recovery"] = case["markers"]["lesion_right_recovery"]
            final_metrics[f"{lesion_name}_left_half_recovery_onset"] = case["markers"][
                "lesion_left_recovery_onset"
            ]
            final_metrics[f"{lesion_name}_right_half_recovery_onset"] = case["markers"][
                "lesion_right_recovery_onset"
            ]
            final_metrics[f"{lesion_name}_recovery_onset_gap"] = case["markers"]["recovery_onset_gap"]
            final_metrics[f"{lesion_name}_recovery_onset_skew"] = case["markers"]["recovery_onset_skew"]
            final_metrics[f"{lesion_name}_support_balance"] = case["markers"]["support_balance"]
            final_metrics[f"{lesion_name}_lesion_half_balance"] = case["markers"]["lesion_half_balance"]
            final_metrics[f"{lesion_name}_preferred_recovery_side"] = case["markers"][
                "preferred_recovery_side"
            ]
            if "boundary_port_peak_disturbance" in case["markers"]:
                final_metrics[f"{lesion_name}_boundary_port_peak_disturbance"] = case["markers"][
                    "boundary_port_peak_disturbance"
                ]
                final_metrics[f"{lesion_name}_distal_core_peak_disturbance"] = case["markers"][
                    "distal_core_peak_disturbance"
                ]
                final_metrics[f"{lesion_name}_whole_body_peak_disturbance"] = case["markers"][
                    "whole_body_peak_disturbance"
                ]
                final_metrics[f"{lesion_name}_distal_sparing_fraction"] = case["markers"][
                    "distal_sparing_fraction"
                ]
                final_metrics[f"{lesion_name}_boundary_locality_ratio"] = case["markers"][
                    "boundary_locality_ratio"
                ]
        for lesion_name, case in no_gradient_results.items():
            final_metrics[f"{lesion_name}_no_gradient_recovery"] = case["markers"]["recovery_fraction"]
            final_metrics[f"{lesion_name}_no_gradient_recovery_probability"] = case["markers"]["recovery_probability"]
            final_metrics[f"{lesion_name}_no_gradient_recovery_steps"] = case["markers"]["recovery_steps"]
            final_metrics[f"{lesion_name}_recovery_advantage_vs_no_gradient"] = float(
                results[lesion_name]["markers"]["recovery_fraction"] - case["markers"]["recovery_fraction"]
            )
        for lesion_name, case in retraining_results.items():
            final_metrics[f"{lesion_name}_retraining_recovery"] = case["markers"]["recovery_fraction"]
            final_metrics[f"{lesion_name}_retraining_recovery_probability"] = case["markers"]["recovery_probability"]
            final_metrics[f"{lesion_name}_retraining_recovery_steps"] = case["markers"]["recovery_steps"]
            final_metrics[f"{lesion_name}_recovery_gap_vs_retraining"] = float(
                results[lesion_name]["markers"]["recovery_fraction"] - case["markers"]["recovery_fraction"]
            )

        port_local = results["port_disruption"]["markers"]
        port_global = results["whole_body_port_disruption"]["markers"]
        final_metrics["port_localization_advantage"] = float(
            port_global["distal_core_peak_disturbance"] - port_local["distal_core_peak_disturbance"]
        )
        final_metrics["port_recovery_advantage_vs_global"] = float(
            port_local["recovery_fraction"] - port_global["recovery_fraction"]
        )
        final_metrics["port_boundary_locality_advantage"] = float(
            port_local["boundary_locality_ratio"] - port_global["boundary_locality_ratio"]
        )

        notes = (
            "Lesion battery executed across cell, tissue, conductance, parameter, field, Z-field, and port perturbations. "
            f"Mean recovery={final_metrics['lesion_recovery_mean']:.4f}; "
            f"no-gradient mean={final_metrics['no_gradient_recovery_mean']:.4f}; "
            f"retraining mean={final_metrics['retraining_recovery_mean']:.4f}; "
            f"probability={final_metrics['lesion_recovery_probability_mean']:.4f}; "
            f"steps_mean={final_metrics['lesion_recovery_steps_mean']:.1f}; "
            f"energy_cost_mean={final_metrics['lesion_energy_cost_mean']:.3f}."
        )
        return AssayResult(
            history=exemplar["history"],
            final_metrics=final_metrics,
            notes=notes,
        )
