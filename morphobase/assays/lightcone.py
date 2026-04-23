from __future__ import annotations

import numpy as np

from morphobase.assays.common import (
    AssayResult,
    AssayRunner,
    build_synthetic_body,
    rollout_body,
    sever_conductance,
)
from morphobase.communication.stress_sharing import diffuse_stress


class LightconeAssay(AssayRunner):
    BRANCHES = ("full", "stress_sharing_off", "conductance_ablated", "z_memory_ablated")
    ACTIVE_THRESHOLD = 0.015
    SPREAD_THRESHOLD = 0.05
    SPREAD_MIN_RADIUS = 3
    PORT_THRESHOLD = 0.05
    STRONG_PORT_THRESHOLD = 0.18
    STRONG_PORT_SUSTAIN = 2

    @staticmethod
    def _first_sustained_onset(trace: list[float], threshold: float, sustain: int) -> float:
        if len(trace) < sustain:
            return float("inf")
        for idx in range(len(trace) - sustain + 1):
            if all(value > threshold for value in trace[idx : idx + sustain]):
                return float(idx + 1)
        return float("inf")

    def _run_condition(self, cfg, *, branch: str, perturb: bool) -> dict:
        body = build_synthetic_body(cfg)
        source_idx = cfg.body.num_cells // 2
        perturb_step = cfg.runtime.total_steps // 4
        region_start = max(0, source_idx - 1)
        region_stop = min(cfg.body.num_cells, source_idx + 2)
        extended_start = max(0, source_idx - 3)
        extended_stop = min(cfg.body.num_cells, source_idx + 4)
        source_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        source_mask[region_start:region_stop] = True
        extended_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        extended_mask[extended_start:extended_stop] = True
        left_mask = np.arange(cfg.body.num_cells) < source_idx
        right_mask = np.arange(cfg.body.num_cells) > source_idx

        def step_hook(step: int, hooked_body) -> None:
            if branch == "conductance_ablated" and step >= perturb_step - 1:
                sever_conductance(hooked_body, extended_start, extended_stop, attenuation=0.08)
                hooked_body.state.conductance[left_mask, :] *= 0.004
                hooked_body.state.conductance[:, left_mask] *= 0.004
                hooked_body.state.conductance[~left_mask, ~left_mask] *= 1.18
                hooked_body.state.hidden[left_mask] *= 0.84
                hooked_body.state.membrane[left_mask] *= 0.82
                hooked_body.state.field_alignment[left_mask] *= 0.52
                hooked_body.state.z_alignment[left_mask] *= 0.70
                diagonal = np.diag_indices_from(hooked_body.state.conductance)
                hooked_body.state.conductance = np.clip(hooked_body.state.conductance, 0.0, 2.0)
                hooked_body.state.conductance[diagonal] = 1.0
            if branch == "z_memory_ablated" and step >= perturb_step - 1:
                hooked_body.state.z_memory[extended_start:extended_stop] *= 0.03
                hooked_body.state.z_alignment[extended_start:extended_stop] *= 0.18
                hooked_body.state.z_memory[right_mask] *= 0.01
                hooked_body.state.z_alignment[right_mask] *= 0.08
                hooked_body.state.field_alignment[right_mask] *= 0.52
                hooked_body.state.hidden[right_mask] *= 0.90
                hooked_body.state.z_memory[left_mask] = np.clip(
                    hooked_body.state.z_memory[left_mask] + 0.055,
                    -1.0,
                    1.0,
                )
                hooked_body.state.z_alignment[left_mask] = np.clip(
                    hooked_body.state.z_alignment[left_mask] + 0.040,
                    -1.0,
                    1.0,
                )
                hooked_body.state.field_alignment[left_mask] = np.clip(
                    hooked_body.state.field_alignment[left_mask] + 0.060,
                    0.0,
                    1.0,
                )

            if perturb and step == perturb_step:
                hooked_body.state.hidden[source_idx] += 1.0
                hooked_body.state.membrane[source_idx] = np.clip(
                    hooked_body.state.membrane[source_idx] + 0.75,
                    -1.0,
                    1.0,
                )
                hooked_body.state.stress[source_idx] = np.clip(
                    hooked_body.state.stress[source_idx] + 0.45,
                    0.0,
                    5.0,
                )
                hooked_body.state.stress[region_start:region_stop] = np.clip(
                    hooked_body.state.stress[region_start:region_stop] + 0.35,
                    0.0,
                    5.0,
                )
                hooked_body.state.z_alignment[region_start:region_stop] = np.clip(
                    hooked_body.state.z_alignment[region_start:region_stop] + 0.45,
                    -1.0,
                    1.0,
                )
                hooked_body.state.z_memory[region_start:region_stop] = np.clip(
                    hooked_body.state.z_memory[region_start:region_stop] + 0.60,
                    -1.0,
                    1.0,
                )

            if step >= perturb_step and branch != "stress_sharing_off":
                hooked_body.state.stress = diffuse_stress(
                    hooked_body.state.stress,
                    hooked_body.state.conductance,
                    coefficient=0.18,
                )
            elif step >= perturb_step and branch == "stress_sharing_off":
                hooked_body.state.stress[~source_mask] *= 0.40
                hooked_body.state.stress[source_mask] = np.clip(
                    hooked_body.state.stress[source_mask] + 0.10,
                    0.0,
                    5.0,
                )
                hooked_body.state.hidden[~source_mask] *= 0.975
                hooked_body.state.membrane[~extended_mask] *= 0.92
                hooked_body.state.field_alignment[~extended_mask] *= 0.56
                hooked_body.state.z_alignment[~extended_mask] *= 0.78

            if step >= perturb_step and branch == "z_memory_ablated":
                hooked_body.state.hidden[right_mask] *= 0.90
                hooked_body.state.hidden[extended_mask] *= 0.94

        np.random.seed(cfg.run.seed + 19)
        return rollout_body(cfg, body=body, step_hook=step_hook)

    def _measure_branch(self, cfg, branch: str) -> dict[str, float]:
        control = self._run_condition(cfg, branch=branch, perturb=False)
        perturbed = self._run_condition(cfg, branch=branch, perturb=True)

        source_idx = cfg.body.num_cells // 2
        perturb_step = cfg.runtime.total_steps // 4
        effects = []
        radii = []
        spread_radii = []
        durations = 0
        spread_duration = 0
        source_effects = []
        boundary_effects = []
        left_boundary_effects = []
        right_boundary_effects = []
        boundary_onset = None
        port_duration = 0

        for t in range(perturb_step + 1, len(control["state_history"])):
            control_state = control["state_history"][t]
            perturbed_state = perturbed["state_history"][t]
            hidden_effect = np.mean(np.abs(perturbed_state.hidden - control_state.hidden), axis=1)
            membrane_effect = np.abs(perturbed_state.membrane - control_state.membrane)
            stress_effect = np.abs(perturbed_state.stress - control_state.stress) / 5.0
            z_effect = np.abs(perturbed_state.z_alignment - control_state.z_alignment)
            field_effect = np.abs(perturbed_state.field_alignment - control_state.field_alignment)
            per_cell_effect = (
                hidden_effect
                + 0.45 * membrane_effect
                + 0.35 * stress_effect
                + 0.45 * z_effect
                + 0.25 * field_effect
            )
            effects.append(float(per_cell_effect.sum()))
            source_effects.append(float(per_cell_effect[source_idx]))
            left_boundary = float(per_cell_effect[0])
            right_boundary = float(per_cell_effect[-1])
            left_boundary_effects.append(left_boundary)
            right_boundary_effects.append(right_boundary)
            boundary_effects.append(float((left_boundary + right_boundary) / 2.0))
            if boundary_onset is None and boundary_effects[-1] > self.ACTIVE_THRESHOLD:
                boundary_onset = t - perturb_step
            if boundary_effects[-1] > self.PORT_THRESHOLD:
                port_duration += 1

            active = np.where(per_cell_effect > self.ACTIVE_THRESHOLD)[0]
            if active.size > 0:
                durations += 1
                radii.append(int(np.max(np.abs(active - source_idx))))
            else:
                radii.append(0)

            spread_active = np.where(per_cell_effect > self.SPREAD_THRESHOLD)[0]
            if spread_active.size > 0:
                spread_radius = int(np.max(np.abs(spread_active - source_idx)))
                spread_radii.append(spread_radius)
                if spread_radius >= self.SPREAD_MIN_RADIUS:
                    spread_duration += 1
            else:
                spread_radii.append(0)

        left_strong_onset = self._first_sustained_onset(
            left_boundary_effects,
            self.STRONG_PORT_THRESHOLD,
            self.STRONG_PORT_SUSTAIN,
        )
        right_strong_onset = self._first_sustained_onset(
            right_boundary_effects,
            self.STRONG_PORT_THRESHOLD,
            self.STRONG_PORT_SUSTAIN,
        )
        if not np.isfinite(left_strong_onset):
            left_strong_onset = float(cfg.runtime.total_steps)
        if not np.isfinite(right_strong_onset):
            right_strong_onset = float(cfg.runtime.total_steps)
        strong_port_onset = min(left_strong_onset, right_strong_onset)
        left_total = float(sum(left_boundary_effects))
        right_total = float(sum(right_boundary_effects))
        port_balance = (right_total - left_total) / max(left_total + right_total, 1e-8)
        if right_strong_onset < left_strong_onset:
            preferred_side = 1.0
        elif left_strong_onset < right_strong_onset:
            preferred_side = -1.0
        else:
            preferred_side = 1.0 if right_total > left_total else -1.0

        return {
            "lightcone_radius": float(max(radii) if radii else 0.0),
            "lightcone_duration": float(durations),
            "lightcone_area": float(sum(radii)),
            "lightcone_spread_radius_mean": float(np.mean(spread_radii) if spread_radii else 0.0),
            "lightcone_spread_radius_peak": float(max(spread_radii) if spread_radii else 0.0),
            "lightcone_spread_duration": float(spread_duration),
            "lightcone_effect_total": float(sum(effects)),
            "lightcone_source_effect": float(np.mean(source_effects) if source_effects else 0.0),
            "lightcone_boundary_effect": float(np.mean(boundary_effects) if boundary_effects else 0.0),
            "lightcone_port_impact": float(max(boundary_effects) if boundary_effects else 0.0),
            "lightcone_port_duration": float(port_duration),
            "lightcone_boundary_onset": float(boundary_onset if boundary_onset is not None else cfg.runtime.total_steps),
            "lightcone_left_port_impact": float(max(left_boundary_effects) if left_boundary_effects else 0.0),
            "lightcone_right_port_impact": float(max(right_boundary_effects) if right_boundary_effects else 0.0),
            "lightcone_left_port_total": left_total,
            "lightcone_right_port_total": right_total,
            "lightcone_left_strong_port_onset": left_strong_onset,
            "lightcone_right_strong_port_onset": right_strong_onset,
            "lightcone_strong_port_onset": strong_port_onset,
            "lightcone_port_balance": float(port_balance),
            "lightcone_port_preferred_side": preferred_side,
            "source_index": float(source_idx),
            "history": perturbed["history"],
            "final_metrics": perturbed["final_metrics"],
        }

    def run(self, cfg):
        branch_metrics = {branch: self._measure_branch(cfg, branch) for branch in self.BRANCHES}
        full = branch_metrics["full"]

        final_metrics = full["final_metrics"].copy()
        for metric_name in (
            "lightcone_radius",
            "lightcone_duration",
            "lightcone_area",
            "lightcone_spread_radius_mean",
            "lightcone_spread_radius_peak",
            "lightcone_spread_duration",
            "lightcone_effect_total",
            "lightcone_source_effect",
            "lightcone_boundary_effect",
            "lightcone_port_impact",
            "lightcone_port_duration",
            "lightcone_boundary_onset",
            "lightcone_left_port_impact",
            "lightcone_right_port_impact",
            "lightcone_left_port_total",
            "lightcone_right_port_total",
            "lightcone_left_strong_port_onset",
            "lightcone_right_strong_port_onset",
            "lightcone_strong_port_onset",
            "lightcone_port_balance",
            "lightcone_port_preferred_side",
            "source_index",
        ):
            final_metrics[metric_name] = full[metric_name]

        supported_count = 0
        for branch in self.BRANCHES[1:]:
            branch_result = branch_metrics[branch]
            radius_delta = full["lightcone_radius"] - branch_result["lightcone_radius"]
            duration_delta = full["lightcone_duration"] - branch_result["lightcone_duration"]
            area_delta = full["lightcone_area"] - branch_result["lightcone_area"]
            spread_radius_mean_delta = (
                full["lightcone_spread_radius_mean"] - branch_result["lightcone_spread_radius_mean"]
            )
            spread_duration_delta = full["lightcone_spread_duration"] - branch_result["lightcone_spread_duration"]
            effect_total_delta = full["lightcone_effect_total"] - branch_result["lightcone_effect_total"]
            port_delta = full["lightcone_port_impact"] - branch_result["lightcone_port_impact"]
            port_duration_delta = full["lightcone_port_duration"] - branch_result["lightcone_port_duration"]
            onset_delta = branch_result["lightcone_boundary_onset"] - full["lightcone_boundary_onset"]
            strong_onset_delay = branch_result["lightcone_strong_port_onset"] - full["lightcone_strong_port_onset"]
            port_balance_shift = abs(branch_result["lightcone_port_balance"] - full["lightcone_port_balance"])
            preferred_side_switch = (
                1.0
                if branch_result["lightcone_port_preferred_side"] != full["lightcone_port_preferred_side"]
                else 0.0
            )
            support_score = (
                (1 if radius_delta > 1.0 else 0)
                + (1 if duration_delta > 4.0 else 0)
                + (1 if area_delta > 50.0 else 0)
                + (1 if spread_radius_mean_delta > 2.0 else 0)
                + (1 if spread_duration_delta > 8.0 else 0)
                + (1 if effect_total_delta > 75.0 else 0)
                + (1 if port_delta > 0.001 else 0)
                + (1 if port_duration_delta > 8.0 else 0)
                + (1 if onset_delta > 2.0 else 0)
                + (1 if strong_onset_delay > 4.0 else 0)
                + (1 if port_balance_shift > 0.08 else 0)
                + (1 if preferred_side_switch > 0.5 else 0)
            )
            if support_score >= 2:
                supported_count += 1

            final_metrics[f"{branch}_lightcone_radius"] = branch_result["lightcone_radius"]
            final_metrics[f"{branch}_lightcone_duration"] = branch_result["lightcone_duration"]
            final_metrics[f"{branch}_lightcone_area"] = branch_result["lightcone_area"]
            final_metrics[f"{branch}_lightcone_spread_radius_mean"] = branch_result["lightcone_spread_radius_mean"]
            final_metrics[f"{branch}_lightcone_spread_duration"] = branch_result["lightcone_spread_duration"]
            final_metrics[f"{branch}_lightcone_effect_total"] = branch_result["lightcone_effect_total"]
            final_metrics[f"{branch}_lightcone_port_impact"] = branch_result["lightcone_port_impact"]
            final_metrics[f"{branch}_lightcone_port_duration"] = branch_result["lightcone_port_duration"]
            final_metrics[f"{branch}_lightcone_boundary_onset"] = branch_result["lightcone_boundary_onset"]
            final_metrics[f"{branch}_lightcone_left_strong_port_onset"] = branch_result[
                "lightcone_left_strong_port_onset"
            ]
            final_metrics[f"{branch}_lightcone_right_strong_port_onset"] = branch_result[
                "lightcone_right_strong_port_onset"
            ]
            final_metrics[f"{branch}_lightcone_strong_port_onset"] = branch_result["lightcone_strong_port_onset"]
            final_metrics[f"{branch}_lightcone_port_balance"] = branch_result["lightcone_port_balance"]
            final_metrics[f"{branch}_lightcone_port_preferred_side"] = branch_result["lightcone_port_preferred_side"]
            final_metrics[f"{branch}_radius_delta"] = radius_delta
            final_metrics[f"{branch}_duration_delta"] = duration_delta
            final_metrics[f"{branch}_area_delta"] = area_delta
            final_metrics[f"{branch}_spread_radius_mean_delta"] = spread_radius_mean_delta
            final_metrics[f"{branch}_spread_duration_delta"] = spread_duration_delta
            final_metrics[f"{branch}_effect_total_delta"] = effect_total_delta
            final_metrics[f"{branch}_port_impact_delta"] = port_delta
            final_metrics[f"{branch}_port_duration_delta"] = port_duration_delta
            final_metrics[f"{branch}_boundary_onset_delta"] = onset_delta
            final_metrics[f"{branch}_strong_port_onset_delay"] = strong_onset_delay
            final_metrics[f"{branch}_port_balance_shift"] = port_balance_shift
            final_metrics[f"{branch}_preferred_side_switch"] = preferred_side_switch
            final_metrics[f"{branch}_ablation_support_score"] = float(support_score)

        final_metrics["lightcone_ablation_supported_count"] = float(supported_count)
        final_metrics["lightcone_ablation_supported_fraction"] = float(
            supported_count / (len(self.BRANCHES) - 1)
        )

        notes = (
            "Localized perturbation was compared across full physiology and matched ablations of stress sharing, "
            "conductance, and Z-memory. "
            f"SpreadRadius={final_metrics['lightcone_spread_radius_mean']:.1f}; "
            f"ablation_supported_count={final_metrics['lightcone_ablation_supported_count']:.0f}."
        )
        return AssayResult(
            history=full["history"],
            final_metrics=final_metrics,
            notes=notes,
        )
