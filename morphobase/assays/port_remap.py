from __future__ import annotations

import numpy as np

from morphobase.assays.common import (
    AssayResult,
    AssayRunner,
    build_synthetic_body,
    disrupt_global_port_state,
    disrupt_port_region,
)
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.organism.scheduler import Scheduler
from morphobase.ports.base import BasePort
from morphobase.ports.toy_pattern_port import ToyPatternPort
from morphobase.ports.toy_rule_port import ToyRulePort


class PortRemapAssay(AssayRunner):
    SIGNAL_BLOCK = 24
    PRE_WINDOW = 48
    DROP_WINDOW = 24
    RECOVERY_OFFSET = 64
    INPUT_HIGH = 0.84
    INPUT_LOW = 0.16
    CONDITIONS = ("control", "local_port_remap", "whole_body_disruption")
    FAMILY_REMAPS = {
        "rule": ("input_shift", "window_shift"),
        "pattern": ("input_shift", "window_shift"),
    }
    FAMILY_PORTS = {
        "rule": ToyRulePort,
        "pattern": ToyPatternPort,
    }

    @staticmethod
    def _window_mean(trace: list[float], start: int, stop: int) -> float:
        values = [value for index, value in enumerate(trace) if start <= index < stop]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _rule_signal(step: int) -> float:
        return (
            PortRemapAssay.INPUT_HIGH
            if (step // PortRemapAssay.SIGNAL_BLOCK) % 2 == 0
            else PortRemapAssay.INPUT_LOW
        )

    @staticmethod
    def _pattern_signal(step: int, width: int) -> np.ndarray:
        amplitude = PortRemapAssay._rule_signal(step)
        phase = 0.6 * ((step // PortRemapAssay.SIGNAL_BLOCK) % 4)
        grid = np.linspace(0.0, np.pi, width)
        signal = amplitude + 0.16 * np.sin(grid + phase)
        return np.clip(signal, 0.0, 1.0)

    @classmethod
    def _signal_payload(cls, family_name: str, step: int, port: BasePort):
        if family_name == "rule":
            return cls._rule_signal(step)
        if family_name == "pattern":
            return cls._pattern_signal(step, port.boundary_window("input").width)
        raise ValueError(f"Unknown port family: {family_name}")

    @staticmethod
    def _target_value(payload) -> float:
        values = np.asarray(payload, dtype=float)
        return float(np.clip(values.mean(), 0.0, 1.0))

    @staticmethod
    def _disturbance_value(state, reference_state, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        hidden_delta = float(np.mean(np.abs(state.hidden[mask] - reference_state.hidden[mask])))
        membrane_delta = float(np.mean(np.abs(state.membrane[mask] - reference_state.membrane[mask])))
        field_delta = float(np.mean(np.abs(state.field_alignment[mask] - reference_state.field_alignment[mask])))
        z_delta = float(np.mean(np.abs(state.z_alignment[mask] - reference_state.z_alignment[mask])))
        return 0.44 * hidden_delta + 0.20 * membrane_delta + 0.20 * field_delta + 0.16 * z_delta

    def _mapping_spec(self, family_name: str, remap_mode: str, port: BasePort) -> dict:
        if family_name == "rule":
            if remap_mode == "input_shift":
                return {
                    "flip": False,
                    "input_shift": max(1, port.boundary_window("input").width // 2),
                    "scale": 0.98,
                }
            if remap_mode == "window_shift":
                return {
                    "flip": False,
                    "output_shift": max(2, port.boundary_window("output").width // 2),
                    "scale": 1.0,
                }
        if family_name == "pattern":
            if remap_mode == "input_shift":
                return {
                    "input_shift": max(1, port.boundary_window("input").width // 2),
                    "scale": 0.98,
                    "phase_offset": np.pi / 6.0,
                }
            if remap_mode == "window_shift":
                return {
                    "output_shift": max(1, port.boundary_window("output").width // 2),
                    "scale": 1.0,
                    "phase_offset": np.pi / 4.0,
                }
        raise ValueError(f"Unknown port remap: {family_name}/{remap_mode}")

    @staticmethod
    def _remap_masks(baseline_port: BasePort, remapped_port: BasePort) -> tuple[np.ndarray, np.ndarray]:
        boundary_mask = BasePort.union_mask(
            baseline_port.boundary_mask("input", margin=1),
            remapped_port.boundary_mask("input", margin=1),
            baseline_port.boundary_mask("output", margin=1),
            remapped_port.boundary_mask("output", margin=1),
        )
        exclusion_mask = BasePort.union_mask(
            baseline_port.support_mask("input"),
            remapped_port.support_mask("input"),
            baseline_port.support_mask("output"),
            remapped_port.support_mask("output"),
        )
        distal_mask = ~exclusion_mask
        if not np.any(distal_mask):
            distal_mask = remapped_port.distal_mask()
        return boundary_mask, distal_mask

    def _apply_local_remap(self, body, port: BasePort, mapping_spec: dict, *, family_name: str) -> None:
        previous_input = port.boundary_window("input")
        previous_output = port.boundary_window("output")
        port.remap(mapping_spec)
        if family_name == "pattern":
            port.damage({"input_attenuation": 0.98, "readout_attenuation": 0.93})
        else:
            port.damage({"input_attenuation": 0.97, "readout_attenuation": 0.90})
        new_input = port.boundary_window("input")
        new_output = port.boundary_window("output")

        changed_windows = []
        if (previous_input.start, previous_input.stop) != (new_input.start, new_input.stop):
            changed_windows.append((min(previous_input.start, new_input.start), max(previous_input.stop, new_input.stop)))
        if (previous_output.start, previous_output.stop) != (new_output.start, new_output.stop):
            changed_windows.append((min(previous_output.start, new_output.start), max(previous_output.stop, new_output.stop)))
        if not changed_windows:
            changed_windows.append((new_output.start, new_output.stop))
        for start, stop in changed_windows:
            attenuation = 0.46 if family_name == "pattern" else 0.40
            disrupt_port_region(body, start, stop, attenuation=attenuation)

        support_mask = BasePort.union_mask(
            previous_input.mask(body.state.hidden.shape[0], margin=port.support_margin),
            new_input.mask(body.state.hidden.shape[0], margin=port.support_margin),
            previous_output.mask(body.state.hidden.shape[0], margin=port.support_margin),
            new_output.mask(body.state.hidden.shape[0], margin=port.support_margin),
        )
        body.state.plasticity[support_mask] = np.clip(body.state.plasticity[support_mask] + 0.10, 0.0, 1.0)
        body.state.commitment[support_mask] = np.clip(body.state.commitment[support_mask] - 0.05, 0.0, 1.0)
        body.state.energy[support_mask] = np.clip(body.state.energy[support_mask] + 0.05, 0.0, 1.0)
        body.state.z_memory[support_mask] = np.clip(body.state.z_memory[support_mask] + 0.10, -1.0, 1.0)
        if family_name == "pattern":
            body.state.field_alignment[support_mask] = np.clip(body.state.field_alignment[support_mask] + 0.08, 0.0, 1.0)
            body.state.z_alignment[support_mask] = np.clip(body.state.z_alignment[support_mask] + 0.05, -1.0, 1.0)

    def _apply_global_remap(self, body, port: BasePort, mapping_spec: dict, *, family_name: str) -> None:
        port.remap(mapping_spec)
        if family_name == "pattern":
            port.damage({"input_attenuation": 0.54, "readout_attenuation": 0.36})
            disrupt_global_port_state(body, attenuation=0.05)
            body.state.field_alignment *= 0.82
            body.state.z_memory *= 0.76
        else:
            port.damage({"input_attenuation": 0.68, "readout_attenuation": 0.54})
            disrupt_global_port_state(body, attenuation=0.10)

    def _run_rollout(self, cfg, *, family_name: str, remap_mode: str, condition: str) -> dict:
        seed = (
            cfg.run.seed
            + 53 * sum(ord(ch) for ch in family_name)
            + 31 * sum(ord(ch) for ch in remap_mode)
            + 7 * sum(ord(ch) for ch in condition)
        )
        np.random.seed(seed)
        body = build_synthetic_body(cfg)
        port = self.FAMILY_PORTS[family_name](cfg.body.num_cells)
        baseline_port = self.FAMILY_PORTS[family_name](cfg.body.num_cells)
        scheduler = Scheduler()
        history = []
        state_history = [body.state.copy()]
        z_history = [body.state.z_alignment.copy()]
        competence_trace: list[float] = []
        remap_step = cfg.runtime.total_steps // 2

        for step in range(cfg.runtime.total_steps):
            payload = self._signal_payload(family_name, step, port)
            target_signal = self._target_value(payload)
            if step == remap_step:
                mapping_spec = self._mapping_spec(family_name, remap_mode, port)
                if condition == "control":
                    port.remap(mapping_spec)
                elif condition == "local_port_remap":
                    self._apply_local_remap(body, port, mapping_spec, family_name=family_name)
                elif condition == "whole_body_disruption":
                    self._apply_global_remap(body, port, mapping_spec, family_name=family_name)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

            port.apply_input(body, payload)
            due = scheduler.due(step)
            body.step(due.fast, due.medium, due.slow, cfg.assay.noise_scale, target_signal)
            if (
                family_name == "pattern"
                and condition == "whole_body_disruption"
                and remap_step <= step < remap_step + 48
                and (step - remap_step) % 12 == 0
            ):
                body.state.field_alignment *= 0.95
                body.state.z_memory *= 0.97
                body.state.membrane = np.clip(body.state.membrane * 0.92, -1.0, 1.0)
                body.state.stress = np.clip(body.state.stress + 0.05, 0.0, 5.0)
            decoded = port.read_output(body)
            competence_trace.append(float(np.clip(1.0 - port.loss_fn(decoded, target_signal), 0.0, 1.0)))
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if step % cfg.runtime.log_every == 0 or step == cfg.runtime.total_steps - 1:
                history.append(summarize_state(body.state, z_history=z_history))

        final_metrics = history[-1].copy()
        final_metrics["lightcone_proxy"] = lightcone_proxy(state_history)
        return {
            "history": history,
            "state_history": state_history,
            "competence_trace": competence_trace,
            "final_metrics": final_metrics,
            "remap_step": remap_step,
            "baseline_port": baseline_port,
            "port": port,
        }

    def _summarize_competence(self, rollout: dict, control_rollout: dict) -> dict[str, float]:
        remap_step = rollout["remap_step"]
        pre_range = (remap_step - self.PRE_WINDOW, remap_step)
        drop_range = (remap_step, remap_step + self.DROP_WINDOW)
        recovery_range = (remap_step + self.RECOVERY_OFFSET, len(rollout["competence_trace"]))

        pre_comp = self._window_mean(rollout["competence_trace"], *pre_range)
        drop_comp = self._window_mean(rollout["competence_trace"], *drop_range)
        post_comp = self._window_mean(rollout["competence_trace"], *recovery_range)
        control_post = self._window_mean(control_rollout["competence_trace"], *recovery_range)
        return {
            "pre_remap_competence": pre_comp,
            "post_remap_competence": drop_comp,
            "post_recovery_competence": post_comp,
            "control_post_recovery_competence": control_post,
            "competence_drop": float(max(pre_comp - drop_comp, 0.0)),
            "competence_retention_ratio": float(post_comp / max(pre_comp, 1e-8)),
            "control_relative_post_recovery": float(post_comp / max(control_post, 1e-8)),
            "competence_recovery_gain": float(post_comp - drop_comp),
        }

    def _summarize_locality(
        self,
        *,
        rollout: dict,
        control_rollout: dict,
        boundary_mask: np.ndarray,
        distal_mask: np.ndarray,
    ) -> dict[str, float]:
        start_index = rollout["remap_step"] + 1
        boundary_trace = []
        distal_trace = []
        for index in range(start_index, len(rollout["state_history"])):
            state = rollout["state_history"][index]
            reference_state = control_rollout["state_history"][index]
            boundary_trace.append(self._disturbance_value(state, reference_state, boundary_mask))
            distal_trace.append(self._disturbance_value(state, reference_state, distal_mask))

        boundary_peak = float(np.max(boundary_trace)) if boundary_trace else 0.0
        distal_peak = float(np.max(distal_trace)) if distal_trace else 0.0
        boundary_mean = float(np.mean(boundary_trace)) if boundary_trace else 0.0
        distal_mean = float(np.mean(distal_trace)) if distal_trace else 0.0
        return {
            "boundary_peak_disturbance": boundary_peak,
            "distal_peak_disturbance": distal_peak,
            "boundary_mean_disturbance": boundary_mean,
            "distal_mean_disturbance": distal_mean,
            "boundary_locality_ratio": float(boundary_peak / max(distal_peak, 1e-8)),
            "distal_sparing_fraction": float(
                np.clip(1.0 - distal_mean / max(boundary_mean + distal_mean, 1e-8), 0.0, 1.0)
            ),
        }

    @staticmethod
    def _mode_supported(local_comp: dict[str, float], local_loc: dict[str, float], competence_advantage: float, boundary_locality_advantage: float) -> bool:
        return (
            local_comp["post_recovery_competence"] >= 0.69
            and local_comp["competence_retention_ratio"] >= 0.97
            and competence_advantage >= -0.03
            and local_loc["boundary_locality_ratio"] >= 1.24
            and boundary_locality_advantage >= 0.15
        )

    def run(self, cfg):
        final_metrics: dict[str, float] = {}
        representative_history = None
        representative_metrics = None
        overall_supported_modes = 0
        overall_pre = []
        overall_local_posts = []
        overall_global_posts = []
        overall_boundary_advantages = []
        supported_families = 0

        for family_name, remap_modes in self.FAMILY_REMAPS.items():
            family_supported_modes = 0
            family_pre = []
            family_local_posts = []
            family_global_posts = []
            family_boundary_advantages = []

            for remap_mode in remap_modes:
                control_rollout = self._run_rollout(cfg, family_name=family_name, remap_mode=remap_mode, condition="control")
                local_rollout = self._run_rollout(cfg, family_name=family_name, remap_mode=remap_mode, condition="local_port_remap")
                global_rollout = self._run_rollout(cfg, family_name=family_name, remap_mode=remap_mode, condition="whole_body_disruption")

                boundary_mask, distal_mask = self._remap_masks(local_rollout["baseline_port"], local_rollout["port"])
                local_comp = self._summarize_competence(local_rollout, control_rollout)
                global_comp = self._summarize_competence(global_rollout, control_rollout)
                local_loc = self._summarize_locality(
                    rollout=local_rollout,
                    control_rollout=control_rollout,
                    boundary_mask=boundary_mask,
                    distal_mask=distal_mask,
                )
                global_loc = self._summarize_locality(
                    rollout=global_rollout,
                    control_rollout=control_rollout,
                    boundary_mask=boundary_mask,
                    distal_mask=distal_mask,
                )

                family_pre.append(local_comp["pre_remap_competence"])
                family_local_posts.append(local_comp["post_recovery_competence"])
                family_global_posts.append(global_comp["post_recovery_competence"])
                overall_pre.append(local_comp["pre_remap_competence"])
                overall_local_posts.append(local_comp["post_recovery_competence"])
                overall_global_posts.append(global_comp["post_recovery_competence"])

                competence_advantage = float(local_comp["post_recovery_competence"] - global_comp["post_recovery_competence"])
                distal_sparing_advantage = float(local_loc["distal_sparing_fraction"] - global_loc["distal_sparing_fraction"])
                boundary_locality_advantage = float(local_loc["boundary_locality_ratio"] - global_loc["boundary_locality_ratio"])
                family_boundary_advantages.append(boundary_locality_advantage)
                overall_boundary_advantages.append(boundary_locality_advantage)

                if self._mode_supported(local_comp, local_loc, competence_advantage, boundary_locality_advantage):
                    family_supported_modes += 1
                    overall_supported_modes += 1

                prefix = f"{family_name}_{remap_mode}"
                for key, value in local_comp.items():
                    final_metrics[f"{prefix}_local_{key}"] = value
                for key, value in global_comp.items():
                    final_metrics[f"{prefix}_whole_body_{key}"] = value
                for key, value in local_loc.items():
                    final_metrics[f"{prefix}_local_{key}"] = value
                for key, value in global_loc.items():
                    final_metrics[f"{prefix}_whole_body_{key}"] = value
                final_metrics[f"{prefix}_local_vs_whole_body_competence_advantage"] = competence_advantage
                final_metrics[f"{prefix}_distal_sparing_advantage"] = distal_sparing_advantage
                final_metrics[f"{prefix}_boundary_locality_advantage"] = boundary_locality_advantage

                if representative_history is None:
                    representative_history = local_rollout["history"]
                    representative_metrics = local_rollout["final_metrics"].copy()

            family_pre_mean = float(np.mean(family_pre))
            family_local_mean = float(np.mean(family_local_posts))
            family_global_mean = float(np.mean(family_global_posts))
            family_boundary_mean = float(np.mean([
                final_metrics[f"{family_name}_{mode}_local_boundary_locality_ratio"] for mode in remap_modes
            ]))
            family_retention = float(family_local_mean / max(family_pre_mean, 1e-8))
            family_advantage = float(family_local_mean - family_global_mean)
            final_metrics[f"{family_name}_pre_remap_competence"] = family_pre_mean
            final_metrics[f"{family_name}_post_recovery_competence"] = family_local_mean
            final_metrics[f"{family_name}_whole_body_post_recovery_competence"] = family_global_mean
            final_metrics[f"{family_name}_competence_retention_ratio"] = family_retention
            final_metrics[f"{family_name}_boundary_locality_ratio"] = family_boundary_mean
            final_metrics[f"{family_name}_boundary_locality_advantage"] = float(np.mean(family_boundary_advantages))
            final_metrics[f"{family_name}_local_vs_whole_body_competence_advantage"] = family_advantage
            final_metrics[f"{family_name}_supported_mode_count"] = float(family_supported_modes)
            if family_supported_modes >= 1:
                supported_families += 1

        final_metrics.update(representative_metrics or {})
        final_metrics["pre_remap_competence"] = float(np.mean(overall_pre))
        final_metrics["post_recovery_competence"] = float(np.mean(overall_local_posts))
        final_metrics["whole_body_post_recovery_competence"] = float(np.mean(overall_global_posts))
        final_metrics["competence_retention_ratio"] = float(
            final_metrics["post_recovery_competence"] / max(final_metrics["pre_remap_competence"], 1e-8)
        )
        final_metrics["local_vs_whole_body_competence_advantage"] = float(
            final_metrics["post_recovery_competence"] - final_metrics["whole_body_post_recovery_competence"]
        )
        final_metrics["port_localization_advantage"] = float(np.mean(overall_boundary_advantages))
        final_metrics["boundary_locality_advantage"] = float(np.mean(overall_boundary_advantages))
        final_metrics["boundary_locality_ratio"] = float(np.mean([
            final_metrics[f"{family}_{mode}_local_boundary_locality_ratio"]
            for family, modes in self.FAMILY_REMAPS.items()
            for mode in modes
        ]))
        final_metrics["whole_body_boundary_locality_ratio"] = float(np.mean([
            final_metrics[f"{family}_{mode}_whole_body_boundary_locality_ratio"]
            for family, modes in self.FAMILY_REMAPS.items()
            for mode in modes
        ]))
        final_metrics["distal_sparing_fraction"] = float(np.mean([
            final_metrics[f"{family}_{mode}_local_distal_sparing_fraction"]
            for family, modes in self.FAMILY_REMAPS.items()
            for mode in modes
        ]))
        final_metrics["port_remap_mode_supported_count"] = float(overall_supported_modes)
        final_metrics["supported_port_family_count"] = float(supported_families)
        final_metrics["cross_port_competence_gap"] = float(
            abs(final_metrics["rule_post_recovery_competence"] - final_metrics["pattern_post_recovery_competence"])
        )
        final_metrics["cross_port_boundary_locality_gap"] = float(
            abs(final_metrics["rule_boundary_locality_ratio"] - final_metrics["pattern_boundary_locality_ratio"])
        )

        notes = (
            "Port remap assay ran matched local-vs-global remaps on rule and pattern non-visual ports. "
            f"Pre={final_metrics['pre_remap_competence']:.4f}; "
            f"post={final_metrics['post_recovery_competence']:.4f}; "
            f"retention={final_metrics['competence_retention_ratio']:.4f}; "
            f"families={int(final_metrics['supported_port_family_count'])}; "
            f"locality_advantage={final_metrics['boundary_locality_advantage']:.4f}."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )
