from dataclasses import dataclass

import numpy as np

from morphobase.organism.body import Body
from morphobase.organism.scheduler import Scheduler
from morphobase.organism.state import OrganismState
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.communication.stress_sharing import diffuse_stress


@dataclass(slots=True)
class AssayResult:
    history: list[dict]
    final_metrics: dict
    notes: str = ''


class AssayRunner:
    def run(self, cfg):
        raise NotImplementedError


def build_synthetic_body(cfg) -> Body:
    state = OrganismState.synthetic(
        cfg.body.num_cells,
        cfg.body.hidden_dim,
        cfg.body.energy_init,
        cfg.body.stress_init,
        cfg.body.plasticity_init,
        cfg.body.z_alignment_init,
    )
    return Body(state)


def recovery_fraction(pre_value: float, lesion_value: float, final_value: float) -> float:
    denominator = pre_value - lesion_value
    if abs(denominator) < 1e-8:
        return 1.0 if final_value >= lesion_value else 0.0
    return float(np.clip((final_value - lesion_value) / denominator, -1.0, 1.0))


def lesion_segment(body: Body, start: int, stop: int, *, energy_scale: float = 0.4, stress_boost: float = 0.6) -> None:
    body.state.hidden[start:stop] = 0.0
    body.state.membrane[start:stop] = 0.0
    body.state.energy[start:stop] = np.clip(body.state.energy[start:stop] * energy_scale, 0.0, 1.0)
    body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + stress_boost, 0.0, 5.0)


def bias_z_field(body: Body, start: int, stop: int, bias: float) -> None:
    body.state.z_alignment[start:stop] = np.clip(body.state.z_alignment[start:stop] + bias, -1.0, 1.0)
    body.state.z_memory[start:stop] = np.clip(body.state.z_memory[start:stop] + bias, -1.0, 1.0)


def bias_conductance_region(
    body: Body,
    start: int,
    stop: int,
    *,
    within_scale: float = 1.45,
    cross_scale: float = 0.55,
) -> None:
    body.state.conductance[start:stop, start:stop] *= within_scale
    body.state.conductance[start:stop, :start] *= cross_scale
    body.state.conductance[start:stop, stop:] *= cross_scale
    body.state.conductance[:start, start:stop] *= cross_scale
    body.state.conductance[stop:, start:stop] *= cross_scale
    body.state.conductance = np.clip(body.state.conductance, 0.0, 2.0)
    diagonal = np.diag_indices_from(body.state.conductance)
    body.state.conductance[diagonal] = 1.0


def bias_stress_region(
    body: Body,
    start: int,
    stop: int,
    *,
    local_boost: float = 0.55,
    diffusion_coefficient: float = 0.35,
) -> None:
    body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + local_boost, 0.0, 5.0)
    body.state.stress = diffuse_stress(
        body.state.stress,
        body.state.conductance,
        coefficient=diffusion_coefficient,
    )


def sever_conductance(body: Body, start: int, stop: int, attenuation: float = 0.20) -> None:
    body.state.conductance[start:stop, :] *= attenuation
    body.state.conductance[:, start:stop] *= attenuation
    diagonal = np.diag_indices_from(body.state.conductance)
    body.state.conductance[diagonal] = 1.0
    body.state.hidden[start:stop] *= 0.75
    body.state.field_alignment[start:stop] *= 0.5
    body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + 0.35, 0.0, 5.0)


def corrupt_field_alignment(body: Body, start: int, stop: int, value: float = 0.0) -> None:
    body.state.field_alignment[start:stop] = value


def corrupt_parameters(
    body: Body,
    start: int,
    stop: int,
    *,
    hidden_noise: float = 0.06,
    membrane_noise: float = 0.18,
    role_noise: float = 1.50,
) -> None:
    body.state.hidden[start:stop] = np.clip(
        body.state.hidden[start:stop] + np.random.normal(0.0, hidden_noise, size=body.state.hidden[start:stop].shape),
        -2.0,
        2.0,
    )
    body.state.membrane[start:stop] = np.clip(
        body.state.membrane[start:stop] + np.random.normal(0.0, membrane_noise, size=body.state.membrane[start:stop].shape),
        -1.0,
        1.0,
    )
    body.state.role_logits[start:stop] += np.random.normal(
        0.0,
        role_noise,
        size=body.state.role_logits[start:stop].shape,
    )
    body.state.plasticity[start:stop] = np.clip(body.state.plasticity[start:stop] * 0.76, 0.0, 1.0)
    body.state.commitment[start:stop] = np.clip(body.state.commitment[start:stop] + 0.10, 0.0, 1.0)
    body.state.field_alignment[start:stop] = np.clip(body.state.field_alignment[start:stop] * 0.58, 0.0, 1.0)
    body.state.z_alignment[start:stop] = np.clip(body.state.z_alignment[start:stop] * 0.64, -1.0, 1.0)
    body.state.energy[start:stop] = np.clip(body.state.energy[start:stop] * 0.985, 0.0, 1.0)
    body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + 0.48, 0.0, 5.0)


def targeted_tissue_ablation(
    body: Body,
    start: int,
    stop: int,
    *,
    margin: int = 2,
) -> None:
    ext_start = max(0, start - margin)
    ext_stop = min(body.state.hidden.shape[0], stop + margin)
    lesion_segment(body, ext_start, ext_stop, energy_scale=0.25, stress_boost=0.85)
    sever_conductance(body, ext_start, ext_stop, attenuation=0.10)
    body.state.z_alignment[ext_start:ext_stop] *= 0.30
    body.state.z_memory[ext_start:ext_stop] *= 0.22
    body.state.field_alignment[ext_start:ext_stop] *= 0.25


def disrupt_port_region(
    body: Body,
    start: int,
    stop: int,
    *,
    attenuation: float = 0.22,
) -> None:
    body.state.hidden[start:stop] *= 0.72
    body.state.membrane[start:stop] = np.clip(body.state.membrane[start:stop] * 0.20, -1.0, 1.0)
    body.state.energy[start:stop] = np.clip(body.state.energy[start:stop] * 0.68, 0.0, 1.0)
    body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + 0.42, 0.0, 5.0)
    body.state.field_alignment[start:stop] = np.clip(body.state.field_alignment[start:stop] * 0.30, 0.0, 1.0)
    body.state.z_alignment[start:stop] *= 0.55
    body.state.z_memory[start:stop] *= 0.78
    body.state.conductance[start:stop, :] *= attenuation
    body.state.conductance[:, start:stop] *= attenuation
    diagonal = np.diag_indices_from(body.state.conductance)
    body.state.conductance[diagonal] = 1.0


def disrupt_global_port_state(
    body: Body,
    *,
    attenuation: float = 0.22,
) -> None:
    body.state.hidden *= 0.58
    body.state.membrane = np.clip(body.state.membrane * 0.12, -1.0, 1.0)
    body.state.energy = np.clip(body.state.energy * 0.55, 0.0, 1.0)
    body.state.stress = np.clip(body.state.stress + 0.65, 0.0, 5.0)
    body.state.field_alignment = np.clip(body.state.field_alignment * 0.18, 0.0, 1.0)
    body.state.z_alignment *= 0.45
    body.state.z_memory *= 0.55
    body.state.plasticity = np.clip(body.state.plasticity * 0.82, 0.0, 1.0)
    body.state.commitment = np.clip(body.state.commitment + 0.08, 0.0, 1.0)
    body.state.conductance *= attenuation
    diagonal = np.diag_indices_from(body.state.conductance)
    body.state.conductance[diagonal] = 1.0


def apply_retraining_correction(
    body: Body,
    reference_state: OrganismState,
    start: int,
    stop: int,
    *,
    strength: float = 0.32,
    support_strength: float = 0.12,
    support_margin: int = 2,
) -> None:
    support_start = max(0, start - support_margin)
    support_stop = min(body.state.hidden.shape[0], stop + support_margin)
    main_slice = slice(start, stop)
    support_slice = slice(support_start, support_stop)

    def blend(current, reference, rate: float):
        return (1.0 - rate) * current + rate * reference

    body.state.hidden[main_slice] = np.clip(
        blend(body.state.hidden[main_slice], reference_state.hidden[main_slice], strength),
        -2.0,
        2.0,
    )
    body.state.hidden[support_slice] = np.clip(
        blend(body.state.hidden[support_slice], reference_state.hidden[support_slice], support_strength),
        -2.0,
        2.0,
    )
    body.state.membrane[main_slice] = np.clip(
        blend(body.state.membrane[main_slice], reference_state.membrane[main_slice], strength),
        -1.0,
        1.0,
    )
    body.state.energy[main_slice] = np.clip(
        blend(body.state.energy[main_slice], reference_state.energy[main_slice], strength),
        0.0,
        1.0,
    )
    body.state.stress[main_slice] = np.clip(
        blend(body.state.stress[main_slice], reference_state.stress[main_slice], strength),
        0.0,
        5.0,
    )
    body.state.field_alignment[main_slice] = np.clip(
        blend(body.state.field_alignment[main_slice], reference_state.field_alignment[main_slice], strength),
        0.0,
        1.0,
    )
    body.state.z_alignment[main_slice] = np.clip(
        blend(body.state.z_alignment[main_slice], reference_state.z_alignment[main_slice], strength),
        -1.0,
        1.0,
    )
    body.state.z_memory[main_slice] = np.clip(
        blend(body.state.z_memory[main_slice], reference_state.z_memory[main_slice], strength),
        -1.0,
        1.0,
    )
    body.state.role_logits[main_slice] = blend(
        body.state.role_logits[main_slice],
        reference_state.role_logits[main_slice],
        strength,
    )
    body.state.conductance[support_slice, :] = np.clip(
        blend(body.state.conductance[support_slice, :], reference_state.conductance[support_slice, :], support_strength),
        0.0,
        2.0,
    )
    body.state.conductance[:, support_slice] = np.clip(
        blend(body.state.conductance[:, support_slice], reference_state.conductance[:, support_slice], support_strength),
        0.0,
        2.0,
    )
    diagonal = np.diag_indices_from(body.state.conductance)
    body.state.conductance[diagonal] = 1.0


def rollout_body(
    cfg,
    *,
    body: Body | None = None,
    allow_growth: bool = True,
    no_gradient: bool = False,
    step_hook=None,
    after_step_hook=None,
    target_schedule=None,
):
    body = body or build_synthetic_body(cfg)
    scheduler = Scheduler()
    history = []
    z_history = [body.state.z_alignment.copy()]
    state_history = [body.state.copy()]

    for step in range(cfg.runtime.total_steps):
        if step_hook is not None:
            step_hook(step, body)

        due = scheduler.due(step)
        target_value = cfg.assay.target_value if target_schedule is None else target_schedule(step, body)
        body.step(
            due.fast,
            due.medium,
            due.slow,
            cfg.assay.noise_scale,
            target_value,
            allow_growth=allow_growth,
            no_gradient=no_gradient,
        )
        if after_step_hook is not None:
            after_step_hook(step, body)
        z_history.append(body.state.z_alignment.copy())
        state_history.append(body.state.copy())
        if step % cfg.runtime.log_every == 0 or step == cfg.runtime.total_steps - 1:
            history.append(summarize_state(body.state, z_history=z_history))

    final_metrics = history[-1].copy()
    final_metrics["lightcone_proxy"] = lightcone_proxy(state_history)
    return {
        "body": body,
        "history": history,
        "final_metrics": final_metrics,
        "state_history": state_history,
        "z_history": z_history,
    }
