# MorphoBASE v1.3 Project Report

Status date: March 10, 2026

## Completion Statement

I agree that v1.3 is complete in the sense intended by the v1.3 master architecture and the current executable build contract.

The strongest basis for that statement is operational, not rhetorical:

- The master ladder passes end to end in [artifacts/master_build_report.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\master_build_report.json).
- Organism-first phases `0` through `10` are green.
- Stack C benchmark robustness is fully closed in [artifacts/benchmark_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\benchmark_phase_robustness.json): `4/4` benchmarks are seed-stable, mechanism-supported, and marked `ready`.
- Stack D robustness is also closed in [artifacts/stack_d_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\stack_d_phase_robustness.json): `2/2` non-visual/control bridges are seed-stable, mechanism-supported, and marked `ready`.
- The main earlier caveat, growth usefulness, is now addressed in [artifacts/growth_usefulness_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\growth_usefulness_assay\final_metrics.json) with positive `growth_utility_gain` and positive `growth_efficiency_advantage`.

v1.3 should still be understood as a completed organism-first architecture milestone, not as the end of the entire research program. Ecology and further frontier promotion remain future work by design.

## Report Scope

This report is based on:

- the executable architecture in `morphobase/*`
- the assay registry in [morphobase/assays/registry.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\assays\registry.py)
- default and assay configs in `configs/*`
- the build contract in [docs/Codex_Master_Build_Plan_v1.3a.md](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\docs\Codex_Master_Build_Plan_v1.3a.md)
- the current artifact inventory in `artifacts/`

Artifact analysis in this report covers:

- the master ladder report
- all top-level robustness reports
- the organism-vs-baseline comparison report
- all top-level `*_assay/final_metrics.json` assay outputs
- the seed and ablation reports that summarize the many per-seed and per-condition run directories in `artifacts/`

## Executive Summary

MorphoBASE v1.3 is now a real organism-first adaptive substrate with:

- bounded survival and identity maintenance
- wound closure and stress-guided recruitment
- selective growth and plasticity maintenance
- setpoint memory and cryptic rewrite behavior
- causal perturbation propagation through a measurable light cone
- broad lesion recovery with no-gradient and retraining controls
- preserved task competence after injury
- localized interface remapping across two non-visual port families
- stable visual continual-learning bridges
- stable non-visual control and symbolic bridges
- exploratory frontier channels for tissue fields, oscillatory coupling, reaction-diffusion, stigmergic highways, and predictive local learning

The important architectural success is that task competence is attached to organismal maintenance, repair, and remapping mechanisms rather than bolted on as an unrelated benchmark model.

## What The Organism Is

MorphoBASE is not a single monolithic neural network. It is a synthetic body composed of cells, communication fields, repair dynamics, slow setpoint memory, and task-facing boundary ports.

The core object is [Body](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\organism\body.py), which evolves an [OrganismState](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\organism\state.py) over three clocks:

- fast clock: local cell-state and metabolic updates
- medium clock: communication fields and coherence channels
- slow clock: growth, repair, stage assignment, and setpoint-guided remodeling

That clock split is one of the most important design choices in the whole project. It lets the organism separate:

- immediate reactivity
- short-horizon coordination
- slower developmental and regenerative responses

## High-Level Processing Loop

1. A port injects task input at a boundary region.
2. The fast cell update pushes local hidden state, membrane state, stress, and plasticity.
3. Medium channels update field alignment, tissue fields, oscillations, predictive traces, reaction-diffusion, stigmergic highways, and Z-memory alignment.
4. Slow physiology decides whether repair or growth is justified.
5. The boundary readout is taken back out through the relevant port.
6. Assays score both task competence and organismal health.

That structure is why v1.3 can be discussed as an organism with ports instead of as a benchmark model with some recovery tricks.

## Organism State: Every Major State Variable

The organism state lives in [morphobase/organism/state.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\organism\state.py).

### Core cell/body state

- `hidden`: main latent cell state.
- `membrane`: a membrane-potential-like fast state used for coordination and conductance updates.
- `plasticity`: current local readiness to change.
- `energy`: local metabolic reserve.
- `stress`: local distress/load signal.
- `role_logits`: soft local role identity representation.
- `commitment`: maturity / specialization commitment.
- `field_alignment`: local agreement with nearby tissue patterning.
- `alive`: active cell mask.
- `conductance`: inter-cell coupling matrix.
- `stages`: categorical developmental stage per cell.
- `growth_pressure`: slow signal representing need for growth-like repair or compensation.

### Setpoint and morphology memory

- `z_alignment`: currently expressed alignment to the anatomical setpoint scaffold.
- `z_memory`: slower hidden memory of setpoint structure.

### Frontier communication channels

- `tissue_field`: bounded regional tissue context channel.
- `oscillation_phase`
- `oscillation_amplitude`
- `morphogen_activator`
- `morphogen_inhibitor`
- `highway_trace`
- `highway_flux`
- `predictive_prediction`
- `predictive_error`
- `predictive_precision`

### Growth-control bookkeeping

- `growth_cooldown`: suppresses repeated unnecessary growth.
- `growth_activity`: recent growth engagement level.
- `recent_growth_signal_mean`
- `recent_growth_event_fraction`
- `recent_growth_energy_transferred`
- `recent_growth_repair_fraction`
- `recent_growth_bottleneck_fraction`
- `recent_growth_decorative_fraction`
- `recent_structural_churn`
- `step_count`

These bookkeeping fields are important because they let the project distinguish helpful repair from pathology such as chronic late growth or decorative churn.

## Developmental Stages

The stage enum is defined in [morphobase/types.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\types.py):

- `seed`
- `exploratory`
- `differentiating`
- `mature`
- `dedifferentiating`
- `prunable`

These are assigned in [morphobase/development/growth.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\development\growth.py) based on plasticity, commitment, stress, and energy.

## Run Verdicts

The run verdict enum in [morphobase/types.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\types.py) is:

- `pass`
- `unknown`
- `unstable`
- `degenerate_lock`
- `chronic_growth`
- `plasticity_loss`
- `dead_field`
- `pseudo_maturity`

This matters because v1.3 intentionally treats several apparently stable but biologically wrong regimes as failures.

## Subsystem Walkthrough

### 1. Fast local cell dynamics

The fast local update is [minimal_cell_update](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\cells\femo.py).

Inputs:

- `hidden`
- `membrane`
- `stress`
- `plasticity`
- `conductance`
- `z_alignment`
- `field_alignment`
- `target_value`
- `noise_scale`

Behavior:

- computes prediction error against `target_value`
- couples each cell to its conductance-weighted neighborhood
- adds Z-drive and field-drive terms
- updates stress, membrane, plasticity, and hidden state

Important internal weights:

- `z_drive = 0.25 * z_alignment`
- `field_drive = 0.10 * (field_alignment - 0.5)`
- stress update: `0.75 * stress + 0.15 * abs(error) + 0.10 * coordination_mismatch`
- membrane update: `0.85 * membrane + 0.10 * error + 0.05 * neighbor_mismatch`
- plasticity update: `0.94 * plasticity + 0.04 * (1 - stress/5) + 0.02 * abs(z_alignment)`
- hidden update: `0.08 * plasticity * error + 0.06 * (neighbor_hidden - pred) + noise`

Interpretation:

- `hidden` carries task/body representation
- `membrane` carries faster electrical-like perturbation
- `stress` penalizes incoherence
- `plasticity` controls willingness to update
- `z_alignment` and `field_alignment` bias updates toward organismal coherence rather than naked error minimization

### 2. Metabolism and budgets

Metabolic consumption is in [morphobase/metabolism/energy.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\metabolism\energy.py).

Important parameters:

- `base_cost = 0.0015`
- additional costs:
  - `+0.0015 * plasticity`
  - `+0.0010 * stress`
  - `+0.0007 * (1 - field_alignment)`
  - `+0.0008 * growth_pressure`
- recovery terms:
  - `+0.0010 * field_alignment`
  - `+0.0008 * z_support`
  - `+0.0006 * (1 - clipped_stress)`

Budget helpers live in [morphobase/metabolism/budgets.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\metabolism\budgets.py):

- `transition_affordable(energy, cost)`
- `reserve_margin(energy, reserve_floor=0.15)`
- `growth_budget(..., reserve_floor=0.15, mobilization=0.2)`

Interpretation:

- high plasticity is metabolically expensive
- incoherent low-field states are metabolically expensive
- alignment and low stress are metabolically protective
- growth is never free

### 3. Conductance coupling

Conductance is updated in [morphobase/communication/conductance.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\conductance.py).

Key formula:

- `out = exp(-(dv + 0.5 * ds))`

Where:

- `dv` is membrane mismatch
- `ds` is stress mismatch

Meaning:

- cells with matching voltage-like state and stress couple more strongly
- stressed or incoherent neighborhoods decouple

`conductance_entropy` is used as a summary of coupling diversity/collapse.

### 4. Local field alignment

Field alignment is updated in [morphobase/communication/fields.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\fields.py).

This gives the organism a local consensus scaffold beyond raw hidden state.

### 5. Tissue fields

`tissue_field` is an exploratory, bounded regional context channel.

Update target weights:

- `0.34 * regional_hidden`
- `0.26 * local_hidden`
- `0.18 * z_memory`
- `0.14 * local_membrane`
- `0.12 * positions`

Update rule:

- `updated = 0.42 * current + 0.58 * target`

Interpretation:

- tissue fields represent slower regional state, not direct task output
- they are positioned between local cell dynamics and whole-body pattern memory

### 6. Z-field and setpoint memory

Defined in [morphobase/communication/z_field.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\z_field.py).

Important parameters:

- `update_z_memory(..., memory_rate=0.0125)`
- `update_z_alignment(..., rate=0.05)`

Interpretation:

- `z_alignment` is the currently expressed setpoint agreement
- `z_memory` is the slower hidden anatomical expectation
- this split is what makes cryptic rewrite possible

### 7. Growth and developmental control

Growth and stage assignment live in [morphobase/development/growth.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\development\growth.py).

#### Growth pressure

Current pressure mix:

- `0.45 * stress`
- `0.22 * (1 - field_alignment)`
- `0.13 * (1 - z_support)`
- `0.20 * predictive_error`
- `-0.28 * cooldown`

Interpretation:

- growth is now driven by real need, not raw size expansion
- stress, field mismatch, setpoint mismatch, and unresolved predictive error all count as need
- cooldown explicitly suppresses chatter

#### Growth eligibility

Default `should_grow(...)` requirements:

- `mean_growth_pressure > 0.30`
- `mean_energy > 0.20`
- `mean_stress > 0.03`
- `active_need_fraction > 0.15`

#### Growth transfer and restraint

`apply_regulated_growth(...)` uses:

- `reserve_floor = 0.18`
- donor mobilization via `growth_budget(..., mobilization=0.14)`
- transfer pool cap `min(sum(donor_surplus) * 0.10, sum(recipient_budget) * 0.75)`
- recipients selected from high-need and low-cooldown cells

#### Growth bookkeeping outputs

- `growth_signal_mean`
- `growth_need_fraction`
- `growth_event_fraction`
- `growth_energy_transferred`
- `growth_repair_fraction`
- `growth_bottleneck_fraction`
- `growth_decorative_fraction`
- `structural_churn`

This is one of the biggest improvements in late v1.3. Growth is no longer just "on" or "off"; it is now classified and audited.

### 8. Repair systems in the body loop

Repair logic lives in [morphobase/organism/body.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\organism\body.py).

There are three major repair paths:

- parameter drift repair
- port-region repair
- general distressed-region repair

#### Parameter drift repair

The body detects incoherence through:

- hidden consensus gap
- membrane neighbor gap
- Z-memory gap
- role disorder

It then repairs hidden, membrane, Z alignment, Z memory, field alignment, plasticity, commitment, energy, and stress in a coordinated way.

This is why parameter corruption is treated as physiological repair, not just a benchmark nuisance.

#### Port-region repair

Boundary repair detects:

- weak coupling near ports
- low field support
- low energy
- hidden gap
- Z-gap

It then performs boundary-local recoupling and donor-support sharing instead of whole-body correction.

This is what makes port remap localization possible.

#### Distressed-region repair

Generic distress repair uses a blend of:

- neighbor hidden state
- local Z-memory

This is the basic setpoint-guided healing loop.

### 9. Oscillatory coupling

Defined in [morphobase/communication/oscillations.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\oscillations.py).

Key parameters:

- intrinsic base frequency `0.12`
- hidden contribution `0.05`
- tissue contribution `0.04`
- membrane contribution `0.03`
- stress penalty `0.03`
- phase coupling weight `0.18`

Amplitude target:

- `0.14`
- `+0.24 * abs(local_membrane)`
- `+0.22 * abs(local_tissue)`
- `+0.14 * abs(sin(updated_phase))`
- `+0.10 * coupling_strength`
- `-0.14 * clipped_stress`

Update:

- amplitude `0.58 * current + 0.42 * target`

### 10. Reaction-diffusion regionalization

Defined in [morphobase/communication/reaction_diffusion.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\reaction_diffusion.py).

Key activator reaction terms:

- `0.14 * activator_drive`
- `0.10 * tissue_drive`
- `0.06 * oscillation_drive`
- `-0.18 * activator`
- `-0.12 * inhibitor * activator`

Key inhibitor reaction terms:

- `0.08 * activator`
- `0.05 * tissue_drive`
- `-0.15 * inhibitor`

Diffusion coefficients:

- activator `0.18`
- inhibitor `0.28`

### 11. Stigmergic highways

Defined in [morphobase/communication/stigmergy.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\stigmergy.py).

Deposition weights:

- `0.22 * hidden_activity`
- `0.30 * traffic`
- `0.26 * field_drive`
- `0.12 * energy_support`

Trace update:

- `0.68 * current_trace`
- `0.12 * smoothed_trace`
- `0.16 * deposition`

Flux update:

- `0.50 * current_flux`
- `0.36 * sqrt(next_trace * (0.15 + traffic) * (0.15 + field_drive))`

### 12. Predictive coding local learners

Defined in [morphobase/communication/predictive_coding.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\predictive_coding.py).

Prediction target weights:

- `0.46 * local_signal`
- `0.28 * neighbor_signal`
- `0.14 * tissue_field`
- `0.12 * z_memory`

Update rules:

- prediction: `0.68 * current + 0.32 * target`
- error: `0.52 * current_error + 0.48 * abs(local_signal - next_prediction)`
- precision target: `1 - error + 0.20 * abs(tissue_field)`
- precision: `0.62 * current + 0.38 * target`

Interpretation:

- these are local prediction-and-error channels, not backpropagation layers

## Ports And Task Interfaces

Ports are the explicit interface between the organism and a task family.

Base behavior is in [morphobase/ports/base.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\ports\base.py).

Common port parameters:

- `width`: width of boundary interaction region
- `support_margin`: nearby support band used for locality analysis
- `input_window`
- `output_window`
- `input_shift`
- `output_shift`
- `scale`
- `flip`
- `input_attenuation`
- `readout_attenuation`

`apply_input(..., gain=0.42)` writes into:

- hidden
- membrane
- field alignment
- z alignment
- z memory
- energy
- stress

Implemented port families:

- rule port
- control port
- MNIST/FashionMNIST-style visual port
- pattern port via remap assays

## Config Parameters: Every Explicit User-Facing Config Field

The config schema is defined in [morphobase/config/schema.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\config\schema.py) and defaulted in [configs/defaults.yaml](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\configs\defaults.yaml).

### `run.*`

- `run.name`: human-readable run name.
- `run.output_dir`: output artifact directory.
- `run.save_plots`: whether plots are emitted.
- `run.summary_name`: summary markdown filename.
- `run.registry_name`: CSV registry filename.
- `run.seed`: run seed.

### `runtime.*`

- `runtime.total_steps`: total organism rollout steps for the assay.
- `runtime.dt`: simulation time step.
- `runtime.log_every`: logging interval.

### `body.*`

- `body.num_cells`: number of cells in the synthetic body.
- `body.hidden_dim`: hidden dimension per cell.
- `body.energy_init`: initial energy.
- `body.stress_init`: initial stress.
- `body.plasticity_init`: initial plasticity.
- `body.z_alignment_init`: initial setpoint alignment.

### `assay.*`

- `assay.name`: registered assay name.
- `assay.noise_scale`: fast update noise scale.
- `assay.target_value`: nominal target used by the local cell update.

### `logging.*`

- `logging.level`: log verbosity.
- `logging.event_log_name`: event log filename.

## Internal Tuning Parameters: Core Defaults And Weights

This section identifies the exposed internal tuning constants that materially shape organism behavior. These are code-level parameters rather than YAML config fields.

### Cell update tuning

Source: [morphobase/cells/femo.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\cells\femo.py)

- `z_drive` weight `0.25`
- `field_drive` weight `0.10`
- stress self-retention `0.75`
- stress error sensitivity `0.15`
- stress coordination penalty `0.10`
- membrane retention `0.85`
- membrane error term `0.10`
- membrane neighbor term `0.05`
- plasticity retention `0.94`
- plasticity anti-stress term `0.04`
- plasticity Z-support term `0.02`
- hidden error correction `0.08`
- hidden neighbor correction `0.06`

### Growth tuning

Source: [morphobase/development/growth.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\development\growth.py)

- growth pressure stress weight `0.45`
- field mismatch weight `0.22`
- Z-support mismatch weight `0.13`
- predictive error weight `0.20`
- cooldown penalty `0.28`
- growth threshold `0.30`
- minimum mean energy `0.20`
- minimum mean stress `0.03`
- minimum active need fraction `0.15`
- reserve floor `0.18`
- mobilization `0.14`
- repair threshold quantile `0.70`
- bottleneck threshold quantile `0.70`
- donor transfer fraction cap `0.10`
- recipient budget cap `0.75`

### Stage assignment tuning

- exploratory: `plasticity > 0.65` and `commitment < 0.2`
- differentiating: `plasticity > 0.3` and `commitment >= 0.2`
- mature: `commitment > 0.7`, `plasticity >= 0.15`, `energy > 0.25`
- dedifferentiating: `stress > 0.4` or `energy < 0.18`
- prunable: `plasticity < 0.08` or `energy < 0.08`

### Metabolism tuning

Source: [morphobase/metabolism/energy.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\metabolism\energy.py)

- base cost `0.0015`
- plasticity cost weight `0.0015`
- stress cost weight `0.0010`
- low-field cost weight `0.0007`
- growth-pressure cost weight `0.0008`
- field recovery weight `0.0010`
- Z-support recovery weight `0.0008`
- low-stress recovery weight `0.0006`

### Budget tuning

Source: [morphobase/metabolism/budgets.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\metabolism\budgets.py)

- `reserve_floor` default `0.15`
- `mobilization` default `0.2`

### Z-field tuning

Source: [morphobase/communication/z_field.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\z_field.py)

- `memory_rate = 0.0125`
- `alignment_rate = 0.05`

### Tissue field tuning

Source: [morphobase/communication/fields.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\fields.py)

- regional hidden `0.34`
- local hidden `0.26`
- Z-memory `0.18`
- local membrane `0.14`
- positional prior `0.12`
- state carryover `0.42`
- target blend `0.58`

### Oscillation tuning

Source: [morphobase/communication/oscillations.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\oscillations.py)

- intrinsic base frequency `0.12`
- hidden contribution `0.05`
- tissue contribution `0.04`
- membrane contribution `0.03`
- stress penalty `0.03`
- phase coupling weight `0.18`
- amplitude base `0.14`
- amplitude membrane contribution `0.24`
- amplitude tissue contribution `0.22`
- amplitude phase contribution `0.14`
- amplitude coupling contribution `0.10`
- amplitude stress penalty `0.14`
- amplitude carryover `0.58`
- amplitude target blend `0.42`

### Predictive coding tuning

Source: [morphobase/communication/predictive_coding.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\predictive_coding.py)

- local signal `0.46`
- neighbor signal `0.28`
- tissue field `0.14`
- Z-memory `0.12`
- prediction carryover `0.68`
- prediction target blend `0.32`
- error carryover `0.52`
- error refresh `0.48`
- precision tissue bonus `0.20`
- precision carryover `0.62`
- precision target blend `0.38`

### Reaction-diffusion tuning

Source: [morphobase/communication/reaction_diffusion.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\reaction_diffusion.py)

- activator drive `0.14`
- tissue drive `0.10`
- oscillation drive `0.06`
- activator decay `0.18`
- inhibitor-on-activator suppression `0.12`
- inhibitor activator term `0.08`
- inhibitor tissue term `0.05`
- inhibitor decay `0.15`
- activator diffusion `0.18`
- inhibitor diffusion `0.28`

### Stigmergy tuning

Source: [morphobase/communication/stigmergy.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\communication\stigmergy.py)

- hidden deposition `0.22`
- traffic deposition `0.30`
- field deposition `0.26`
- energy-support deposition `0.12`
- trace carryover `0.68`
- smoothed-trace contribution `0.12`
- new deposition contribution `0.16`
- flux carryover `0.50`
- flux target weight `0.36`

### Port tuning

Source: [morphobase/ports/base.py](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\morphobase\ports\base.py)

- boundary injection gain `0.42`
- window width
- support margin
- input attenuation
- readout attenuation
- input shift
- output shift
- scale
- optional flip

## Safe User-Tunable Parameters

These are the parameters a user can change with the lowest risk of invalidating the architecture.

### Safe config-level tuning

- `run.seed`
- `run.output_dir`
- `run.save_plots`
- `runtime.total_steps`
- `runtime.dt`
- `runtime.log_every`
- `body.num_cells`
- `body.hidden_dim`
- `body.energy_init`
- `body.stress_init`
- `body.plasticity_init`
- `body.z_alignment_init`
- `assay.noise_scale`
- `assay.target_value`
- `logging.level`

### Semi-safe code-level tuning

These are meaningful, but changing them changes the organism itself:

- growth threshold and reserve floor
- growth mobilization and transfer caps
- cell update weights
- metabolic cost weights
- Z-memory rates
- conductance sensitivity
- port width and support margin
- probe variant loads such as `growth_probe` or `repair_probe`

### Recommended user workflow for tuning

1. Tune YAML config first.
2. Only tune code-level constants one subsystem at a time.
3. Re-run the master build after any physiology change.
4. Re-check:
   - [artifacts/master_build_report.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\master_build_report.json)
   - [artifacts/benchmark_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\benchmark_phase_robustness.json)
   - [artifacts/stack_d_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\stack_d_phase_robustness.json)

## Assay Inventory

Current registered assays:

- `smoke`
- `identity`
- `wound_closure`
- `stress_recruitment`
- `growth_usefulness`
- `compensation_block`
- `plasticity_stress`
- `setpoint_rewrite`
- `lightcone`
- `lesion_battery`
- `lesion_preserves_competence`
- `port_remap`
- `mnist_sanity`
- `split_mnist`
- `permuted_mnist`
- `split_fashion_mnist`
- `permuted_fashion_mnist`
- `gridworld_remap`
- `sequential_rules`
- `lesion_sequential_rules`
- `lesion_gridworld_remap`
- `lesion_split_mnist`
- `tissue_field_probe`
- `oscillatory_coupling_probe`
- `reaction_diffusion_probe`
- `stigmergic_highway_probe`
- `predictive_coding_probe`

## Artifact Folder Inventory

The `artifacts/` folder currently contains:

- canonical `*_assay/` run directories
- per-seed benchmark and Stack D run directories
- per-condition ablation run directories
- top-level robustness JSON/Markdown reports
- the master build report
- run registry CSV

Important top-level reports:

- [artifacts/master_build_report.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\master_build_report.json)
- [artifacts/benchmark_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\benchmark_phase_robustness.json)
- [artifacts/stack_d_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\stack_d_phase_robustness.json)
- [artifacts/single_organism_vs_transformer.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\single_organism_vs_transformer.json)

## Results Analysis

### Master build result

The master ladder is fully passing.

That means the build now clears:

- phase 0: substrate survival
- phase 1: identity and repair
- phase 2: growth, compensation, and plasticity
- phase 3: setpoint memory and perturbation reach
- phase 4 onward visual bridge phases
- phase 9: control remap bridge
- phase 10: symbolic non-visual bridge

This is the cleanest project-level indicator that v1.3 is complete as an executable architecture.

### Phase 0: Substrate survival

Representative result: [artifacts/master_build_report.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\master_build_report.json)

The smoke assay shows:

- bounded energy
- low stress
- live active cell counts
- nonzero field, Z, tissue, predictive, and frontier channels

Interpretation:

- the organism is not collapsing or freezing before more meaningful claims begin

### Phase 1: Identity and repair

Assays:

- `identity`
- `wound_closure`
- `stress_recruitment`

Interpretation:

- the body maintains coherence under mild perturbation
- wound closure and stress recruitment are not flatline behaviors
- organismal coordination signals become stronger without obvious instability

### Phase 2: Growth, compensation, and plasticity

#### Growth usefulness

Source: [artifacts/growth_usefulness_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\growth_usefulness_assay\final_metrics.json)

Current key results:

- `growth_utility_gain = 0.02095`
- `growth_efficiency_advantage = 0.01128`
- `mean_growth_decorative_fraction = 0.0`
- `late_growth_event_fraction_mean = 0.0`
- `energy_advantage = -0.02704`
- `lesion_field_advantage = 0.13995`

Interpretation:

- the earlier growth problem is largely fixed
- growth is now selectively useful and efficient in lesion-focused terms
- total global energy still does not simply improve in every comparison
- that is acceptable because the important criterion is now repair/competence benefit per cost, not raw unused reserve

#### Compensation and plasticity

`compensation_block` and `plasticity_stress` pass in the master ladder, which means:

- the organism can tolerate blocked channels without trivial collapse
- plasticity is retained rather than silently locking into pseudo-mature pathology

### Phase 3: Setpoint memory, lesion recovery, and perturbation reach

#### Setpoint rewrite

Source: [artifacts/setpoint_rewrite_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\setpoint_rewrite_assay\final_metrics.json)

Key results:

- `rewrite_mode_supported_count = 3`
- `rewrite_mode_supported_fraction = 1.0`
- `strong_cryptic_mode_count = 2`
- `rewrite_persistence = 0.2471`
- `hidden_z_memory_gap_advantage = 0.3984`
- best rewrite mode: `stress_bias`

Interpretation:

- all three rewrite modes are supported
- hidden setpoint memory remains clearly separable from visible state
- repeated persistence across lesion cycles is present
- honest caveat: aggregate top-level `cryptic_shift` is currently negative, so the strongest evidence is repeated persistence and hidden-memory separation rather than a uniformly positive single summary scalar

#### Lightcone

Source: [artifacts/lightcone_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\lightcone_assay\final_metrics.json)

Key results:

- `lightcone_ablation_supported_count = 3`
- `lightcone_area = 4032`
- `lightcone_duration = 168`
- `lightcone_effect_total = 1134.51`
- `lightcone_port_duration = 168`
- strong causal deltas under:
  - `stress_sharing_off`
  - `conductance_ablated`
  - `z_memory_ablated`

Interpretation:

- perturbation spread is large
- more importantly, it changes in interpretable ways under channel-specific ablation
- the light cone is therefore causal evidence, not just a large activity plume

#### Lesion battery

Source: [artifacts/lesion_battery_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\lesion_battery_assay\final_metrics.json)

Key results:

- `lesion_recovery_mean = 0.8495`
- `lesion_recovery_probability_mean = 0.7778`
- `lesion_recovery_steps_mean = 69.56`
- `organismal_recovery_vs_retraining_ratio = 0.7664`
- `recovery_retention_without_gradients = 1.1684`
- `repeated_injury_vs_retraining_ratio = 0.8337`
- `port_localization_advantage = 0.2941`
- `port_boundary_locality_advantage = 3.4945`

Interpretation:

- lesion competence is broad, not limited to one injury class
- no-gradient recovery remaining strong is important evidence against hidden retraining during rollout
- raw isolated port-disruption recovery remains the weakest lesion family in current metrics
- the system nevertheless shows strong locality and meaningful repeated-injury resilience

#### Lesion preserves competence

Source: [artifacts/lesion_preserves_competence_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\lesion_preserves_competence_assay\final_metrics.json)

Key results:

- `competence_retention_ratio = 0.9894`
- `competence_supported_task_count = 2`
- `post_recovery_competence = 0.7304`
- `no_gradient_post_recovery_competence = 0.6837`
- `organismal_competence_advantage_vs_no_gradient = 0.0467`
- `organismal_competence_vs_retraining_ratio = 1.0012`

Interpretation:

- this closes one of the most important Tier-5 requirements
- recovery is not only morphological
- prior competence is substantially preserved after lesion on more than one toy task family

#### Port remap

Source: [artifacts/port_remap_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\port_remap_assay\final_metrics.json)

Key results:

- `supported_port_family_count = 2`
- `port_remap_mode_supported_count = 3`
- `boundary_locality_ratio = 1.6853`
- `competence_retention_ratio = 0.9824`
- `post_recovery_competence = 0.7811`
- `pattern_boundary_locality_ratio = 1.9568`
- `rule_boundary_locality_ratio = 1.4137`

Interpretation:

- remapping is localized near the interface rather than diffusing through the body indiscriminately
- this is a strong proof that boundary adaptation is real across two non-visual port families
- honest caveat: `local_vs_whole_body_competence_advantage` is slightly negative overall, so the strongest evidence is locality plus competence preservation, not universally better task score than whole-body disruption

### Stack C: Visual benchmark bridge analysis

Source: [artifacts/benchmark_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\benchmark_phase_robustness.json)

Summary:

- `benchmark_count = 4`
- `stable_seed_benchmark_count = 4`
- `full_mechanism_support_count = 4`
- `ready_benchmark_count = 4`
- `attention_required_count = 0`

#### MNIST sanity

Source: [artifacts/mnist_sanity_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\mnist_sanity_assay\final_metrics.json)

- `mnist_accuracy_advantage = 0.3`

Interpretation:

- the visual port bridge is genuinely above chance

#### Split-MNIST

Base assay metrics from [artifacts/split_mnist_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\split_mnist_assay\final_metrics.json):

- `final_accuracy_mean = 0.3833`
- `peak_accuracy_mean = 0.4833`
- `mean_forgetting = 0.125`
- `bwt = -0.125`

Seed robustness from [artifacts/benchmark_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\benchmark_phase_robustness.json):

- `final_accuracy_mean = 0.3567`
- `std = 0.0416`
- stable across 5 seeds
- full mechanism support `3/3`

Interpretation:

- this is a real continual-learning bridge, not a one-seed artifact
- growth, stress, and Z-field matter in the matched chamber

#### Permuted-MNIST

Base assay metrics from [artifacts/permuted_mnist_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\permuted_mnist_assay\final_metrics.json):

- `final_accuracy_mean = 0.2920`
- `peak_accuracy_mean = 0.3400`
- `mean_forgetting = 0.0600`
- `bwt = -0.0550`

Seed robustness summary:

- `final_accuracy_mean = 0.2744`
- `std = 0.0149`
- stable across 5 seeds
- full mechanism support
- growth-probe chamber also stable and fully mechanism-supported

Interpretation:

- this is the harder distribution-shift bridge
- the standard benchmark is now stable
- the causal chamber is stronger than the raw accuracy story, which is appropriate for this benchmark's role

#### Split-FashionMNIST

Base assay metrics from [artifacts/split_fashion_mnist_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\split_fashion_mnist_assay\final_metrics.json):

- `final_accuracy_mean = 0.4333`
- `peak_accuracy_mean = 0.5833`
- `mean_forgetting = 0.1875`
- `bwt = -0.1875`

Seed robustness summary:

- `final_accuracy_mean = 0.4733`
- `std = 0.0544`
- stable across seeds
- full mechanism support

Interpretation:

- strongest visual benchmark family in v1.3 on raw continual-learning accuracy

#### Permuted-FashionMNIST

Base assay metrics from [artifacts/permuted_fashion_mnist_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\permuted_fashion_mnist_assay\final_metrics.json):

- `final_accuracy_mean = 0.2640`
- `peak_accuracy_mean = 0.2680`
- `mean_forgetting = 0.0050`
- `bwt = 0.0250`

Seed robustness summary:

- `final_accuracy_mean = 0.2696`
- `std = 0.0157`
- stable across seeds
- full mechanism support

Interpretation:

- raw accuracy is modest
- forgetting is very low and BWT is positive
- that makes this an unusually retention-friendly bridge in the current stack

### Stack D: Non-visual control and symbolic bridge analysis

Source: [artifacts/stack_d_phase_robustness.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\stack_d_phase_robustness.json)

Summary:

- `assay_count = 2`
- `stable_seed_assay_count = 2`
- `full_mechanism_support_count = 2`
- `ready_assay_count = 2`

#### Gridworld remap

Base assay metrics from [artifacts/gridworld_remap_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\gridworld_remap_assay\final_metrics.json):

- `final_success_mean = 0.8167`
- `peak_success_mean = 0.8167`
- `mean_forgetting = 0.0`
- `bwt = 0.0`
- `efficiency_mean = 0.5833`

Seed robustness:

- `final_success_mean = 0.8167`
- `std = 0.0`
- `forgetting_max = 0.0`

Interpretation:

- this is one of the cleanest results in the whole repository
- after benchmark stabilization, the standard chamber is perfectly stable across the tested seeds
- repair-probe ablations are no longer flat and now show real dependency, especially on Z-field support

#### Sequential rules

Base assay metrics from [artifacts/sequential_rules_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\sequential_rules_assay\final_metrics.json):

- `final_accuracy_mean = 0.8`
- `peak_accuracy_mean = 0.8167`
- `mean_forgetting = 0.0208`
- `bwt = -0.0208`

Robustness summary:

- `final_accuracy_mean = 0.8467`
- `std = 0.0245`
- `min = 0.8`
- full mechanism support in the repair probe

Interpretation:

- this is the strongest non-visual symbolic bridge
- forgetting is very low
- standard chamber is stable
- ablation story is now informative rather than flat

### Lesion-aware competence bridge analysis

Source: [artifacts/single_organism_vs_transformer.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\single_organism_vs_transformer.json)

Summary:

- `task_count = 4`
- `baseline_trained_retention_task_count = 3`
- `organism_mechanism_drop_supported_task_count = 3`
- `organism_energy_efficiency_reported_task_count = 4`
- `organism_score_per_energy_mean = 3.7331`

Important interpretation:

- lesion tasks are evaluated with `baseline_trained_retained_competence`
- this is the honest contract for damage-aware comparisons
- the readout is trained on baseline organism embeddings, then evaluated after lesion

This makes the comparison more meaningful than retraining every condition separately.

#### Lesion sequential rules

- organism retained accuracy `0.5833`
- `no_growth` drops to `0.5000`
- `no_z_field` drops to `0.4667`

Interpretation:

- both growth and Z-field now matter under lesion in the symbolic task

#### Lesion gridworld remap

- organism retained success `0.7833`
- `no_growth` drops to `0.7667`
- `no_z_field` collapses to `0.1500`

Interpretation:

- strongest current lesion-aware control proof
- Z-field dependence is especially strong

#### Lesion Split-MNIST

- organism retained accuracy `0.3667`
- `no_growth` drops to `0.3333`
- `no_z_field` drops to `0.2500`

Interpretation:

- visual lesion competence is now part of the proof stack

### Frontier probe analysis

These probes are intentionally non-gating. They are there to test whether later frontier mechanisms can produce bounded, interpretable, local behavior without breaking the completed organism core.

#### Tissue field probe

Source: [artifacts/tissue_field_probe_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\tissue_field_probe_assay\final_metrics.json)

- `tissue_field_localization_ratio = 3.0731`

Interpretation:

- tissue-field response is local, not globally smeared

#### Oscillatory coupling probe

Source: [artifacts/oscillatory_coupling_probe_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\oscillatory_coupling_probe_assay\final_metrics.json)

- `oscillation_localization_ratio = 22.5731`
- `oscillation_coherence_advantage = 0.0081`

Interpretation:

- driven oscillatory entrainment is strong and localized

#### Reaction-diffusion probe

Source: [artifacts/reaction_diffusion_probe_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\reaction_diffusion_probe_assay\final_metrics.json)

- `reaction_diffusion_localization_ratio = 10.1036`

Interpretation:

- patterning response is measurable and spatially selective

#### Stigmergic highway probe

Source: [artifacts/stigmergic_highway_probe_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\stigmergic_highway_probe_assay\final_metrics.json)

- `stigmergic_localization_ratio = 12.6442`
- `stigmergic_persistence_retention = 0.8991`

Interpretation:

- traces persist enough to function as a highway-like channel without becoming unstable

#### Predictive coding probe

Source: [artifacts/predictive_coding_probe_assay/final_metrics.json](G:\My Drive\AI\AI_Projects\MorphoBase\V1.3\MorphoBASE_v1.3a\artifacts\predictive_coding_probe_assay\final_metrics.json)

- `predictive_localization_ratio = 21.4543`
- `predictive_error_recovery_gain = 0.1918`

Interpretation:

- predictive local learners are behaving like local repair-sensitive error channels, not as a diffuse extra feature map

## How We Know The Project Proceeded As Intended

There are five strong reasons.

### 1. The build order stayed organism-first

The project did not start by optimizing benchmarks. It established:

- survival
- identity
- repair
- growth control
- setpoint memory
- lesion competence
- remap capacity

before leaning on broader benchmark claims.

### 2. Every important mechanism is tied to an assay

The project repeatedly enforced:

- implementation
- diagnostics
- assay
- ablation

That discipline is the main reason the current results are credible.

### 3. Visual benchmarks are stable and mechanism-supported

Stack C is now fully ready. This is a major milestone because it means the visual benchmark bridge is no longer fragile or one-seed-only.

### 4. Non-visual bridges are also stable

Stack D is ready as well. This matters because it prevents the project from becoming a visual-benchmark-only story.

### 5. Frontier mechanisms are exploratory, not contaminating the proof stack

Tissue fields, oscillations, reaction-diffusion, stigmergy, and predictive coding were added only after the main organism claims were already stable. That is exactly the right sequencing.

## Honest Residual Caveats

v1.3 is complete, but not perfect.

### Caveat 1: Setpoint rewrite is stronger in persistence than in one-number cryptic shift

Current `setpoint_rewrite` is clearly supported on repeated persistence and hidden-memory separation, but the single top-level `cryptic_shift` summary is negative in the current canonical metrics.

### Caveat 2: Port disruption remains the weakest lesion family

The lesion battery is broadly strong, but isolated port disruption is still less robust than the best lesion classes in raw organismal recovery terms.

### Caveat 3: Benchmark performance is not the main win

The project now has stable benchmark bridges, but raw accuracy is not the headline claim. The headline claim is organismal maintenance, remapping, and repair-linked competence.

### Caveat 4: Frontier mechanisms are not yet promoted into the gated core

That is the correct state for v1.3, but it means these channels are promising rather than fully established organism essentials.

## What v1.3 Delivers

v1.3 delivers a completed organism-first substrate with:

- explicit body clocks
- local cell dynamics
- metabolic budgets
- conductance coupling
- field alignment
- Z-memory and setpoint-guided repair
- selective growth
- lesion-aware competence preservation
- port-local remapping
- stable visual CL bridges
- stable non-visual CL bridges
- frontier physiological channels added without destabilizing the core

That is a substantial architecture milestone.

## Recommended Post-v1.3 Direction

If the project continues into v1.4-style work, the cleanest next directions are:

1. Keep single-organism pressure tests going before ecology.
2. Promote only the frontier channels that improve repair, remap, or transfer.
3. Leave ecology for after the single-organism story is fully exhausted.
4. Use the organism-vs-baseline comparison stack to keep the project honest about efficiency and competence retention.

## Final Assessment

MorphoBASE v1.3 is complete as planned.

It is now best described as a repair-capable adaptive organism with task-facing ports, not merely a benchmark model. The build is reproducible, the phases are gated, the major bridges are stable, and the artifact record supports the central claims.
