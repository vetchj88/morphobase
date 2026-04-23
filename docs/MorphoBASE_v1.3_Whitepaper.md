# MorphoBASE v1.3: A Computational Substrate for Organism-First Adaptive Intelligence

Author: Justin
Date: March 2026

## Abstract

We present MorphoBASE v1.3, a computational substrate that inverts the standard machine learning design sequence: instead of building a task optimizer and retrofitting adaptation, we build a synthetic multicellular body with explicit physiology — metabolism, stress, repair, growth control, conductance, and setpoint memory — and attach task competence through boundary ports. The system is organized around principles drawn from developmental biology and bioelectric signaling research, particularly the multi-scale competency architecture described by Levin and colleagues.

v1.3 demonstrates several organism-level properties through a suite of 25 assays with matched ablation controls:

- **Lesion recovery**: 84.95% mean morphological recovery across lesion types, with 76.64% recovery relative to full retraining
- **Competence preservation**: 98.94% task competence retention after injury, measured across multiple task families
- **Gradient-free recovery**: organisms recover function without gradient-based retraining during rollout (recovery-without-gradients ratio: 1.17)
- **Setpoint memory**: hidden anatomical memory persists separately from expressed state (hidden-memory gap advantage: 0.40)
- **Selective growth**: growth events are repair-oriented with zero decorative growth fraction
- **Boundary-local adaptation**: port remapping remains localized (boundary locality ratio: 1.69) while preserving 98.24% competence
- **Measurable perturbation propagation**: a causal "light cone" that changes interpretably under channel-specific ablation

Task performance on standard continual-learning benchmarks is modest (27–47% across MNIST/FashionMNIST variants, 82–85% on non-visual tasks). This is expected: the system was not designed to maximize benchmark scores. The benchmarks serve as validation that organismal mechanisms produce real task competence, not as the primary contribution.

The primary contribution is a working, testable computational substrate in which repair, memory, and adaptation are properties of the body rather than features of a task-specific model.

## 1. Introduction

### 1.1 Motivation

Most machine learning systems are designed around task optimization. Adaptation, robustness, and continual learning are added as secondary properties — regularization terms, replay buffers, architectural tricks — layered onto systems whose core design is benchmark performance.

Biological organisms take the opposite approach. A planarian regenerates its head not because it has a loss function for head-having, but because it maintains an anatomical setpoint that guides repair. Its competence at navigating its environment survives the injury because that competence is embedded in a body that actively maintains itself. The body is primary; task performance is a consequence.

This perspective has been developed extensively in the work of Levin and colleagues on bioelectric signaling, multi-scale competency, and goal-directed behavior in biological systems [1–5]. The key insight is that biological intelligence operates at multiple scales simultaneously: individual cells make local homeostatic decisions, groups of cells coordinate through bioelectric and chemical signaling, and the whole organism exhibits goal-directed behavior that emerges from — but is not reducible to — these lower-level dynamics.

MorphoBASE asks whether this organizational principle can be instantiated computationally: can we build a synthetic body where task competence emerges from organismal maintenance rather than being engineered directly?

### 1.2 Scope and honest framing

MorphoBASE v1.3 is a research prototype. It is not a state-of-the-art continual learning system by benchmark standards. Its visual task performance (27–47% on MNIST/FashionMNIST variants) is well below specialized continual learning methods.

What it does demonstrate is that a system built around explicit physiology — with no benchmark-specific logic in its core — can exhibit measurable repair, memory, competence preservation, and adaptive growth. These properties are tested through ablation controls, not claimed through analogy.

The intended audience for this work is researchers interested in organism-inspired computation, bioelectric signaling models, and alternative computational substrates — not the continual learning leaderboard community.

## 2. Architecture

### 2.1 The body

The central runtime object is a `Body` that evolves a multicellular `OrganismState` across three clocks:

- **Fast clock** (every step): local cell-state updates, metabolic dynamics, stress and plasticity regulation
- **Medium clock** (every 4 steps): communication fields, conductance coupling, Z-memory alignment, tissue fields, oscillatory and predictive channels
- **Slow clock** (every 16 steps): repair decisions, growth control, developmental stage assignment, setpoint-guided remodeling

This three-clock separation is a deliberate design choice. It prevents fast task dynamics from overwriting slow anatomical structure, mirroring how biological systems separate immediate reactivity from developmental timescales.

### 2.2 State variables

Each cell in the body carries:

- **Core state**: hidden representation, membrane potential, energy, stress, plasticity, commitment
- **Identity**: role logits, developmental stage (seed → exploratory → differentiating → mature → dedifferentiating → prunable)
- **Setpoint memory**: z_alignment (currently expressed setpoint agreement) and z_memory (slower hidden anatomical memory)
- **Connectivity**: conductance matrix governing inter-cell coupling strength
- **Growth control**: growth pressure, cooldown, activity tracking, and bookkeeping for repair vs. decorative growth classification

### 2.3 Local cell dynamics

The fast cell update combines:
- Prediction error against a nominal target
- Conductance-weighted neighborhood coupling
- Z-alignment drive (weight 0.25) and field-alignment drive (weight 0.10)
- Stress and plasticity regulation

All update weights are deliberately small. The organism coheres through many weakly coupled mechanisms rather than one dominant controller.

### 2.4 Metabolism

Metabolic costs rise with plasticity, stress, field mismatch, and growth pressure. Recovery rises with field alignment, Z-support, and low stress. This makes adaptation genuinely costly and prevents the organism from appearing adaptive by spending unbounded energy.

### 2.5 Conductance and communication

Cells with matching membrane state and stress couple more strongly; stressed or incoherent neighborhoods decouple. Coordination is therefore a consequence of local physiological agreement, not a static wiring assumption.

### 2.6 Setpoint memory (Z-field)

The Z-field is one of the most important architectural components:
- `z_alignment` tracks currently expressed agreement with the anatomical scaffold
- `z_memory` stores a slower hidden version of that scaffold (update rate: 0.0125)

This split enables what we call "cryptic rewrite": a hidden anatomical change can persist even when visible phenotype is partially overwritten. This is analogous to how biological organisms maintain morphogenetic memory that is not directly visible in current tissue state.

### 2.7 Repair

Three repair paths operate in the body:
1. **Parameter drift repair**: detects incoherence through hidden consensus gap, membrane neighbor gap, Z-memory gap, and role disorder; repairs toward neighbor consensus blended with stored Z-memory
2. **Port-region repair**: boundary-local recoupling and donor-support sharing for port-adjacent injuries
3. **Distressed-region repair**: general setpoint-guided healing using neighbor state and Z-memory

### 2.8 Growth control

Growth is driven by real need (stress, field mismatch, setpoint mismatch, predictive error) and suppressed by cooldown. Growth eligibility requires sufficient pressure, energy, stress, and active-need fraction simultaneously. Every growth event is classified as repair-oriented, bottleneck-oriented, or decorative, and audited.

### 2.9 Ports

Ports are explicit boundary interfaces between the organism and task environments. They define boundary windows, support bands, input/readout attenuation, and remap shifts. Task-specific code lives entirely in the port layer; the body core remains task-agnostic.

Implemented port families: visual (MNIST/FashionMNIST), rule (sequential symbolic), control (gridworld), and pattern (remap assays).

## 3. Experimental Methodology

### 3.1 Assay-gated progression

The v1.3 build used a strict phase-gated progression. No later phase was accepted until earlier phases passed:

| Phase | Focus | Assays |
|-------|-------|--------|
| 0 | Substrate survival | smoke |
| 1 | Identity and repair | identity, wound_closure, stress_recruitment |
| 2 | Growth and plasticity | growth_usefulness, compensation_block, plasticity_stress |
| 3 | Memory and perturbation | setpoint_rewrite, lightcone, lesion_battery, lesion_preserves_competence, port_remap |
| C | Visual bridges | split_mnist, permuted_mnist, split_fashion_mnist, permuted_fashion_mnist |
| D | Non-visual bridges | gridworld_remap, sequential_rules |
| — | Lesion+task bridges | lesion_split_mnist, lesion_sequential_rules, lesion_gridworld_remap |

### 3.2 Evidence standards

A mechanism was accepted only when it had:
1. Implementation
2. Diagnostics
3. At least one dedicated assay
4. Ablation or matched control

This means the results reported below are supported by causal evidence (what happens when you remove the mechanism), not just correlation.

### 3.3 Seed robustness

All benchmark results are reported across 5 random seeds. A result is considered stable only if the standard deviation is bounded and the mechanism-support pattern holds across seeds.

## 4. Results

### 4.1 Lesion recovery

**Source**: lesion_battery assay (8 lesion types including parameter corruption, region knockout, conductance disruption, port disruption)

| Metric | Value |
|--------|-------|
| Mean morphological recovery | 84.95% |
| Recovery vs. full retraining | 76.64% |
| Recovery without gradients | 116.84% of gradient recovery |
| Repeated injury resilience vs. retraining | 83.37% |
| Recovery probability (fraction of lesions showing recovery) | 77.78% |
| Mean recovery time | 69.56 steps |
| Port boundary locality advantage | 3.49 |

The gradient-free recovery ratio exceeding 1.0 is noteworthy: the organism's physiological repair mechanisms (neighbor consensus, Z-memory guidance, port-region recoupling) actually outperform what gradient-based correction alone achieves during rollout. This is evidence that recovery is driven by the organism's own maintenance dynamics, not by hidden retraining.

**Honest caveat**: isolated port-disruption recovery remains the weakest lesion family.

### 4.2 Competence preservation after injury

**Source**: lesion_preserves_competence assay

| Metric | Value |
|--------|-------|
| Competence retention ratio | 98.94% |
| Supported task families | 2 |
| Post-recovery competence | 73.04% |
| No-gradient post-recovery competence | 68.37% |
| Organismal competence vs. retraining | 100.12% |

This is the result that most directly tests the Levin-inspired thesis: injury to the body should not destroy task competence if that competence is embedded in organismal maintenance rather than fragile parameter tuning. The 98.94% retention ratio across two task families supports this claim.

### 4.3 Setpoint memory and cryptic rewrite

**Source**: setpoint_rewrite assay

| Metric | Value |
|--------|-------|
| Supported rewrite modes | 3/3 |
| Strong cryptic rewrite modes | 2 |
| Rewrite persistence | 0.247 |
| Hidden z_memory gap advantage | 0.398 |

All three rewrite modes (stress bias, field bias, direct injection) produce measurable anatomical change. The hidden z_memory gap advantage of 0.40 means the slow anatomical memory diverges substantially from expressed state — evidence for a genuine hidden morphogenetic scaffold.

**Honest caveat**: the aggregate top-level `cryptic_shift` metric is currently negative. The strongest evidence is persistence and hidden-memory separation, not a uniformly positive single summary scalar.

### 4.4 Perturbation propagation (light cone)

**Source**: lightcone assay

| Metric | Value |
|--------|-------|
| Light cone area | 4,032 |
| Light cone duration | 168 steps |
| Total effect magnitude | 1,134.51 |
| Ablation-supported mechanisms | 3 (stress sharing, conductance, Z-memory) |

The light cone is large, but more importantly it changes in interpretable ways under channel-specific ablation. Removing stress sharing, conductance, or Z-memory each produce measurably different light cone profiles. This makes the light cone causal evidence for the organism's communication architecture, not just a large activity plume.

### 4.5 Selective growth

**Source**: growth_usefulness assay

| Metric | Value |
|--------|-------|
| Growth utility gain | 0.021 |
| Growth efficiency advantage | 0.011 |
| Decorative growth fraction | 0.0 |
| Late growth event fraction | 0.0 |
| Lesion field advantage | 0.140 |

Growth is now selective: it occurs in response to need (stress, field mismatch, setpoint mismatch), is repair-oriented, and produces measurable benefit by lesion-focused value-per-cost criteria. Decorative growth — growth that consumes resources without improving repair or competence — has been completely suppressed.

**Honest caveat**: total global energy does not simply improve in every comparison. The criterion is repair benefit per cost, not raw unused reserve.

### 4.6 Boundary-local adaptation (port remap)

**Source**: port_remap assay

| Metric | Value |
|--------|-------|
| Supported port families | 2 |
| Supported remap modes | 3 |
| Boundary locality ratio | 1.69 |
| Competence retention ratio | 98.24% |
| Post-recovery competence | 78.11% |

When the interface to the organism is remapped (analogous to sensory rewiring), adaptation remains localized near the boundary rather than diffusing through the body. This is consistent with the biological observation that sensory remapping produces local cortical reorganization, not global brain rewiring.

### 4.7 Benchmark bridges

These benchmarks validate that the organism's internal mechanisms produce real task competence. They are not the primary contribution.

#### Visual continual learning (Stack C)

| Benchmark | Mean Accuracy | Std | Forgetting | Seed-Stable | Mechanism-Supported |
|-----------|--------------|-----|------------|-------------|-------------------|
| Split-MNIST | 35.67% | 4.16% | 0.125 | Yes | 3/3 ablations |
| Permuted-MNIST | 27.44% | 1.49% | 0.060 | Yes | 3/3 ablations |
| Split-FashionMNIST | 47.33% | 5.44% | 0.188 | Yes | 3/3 ablations |
| Permuted-FashionMNIST | 26.96% | 1.57% | 0.005 | Yes | 3/3 ablations |

For context: random chance on 10-class classification is 10%. These scores are 2.7–4.7x chance. State-of-the-art continual learning methods achieve 90%+ on these benchmarks. MorphoBASE is not competing on this axis.

What matters here is:
1. All four benchmarks are **seed-stable** across 5 seeds
2. All four are **mechanism-supported**: ablating growth, stress, or Z-field produces measurable performance changes, confirming that task competence depends on organismal mechanisms
3. Permuted-FashionMNIST shows **very low forgetting** (0.005) and **positive backward transfer** (BWT = +0.025), meaning earlier tasks actually improve slightly after later learning — a signature of genuine knowledge consolidation rather than interference

#### Non-visual control and symbolic tasks (Stack D)

| Benchmark | Mean Score | Std | Forgetting | Seed-Stable | Mechanism-Supported |
|-----------|-----------|-----|------------|-------------|-------------------|
| Gridworld Remap | 81.67% success | 0.0% | 0.0 | Yes | 3/3 ablations |
| Sequential Rules | 84.67% accuracy | 2.45% | 0.021 | Yes | 3/3 ablations |

These non-visual tasks are substantially stronger. Gridworld Remap is the cleanest result in the entire project: perfectly stable across seeds with zero forgetting. Sequential Rules shows low forgetting (0.021) with a tight standard deviation.

These results demonstrate that the organism is a general substrate, not a visual-task-specific architecture.

#### Lesion-aware task bridges

Under lesion conditions, the organism retains task competence while ablation baselines degrade:

- **Lesion Sequential Rules**: organism retains 58.33% accuracy; no-growth drops to 50.00%, no-Z-field drops to 46.67%
- **Lesion Gridworld Remap**: organism retains function; ablation shows Z-field dependency
- **Lesion Split-MNIST**: organism retains function under damage

### 4.8 Organism vs. baseline comparison

| Metric | Value |
|--------|-------|
| Tasks with trained-retention competence | 3/4 |
| Tasks with mechanism-drop support | 3/4 |
| Mean organism score per energy unit | 3.73 |

The organism is strongest when the task rewards repair, remap, and persistence. It is not trying to beat transformers or MLPs on raw undamaged benchmark score.

## 5. Frontier mechanisms (exploratory)

Five additional communication channels are implemented but designated as non-gating exploratory probes:

| Channel | Localization Ratio |
|---------|-------------------|
| Tissue fields | 3.07 |
| Oscillatory coupling | 22.57 |
| Reaction-diffusion | 10.10 |
| Stigmergic highways | 12.64 |
| Predictive coding | 21.45 |

These channels are active, bounded, and locally interpretable. They are not yet promoted to required organism claims because their causal contribution to task competence has not been fully isolated.

## 6. Limitations

1. **Visual benchmark performance is low.** 27–47% on MNIST/FashionMNIST variants is well below specialized continual learning methods. This is a real limitation if the goal is task performance; it is expected if the goal is demonstrating organism-first principles.

2. **Setpoint rewrite evidence is mixed.** The strongest evidence is persistence and hidden-memory separation. The aggregate cryptic_shift metric is not uniformly positive.

3. **Port disruption is the weakest lesion class.** The organism recovers well from most injury types, but isolated port disruption remains the least comfortable recovery story.

4. **Scale is small.** The current organism has 64 cells with 32-dimensional hidden state. Whether these properties survive scaling is an open question.

5. **No ecology.** The current system is a single organism. Multi-organism interaction, competition, and collective behavior are future work.

6. **Frontier channels are unproven.** Oscillatory coupling, reaction-diffusion, stigmergy, and predictive coding are implemented and active, but their causal necessity is not yet established.

7. **The system was developed with AI assistance.** Architecture design, implementation, and debugging were conducted in collaboration with Claude (Anthropic). The experimental results are from actual code execution, not generated text.

## 7. Discussion

### 7.1 What this demonstrates

MorphoBASE v1.3 provides evidence that a computational system organized around explicit physiology can exhibit properties that are typically difficult to achieve in standard neural networks:

- **Repair without retraining**: the organism recovers from diverse injuries through its own maintenance dynamics, not through gradient descent during recovery
- **Competence preservation through body maintenance**: task performance survives injury because the body that supports it survives injury
- **Anatomical memory**: a hidden setpoint scaffold persists independently of expressed state and guides repair
- **Selective growth**: the system adds resources only when physiological need justifies the metabolic cost
- **Boundary-local adaptation**: interface changes produce localized reorganization, not global disruption

These properties are individually achievable through specialized ML techniques. What is novel is that they emerge together from a single coherent physiological substrate without benchmark-specific engineering.

### 7.2 Relationship to Levin's framework

MorphoBASE is a computational exploration of ideas developed in the biological context by Levin and colleagues. Several correspondences are intentional:

- **Multi-scale competency**: individual cells make local homeostatic decisions; coordination emerges through coupling, not central control
- **Bioelectric setpoints**: the Z-memory field serves as an anatomical target state analogous to bioelectric prepatterns
- **Gap junction-mediated coordination**: the conductance matrix modulates inter-cell coupling based on physiological agreement, analogous to gap junction gating
- **Repair as goal-directed behavior**: recovery is guided by stored anatomical memory, not by externally imposed correction

MorphoBASE is not a faithful simulation of biological tissue. It is a computational substrate that takes these organizational principles seriously and tests whether they produce measurable functional properties.

### 7.3 What this does not demonstrate

- This is not a competitive continual learning system. Specialized methods substantially outperform it on benchmarks.
- This is not a model of real biological tissue. The correspondences to biology are organizational, not mechanistic.
- This does not prove that organism-first design is superior to standard ML for practical applications.
- This does not demonstrate multi-scale competency in the full sense described by Levin — v1.3 has one scale of cellular organization, not nested hierarchies.

## 8. Future Directions (v1.4)

v1.4 extends the organism with:

- **Explicit shared genome**: a sparse signed regulatory matrix shared by all cells
- **Per-cell epigenome**: each cell gates the shared genome with its own accessibility mask, enabling specialization without per-cell dense matrices
- **Dual-track bioelectric state**: slow morphogenetic membrane + fast task-relay membrane, preventing fast dynamics from overwriting anatomical structure
- **Local homeostatic controller**: belief state, prediction error, precision, and frustration computation per cell
- **Voltage-gated junctions**: gap-junction openness dependent on voltage mismatch, stress mismatch, and setpoint agreement

These extensions aim to test four organism-level properties motivated by Levin's 2026 research:

1. **Goal retargetability**: can the organism adopt a changed target and reorganize?
2. **Boundary management**: can coordinated control be enlarged or contracted through junction manipulation?
3. **Memory reinterpretation**: can stored traces be reused after changed context?
4. **Cognitive light cone expansion**: does coordinated scope enlarge when coupling and memory support it?

## 9. Availability

The complete codebase, configuration files, assay definitions, and raw experimental artifacts are available at https://github.com/justin/morphobase. The system is implemented in Python with PyTorch and requires no specialized hardware.

## References

[1] Levin, M. (2019). The Computational Boundary of a "Self": Developmental Bioelectricity Drives Multicellularity and Scale-Free Cognition. *Frontiers in Psychology*, 10, 2688.

[2] Levin, M. (2022). Technological Approach to Mind Everywhere: An Experimentally-Grounded Framework for Understanding Diverse Bodies and Minds. *Frontiers in Systems Neuroscience*, 16.

[3] Levin, M. (2023). Darwin's agential materials: evolutionary implications of multiscale competency in developmental biology. *Cellular and Molecular Life Sciences*, 80, 142.

[4] Fields, C., & Levin, M. (2022). Competency in Navigating Arbitrary Spaces as an Invariant for Analyzing Cognition in Diverse Embodiments. *Entropy*, 24(6), 819.

[5] Levin, M. (2026). Self-Improvising Memory: A Perspective on Memories as Agential, Constructed, and Purposeful. *Preprint*.
