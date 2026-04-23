# MorphoBASE Build Checklist v1.3a
## A phased execution plan for fast iteration, hard gates, deep debugging, benchmark-agnostic organism development, and research-hardened plasticity/Z-field validation

**Author:** Justin M. Vetch  
**Aligned with:** MorphoBASE Master Architecture v1.3a  
**Purpose:** Turn the v1.3 master spec into a practical, smooth, low-waste build path  
**Key design rule:** Build the smallest organism that is still truly organismal

---

## 0. Why this checklist is different

The old phased plan tried to move too quickly from architectural ideas to benchmark training. That creates a predictable problem: when the system fails, you do not know whether the culprit is:

- the cell update rule,
- communication,
- plasticity gating,
- growth control,
- global field dynamics and Z-field health,
- the training loop,
- or the benchmark itself.

This v1.3a checklist keeps that structure and adds four hardening goals from recent research: explicit plasticity maintenance, cryptic phenotype / setpoint rewrite testing, perturbation-based light-cone measurement, and restored Z-field specificity. It therefore enforces the following principles:

1. **Substrate before benchmark.**  
   The organismal substrate must be debugged on tiny assays before being judged on standard continual-learning datasets.

2. **Fast assays before expensive assays.**  
   First use tasks that train in seconds to minutes and isolate one mechanism at a time.

3. **Instrumentation before interpretation.**  
   If you cannot see conductance, stress, stage transitions, energy use, structural churn, and recovery trajectories, you do not know what the organism is doing.

4. **Empirical gates, not narrative phases.**  
   Mechanisms may exist in code early, but a phase is only considered passed once its exit evidence is met.

---

## 1. Global Development Strategy

The build path is divided into three layers:

### Layer I — Substrate proving
Goal:
- verify that the organismal machinery actually works in isolation.

Typical assay properties:
- tiny,
- synthetic,
- deterministic or semi-deterministic,
- easy to visualize,
- fast enough for many runs per hour.

### Layer II — Sequential competence
Goal:
- show that the substrate can preserve and adapt competence over nonstationary task sequences.

Typical assay properties:
- still small,
- now includes forgetting/transfer pressure,
- introduces stress on plasticity, growth, and memory.

### Layer III — Regeneration and generality
Goal:
- show that the same substrate supports recovery, remapping, and more than one assay family.

Typical assay properties:
- interventional,
- cross-port,
- cross-domain,
- still staged carefully by cost.

---

## 2. Development Rules

These are mandatory process rules for the whole project.

### 2.1 Never add a major mechanism without a minimal assay
Every mechanism must come with:
- a unit test,
- a tiny behavioral assay,
- a visual diagnostic,
- and at least one ablation.

### 2.2 Every run must be classifiable
Every run should end in one of the following labels:
- pass,
- expected fail,
- unstable,
- degenerate lock,
- chronic growth,
- no-effect mechanism,
- instrumentation failure,
- benchmark bottleneck,
- or unknown.

### 2.3 Benchmarks are assay chambers
MNIST-like tasks are allowed early because they are fast and controllable.
They are not considered evidence of a general organism by themselves.

### 2.4 Expensive runs require prior survival on cheaper runs
No long sweep should start until the same config survives a smaller assay ladder.

### 2.5 Every phase has a red-team question
Before calling a phase complete, ask:
**What is the most boring non-organism explanation of this result?**
If that explanation is still plausible, the phase is not done.

---

## 3. Instrumentation First: Required Before Real Development

Before any substantive training work continues, build the following infrastructure.
This is Phase 0 in practice, even if some code already exists.

### 3.1 Unified logging schema
Every run must log:

**Core outcome metrics**
- task competence metric(s)
- average accuracy / return
- forgetting / BWT / FWT where applicable
- recovery success / time-to-recovery where applicable

**Organism state metrics**
- active cell count
- total edges / mean conductance / conductance entropy
- mean and variance of stress
- mean and variance of energy
- global-field magnitude / drift
- Z-field magnitude / drift / alignment
- dormant-cell fraction and active-unit diversity
- late-task learning slope
- light-cone probe metrics
- stage occupancy (seed, exploratory, differentiating, mature, dedifferentiating, prunable)
- organogenesis event count
- pruning event count
- immune intervention count
- dedifferentiation attempts / successes

**Efficiency metrics**
- wall time per step
- samples per second
- VRAM peak
- CPU RAM peak
- checkpoint size
- active parameters or effective capacity proxy

### 3.2 Required plots for every experiment
Automatically produce:
- competence curve
- forgetting / BWT curve
- active cell count over time
- growth and pruning event rate over time
- stage occupancy stacked plot
- stress heatmap snapshots
- conductance map snapshots
- global-field summary trace
- Z-field alignment / drift trace
- plasticity-health dashboard
- light-cone area or influence-radius trace
- energy usage trace
- recovery curve after intervention

### 3.3 Required run artifacts
Every run saves:
- config yaml/json
- git commit or version hash
- seed
- compact metrics json
- full event log
- best checkpoint
- last checkpoint
- summary markdown
- generated plots

### 3.4 Hard fail alerts
Abort or flag runs when:
- NaNs appear,
- all conductances collapse to zero or one,
- stress becomes constant everywhere,
- active cell count explodes above configured hard limit,
- active cell count falls below viability threshold,
- stage occupancy freezes early,
- dormant-cell fraction rises above configured ceiling,
- Z-field goes flat while task regime is changing,
- throughput drops below expected floor,
- or the benchmark dataloader is the bottleneck.

### 3.5 Run registry
Maintain a single CSV or parquet registry with one row per run, including:
- date,
- run name,
- assay family,
- phase,
- seed,
- ablations,
- key metrics,
- verdict,
- notes.

This becomes your memory and prevents rerunning old mistakes.

---

## 4. Assay Ladder: What to train on first

This is the most important practical section.

The old plan leaned too quickly toward canonical CL datasets. Those are useful later, but the fastest path now is a **mechanism ladder**. Each assay should isolate a small claim.

## Tier 0 — Zero-cost / almost-zero-cost tests

### T0.1 Deterministic unit tests
Goal:
- verify pure logic and invariants.

Examples:
- conductance bounds remain within allowed range
- stage transitions obey gating logic
- growth and pruning hysteresis respected
- immune interventions cannot fire in forbidden states
- field update rate is slower than fast state update
- dedifferentiation counterfactual gate blocks improper rollback

Training required:
- none.

Runtime target:
- milliseconds to seconds.

### T0.2 Rollout stability smoke test
Goal:
- ensure the organism can roll forward with no task and not blow up.

Conditions:
- random seed body,
- null input,
- low noise,
- no optimizer step.

Pass signs:
- no divergence,
- no uncontrolled growth,
- bounded state variables,
- no spontaneous pathological lock unless expected.

Runtime target:
- seconds.

---

## Tier 1 — Tiny substrate assays (seconds to a few minutes)

These should become your daily bread.

### T1.1 Identity maintenance assay
Goal:
- can a seeded body maintain a simple target state under mild perturbation?

Task:
- present a simple spatial or graph pattern,
- organism must preserve or reconstruct it.

Examples:
- binary blob maintenance,
- line restoration,
- small symbol reconstruction,
- one-hot field maintenance.

What it tests:
- stability,
- local repair,
- field bias,
- communication usefulness.

### T1.2 Denoising / wound closure assay
Goal:
- can the body repair local corruption without gradient steps during rollout?

Task:
- perturb a stable target pattern,
- organism restores it from partial corruption.

What it tests:
- counterfactual memory,
- regeneration,
- local coordination.

### T1.3 Role differentiation assay
Goal:
- do stable role clusters emerge under a simple repeated demand?

Task:
- repeated input-output mapping with spatial bottlenecks,
- or local relay tasks where some cells become bridges.

What it tests:
- differentiation,
- route specialization,
- tissue formation.

### T1.4 Stress-sharing assay
Goal:
- does localized mismatch propagate useful recruitment rather than global panic?

Task:
- inject failure into one region,
- measure stress spread radius and support recruitment.

What it tests:
- stress diffusion,
- non-uniform communication,
- repair recruitment.

### T1.5 Growth usefulness assay
Goal:
- does growth occur only when needed, and does it improve the objective?

Task:
- compare identical runs with growth enabled vs disabled under a local bottleneck.

What it tests:
- organogenesis necessity,
- avoidance of decorative growth.

### T1.6 Channel-block compensation assay
Goal:
- can the body preserve safe function when one signaling channel or message family is disabled?

Task:
- block one message family or state dimension,
- require maintenance of a simple objective,
- measure whether remaining pathways compensate.

What it tests:
- homeostatic compensation,
- channel redundancy without centralization,
- resilience of physiology rather than just outputs.

Recommendation:
These Tier 1 assays should be the first place you spend real time. If the organism cannot pass these quickly, there is no reason to move to CL benchmarks yet.

---

## Tier 2 — Tiny sequential assays (minutes)

These introduce memory and plasticity pressure without the baggage of full image classification.

### T2.1 Sequential rule-switch toy tasks
Goal:
- test task changes on tiny synthetic inputs.

Examples:
- binary vector parity then majority
- XOR then XNOR
- threshold task then inverted threshold
- simple sequence transform A then B

What it tests:
- sequential adaptation,
- local plasticity gating,
- retention under small shifts.

### T2.2 Port remap toy assay
Goal:
- keep body substrate fixed while changing a small boundary mapping.

Task:
- same core problem, but input or output channel assignment changes.

What it tests:
- whether adaptation is localized near ports or destabilizes whole body.

### T2.3 Sequential pattern families
Goal:
- preserve competence across a sequence of small reconstruction tasks.

Examples:
- reconstruct symbol family 1, then 2, then 3,
- each with limited overlap.

What it tests:
- morphological memory,
- transfer,
- selective change.

### T2.4 Injury during sequential learning
Goal:
- test whether the body can keep learning while damaged.

Task:
- switch toy tasks while occasionally masking cells or severing edges.

What it tests:
- persistence under perturbation,
- not just recovery after training.

### T2.5 Plasticity-loss stress test
Goal:
- detect whether the system is becoming stable by losing the ability to learn.

Task:
- run a longer toy sequence,
- track late-task learning slope, dormant-cell fraction, diversity proxies, and response to refresh interventions.

What it tests:
- plasticity maintenance,
- pseudo-maturity versus healthy morphostasis,
- whether renewal mechanisms are truly needed.

---

## Tier 3 — Fast canonical vision CL assays (minutes to low hours)

These are useful, but only now.

### T3.1 Split-MNIST
Use it as:
- a fast sanity benchmark,
- a first conventional CL bridge,
- a way to compare against simple baselines.

Do not use it as:
- evidence of generality,
- or the main architecture driver.

### T3.2 Permuted-MNIST
Use it as:
- a controlled distribution-shift and task-sequence stressor.

Caution:
- it may over-reward low-level adaptation tricks and under-test genuine organismal structure.

### T3.3 Fashion-MNIST variants or KMNIST
Use them as:
- slightly richer but still cheap visual assay families.

Recommendation:
Only move here once Tier 2 is producing interpretable behavior.

---

## Tier 4 — Slightly harder but still practical assays

### T4.1 Split-FashionMNIST / KMNIST / EMNIST
Goal:
- test whether organismal behavior survives slightly richer input diversity.

### T4.2 Small control tasks
Examples:
- CartPole with observation remaps,
- simple gridworld with shifted reward zones,
- tiny continuous control with regime shifts.

Goal:
- force non-visual organism competence.

### T4.3 Small reasoning or algorithmic tasks
Examples:
- short symbol transformation sequences,
- mini RuleWorld-style tasks,
- simple grammar remaps.

Goal:
- prove the architecture is not merely a spatial image substrate.

This tier is the earliest point where you can start claiming the organism is becoming general rather than visual.

---

## Tier 5 — Regeneration and transfer proof suite

### T5.1 Lesion battery
- random cell ablation
- targeted organ/tissue ablation
- conductance severance
- parameter corruption
- field corruption
- Z-field corruption
- port disruption

### T5.2 Recovery evaluation
Measure:
- probability of recovery,
- steps to recover,
- energy cost,
- whether recovery preserves prior tasks.

### T5.3 Cryptic phenotype / setpoint rewrite
- apply a brief perturbation to conductance, stress sharing, or Z-field bias,
- return to nominal conditions,
- later lesion the body,
- test whether recovery follows a rewritten attractor.

### T5.4 Light-cone probe suite
- perturb a local cell or region,
- measure downstream effect radius, duration, and port impact,
- compare with stress-sharing, conductance, and Z-field ablations.

### T5.5 Cross-port transfer
Same substrate, different boundary ports.
This is one of the most important long-term proof points.

---

## 5. Revised Build Phases

The build phases below align to v1.3 but are made more practical.

## Phase 0 — Tooling, Observability, and Determinism

**Goal:** Make the system inspectable and reproducible before serious iteration.

### Entry
- codebase exists.

### Exit criteria
- all required logs and plots generate automatically
- deterministic replay works within tolerance for fixed seed/config
- run registry operational
- one-command tiny experiment execution works end-to-end
- hard fail alerts catch obvious numerical/pathology failures

### Tasks
- [ ] Create a unified run config schema
- [ ] Create a unified metrics/event logger
- [ ] Add run summary generator
- [ ] Add run registry CSV/parquet updater
- [ ] Add deterministic seed handling across Python / NumPy / Torch / CUDA
- [ ] Add throughput, wall-time, VRAM, RAM logging
- [ ] Add automatic plot generation
- [ ] Add compact snapshot export for stress / conductance / stage state
- [ ] Add hard fail alert system
- [ ] Build a one-command `tiny_assay_smoke.py`

### Debugging requirements
- [ ] Verify repeated same-seed runs are materially similar
- [ ] Verify different-seed runs remain within expected variance band
- [ ] Verify plotting itself never crashes a run

---

## Phase 1 — Viable Cell Substrate

**Goal:** Prove the cell update rule and local physiology are stable.

### Assays
- T0.2 rollout stability smoke test
- T1.1 identity maintenance

### Exit criteria
- seeded body remains viable across long no-task rollouts
- state variables remain bounded
- local plasticity gate varies meaningfully across conditions
- simple target maintenance works better than a no-communication or no-field baseline

### Tasks
- [ ] Verify cell state update ordering
- [ ] Verify membrane/excitability dynamics are bounded
- [ ] Verify plasticity gate responds to mismatch / energy / maturity inputs
- [ ] Add local state histograms and drift diagnostics
- [ ] Implement identity maintenance assay
- [ ] Run no-field, no-communication, no-plasticity ablations

### Required plots
- hidden-state norm trace
- membrane variable trace
- plasticity gate trace
- viability / alive count trace

### Fast sweep parameters
- body size
- local update step size
- state noise
- membrane decay / hysteresis

---

## Phase 2 — Adaptive Communication

**Goal:** Show that communication becomes differentiated and useful.

### Assays
- T1.3 role differentiation
- T1.4 stress-sharing

### Exit criteria
- conductance maps are non-uniform and reproducibly task-sensitive
- localized mismatch recruits help without whole-body panic
- communication ablation hurts performance on relevant assays
- routing changes are causally useful, not merely decorative

### Tasks
- [ ] Add conductance entropy and edge-utilization metrics
- [ ] Add stress spread radius metric
- [ ] Add causal edge ablation probe
- [ ] Visualize conductance over time
- [ ] Visualize stress wave propagation
- [ ] Test static-routing baseline vs adaptive routing
- [ ] Test no-stress-sharing ablation

### Debugging questions
- Does conductance saturate at 0 or 1?
- Is communication change correlated with useful error reduction?
- Does stress spread too far, too fast, or not at all?

### Fast sweep parameters
- conductance update gain
- stress diffusion coefficient
- communication sparsity penalty
- role compatibility term

---

## Phase 3 — Controlled Growth and Structural Usefulness

**Goal:** Prove that growth and pruning are triggered selectively and improve competence.

### Assays
- T1.5 growth usefulness
- T2.1 sequential rule-switch toy tasks

### Exit criteria
- growth occurs primarily in bottleneck regions
- growth-enabled runs outperform matched no-growth runs when capacity is needed
- pruning removes low-utility tissue without catastrophic collapse
- hysteresis prevents growth-prune chatter

### Tasks
- [ ] Add growth event cause tagging
- [ ] Add pruning reason tagging
- [ ] Add structural churn metric
- [ ] Add utility / saliency / uniqueness score logging for pruned cells
- [ ] Implement matched growth-on vs growth-off experiments
- [ ] Implement refractory/hysteresis verification tests
- [ ] Add max-body hard cap enforcement

### Debugging questions
- Is the organism growing because it needs capacity or because the gate is too permissive?
- Are pruned cells actually redundant?
- Does growth persist after the mismatch is solved?

### Fast sweep parameters
- growth threshold
- growth cooldown
- pruning threshold
- Z-protection weight
- energy cost of new cells

---

## Phase 4 — Maturation, Morphostasis, and Plasticity Health

**Goal:** Reach a regime where useful structures stabilize.

### Assays
- T2.3 sequential pattern families
- T3.1 Split-MNIST only after Tier 2 success

### Exit criteria
- mature tissue fraction becomes non-zero and sustained
- chronic late growth decreases sharply
- active cell count stabilizes after initial adaptation period
- competence remains stable over longer rollouts
- pseudo-maturity diagnostics are in place and negative
- late-task learning slope remains above configured floor
- Z-field is stable enough to guide recovery but not flatlined

### Tasks
- [ ] Rework stage definitions into explicitly logged states
- [ ] Add stage transition matrix logging
- [ ] Add pseudo-maturity detector
- [ ] Add plasticity-health dashboard
- [ ] Add dormant-cell and diversity diagnostics
- [ ] Add morphostasis score combining low churn + sustained competence
- [ ] Test maturation with and without global field consolidation
- [ ] Verify dedifferentiation counterfactual gate on synthetic cases

### Debugging questions
- Is the body stuck in pre-mature limbo?
- Is maturity suppressing useful plasticity too early?
- Is lock score rising while competence stays poor?
- Is the system stable because it is healthy, or because it can no longer learn?

### Fast sweep parameters
- maturation thresholds
- commitment gain
- dedifferentiation difficulty
- global field update rate
- maturity-protected plasticity floor

---

## Phase 5 — Sequential Continual Learning Bridge

**Goal:** Demonstrate continual-learning competence on fast canonical datasets without letting them define the architecture.

### Assays
- T2.5 plasticity-loss stress test
- T3.1 Split-MNIST
- T3.2 Permuted-MNIST
- optionally Fashion-MNIST or KMNIST variants

### Exit criteria
- beats naive baseline convincingly
- forgetting and/or BWT improve over earlier versions
- primary metrics and biomarkers tell a consistent story
- performance is stable across seeds, not one lucky run
- plasticity-health metrics do not show late-stage collapse

### Tasks
- [ ] Standardize baseline suite (naive MLP, fixed NCA, no-growth organism, no-field organism)
- [ ] Create seed-robust evaluation protocol
- [ ] Build sweep score using primary metrics first, biomarkers second
- [ ] Add benchmark-specific throughput comparison
- [ ] Add checkpoint resume and compare stability across resumed runs

### Important rule
No sweep ranker may place lock/coherence above competence retention and recovery.

### Debugging questions
- Is the organism actually learning or just using a brittle shortcut?
- Are canonical CL gains coming from the biological mechanisms or from the port/encoder?

---

## Phase 6 — Non-Visual Generality

**Goal:** Show the organism is not an MNIST machine.

### Assays
- T4.2 small control tasks
- T4.3 small reasoning / algorithmic tasks
- T2.2 port remap toy assay

### Exit criteria
- the same substrate principles work in at least one non-visual family
- adaptation can be partially localized near ports when appropriate
- internal organismal diagnostics remain meaningful outside visual tasks

### Tasks
- [ ] Implement explicit boundary-port abstraction in code
- [ ] Create at least one non-visual port
- [ ] Create port remap assay
- [ ] Compare whole-body disruption vs local-port adaptation
- [ ] Evaluate cross-domain transfer of organism-level biomarkers

### Debugging questions
- Did the body generalize, or did you just build another head?
- Are the same internal mechanisms active across domains?

---

## Phase 7 — Regeneration, Setpoint Rewrite, and Interventional Validation

**Goal:** Earn the organismal claims.

### Assays
- T5.1 lesion battery
- T5.2 recovery evaluation
- T5.3 cryptic phenotype / setpoint rewrite
- T5.4 light-cone probe suite
- cross-port disruption and remapping

### Exit criteria
- recovery occurs after multiple injury classes with bounded cost
- recovery does not rely entirely on retraining
- regeneration preserves some prior competence
- morphological memory claim has direct evidence
- Z-field lesion or transient bias produces interpretable changes in recovery
- at least one cryptic-phenotype or setpoint-rewrite result is observed or convincingly falsified

### Tasks
- [ ] Implement lesion battery runner
- [ ] Add lesion localization controls
- [ ] Add no-gradient recovery mode
- [ ] Add Z-field corruption / bias operators
- [ ] Add cryptic phenotype protocol runner
- [ ] Add perturbation-based light-cone probe runner
- [ ] Measure time-to-recovery and energy-to-recovery
- [ ] Add recovery-vs-retraining comparison
- [ ] Test repeated injury / re-amputation cases

### Debugging questions
- Is recovery really organismal repair, or is it just fast retraining?
- Does the field actually help resolve damage ambiguity?

---

## Phase 8 — Open-Ended Extensions

Do not touch this until Phase 7 is credible.

Possible work:
- organ nesting
- oscillatory coordination
- immune pathology games
- ecosystem / multiple organisms
- richer active-inference formalization
- metamorphosis / life-stage transitions

---

## 6. Debugging Doctrine

This should be followed every week.

### 6.1 One new mechanism at a time
Do not simultaneously add:
- new stage logic,
- new loss terms,
- new growth gates,
- and new benchmark ports.

That destroys interpretability.

### 6.2 Ablate immediately
Whenever a mechanism is added, create:
- mechanism on
- mechanism off
- mechanism weak
- mechanism strong

### 6.3 Prefer paired experiments
Compare matched runs that differ by one thing only.

### 6.4 Use tiny runs to debug long-run failures
If a long run fails mysteriously, reproduce the failure on the smallest possible assay.

### 6.5 Freeze the benchmark when debugging the substrate
When debugging communication or growth, do not keep changing the dataset.

### 6.6 Red-team plasticity health explicitly
Whenever morphostasis improves, ask whether it improved because the organism matured or because it lost the ability to change.

---

## 7. Sweep Strategy

### 7.1 Sweep order
Always sweep in this order:
1. numerical stability parameters
2. communication parameters
3. growth/pruning parameters
4. maturation parameters
5. benchmark-specific training parameters

### 7.2 Sweep sizes
Use three sweep scales:
- micro: 4–8 runs, seconds/minutes, same day decision
- meso: 8–24 runs, minutes/hour-scale, weekly direction
- macro: only after a config has already passed smaller ladders

### 7.3 Sweep score hierarchy
Rank by:
1. primary outcome metrics
2. recovery / retention
3. morphostasis
4. efficiency
5. biomarkers

### 7.4 Keep benchmark difficulty behind substrate confidence
Do not scale benchmark difficulty just because the sweeps are easy to launch.

---

## 8. Performance Tracking and Speed Optimization

### 8.1 Targets for iteration velocity
Aim for:
- Tier 0–1 assays: seconds to 2 minutes
- Tier 2 assays: under 5 minutes
- Tier 3 fast benchmark sanity runs: under 20–30 minutes
- only a small number of runs should exceed 1 hour

### 8.2 Practical speed levers
- tiny body sizes first
- tiny latent/state widths first
- limited rollouts first
- reduced logging during inner-loop sweeps, full logging only for finalists
- persistent dataloaders
- mixed precision where safe
- batched small assays
- checkpoint only key states on micro sweeps

### 8.3 Never optimize speed blindly
Track whether speed-ups hide mechanism behavior.
A fast run that removes needed observability is usually not worth it.

---

## 9. Concrete Suggested Starting Task Stack

If I were running this next week, I would use this order.

### Stack A — immediate substrate proving
1. rollout stability smoke test
2. identity maintenance
3. denoising / wound closure
4. stress-sharing localized failure
5. growth usefulness under bottleneck

### Stack B — first sequential pressure
6. sequential rule-switch toy tasks
7. sequential pattern families
8. injury during sequential toy learning

### Stack C — canonical CL bridge
9. Split-MNIST
10. Permuted-MNIST
11. Fashion-MNIST or KMNIST variant

### Stack D — generality bridge
12. simple gridworld or CartPole remap
13. small symbol transformation / RuleWorld-mini
14. explicit port remap assay

That is the fastest path to meaningful evidence with minimal wasted compute.

---

## 10. Red Flags That Should Pause Development

Stop adding features and debug if any of these happen:

- biomarkers improve while competence flatlines
- Z-field becomes flat or meaningless while tasks are still changing
- late-task learning slope collapses even as stability metrics improve
- stage occupancy collapses into one state for most runs
- growth remains high late in training
- recovery only works with gradients on
- different seeds produce completely different qualitative regimes
- communication maps look rich but edge ablations do nothing
- performance gains vanish when port assumptions change
- throughput is too slow to support multiple runs per day

---

## 11. Deliverables by Milestone

### Milestone M1
Tooling done, tiny assays automated.

### Milestone M2
Stable cell substrate and useful communication on Tier 1 assays.

### Milestone M3
Selective growth and morphostasis visible on Tier 2 assays.

### Milestone M4
Canonical CL bridge works on Split-MNIST without misleading sweep metrics.

### Milestone M5
At least one non-visual port/assay succeeds.

### Milestone M6
Regeneration, cryptic phenotype, and Z-field claims supported by lesion battery plus setpoint-rewrite assays.

Only after M6 should you heavily expand frontier biology.

---

## 12. The Core Mindset

This checklist is designed to keep you from making two classic mistakes:

1. building a biologically decorated benchmark model, or
2. building a beautiful organism simulator that never becomes competent.

The right path is narrower:
- tiny fast assays,
- deep observability,
- hard empirical gates,
- and progressive broadening of assay families.

If a mechanism matters, it should show up first on the cheap assays.
If it only “matters” on a giant late benchmark, you probably do not understand it yet.

That is the v1.3 build philosophy.
