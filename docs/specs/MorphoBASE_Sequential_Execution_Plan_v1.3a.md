# MorphoBASE Sequential Execution Plan v1.3a
## Fresh-code, organism-first implementation order aligned to the research-hardened v1.3a master architecture

**Author:** Justin M. Vetch  
**Purpose:** A strict, step-by-step build order for implementing MorphoBASE from scratch while staying faithful to the v1.3a master architecture and the v1.3a build checklist.  
**Key rule:** The body comes first. Benchmarks come later.  
**Secondary rule:** No benchmark-specific assumptions are allowed to leak into core organism code.  
**New v1.3a emphasis:** Plasticity maintenance, Z-field specificity, cryptic phenotype testing, perturbation-based light-cone measurement, and channel-block compensation are mandatory, not optional garnish.

---

## 0. What changed in v1.3a

Relative to the earlier execution plan, v1.3a makes five changes:

1. **The Z-field is restored as a named, load-bearing mechanism.**  
   It is not merely a generic global field. It is the slow setpoint scaffold for counterfactual recovery and durable setpoint rewrite.

2. **Plasticity maintenance becomes explicit.**  
   The codebase must detect and respond to plasticity collapse instead of mistaking it for maturity.

3. **Cryptic phenotype / setpoint rewrite assays become first-class gates.**  
   Ordinary lesion recovery is no longer enough for strong morphological-memory claims.

4. **Light-cone language must become perturbation-based measurement.**  
   Influence radius, duration, and causal reach are now measured, not just discussed.

5. **Compensation after channel loss is required.**  
   The organism should preserve safe function when one signaling mode is blocked.

---

## 1. Non-Negotiable Execution Rules

### Rule 1 — Fresh code means fresh code
Do not copy old modules into the new repo unless they are rewritten deliberately.
Legacy code may be referenced for ideas only.

### Rule 2 — One source of truth
There must be one authoritative config path, one state schema, one logging schema, and one experiment runner.

### Rule 3 — Interfaces before internals
Before implementing a mechanism, define:
- what state it reads,
- what state it writes,
- what clock it runs on,
- what diagnostics prove it is active,
- and what perturbation proves it matters.

### Rule 4 — No benchmark leakage into organism core
The organism package must not import MNIST datasets, image encoders, classifier heads, or benchmark-specific losses.
All world contact happens through explicit port modules.

### Rule 5 — Every mechanism gets four things immediately
A mechanism is not “added” until it has:
- implementation,
- unit tests,
- visualization,
- and at least one ablation.

### Rule 6 — Tiny assays are primary until they are boring
If a mechanism cannot prove itself on a cheap assay, do not move it to an expensive one.

### Rule 7 — Sequential means sequential
You do not start Step N+1 until Step N exit criteria are met.
If a later idea becomes tempting, put it in a deferred notes file.

---

## 2. Target Repo Layout

```text
morphobase_v13a/
  README.md
  pyproject.toml
  configs/
    defaults.yaml
    assay/
    sweep/
  docs/
    debugging_playbook.md
    experiment_log.md
    research_notes/
  scripts/
    run_assay.py
    run_sweep.py
    export_summary.py
  morphobase/
    __init__.py
    types.py
    clocks.py
    seeds.py
    registry.py
    config/
      schema.py
      validate.py
    organism/
      state.py
      body.py
      scheduler.py
      snapshot.py
    cells/
      femo.py
      genome.py
      local_model.py
    communication/
      conductance.py
      stress_sharing.py
      fields.py
      z_field.py
    development/
      growth.py
      pruning.py
      maturation.py
      dedifferentiation.py
      renewal.py
    metabolism/
      energy.py
      budgets.py
    pathology/
      audit.py
      lesions.py
    ports/
      base.py
      toy_pattern_port.py
      toy_rule_port.py
      mnist_port.py
      control_port.py
    training/
      losses.py
      trainer.py
      interventions.py
    assays/
      smoke.py
      identity.py
      wound_closure.py
      stress_recruitment.py
      growth_usefulness.py
      compensation_block.py
      sequential_rules.py
      plasticity_stress.py
      setpoint_rewrite.py
      lightcone.py
      lesion_battery.py
      port_remap.py
    diagnostics/
      logger.py
      metrics.py
      plots.py
      alerts.py
      summaries.py
  tests/
    unit/
    integration/
    regression/
```

---

## 3. Strict Build Order

## Step 1 — Initialize repo and lock the development contract

**Goal:** Stand up the repo skeleton, config system, and execution conventions before any organism logic exists.

Create first:
- `README.md`
- `pyproject.toml`
- `morphobase/types.py`
- `morphobase/config/schema.py`
- `morphobase/config/validate.py`
- `morphobase/seeds.py`
- `morphobase/clocks.py`
- `morphobase/registry.py`
- `scripts/run_assay.py`

Exit criteria:
- one config file can instantiate a no-op run,
- same seed produces identical stub output,
- run artifacts land in the correct folder,
- registry row writes successfully.

## Step 2 — Build diagnostics before the organism

**Goal:** Make the future system observable before it exists.

Build now:
- `diagnostics/logger.py`
- `diagnostics/metrics.py`
- `diagnostics/plots.py`
- `diagnostics/alerts.py`
- `diagnostics/summaries.py`
- `organism/snapshot.py`

Must-support outputs:
- scalar metrics logging,
- event logging,
- snapshot serialization,
- stage occupancy plot,
- active cell count plot,
- stress summary plot,
- conductance summary plot,
- energy trace plot,
- Z-field alignment / drift trace,
- plasticity-health dashboard,
- light-cone trace.

Exit criteria:
- diagnostic stack works on synthetic data,
- plots and summaries render on degenerate inputs,
- run summary export works end to end.

## Step 3 — Define canonical organism state and update clocks

**Goal:** Create the one true organism-state schema before implementing any mechanism.

Build now:
- `organism/state.py`
- `organism/scheduler.py`
- `organism/body.py`
- `cells/femo.py` (state definition only)

State schema must include:
- hidden state,
- membrane/excitability state,
- plasticity gate,
- energy/budget state,
- stress state,
- role embedding/logits,
- commitment/maturity,
- local field alignment,
- Z-field alignment,
- viability/aliveness,
- edge state,
- stage label.

Exit criteria:
- body object can step through clocks with dummy mechanisms,
- snapshots serialize cleanly,
- stage labels and counters advance correctly.

## Step 4 — Define the boundary-port abstraction before any task code

**Goal:** Prevent benchmark assumptions from contaminating the organism.

Build now:
- `ports/base.py`
- `ports/toy_pattern_port.py`
- `ports/toy_rule_port.py`

Base port contract:
- `encode(external_input) -> boundary_signals`
- `decode(boundary_state) -> external_output`
- `loss_fn(external_output, target)`
- `remap(mapping_spec)`
- `damage(mask_spec)`

Exit criteria:
- a dummy body can receive toy port input and emit dummy output,
- ports can be swapped without changing organism code.

## Step 5 — Implement the minimal viable cell substrate

**Goal:** Create a stable cell that can update without growth and without training.

Build now:
- `cells/genome.py`
- `cells/local_model.py`
- `cells/femo.py`
- `metabolism/energy.py`

Implement now:
- hidden-state update,
- membrane/excitability update,
- local mismatch estimate,
- local plasticity gate,
- energy consumption per update,
- optional local goal-state interface.

Exit criteria:
- long no-task rollouts stay bounded,
- cell update remains stable across seeds and noise levels,
- energy remains interpretable.

## Step 6 — Build the first no-learning assays

**Goal:** Prove the substrate can exist and preserve a simple target before introducing adaptation pressure.

Build now:
- `assays/smoke.py`
- `assays/identity.py`
- `assays/wound_closure.py`

Exit criteria:
- all three assays run in under 2 minutes,
- body outperforms trivial no-dynamics baseline on identity maintenance,
- wound closure works at least partially without training steps during rollout.

## Step 7 — Add adaptive communication and stress sharing

**Goal:** Make communication physiological and useful.

Build now:
- `communication/conductance.py`
- `communication/stress_sharing.py`
- `communication/fields.py` (local and tissue fields only)

Immediate assays:
- stress recruitment,
- role differentiation,
- conductance ablation.

Exit criteria:
- conductance is non-uniform and task-sensitive,
- stress spreads locally before globally,
- disabling communication measurably hurts relevant assays.

## Step 8 — Add the Z-field explicitly

**Goal:** Introduce the named slow setpoint scaffold before growth and before strong memory claims.

Build now:
- `communication/z_field.py`
- Z-field metrics in `diagnostics/metrics.py`
- Z-field plots in `diagnostics/plots.py`

Implement now:
- Z-field state,
- Z-field update rule slower than local state and usually slower than plasticity,
- local-to-Z alignment measure,
- Z-field corruption / bias operator,
- Z-field archive interface for counterfactual recovery.

Assays to rerun:
- identity maintenance,
- wound closure,
- stress recruitment,
- Z-field corruption / restoration.

Exit criteria:
- Z-field changes more slowly than local state,
- Z-field improves recovery or stability on at least one tiny assay,
- Z-field corruption produces interpretable degradation,
- Z-field is not flat and not overbearing.

## Step 9 — Add growth, pruning, and hard structural controls

**Goal:** Make structural change possible, selective, and expensive.

Build now:
- `development/growth.py`
- `development/pruning.py`
- `metabolism/budgets.py`

Immediate assays:
- growth usefulness,
- bottleneck relief,
- prune-redundancy.

Exit criteria:
- growth happens in meaningful regions,
- growth-on beats growth-off only when capacity is needed,
- pruning removes redundancy without collapse,
- no growth-prune chatter.

## Step 10 — Add maturation, commitment, and controlled dedifferentiation

**Goal:** Move from raw growth to morphostasis without plasticity collapse.

Build now:
- `development/maturation.py`
- `development/dedifferentiation.py`

Immediate assays:
- sequential pattern-family toy assay,
- pseudo-maturity detection assay,
- synthetic repair-vs-dediff cases.

Exit criteria:
- mature states become non-zero and sustained,
- body size stabilizes after initial adaptation,
- late growth drops substantially,
- dedifferentiation is rare and interpretable.

## Step 11 — Add explicit plasticity maintenance and renewal

**Goal:** Prevent the organism from achieving stability by becoming unable to learn.

Build now:
- `development/renewal.py`
- plasticity-health metrics in `diagnostics/metrics.py`

Implement now:
- dormant-cell detection,
- active-unit diversity proxy,
- selective refresh or reseeding of underused cells,
- budgeted renewal intervention,
- optional low-amplitude diversity injection.

Immediate assays:
- `assays/plasticity_stress.py`
- late-task learning slope probe,
- refresh-on vs refresh-off paired experiments.

Exit criteria:
- late-task learning slope stays above floor on toy sequential runs,
- dormant-cell fraction remains controlled,
- renewal interventions improve learning without wrecking stability.

## Step 12 — Add the compensation / channel-block assay

**Goal:** Prove the organism can preserve safe function when one message family is blocked.

Build now:
- `assays/compensation_block.py`

Protocol:
- disable one signaling channel or message family,
- require maintenance of a simple target,
- measure whether remaining pathways compensate.

Exit criteria:
- predictable degradation without total collapse,
- at least partial compensation through remaining pathways,
- compensation is visible in conductance, stress, or Z-field adaptation traces.

## Step 13 — Add the audit / pathology layer

**Goal:** Detect and suppress pathological local behavior without turning the whole system into a centralized controller.

Build now:
- `pathology/audit.py`
- `pathology/lesions.py`

Exit criteria:
- pathology events can be induced on toy setups,
- audit catches them at useful rates,
- audit interventions are less harmful than the pathology itself.

## Step 14 — Add the minimal training loop only after the substrate is alive

**Goal:** Introduce learning in the least invasive way possible.

Build now:
- `training/losses.py`
- `training/trainer.py`
- `training/interventions.py`

First training tasks:
- tiny sequential rule-switch tasks,
- tiny pattern-family sequence tasks,
- injury-during-learning toy tasks.

Exit criteria:
- organism can adapt across tiny sequential tasks without immediate collapse,
- learning signals do not destroy morphostasis or plasticity-health diagnostics,
- runs remain fast enough for many iterations per day.

## Step 15 — Add the cryptic phenotype / setpoint rewrite assay before MNIST

**Goal:** Test whether a brief intervention can durably rewrite later recovery tendencies.

Build now:
- `assays/setpoint_rewrite.py`

Protocol:
- apply transient bias to conductance, stress, or Z-field,
- return to nominal conditions,
- later lesion the body,
- measure whether recovery converges to original or rewritten attractor.

Exit criteria:
- either a genuine setpoint rewrite is observed or a strong falsification case is documented,
- baseline-looking states can still reveal altered recovery tendencies after later lesions,
- Z-field instrumentation makes the result interpretable.

## Step 16 — Add perturbation-based light-cone probes

**Goal:** Turn “cognitive light cone” into a real engineering metric.

Build now:
- `assays/lightcone.py`

Protocol:
- perturb one cell or region at time \(t_0\),
- measure downstream effect radius, duration, and port-output impact,
- compare under communication, stress, and Z-field ablations.

Exit criteria:
- influence radius and duration are measurable,
- stress sharing expands or reshapes the cone in a consistent way,
- Z-field and conductance ablations produce interpretable cone changes.

## Step 17 — Add the first benchmark bridge: MNIST as a port, not as ontology

**Goal:** Use MNIST only as a conventional bridge assay once the substrate has earned it.

Build now:
- `ports/mnist_port.py`

Benchmark order:
1. tiny MNIST subset sanity run
2. Split-MNIST
3. only after success, Permuted-MNIST

Exit criteria:
- same body substrate proven on toy assays runs the benchmark,
- primary metrics dominate sweep ranking,
- biomarkers support but do not replace competence metrics,
- plasticity-health metrics do not collapse late in the run.

## Step 18 — Add a non-visual bridge before making any “general organism” claim

**Goal:** Prove the architecture is not just a spatial image body.

Build now:
- `ports/control_port.py` or minimal symbolic port
- `assays/port_remap.py`

Exit criteria:
- same core body supports a non-visual port,
- port remapping changes behavior mostly near the boundary when appropriate,
- internal diagnostics remain meaningful across domains.

## Step 19 — Add the lesion and regeneration battery

**Goal:** Earn the organismal claims with interventional validation.

Build now:
- `assays/lesion_battery.py`
- extend `pathology/lesions.py`

Lesions to support:
- cell masking,
- cell deletion,
- edge severance,
- parameter corruption,
- field corruption,
- Z-field corruption,
- port damage,
- repeated injury.

Exit criteria:
- some recovery works without full retraining,
- recovery preserves part of prior competence,
- multiple injury classes are tolerated,
- Z-field lesion/bias changes recovery in interpretable ways.

## Step 20 — Only now consider frontier enrichments

Possible later candidates:
- richer tissue fields,
- oscillatory coupling,
- reaction-diffusion regionalization,
- stigmergic highways,
- stronger predictive coding local learners,
- ecological multi-organism work.

Hard rule:
No frontier mechanism enters the main branch until baseline regeneration, setpoint rewrite, and one non-visual bridge are credible.

---

## 4. First 20 Concrete Build Actions

1. Create new repo and folder layout.
2. Add `pyproject.toml`, formatter, tests, linting.
3. Create config schema and validation.
4. Create seed utilities and run registry.
5. Create run-assay script scaffold.
6. Build logger and summary exporter.
7. Build plotting stack on fake data.
8. Define organism state dataclasses/tensors.
9. Define fast/medium/slow scheduler.
10. Define base port interface.
11. Implement toy pattern port.
12. Implement toy rule port.
13. Implement minimal cell state update.
14. Implement energy accounting.
15. Run no-task bounded-rollout smoke test.
16. Implement identity maintenance assay.
17. Implement wound closure assay.
18. Implement dynamic conductance module.
19. Implement stress-sharing module.
20. Implement explicit Z-field state and diagnostics.

Only after those 20 are working should you proceed to growth.

---

## 5. Second 20 Concrete Build Actions

21. Add Z-field corruption / bias operator.
22. Re-run repair assays with and without Z-field.
23. Implement growth trigger and spawn logic.
24. Implement new-cell initialization.
25. Implement pruning score and pruning action.
26. Add structural churn metrics.
27. Run growth-usefulness assay.
28. Add refractory / hysteresis logic.
29. Add hard max-body cap.
30. Implement maturation rules.
31. Implement commitment state and transitions.
32. Implement controlled dedifferentiation eligibility.
33. Add pseudo-maturity detector.
34. Implement renewal / reseeding hooks.
35. Add dormant-cell and diversity diagnostics.
36. Run plasticity-loss stress test on toy sequences.
37. Implement channel-block compensation assay.
38. Implement audit / pathology layer.
39. Add tiny sequential rule-switch training.
40. Re-run all earlier assays with learning enabled.

Only after those 40 are working should you proceed to MNIST.

---

## 6. Third 20 Concrete Build Actions

41. Implement setpoint rewrite / cryptic phenotype assay.
42. Implement perturbation-based light-cone assay.
43. Implement MNIST port.
44. Run tiny MNIST subset sanity check.
45. Run Split-MNIST with minimal sweep.
46. Build primary-metric-first sweep ranking.
47. Run seed robustness check on Split-MNIST.
48. Run matched no-growth / no-stress / no-Z-field ablations.
49. Implement non-visual port.
50. Run first non-visual toy bridge.
51. Implement port-remap assay.
52. Run port-remap adaptation tests.
53. Implement lesion battery runner.
54. Add no-gradient recovery mode.
55. Run cell-masking recovery.
56. Run parameter-corruption recovery.
57. Run edge-severance recovery.
58. Run Z-field-corruption recovery.
59. Compare recovery vs fast retraining.
60. Freeze baseline and only then explore frontier enrichments.

---

## 7. What to defer, even if it is tempting

Do not implement these early:
- CIFAR or large visual datasets,
- language tasks,
- multi-organism ecology,
- full oscillatory coupling,
- reaction-diffusion patterning,
- rich immune politics,
- MoE hybrids,
- full predictive-coding replacement of the training loop.

---

## 8. Recommended first task stack

1. rollout stability smoke test
2. identity maintenance
3. wound closure / denoising
4. stress recruitment
5. Z-field corruption / restoration
6. growth usefulness under bottleneck
7. sequential rule-switch toy tasks
8. sequential pattern-family tasks
9. plasticity-loss stress test
10. channel-block compensation
11. cryptic phenotype / setpoint rewrite
12. perturbation-based light-cone probes
13. tiny MNIST subset
14. Split-MNIST
15. non-visual bridge
16. lesion battery

---

## 9. Final guidance

If you stay faithful to this execution plan, the new codebase should feel very different from the old one.
It should be:
- smaller at the start,
- cleaner in ontology,
- easier to debug,
- faster to iterate,
- and much harder to fool with pretty-but-empty biology.

The real discipline is not technical difficulty.
It is refusing to move on before the substrate has earned the next step, and refusing to call a generic global signal a Z-field unless it actually behaves like a setpoint-carrying recovery scaffold.

That is how you stay true to the master architecture.
