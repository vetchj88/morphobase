# MorphoBASE Master Architecture v1.3a
## A benchmark-agnostic master specification for a general computational organism

**Author:** Justin M. Vetch  
**Version:** 1.3a  
**Status:** Working master specification  
**Scope:** General computational organism, not a benchmark-specific model  
**Role of MNIST:** A simple assay, not the definition of the organism

---

## 0. Executive Summary

MorphoBASE is not a neural network with biological names. It is an attempt to build a **general computational organism**: a distributed population of semi-autonomous computational cells that can grow, differentiate, coordinate, stabilize, repair damage, and reconfigure themselves while preserving competence on long-horizon goals.

The core claim is unchanged from v1.2, but v1.3 makes it more disciplined:

1. **The organism is primary; the benchmark is secondary.**  
   Benchmarks are assay environments used to probe competence. They are not the ontology of the system.

2. **Advancement is empirical, not rhetorical.**  
   Mechanisms may overlap in code, but phase claims are gated by measurable capabilities.

3. **Accuracy is not enough, but phenotype is not enough either.**  
   A beautiful internal morphology with poor retained competence is failure. A high score with no growth, no regeneration, and no interventional robustness is also failure.

4. **Memory is multi-layered and partially counterfactual.**  
   Memory is not only weights. It is also field state, tissue arrangement, conductance pattern, developmental commitments, and latent target constraints that help restore function after perturbation.

5. **The architecture must remain benchmark-agnostic.**  
   The same organismal substrate should, in principle, support vision, language, robotics, control, scientific reasoning, and multimodal sensing through different boundary interfaces.

6. **Plasticity maintenance is a survival function, not an optimization afterthought.**  
   A body that becomes stable by losing the ability to learn is not mature; it is pathologically locked. Long-horizon viability therefore requires explicit renewal, reserve capacity, and measurable plasticity health.

7. **The Z-field remains a first-class mechanism.**  
   MorphoBASE does not reduce deep setpoint memory to a vague global bias. The Z-field is the organism’s slow, editable target scaffold for counterfactual recovery, long-horizon alignment, and durable setpoint rewrites.

v1.3a therefore separates:
- **non-negotiable invariants,**
- **required substrate mechanisms,**
- **optional later enrichments,**
- **primary success metrics vs supporting biomarkers,**
- **plasticity health vs apparent stability,**
- **generic field effects vs the specific Z-field setpoint scaffold,**
- **and execution reality vs long-horizon theory.**

---

## 1. Why v1.3a Exists

v1.2 was strong as a research manifesto but too loose as an execution spec. It mixed:

- hard invariants,
- current implementation assumptions,
- ambitious future mechanisms,
- and benchmark-specific expectations.

That created four avoidable problems:

1. **Phase ambiguity.**  
   The architecture implied overlapping phases, while implementation work often treated phases as strict gates.

2. **Over-coupling to early assays.**  
   Split-MNIST and Permuted-MNIST were useful scaffolds, but the document risked letting those assays define the architecture.

3. **Biomarker confusion.**  
   Internal coherence, lock-in, Z-alignment, or tissue-stage occupancy can be informative, but they are not substitutes for competence retention and perturbation recovery.

4. **Insufficient failure diagnostics.**  
   The architecture named mechanisms elegantly but did not specify enough concrete collapse modes.

v1.3 fixed those issues while preserving the strongest ideas from v1.2. v1.3a keeps that structure and hardens it with recent biology and continual-learning evidence around plasticity maintenance, cryptic setpoint rewrites, perturbation-defined cognitive boundaries, compensatory adaptation after channel loss, and a more explicit restoration of the Z-field as a named, load-bearing mechanism rather than a generic background field.

---

## 2. Foundational Philosophy

MorphoBASE adopts the following philosophical position:

**Intelligence is the competence of an organized collective to pursue and restore goal states across space, time, and perturbation.**

This definition is deliberately wider than conventional machine learning.
A thermostat shows a tiny form of goal maintenance. An embryo shows far richer goal maintenance. A regenerating planarian shows still richer competence because it can reach the same target morphology through many alternate micro-trajectories after injury. A useful artificial organism should move in that direction: not toward anthropomorphic theater, but toward increasing scale, robustness, flexibility, and self-maintenance.

The important unit of analysis is therefore not the static parameter graph. It is the **process** by which local agents:
- sense local and field state,
- negotiate with neighbors,
- update internal models,
- assume or relinquish roles,
- and collectively restore target organization.

A trained MorphoBASE instance is not merely a solution. It is a **living solution process**.

---

## 3. Non-Negotiable Invariants

These are the invariants that define MorphoBASE. A system that violates them may still be interesting, but it is not MorphoBASE.

### 3.1 Cell-First Ontology
The primitive unit is a computational cell, not a layer and not a token.
Each cell must have:
- internal state,
- local update dynamics,
- local plasticity conditions,
- an interface to neighboring cells,
- and the ability to change functional role over time.

### 3.2 Physiological Communication
Routing is not fixed. Communication must depend on the physiological and relational state of cells and tissues.
A communication edge is never just “there”; it has state, permeability, and context.

### 3.3 Physiological Plasticity
Plasticity must be locally gated.
Global optimization may still be used as a scaffold during training, but the substrate itself must expose local conditions under which a cell is more or less plastic.

### 3.4 Developmental Topology
Growth, pruning, dedifferentiation, and stabilization are first-class processes.
Architecture size and topology are outcomes, not only hyperparameters.

### 3.5 Morphological Memory
Memory exists at multiple levels:
- transient state,
- long-lived parameters,
- communication structure,
- tissue arrangement,
- and global target constraints or attractors.

### 3.6 Interventional Validation
Claims about regeneration, canalization, or target morphology must be tested through perturbation:
- lesion,
- severance,
- corruption,
- input remapping,
- state clamping,
- and recovery without full retraining.

### 3.7 Benchmark Agnosticism
No benchmark is allowed to dictate the ontology of the organism.
Benchmarks are assay chambers, not the organism itself.

### 3.8 Energy and Budget Realism
All growth, persistence, signaling, and recovery happen under finite budgets.
The system must reason, implicitly or explicitly, under resource constraints.

### 3.9 Multi-Scale Agency
Agency exists at multiple nested scales:
- cell,
- tissue,
- organ,
- organism,
- and possibly collectives of organisms.
The architecture must support nested organization, not only flat swarms.

### 3.10 Partial Separability of Form and Skill
The system should increasingly learn to preserve useful organization even when low-level details change.
Not full abstraction in the human symbolic sense, but some partial transportability of form across tasks and perturbations.


### 3.11 Plasticity Maintenance and Renewal
The organism must actively preserve its ability to keep learning.
Plasticity is not assumed to persist automatically under long-horizon training.
The architecture must therefore support:
- renewable adaptive capacity,
- selective refresh or reseeding of underused structure,
- measured protection against dormancy and overcommitment,
- and explicit distinction between mature stability and pathological loss of plasticity.

### 3.12 Z-Field Specificity
A generic field channel is not enough.
MorphoBASE requires a named setpoint-carrying mechanism — the Z-field or an explicitly equivalent construct — that stores slow organism-level tendencies about what the body should restore, preserve, or become under perturbation.

---

## 4. What the Organism Is and Is Not

### It is:
- a distributed adaptive substrate,
- a goal-stabilizing developmental process,
- a benchmark-agnostic computational body,
- a platform for studying growth, repair, and collective intelligence in silico.

### It is not:
- an MLP with extra regularizers,
- an MoE whose experts have been renamed as tissues,
- a static graph with post hoc biological metaphors,
- a benchmark trick for MNIST,
- or a claim that biology can be copied literally into PyTorch.

MorphoBASE is a **translation layer**, not a literal simulation.
The goal is to capture organizing principles, not to reproduce every biochemical detail.

---

## 5. The General Computational Organism

A MorphoBASE organism is a population of computational cells embedded in a substrate. Each cell maintains:

- a hidden state,
- a membrane-like control state,
- a local generative or predictive model,
- a plasticity state,
- an energetic state,
- a role / identity distribution,
- and a set of communication edges.

The organism as a whole maintains:

- a body layout or graph topology,
- local tissue fields,
- a global morphological field,
- developmental commitments,
- energy / budget ledgers,
- and a target manifold describing what successful organization feels like.

This target manifold is not necessarily a fixed template.
In a general organism, “target morphology” means the set of states and trajectories that preserve competence under the current context. In one setting that may correspond to a visual classifier phenotype. In another it may be a control policy. In another it may be a distributed scientific reasoning collective.

Thus, target morphology is not a picture of the final body.
It is a **goal-conditioned attractor family**.

---

## 6. Core Cell Design: FEMO v1.3

The cell remains FEMO-like in spirit, but v1.3 broadens the abstraction.

### 6.1 Minimum Cell State
Each cell \(i\) should expose, conceptually, the following state:

- `h_i`: internal latent state
- `v_i`: membrane / excitability state
- `p_i`: plasticity gate or learning readiness
- `e_i`: energy / budget reserve
- `s_i`: stress / mismatch estimate
- `r_i`: role logits or role embedding
- `c_i`: developmental commitment / maturity
- `z_i`: local alignment to organism-level field or target
- `phi_i`: optional oscillatory phase or timing state
- `m_i`: local memory traces
- `alive_i`: viability state

Not every experiment needs every variable, but these are the canonical slots.

### 6.2 What a Cell Does
A cell performs five recurrent functions:

1. **Sense**
   - read local input,
   - read neighbor signals,
   - read tissue/global fields,
   - read its own mismatch.

2. **Infer**
   - update local internal belief about what state it and its neighborhood should occupy.

3. **Act**
   - modify hidden state,
   - send signals,
   - open or close communication,
   - alter plasticity,
   - request growth, repair, or role change.

4. **Remember**
   - retain useful local traces,
   - compress repeated experience into more durable forms.

5. **Negotiate**
   - reconcile local and collective goals through communication, stress signals, and field alignment.

### 6.3 Cell Internal Model
A cell should carry a minimal generative or predictive capability.
This does not have to be a full active-inference implementation, but it should be able to represent:
- current state,
- expected next state,
- local prediction error,
- and an estimate of how controllable the mismatch is.

This is crucial because v1.3 distinguishes:
- **recoverable mismatch** → local adaptation,
- **persistent uncloseable mismatch** → recruitment, dedifferentiation, or structural change.

### 6.4 Cell Identity Is Soft
Cells do not belong to hard-coded modules.
Identity is a dynamic role distribution such as:
- sensory,
- integrative,
- relay,
- stabilizer,
- repairer,
- archive,
- exploratory,
- metabolic / budget regulator,
- immune / audit role.

These roles are not permanent classes. They are attractor basins.

---

## 7. Boundary Interfaces: How the Organism Touches the World

One of the biggest missing pieces in v1.2 was a more general treatment of the organism boundary.
v1.3 adds it explicitly.

### 7.1 The Boundary Problem
A general organism cannot assume that inputs are images and outputs are logits.
It must support arbitrary interfaces:
- spatial sensory streams,
- token streams,
- continuous control,
- graph events,
- tool observations,
- external memory surfaces,
- robotic proprioception,
- symbolic constraints.

### 7.2 Boundary Ports
MorphoBASE therefore uses **ports**, not task-specific heads as the primary abstraction.

A port is a boundary tissue that translates between external formats and organism-internal physiology.
Ports may include:
- sensory ports,
- action ports,
- query ports,
- memory ports,
- reward / evaluation ports,
- embodiment ports.

### 7.3 Port Design Rules
Ports may be engineered initially, but they should obey the same organismal logic as the rest of the body:
- they have local state,
- they can adapt,
- they can become damaged,
- they can specialize,
- and they can be remapped.

### 7.4 Benchmark Ports
For assays like MNIST, the image encoder is just a temporary visual port.
For language, a tokenizer or latent front-end is a language port.
For robotics, sensor fusion becomes a sensorimotor port.
This preserves a single ontology while allowing many task families.

---

## 8. Multi-Scale Organization

Recent biology and collective-intelligence work strongly suggest that cognition-like competence can exist across scales. v1.3 therefore formalizes four nested levels.

### 8.1 Level 0: Cell
Local update, local mismatch, local action.

### 8.2 Level 1: Tissue
A tissue is a persistent cluster of cells with:
- correlated communication,
- shared timescale,
- partially shared field alignment,
- and recurring functional role.

### 8.3 Level 2: Organ
An organ is a tissue collective with a semi-stable interface to the rest of the organism.
Examples:
- sensory organ,
- integrative organ,
- archive organ,
- repair organ,
- planning organ,
- motor organ.

In early versions, organs may be mostly analytical concepts. Over time they should emerge more clearly.

### 8.4 Level 3: Organism
The whole body expresses:
- global viability,
- target coherence,
- competence on tasks,
- perturbation recovery,
- and developmental identity.

### 8.5 Level 4: Ecosystem
Multiple organisms may eventually coordinate or specialize together. This is not the immediate target, but the architecture should not preclude it.

---

## 9. Communication and Coordination

### 9.1 Three Communication Modes
The organism should support at least three qualitatively distinct signaling systems:

1. **Direct edge communication**  
   Pairwise signals along adjustable conductance edges.

2. **Diffusive field communication**  
   Local or tissue-scale spread of scalar/vector state such as stress, novelty, or morphogen-like signals.

3. **Global field influence**  
   Slower, lower-bandwidth organism-level biasing that encodes target tendencies, constraints, or regime shifts.

### 9.2 Gap-Junction Analogue
The current gap-junction idea remains good, but v1.3 clarifies its role:
- it is not just routing,
- it is a **relationship variable** expressing willingness, compatibility, and permeability.

Conductance should depend on some combination of:
- spatial or graph proximity,
- role compatibility,
- voltage or state similarity,
- stress correlation,
- trust / historical usefulness,
- and organism-level policy.

### 9.3 Stress Sharing
Stress is elevated to a first-class coordination primitive.
It should not be a crude loss mirror.
Stress should represent local inability to maintain expected function under current conditions.

Stress may:
- spread to neighbors,
- recruit support,
- bias growth,
- trigger role redistribution,
- or mark the onset of pathology.

### 9.4 Communication Plasticity
Edges must be able to:
- strengthen,
- weaken,
- close,
- reopen,
- reroute,
- bundle into highways,
- or dissolve.

### 9.5 Oscillatory and Timing Structure
Oscillatory coordination is optional but valuable.
Phase-like timing can help:
- avoid destructive synchrony,
- define windows for plasticity,
- and support long-range coordination without global locks.

---

## 10. Fields: Local, Tissue, Global, and Z-Field

v1.2 already used a Z-field idea. v1.3 keeps it but clarifies the field stack.

### 10.1 Local Alignment Field
Each cell tracks how well its current state aligns with the local developmental context.

### 10.2 Tissue Fields
Tissues may maintain slower shared variables:
- tissue identity,
- activity setpoint,
- repair demand,
- archive density,
- plasticity climate.

### 10.3 Global Morphological Field
The organism maintains a slow global field that encodes soft constraints on what kind of body it is trying to be.
This field should:
- bias local attractors,
- shape communication,
- help resolve ambiguity after injury,
- and preserve large-scale organization.

### 10.4 Counterfactual Role of the Global Field
The global field is not a static template. It is most valuable when the organism is damaged, unfamiliar, or partially disassembled.
Its purpose is to answer:
- what should this region become again,
- what functions are missing,
- and which configurations count as acceptable recovery.

### 10.5 Field Update Rule
The field should update more slowly than local state and usually more slowly than local plasticity.
It should integrate:
- persistent successes,
- recurring failures,
- long-horizon target tendencies,
- and developmental commitments.

### 10.6 The Z-Field
The Z-field is a special field, not just the global field under another name.
It is the organism’s slow setpoint archive: a latent record of what successful organization feels like across perturbations and partial damage.

The Z-field should:
- encode organism-level correction tendencies,
- bias ambiguous recovery toward coherent wholes,
- preserve deep constraints while allowing multiple local realizations,
- support durable setpoint rewrites after brief intervention,
- and remain partially decoupled from moment-to-moment task noise.

A healthy Z-field is neither dead nor tyrannical.
If it becomes too weak, recovery becomes shallow and brittle.
If it becomes too strong, it prevents adaptive reconfiguration and smothers useful novelty.

---

## 11. Memory Architecture

Memory is one of the most important sections to upgrade.

### 11.1 Memory Is Layered
MorphoBASE uses at least five memory strata:

1. **Fast state memory**  
   Short-lived hidden and voltage-like state.

2. **Synaptic / parametric memory**  
   Slower, optimization-shaped weights.

3. **Relational memory**  
   Durable communication pathways, conductance patterns, and tissue adjacency structure.

4. **Morphological memory**  
   Stable role allocations, tissue geometry, organ boundaries, commitment landscapes.

5. **Counterfactual memory**  
   Latent knowledge of what the organism should restore when actual state is corrupted.

### 11.2 Memory Is Reinterpretable
Memory is not a frozen table.
The same memory substrate may support multiple future uses depending on context.
This implies:
- memory retrieval is reconstructive,
- role of stored information may change with organism state,
- and not all preservation should be literal replay.

### 11.3 Memory Compression
As skills stabilize, the organism should compress:
- repeated pathways,
- recurrent tissue arrangements,
- and re-used attractors.

Compression is not just for efficiency. It is part of maturation.

### 11.4 Protective Memory
Some cells or structures may become protected because they hold globally important constraints.
This does not make them immutable; it raises the bar for pruning.

### 11.5 Cryptic Memory
The organism may harbor competencies not visible under baseline conditions.
These should be testable through perturbation and context shifts.

### 11.6 Z-Field Memory
The Z-field is the most explicit substrate for deep counterfactual memory in MorphoBASE.
It does not store every detail literally. Instead it stores the low-bandwidth setpoint tendencies that let the body infer what kind of organization is missing and how far current state is from an acceptable recovery basin.

### 11.7 Setpoint Rewrite and Cryptic Phenotypes
A serious computational organism should be able to undergo durable rewrites of its recovery tendencies after brief intervention.
This does not mean arbitrary memory editing. It means that a transient perturbation to conductance, stress propagation, or Z-field bias may create a lasting change in what future recovery converges toward.

The strongest form of this claim is a cryptic phenotype:
- the organism appears normal at baseline,
- but later lesions reveal that the latent recovery attractor has changed.

This is a stricter test of morphological memory than ordinary robustness.
The architecture therefore treats setpoint rewrite assays as first-class evidence for deep counterfactual memory.

---

## 12. Developmental Program

MorphoBASE development proceeds through a cyclic developmental logic, not just a feedforward computation.

### 12.1 Seed
Start from a small viable seed, not a full body.
The seed must be sufficient to:
- remain alive,
- interpret boundary inputs at a primitive level,
- and recruit additional structure.

### 12.2 Exploration and Growth
When mismatch persists and resources allow, the organism can grow.
Growth may mean:
- adding cells,
- increasing edge density,
- opening plasticity in existing tissue,
- or recruiting dormant regions.

### 12.3 Differentiation
Cells and tissues specialize when:
- their local statistics stabilize,
- their communication profile becomes characteristic,
- their contribution becomes reliable,
- and their developmental commitments consolidate.

### 12.4 Maturation
Maturation is a checkpointed reduction in volatility.
Mature tissue is less exploratory, more stable, more energy-efficient, and harder to repurpose.

### 12.5 Morphostasis
Morphostasis is the regime in which:
- gross body plan stabilizes,
- growth pressure decreases,
- routine mismatch is handled locally,
- and the organism can maintain competence with low structural churn.

### 12.6 Repair and Dedifferentiation
When damage or regime shift exceeds local repair capacity, some tissue may loosen its commitment and re-enter a more plastic state.
This must be controlled carefully to avoid pathological collapse.

### 12.7 Regeneration
True regeneration means the organism can restore useful function after partial destruction with limited or no gradient updates.
Regeneration may restore exact prior structure or a functionally equivalent alternative.

### 12.8 Metamorphosis
Long-horizon organisms should support slower body-plan changes across life stages or task regimes.
This is optional in early versions, but the architecture should not prohibit it.

---

## 13. Plasticity and Learning

### 13.1 Three Kinds of Learning
MorphoBASE distinguishes:

1. **State adaptation**  
   Rapid changes in hidden/field state.

2. **Parameter learning**  
   Slower changes in local models and communication machinery.

3. **Structural learning**  
   Growth, pruning, role reassignment, tissue splitting or merging.

### 13.2 Local Plasticity Gating
A cell’s effective plasticity should depend on:
- mismatch magnitude,
- uncertainty,
- energy reserve,
- maturity,
- tissue demands,
- and global danger / repair mode.

### 13.3 Global Optimization Is a Scaffold, Not the Theory
Backprop or gradient-based optimization may still be used as a practical training tool.
But the conceptual learning signal for the organism is not “the optimizer decided.”
The organism should expose local conditions under which change is allowed or suppressed.

### 13.4 Controllability Matters
Not all mismatch should trigger growth.
The system should distinguish:
- mismatch I can fix locally,
- mismatch my neighbors can help with,
- mismatch that requires recruiting structure,
- mismatch that indicates the task is incompatible with current body plan.

### 13.5 Learning Without Full Forgetting
The goal is not zero change. The goal is selective change that preserves deeper constraints while reconfiguring shallow details.
This is where morphological memory should matter most.

### 13.6 Plasticity Maintenance and Renewal
The organism must explicitly monitor whether it is still capable of useful change.
Modern continual-learning systems can become apparently stable because they have lost diversity, accumulated overcommitted units, or let too much structure harden beyond reuse. MorphoBASE therefore treats plasticity maintenance as a physiological control problem.

Plasticity maintenance should include some combination of:
- reserve capacity that remains weakly committed,
- selective refresh of underused or dormant cells,
- budgeted dedifferentiation or reseeding,
- diversity-preserving noise or variability injection,
- and biomarkers that distinguish healthy maturity from pseudo-maturity.

Primary signs of plasticity health include:
- continued ability to improve on late tasks,
- low dormant-cell fraction,
- preserved representational diversity,
- and the ability to recover learning after refresh interventions.

### 13.7 Compensation and Channel-Block Adaptation
The organism should not depend on any single signaling path remaining intact.
When one channel, message type, or conductance family is disrupted, the body should be able to compensate by adjusting remaining pathways where possible.

This is not redundancy for its own sake.
It is evidence that the organism has learned safe operating regimes and can preserve competence by rebalancing its physiology under perturbation.

---

## 14. Energy, Budget, and Metabolism

v1.2 under-emphasized metabolism. v1.3 makes it explicit.

### 14.1 Every Process Has Cost
Costs should attach to:
- cell maintenance,
- signaling,
- state update complexity,
- structural growth,
- prolonged high plasticity,
- repair,
- and global-field updates.

### 14.2 Energy State
Each cell has an energy-like variable or effective budget pressure.
The organism also has a whole-body budget.

### 14.3 Functional Consequences
Low energy should bias toward:
- local repair before growth,
- pruning of low-utility tissue,
- reduced signaling range,
- and reuse of existing pathways.

### 14.4 Metabolic Intelligence
A useful organism learns not only to solve tasks, but to solve them in ways that are energetically sustainable.
A brittle solution that requires constant high-growth/high-signal expenditure is not mature.

---

## 15. Pathology and Immune Function

Pathology is not an edge case. It is an expected mode of collective systems.

### 15.1 Main Pathologies
At minimum, the architecture should recognize:

- **runaway growth**  
  chronic organogenesis or body inflation,

- **pseudo-maturity**  
  tissue appears locked but is not truly competent,

- **plasticity collapse**  
  the body stabilizes by losing the ability to adapt or recruit diversity,

- **field collapse**  
  the global field becomes trivial, uninformative, or overbearing,

- **route monoculture**  
  all traffic collapses through one brittle corridor,

- **dedifferentiation cascade**  
  a local perturbation causes whole-body loss of role identity,

- **silent necrosis**  
  portions of the body stop contributing but remain metabolically expensive,

- **oncogenic specialization**  
  local tissue optimizes its own persistence at the expense of organism-level function.

### 15.2 Immune / Audit Layer
The organism should include a slow auditing process that:
- scans for mismatched local vs organism goals,
- detects chronic non-contributors,
- identifies unstable positive feedback loops,
- and applies interventions.

### 15.3 Allowed Interventions
Audit interventions may include:
- suppressing growth,
- closing or narrowing edges,
- lowering plasticity,
- isolating tissue,
- forcing cooling periods,
- marking tissue for pruning,
- or temporarily boosting repair partners.

### 15.4 Immune Humility
The immune layer should not become a disguised centralized controller.
Its role is supervisory constraint, not micromanagement.

---

## 16. Organogenesis and Pruning

### 16.1 When Growth Is Allowed
Growth should require a conjunction of conditions such as:
- persistent mismatch,
- sufficient energy,
- local or regional inability to close the gap,
- absence of systemic cascade failure,
- and evidence that new structure would be useful.

### 16.2 What Growth Can Mean
Growth is broader than adding nodes.
It may mean:
- new cells,
- new edges,
- temporary repair scaffold,
- new tissue boundary,
- or a new port interface.

### 16.3 Differentiated Organogenesis
Growth should not be uniform.
Growth zones emerge where:
- novelty enters,
- bottlenecks recur,
- stress accumulates,
- or the body needs redundancy.

### 16.4 Pruning
Pruning is not punishment.
It is recycling.
Candidates for pruning include tissue that is:
- low utility,
- low uniqueness,
- energetically expensive,
- chronically unstable,
- or redundant.

### 16.5 Hysteresis
Growth and pruning should be separated by refractory periods.
Otherwise the body chatters.

### 16.6 Z-Role Protection
Cells storing globally important constraints or serving irreplaceable relay roles should receive extra protection, though not absolute immunity.

---

## 17. Maturation, Commitment, and Dedifferentiation

### 17.1 Commitment
Commitment measures how strongly a cell or tissue has crystallized into a role and local attractor.

### 17.2 Maturation Signals
No single signal should force commitment.
Maturation should depend on multiple jointly satisfied indicators:
- repeated competence,
- low unresolved mismatch,
- stable communication,
- sustainable energy use,
- and predictable contribution.

### 17.3 What Maturity Does
Maturity should:
- lower plasticity,
- narrow exploratory range,
- stabilize communication,
- reduce energy waste,
- and create more reliable tissue identity.

### 17.4 Dedifferentiation
Dedifferentiation is allowed, but should be difficult.
It should require evidence that:
- local repair is failing,
- global context supports repurposing,
- and the likely gain exceeds destabilization cost.

### 17.5 Counterfactual Gate
A key v1.3 rule:
Dedifferentiation should be blocked if a forward repair attempt suggests the gap is still closeable without structural rollback.

This preserves structure whenever possible and prevents panic responses.

---

## 18. Time Structure

### 18.1 Fast Timescale
- hidden state
- excitability / membrane state
- local signaling
- edge gating
- immediate action

### 18.2 Intermediate Timescale
- parameter updates
- role drift
- tissue statistics
- port adaptation
- stress propagation

### 18.3 Slow Timescale
- global field update
- maturation checks
- immune audits
- organogenesis / pruning decisions
- budget redistribution

### 18.4 Ultra-Slow Timescale
- life-stage shifts
- new organ emergence
- long-horizon compression
- embodiment transfer
- meta-learning of developmental priors

The single most important timing rule is this:
**structural change must usually be slower than plastic change, and plastic change must usually be slower than state adaptation.**

---

## 19. Training Philosophy

MorphoBASE does not “train a model” in the ordinary sense.
It develops a body under repeated perturbation and assay.

### 19.1 Three Training Views
Each experiment should explicitly state which of these is primary:

- **developmental training:** can the body grow into competence?
- **persistence training:** can it hold competence over time?
- **repair training:** can it restore competence after damage?

### 19.2 Overlapping Mechanisms, Gated Claims
Mechanisms may coexist in code. However, claiming entry into a new phase requires empirical gate satisfaction.
This resolves the old ambiguity.

### 19.3 Assay Families
Training and evaluation should span more than one assay family:
- simple visual tasks,
- nonstationary control,
- sequential reasoning,
- port remapping,
- lesion recovery,
- distribution shift,
- embodiment transfer.

### 19.4 Curriculum Principle
Curricula should challenge:
1. growth,
2. stability,
3. recovery,
4. transfer,
5. and economical self-maintenance.

Not every run needs all five, but the master program does.

### 19.5 No Benchmark Overfitting of Ontology
If a mechanism only makes sense on MNIST-like grids and cannot be re-expressed as a general organismal principle, it should not define the master architecture.

---

## 20. Build Phases (Revised)

These are empirical gates, not merely narrative chapters.

### Phase A — Viable Cell Substrate
Goal:
- stable cell update,
- local state dynamics,
- local plasticity gating,
- viable seed.

Exit evidence:
- seed remains alive across extended rollouts,
- local plasticity meaningfully varies by state,
- no trivial divergence/collapse.

### Phase B — Adaptive Communication
Goal:
- dynamic conductance,
- stress sharing,
- non-uniform routing,
- tissue-like clusters begin to appear.

Exit evidence:
- communication map differentiates under task pressure,
- stress propagation is measurable and useful,
- routing is not static and not uniformly open.

### Phase C — Growth Competence
Goal:
- the body can recruit or reorganize structure when mismatch persists.

Exit evidence:
- controlled growth beyond seed,
- growth occurs preferentially in meaningful regions,
- no chronic cascade failure.

### Phase D — Morphostasis
Goal:
- self-terminating growth,
- stable tissue roles,
- reduced unnecessary churn,
- sustainable energy profile,
- and preserved plasticity reserve.

Exit evidence:
- body size stabilizes after challenge onset,
- mature tissues emerge,
- late growth decreases sharply,
- competence remains stable over long rollouts,
- and late-task learning slope does not collapse to near-zero.

### Phase E — Regeneration
Goal:
- partial destruction can be repaired without full retraining.

Exit evidence:
- recovery after lesions, severance, corruption, or remapping,
- function restored within bounded steps or bounded energy,
- multiple injury types tolerated,
- and Z-field corruption or transient bias produces interpretable, testable changes in recovery trajectories.

### Phase F — Transferable Organism
Goal:
- the same substrate principles support multiple assay families and interface ports.

Exit evidence:
- at least two qualitatively different task families supported,
- port remapping feasible,
- organismal principles still explanatory.

### Phase G — Open-Ended Development
Goal:
- multi-stage life history,
- organ reconfiguration,
- meta-developmental priors,
- possible multi-organism ecology.

Exit evidence:
- not required for early claims.

---

## 21. Metrics: Primary vs Biomarker

v1.3 explicitly separates outcome metrics from internal biomarkers.

### 21.1 Primary Metrics
These are the load-bearing metrics.

1. **Task competence**
   - accuracy, reward, return, control quality, reasoning score, depending on assay.

2. **Retention / transfer**
   - forgetting,
   - positive/negative backward transfer,
   - forward transfer,
   - competence under sequential nonstationarity.

3. **Recovery**
   - probability of recovery,
   - time to recover,
   - energy cost of recovery,
   - post-recovery competence.

4. **Morphostasis**
   - stabilization of body size,
   - reduced structural churn,
   - sustained function over long rollouts.

5. **Efficiency**
   - parameters or active cells per competence,
   - signaling cost,
   - energy burden,
   - recovery cost.

### 21.2 Biomarkers
These are supporting diagnostics, not victories by themselves.

- stage occupancy,
- Z-error / field alignment,
- Z-field drift,
- Z-field lesion sensitivity,
- conductance differentiation,
- conductance entropy,
- stress map quality,
- tissue modularity,
- oscillatory coherence,
- synergy / integration markers,
- field drift,
- cognitive light cone size,
- dormant-cell fraction,
- active-unit diversity,
- learning slope on late tasks,
- canalization index,
- immune intervention rate.

### 21.3 Perturbation-Based Light-Cone Measurement
Light-cone language is only useful if it is measurable.
In MorphoBASE, cognitive light-cone estimates should come primarily from perturbation probes rather than topology alone.

Canonical protocol:
- perturb one cell or small region at time \(t_0\),
- track downstream effect size across space and time,
- define cone area or influence radius as the region where effect exceeds a threshold,
- compare this cone under stress-sharing, Z-field, and conductance ablations.

This converts a philosophical notion of expanding agency into an engineering diagnostic.

### 21.4 The Rule
A biomarker may explain success or failure.
It may not replace success.

---

## 22. Assay Suite

The master architecture should be evaluated by a suite, not a single benchmark.

### 22.1 Basic Development Assays
- grow from seed to competence,
- remain viable under longer rollout,
- tolerate input noise.

### 22.2 Sequential Learning Assays
- simple image sequences,
- distribution-shifted variants,
- control tasks with regime change,
- small reasoning curricula.

### 22.3 Regeneration Assays
- node ablation,
- edge severance,
- parameter corruption,
- state clamping,
- port disruption.

### 22.4 Field / Counterfactual Assays
- pulse intervention,
- re-amputation,
- alternate recovery paths,
- cryptic phenotype exposure,
- Z-field corruption and restoration,
- setpoint rewrite after transient perturbation,
- forced role swaps.

### 22.5 Compensation / Channel-Block Assays
- disable one signaling channel or message family,
- require maintenance of safe operating regime,
- measure compensation through remaining pathways,
- compare local rescue versus organism-wide collapse.

### 22.6 Boundary / Port Assays
- same body with new sensory port,
- remapped outputs,
- cross-embodiment transfer,
- partial body with substituted port tissue.

### 22.7 Ecology Assays
Optional later:
- two organisms cooperating,
- organism competition for budget,
- shared environment signaling.

---

## 23. Failure Diagnosis Matrix

This section is mandatory in v1.3.

### 23.1 High S2 / low mature / low recovery
Likely causes:
- commitment thresholds misaligned,
- body trapped in pre-mature limbo,
- insufficient stabilization reward,
- global field not informative enough.

### 23.2 Strong lock / poor competence
Likely causes:
- pseudo-maturity,
- trivial field collapse,
- Z-field domination without flexibility,
- over-suppressed plasticity,
- biomarkers optimized instead of function.

### 23.3 Persistent late growth
Likely causes:
- morphostasis controller too weak,
- mismatch never truly resolved,
- organogenesis gate too permissive,
- chronic under-capacity.

### 23.4 Differentiated routing / no task gain
Likely causes:
- communication structure has emerged but is not causally useful,
- ports are bottlenecking,
- cell internal models too weak.

### 23.5 Good immediate learning / bad retention
Likely causes:
- changes stored in fragile fast pathways,
- insufficient protective memory,
- too much dedifferentiation,
- poor field consolidation.

### 23.6 Recovery only with gradients
Likely causes:
- regeneration is really retraining,
- counterfactual memory is absent or weak,
- repair scaffold not learned.

### 23.7 Low energy stability / high competence
Likely causes:
- brute-force body,
- immature metabolic policy,
- unsustainable communication density.

### 23.8 Plasticity-health collapse
Likely causes:
- too much commitment too early,
- insufficient renewal or reseeding,
- dormant-cell accumulation,
- Z-field refusing useful reconfiguration.

### 23.9 High immune intervention frequency
Likely causes:
- chronic pathology,
- poor local governance,
- immune layer compensating for bad substrate design.

---

## 24. Minimal v1.3 Must-Haves vs Deferred Mechanisms

### 24.1 Must-Haves
The following are required for claiming MorphoBASE v1.3:

- cell-first substrate,
- local plasticity gating,
- dynamic conductance,
- stress sharing,
- slow global field plus an explicit Z-field or equivalent setpoint archive,
- controlled growth/pruning,
- maturation and dedifferentiation logic,
- explicit energy/budget terms,
- benchmark-agnostic port abstraction,
- explicit plasticity-maintenance machinery or at least plasticity-health diagnostics,
- interventional regeneration assays,
- primary-vs-biomarker metric separation,
- failure diagnosis tooling.

### 24.2 Strongly Desirable
- tissue fields,
- protected archive regions,
- immune audits,
- phase / oscillatory coordination,
- organ-level nesting.

### 24.3 Deferred / Frontier
- full active-inference formalization everywhere,
- multi-organism ecosystems,
- fully emergent ports,
- developmental reproduction,
- rich metamorphosis across life stages,
- open-ended ecology and niche construction.

These are exciting, but they should not bloat the minimal proof path.

---

## 25. Implementation Guidance

### 25.1 Preserve the General Ontology
When adding a feature, ask:
Does this generalize beyond the current assay?
If not, isolate it as a temporary scaffold.

### 25.2 Keep Ports Explicit
Do not let visual encoder assumptions leak into the body ontology.
The body should only see boundary signals and port states.

### 25.3 Make State Observable
Every major mechanism needs diagnostics:
- field maps,
- stress maps,
- conductance maps,
- stage transitions,
- energy budgets,
- recovery curves.

### 25.4 Prefer Soft Constraints Over Hard Templates
Target morphology should bias recovery without dictating a single brittle final arrangement.

### 25.5 Track Structural Causality
Whenever a new tissue or edge bundle emerges, ask whether removing it changes function.
Morphology without causal necessity is decoration.

### 25.6 Preserve Z-Field Specificity
Do not let the Z-field blur into a generic catch-all field term.
If a mechanism is carrying counterfactual setpoint memory, name it, instrument it, lesion it, and measure what changes when it is biased or corrupted.

### 25.7 Protect Against Beautiful Failure
Some runs will look organismal while performing badly.
The architecture and tooling must make that obvious.

---

## 26. Research Positioning

MorphoBASE v1.3 sits at the intersection of:

- developmental bioelectricity,
- collective intelligence,
- basal cognition,
- embodied intelligence,
- active inference / self-maintaining systems,
- continual learning,
- and adaptive morphology.

Its distinctive bet is this:

**A useful path to general intelligence may not be bigger static models alone, but computational bodies that can preserve and restore competence by developmental means.**

This does not reject scaling. It reframes scaling.
The relevant scaling variables include:
- size,
- communication richness,
- memory layering,
- recovery depth,
- developmental horizon,
- and the spatial/temporal extent of goals the system can coherently maintain.

---

## 27. What Counts as Success

A strong MorphoBASE result is not:
- one benchmark score,
- one pretty heatmap,
- or one metaphor.

A strong result is a body that can:
1. grow into competence from a seed,
2. retain competence through sequential change,
3. stabilize without endless growth,
4. recover from partial destruction,
5. do so efficiently enough to be a plausible substrate,
6. and transfer the same principles across more than one assay family.

That is the bar.

---

## 28. Immediate Practical Consequences for the Current Build

For the current codebase, this v1.3 master spec implies:

1. Treat MNIST as an assay, not the organism definition.
2. Keep the cell/field/stress/topology core.
3. Add explicit boundary-port abstractions early.
4. Separate primary outcomes from morphology biomarkers in all sweep scoring.
5. Add energy/budget accounting as a genuine controller input.
6. Keep the Z-field explicit as the counterfactual setpoint scaffold; do not collapse it into a generic global field.
7. Diagnose pre-mature limbo, pseudo-maturity, and plasticity-health collapse explicitly.
8. Make phase claims strictly empirical.
9. Demand regeneration and setpoint-rewrite evidence before strong morphological-memory claims.
10. Move aspirational Phase 5+ mechanisms into a roadmap, not the proof path.
11. Require at least one non-visual assay family before claiming a general computational organism.

---

## 29. v1.3 Design Mandate

Build the smallest organism that is still truly organismal.

Do not chase decorative biology.
Do not let easy benchmarks define the architecture.
Do not call internal coherence success if the organism forgets.
Do not call accuracy success if the organism cannot repair.

The aim is a body that can maintain and recover goals by developmental means.

That is MorphoBASE.

---

## 30. Appendix A — Canonical Questions for Every New Mechanism

Before adding any mechanism, answer:

1. What organism-level problem does it solve?
2. Is it required now, or is it roadmap material?
3. Does it generalize beyond the current benchmark?
4. At what timescale does it operate?
5. What state variables does it read and write?
6. What failure mode does it prevent?
7. What new pathology could it create?
8. How will we ablate it?
9. What primary metric should improve if it matters?
10. What biomarker should move if it is actually active?

---

## 31. Appendix B — Suggested Next Revision Targets

These are not yet part of the hard spec, but are likely next:

- formal organism-level belief state,
- richer tissue identity formation,
- port tissue self-repair,
- organ-level working memory,
- ecological multi-agent morphogenesis,
- cross-substrate transfer of developmental priors,
- stronger mathematical treatment of controllability and “gap closeability.”

---

## 32. Closing Statement

MorphoBASE v1.3 is a commitment to a different research style.
Instead of asking only how to optimize a fixed network, it asks how to cultivate a body of computation that can acquire, stabilize, and restore competence over time.

The goal is not to imitate biology cosmetically.
The goal is to extract from biology the principles that let matter become goal-directed, self-repairing, and developmentally intelligent, and then reinstantiate those principles in a new substrate.

That is the project.