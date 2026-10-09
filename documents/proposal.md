# **Autonomous Dynamic-Nesting Networks: Controlling Module Expansion Rate in Continual Learning**

## **1\. Executive summary**

Isolation-based continual learning methods eliminate catastrophic forgetting by allocating separate parameters to separate tasks. This works, and it is the reason the family exists. Its cost is module proliferation: parameter count grows with the number of tasks learned, and the system becomes progressively more expensive to store, route through, and deploy.

The field addresses this almost entirely at the back end. Modules are created first and compressed afterwards through pruning, distillation, or merging. This project asks a different question: can proliferation be reduced at the **front end**, by building a learner that absorbs more of the data stream before it needs to delegate at all?

The proposed system combines three mechanisms toward that end. A main learner with nested multi-timescale internal structure holds the frontier of novel data. Spawned offspring modules remain plastic rather than frozen, so recurring domains are absorbed by existing modules instead of triggering new ones. A non-parametric control plane manages the module population using accumulated statistics rather than trained weights, so the management layer is not itself subject to forgetting.

A capacity-triggered offline consolidation stage, the Dream Phase, acts as a backstop when front-end reduction is insufficient.

---

## **2\. Problem statement and hypothesis**

**Primary hypothesis.** Module expansion rate is a controllable design variable rather than a fixed consequence of task count. Multi-timescale nesting and module plasticity each reduce it, and their combination reduces it further, without a corresponding loss in retention.

**Primary metric.** Spawn rate: modules created per task, measured over long streams. This is the quantity the architecture uniquely claims to improve.

**On the metric that is not used as a headline.** Backward transfer approaching zero is arithmetic under parameter freezing, not an experimental finding. Under the plastic-module design proposed here, forgetting becomes possible again, which is what makes retention measurements meaningful. Retention is reported, but as a constraint the system must satisfy rather than as evidence of success.

**Secondary question.** Once front-end reduction is applied, how much additional compression does back-end consolidation provide, and at what cost to routing resolution and retention?

---

## **3\. Positioning**

The architecture sits in the isolation family alongside Progressive Neural Networks, Dynamically Expandable Networks, and CN-DPM. Those systems are not comparison points for retention, where all isolation methods perform similarly by construction. They are comparison points for **expansion rate**, which is the axis this work targets.

Three specific relationships to existing work:

**Task-free expansion.** Loss-plateau triggering for task boundary detection is established practice. This project adopts it rather than claiming it, and the contribution lies in what determines when the plateau is reached.

**Multi-timescale learning.** Nested update frequencies overlap with Nested Learning / HOPE, with Complementary Learning Systems approaches such as CLS-ER and DualNet, and with fast-weight literature going back to Hinton and Plaut. The open question addressed here is whether the effect survives at small scale in vision continual learning under standard protocols, and whether it withstands the deflationary explanation that it reduces to per-layer learning-rate scheduling.

**Plastic experts.** Mixture-of-experts continual learning has moved toward updatable experts, and has identified two failure modes in doing so: router drift and expert drift. That work is almost entirely LoRA adapters on frozen pretrained backbones. The configuration proposed here, where autonomously spawned self-contained learners take over training for their own domain, is not covered by it.

---

## **4\. Architecture**

### **4.1 Nested main learner**

A learner with three internal levels updating at different frequencies. Slow levels are intended to accumulate stable general structure; fast levels handle immediate specifics.

Its role in the system is the **novelty handler**. It receives only data that fails to route to any existing module, so it never relearns what offspring already hold, and it serves as the incubator where unclaimed data accumulates until there is enough of it to justify spawning.

The concrete meaning of "different update frequency" is deliberately left as a design variable to be resolved experimentally (Section 7.3). The candidates are not equivalent, and at least two of them have deflationary explanations that must be ruled out.

### **4.2 Plastic offspring**

Spawned modules are clones of the main learner at the time of spawning, and retain the same nested structure. They are not frozen. When the router assigns data to a module, that module trains on it.

The theoretical justification: if routing is accurate, each module sees only data from its own distribution, so its input stream is approximately stationary. Forgetting is a non-stationarity problem, and correct routing restores local stationarity. This also identifies precisely where the design fails, which is when routing is wrong.

### **4.3 Warm-start cloning**

On spawning, the main learner clones itself and continues from its current weights rather than reinitializing. Resetting would discard exactly the slow-level structure the nesting is meant to accumulate, at the moment it is most valuable. Warm starting preserves it and provides forward transfer to each successive offspring.

The cost is that offspring are snapshots of a drifting learner rather than independently initialized specialists. This is also, unexpectedly, an asset: common ancestry is the condition under which weight-space merging is viable (Section 6.3).

### **4.4 Non-parametric control plane**

A statistics layer above the modules, storing per module:

* a prototype, or set of centroids, summarizing its input distribution  
* a firing count  
* a running competence score  
* spawn generation, for ancestry tracking  
* last-fired timestamp

All entries update by accumulation. None are trained by gradient descent. This is the central design constraint: a learned management layer trained on a non-stationary stream would itself forget, which recurses the problem rather than solving it. Router drift has been measured to induce forgetting independently of any change in the underlying representations, which is direct evidence for keeping this layer non-parametric.

Every management decision is a query against this table. Which module owns an input is nearest-prototype. Whether two modules are redundant is prototype distance plus ancestry distance. Whether a module is dead weight is firing count over a window. Whether the main learner is saturated is competence plateau under sustained unrouted volume. Whether a module is protected from consolidation is minimum distance to all other prototypes.

The tradeoff is bluntness. A learned router could discover that superficially different inputs belong together; a prototype tracker sees only distance in the representation it is given. For a project whose thesis concerns long streams, stability is the correct trade, and the accuracy gap against a learned router is itself a measurement worth reporting.

### **4.5 Prototype routing with abstain**

An incoming sample is compared against all module prototypes. If the nearest is within threshold, the sample is routed there and the module trains on it. If no prototype is within threshold, the router abstains and the sample goes to the main learner.

The abstain gate is a **correctness mechanism, not an efficiency optimization**. With frozen modules a misroute costs one wrong prediction. With plastic modules a misroute writes wrong data into a module and pulls its prototype toward the wrong region, making subsequent misroutes more likely. The gate exists to break that feedback loop.

### **4.6 Prototype re-anchoring**

Because modules keep learning, a prototype computed at spawn time gradually stops describing the module it labels. Prototypes are recomputed periodically from a small held-out sample of each module's own data, so the prototype tracks the module rather than going stale.

### **4.7 Per-offspring anchor sets**

Each module retains a small buffer of its own real samples, replayed alongside incoming data. Three functions:

1. Resists corruption when a run of misrouted samples arrives  
2. Converts silent corruption into a visible competence drop the control plane can act on  
3. Serves as a real-data transfer set for distillation, removing the need for a generator during consolidation

---

## **5\. Lifecycle**

1. **Cold start.** No modules exist. All data flows to the main learner.  
2. **Saturation.** The control plane observes competence plateauing while unrouted volume remains high, and triggers a spawn.  
3. **Spawn.** The main learner clones into an offspring, warm-started. The control plane computes the offspring's prototype and initializes its counters. The main learner continues on the frontier.  
4. **Routing.** Incoming samples are matched against prototypes. Confident matches train their assigned module. Abstentions go to the main learner.  
5. **Re-anchoring.** After each training block, module prototypes are recomputed from held-out samples.  
6. **Housekeeping.** The control plane continuously flags merge candidates, retirement candidates, and protected outliers.  
7. **Consolidation.** On hitting the capacity threshold, external learning pauses and the Dream Phase runs.

---

## **6\. The Dream Phase**

Offline consolidation during a pause in external learning: related modules are clustered, compressed, and the unique ones protected.

### **6.1 Trigger and clustering**

Triggered by module count crossing a fixed budget. Clusters are formed on two criteria jointly:

* **Prototype distance** — are these modules handling related data  
* **Ancestry proximity** — were they spawned close enough in time to remain weight-compatible

Merging cousins is safe. Merging distant relatives is not, because prolonged independent training may have moved them into different regions of parameter space where averaging is meaningless. Pairs that are close in prototype space but distant in ancestry fall back to distillation or are declined.

### **6.2 Outlier immunity**

Any module whose distance to every other prototype exceeds an immunity threshold is excluded from consolidation permanently. Compression averages away detail, which is an acceptable trade when modules overlap and a pure loss when they do not. A module with no near neighbours is the only holder of its knowledge.

Under the control plane this requires no separate mechanism. It is one query against the prototype table.

### **6.3 Default operator: asymmetric weight merging**

Slow levels are merged across the cluster. Fast levels are retained as per-domain heads.

The rationale is structural. Distillation reconstructs behaviour from input-output matching, and nothing in that objective requires the student to rebuild internal timescale organization. A student can match outputs while being architecturally hollow, pass a retention check, and then saturate early — which would raise spawn rate, defeating the project's primary claim. Merging never dismantles the structure in the first place, because corresponding components are combined directly.

The asymmetry follows the intended semantics of the levels. Slow levels hold general reusable structure and are the natural target for consolidation. Fast levels hold the specialization that distinguishes the modules and are worth preserving separately. Since slow levels carry most of the parameters, most of the compression is retained.

Merging is also close to free computationally, and eliminates the generator entirely.

### **6.4 Commit gate**

Before any originals are purged, the consolidated module is evaluated on the pooled anchor sets of every cluster member. If any member's retention falls below tolerance, **the merge is rejected and those modules are left intact**. Deletion is irreversible; a forced merge destroys a specialist permanently and the consequence surfaces many tasks later.

### **6.5 Post-consolidation routing resolution**

A merged module inherits a catchment area in feature space that is wider and blurrier than any of its constituents. Since routing accuracy is the system's accuracy ceiling, every merge lowers that ceiling slightly, and a blurrier prototype attracts more misroutes, which under plasticity means corruption rather than mere misclassification.

Two responses. First, merged modules retain **multiple centroids** rather than a single prototype, preserving the shape of their territory. Second, mean inter-prototype distance is tracked before and after each consolidation as an explicit metric (Section 7.4). The relevant quantity is memory saved per unit of routing resolution lost, not memory saved alone.

### **6.6 A note on frequency of use**

If the primary hypothesis holds, the capacity threshold will be reached rarely and the Dream Phase will seldom fire. This is the correct relationship between the two mechanisms rather than a defect. Front-end reduction is primary; back-end consolidation is the backstop. A finding that consolidation adds little once spawn-rate reduction is applied supports the main thesis.

---

## **7\. Experimental design**

### **7.1 Consolidation operator**

Asymmetric merge (slow merged, fast retained) · uniform weight merge · anchor-set distillation · generative distillation · no consolidation.

Distillation variants are retained specifically to test whether the hollow-student concern is real, and to cover the case where ancestry has diverged too far for merging.

### **7.2 Nesting depth**

1, 2, and 3 levels. This is the headline curve for the nesting component.

### **7.3 Update-frequency implementation**

The specification gap that currently blocks implementation. Candidates:

* optimizer step applied every *k* steps with gradient accumulation  
* per-layer learning-rate ratios  
* EMA coupling between levels  
* each level optimizing a distinct objective over the level below's gradient stream

Only the last is architecturally non-trivial. Gradient accumulation approximates a smaller learning rate with a larger batch; per-layer rates are layer-wise learning-rate decay. **The mandatory control is a flat network with a per-layer learning-rate schedule matched to the nesting ratios.** If nesting does not beat it, the result is a schedule, not an architecture. This control runs first, before the full system is built.

### **7.4 Frequency ratios and ordering**

(1,4,16) versus (1,16,256), plus **reversed ordering** with fast-deep rather than slow-deep. If both orderings help equally, the effect is that some layers move less, not that multi-timescale structure matters.

### **7.5 Plasticity**

Plastic versus frozen offspring, holding everything else constant. Plasticity is an independent spawn-rate reducer, and the experiment must separate its contribution from nesting's or the result is uninterpretable.

### **7.6 Routing**

Prototype versus learned keys. Single versus multi-centroid prototypes. Abstain threshold sensitivity.

### **7.7 Ancestry threshold**

How many generations of independent training can separate two modules before weight merging degrades. Determines when the distillation fallback is needed.

---

## **8\. Metrics**

**Primary**

* Spawn rate: modules per task, over 20+ task streams

**Retention and capability**

* Average accuracy across all tasks  
* Forward transfer  
* Post-consolidation retention, per task, before versus after each Dream Phase  
* Plasticity: accuracy on each new task as sequence length grows

**Routing**

* Routing accuracy as a function of sequence length  
* Misroute-to-corruption rate: competence drop attributable to misrouted training  
* Mean inter-prototype distance before and after each consolidation

**Structural**

* Level-wise drift in merged modules: do slow levels still move less than fast levels, or has nesting become cosmetic  
* Time-to-saturation for merged versus freshly spawned modules: does consolidation quietly raise spawn rate

**Cost**

* Parameters, VRAM, inference FLOPs, wall-clock, including consolidation overhead

All results over at least five seeds with variance reported.

---

## **9\. Baselines**

* Parameter-matched single network with a byte-matched replay buffer  
* Flat network with per-layer learning-rate decay (the deflationary control)  
* Frozen-offspring version of this same system  
* PNN, DEN, CN-DPM, compared on expansion rate  
* EWC, LwF, DER++, ER, GDumb for context

---

## **10\. Scope and roadmap**

**Benchmarks.** Split-CIFAR-100 class-IL with a small ResNet, 20+ tasks, plus a task-free blurry-boundary stream where boundaries are not clean. Split-MNIST and Permuted-MNIST are used for debugging only; both are saturated and the multi-timescale hypothesis has no feature hierarchy to exploit in a small MLP. Class-incremental versus task-incremental setting is stated explicitly in all results.

**Implementation.** Avalanche plugin from the outset, so baselines, benchmarks, and metrics are inherited and results are directly comparable.

### **Phases**

**Phase 0 — the deflationary control.** Flat network with per-layer learning-rate schedule versus nested learner, matched on parameters, total steps, and effective learning rate. Roughly one week. Determines whether the rest is worth building.

**Phase 1 — core system.** Nested learner, spawn trigger, warm-start cloning, control plane, prototype routing with abstain, anchor sets. No consolidation. Repository public from the start.

**Phase 2 — the primary experiment.** Spawn-rate curves against all baselines over long streams. Nesting depth and frequency-ratio sweeps. Plastic versus frozen ablation. This is the experiment the project exists for.

**Phase 3 — Dream Phase.** Consolidation operators compared. Evaluated as an add-on: how much additional memory does it buy once front-end reduction is applied, and at what cost.

**Phase 4 — write-up.** Target CoLLAs. Negative results reported.

### **Out of scope**

Cross-offspring compositionality · cache hierarchies · sparse and orthogonal representations as an additional spawn-rate reducer · adapters on a pretrained backbone.

The last is the step that would make the work relevant to current practice and is the natural follow-up, but it comes after the core result.

---

## **11\. Open questions**

* Which implementation of "different update frequency" is used, and whether any of them survives the per-layer learning-rate control  
* How far ancestry can diverge before weight merging fails  
* Whether a merged module retains functional nesting or only its shape  
* Whether prototypes should be computed in input space or in a learned representation that itself drifts  
* Whether offspring should ever hand control back to the main learner, or only ever accumulate

---

## **12\. Risks**

**Routing accuracy is the system's ceiling.** It will consume most of the engineering effort, and it degrades under exactly the conditions the project is designed to explore. The abstain gate is built before anything else.

**Two independent spawn-rate reducers.** Nesting and plasticity both reduce expansion. The ablation must separate them or the headline result is uninterpretable.

**Consolidation may work against the thesis.** If merged modules saturate faster than fresh ones, the Dream Phase raises spawn rate while appearing to lower memory. Time-to-saturation is the check.

**The nesting effect may not survive the control.** Phase 0 exists to find this out cheaply. A clean negative result on whether multi-timescale nesting reduces expansion rate in small-scale vision continual learning is a publishable outcome and is planned for as such.

## **13\. Research**

* Rusu et al. (2016), *Progressive Neural Networks* — the isolation baseline and the origin of the proliferation problem  
* Aljundi et al. (2019), *Task-Free Continual Learning* — loss-plateau boundary detection, CVPR  
* Lee et al. (2020), *A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning* — principled expansion, ICLR  
* Behrouz et al. (2025), *Nested Learning* / HOPE, NeurIPS — your multi-timescale mechanism, including the Hope-1/2/3 depth ablation  
* van de Ven & Tolias (2019), *Three Scenarios for Continual Learning* — the task-IL vs domain-IL vs class-IL taxonomy you must state explicitly  
* Kermiche (2026), *Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery* — arXiv 2604.14375, your nearest neighbour

**Expansion and isolation**

* Yoon et al. (2018), *Lifelong Learning with Dynamically Expandable Networks*  
* Mallya & Lazebnik (2018), *PackNet*  
* Aljundi et al. (2017), *Expert Gate: Lifelong Learning with a Network of Experts* — reconstruction-based routing  
* Hung et al. (2019), *Compacting, Picking and Growing* (CPG) — pruning before expanding  
* Ramesh & Chaudhari (2022), *Model Zoo: A Growing Brain That Learns Continually*

**Multi-timescale and complementary learning systems**

* Hinton & Plaut (1987), *Using Fast Weights to Deblur Old Memories* — the original slow/fast synapse  
* McClelland, McNaughton & O'Reilly (1995), *Why There Are Complementary Learning Systems in the Hippocampus and Neocortex*  
* Arani et al. (2022), *Learning Fast, Learning Slow: A Complementary Learning System Approach* (CLS-ER), ICLR — plastic and stable models at different EMA rates  
* Pham et al. (2021), *DualNet: Continual Learning, Fast and Slow*, NeurIPS

**Plastic experts and routing drift** — directly relevant to your non-frozen offspring

* *SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning* — arXiv 2602.01990, names and measures router drift vs expert drift  
* *LLaVA-DyMoE: Dynamic MoE with Drift-Aware Token Assignment* — arXiv 2603.27481  
* *PASs-MoE* — arXiv 2601.13020, co-drift between router and experts  
* Lu et al., *MoRAM: Little by Little* — arXiv 2506.21035, expert granularity and routing confusion

**Consolidation: replay, distillation, merging**

* Shin et al. (2017), *Continual Learning with Deep Generative Replay*  
* van de Ven, Siegelmann & Tolias (2020), *Brain-Inspired Replay*, Nature Communications — why pixel-space generative replay fails to scale and latent replay fixes it  
* Li & Hoiem (2017), *Learning without Forgetting*  
* Wortsman et al. (2022), *Model Soups* — the linear-connectivity result your warm-start merging depends on  
* Ilharco et al. (2023), *Editing Models with Task Arithmetic*  
* Yadav et al. (2023), *TIES-Merging* — handles sign conflicts, relevant to your asymmetric merge

**Evaluation and critique** — read before you finalize your protocol

* Farquhar & Gal (2018), *Towards Robust Evaluations of Continual Learning* — why Split-MNIST and Permuted-MNIST mislead  
* Prabhu et al. (2020), *GDumb: A Simple Approach that Questions Our Progress*, ECCV — the humbling baseline  
* Davari et al. (2022), *Probing Representation Forgetting*, CVPR — accuracy misleads about what's actually retained  
* Chaudhry et al. (2018), *Riemannian Walk* — the standard forgetting and intransigence metrics

**Biological consolidation**

* Tadros et al. (2022), *Sleep-Like Unsupervised Replay Reduces Catastrophic Forgetting*, Nature Communications  
* Tononi & Cirelli (2014), *Sleep and the Price of Plasticity* — synaptic homeostasis  
* Teyler & Rudy (2007), *The Hippocampal Indexing Theory* — your key-value routing, biologically

**Tooling**

* Lomonaco et al. (2021), *Avalanche: An End-to-End Library for Continual Learning*, CVPR workshops  
* Boschini et al., *Mammoth* — the DER/DER++ codebase, useful for baselines

**Primary — spawn rate over long streams**

The headline metric needs many increments, which rules out most standard benchmarks.

* **CORe50, NICv2-391 protocol.** 391 incremental batches, 50 objects across 11 sessions with varying backgrounds and lighting. This is the best fit you'll find for a spawn-rate study — almost nothing else gives you hundreds of increments with natural domain variation. Built into Avalanche.  
* **Split-CIFAR-100**, 20 tasks × 5 classes. Your workhorse for class-IL. Standard, every baseline reports on it, fast to iterate on.  
* **Split-Tiny-ImageNet**, 40 tasks × 5 classes. Doubles the sequence length at moderate cost.

**Task-free and blurry boundaries** — where autonomous spawning actually gets tested

* **i-Blurry** (Koh et al., ICLR 2022). Classes overlap across increments rather than partitioning cleanly, so there is no boundary to detect. This is the honest test of your plateau trigger; on clean Split-CIFAR the trigger is trivially correct and proves nothing.  
* **CLEAR** (Lin et al., NeurIPS 2021 Datasets track). Ten years of real image data with natural temporal drift. No synthetic task boundaries at all.  
* **Stream-51**. Temporally correlated video frames, novelty detection built into the protocol.

**Debugging only**

* Split-MNIST and Permuted-MNIST. Use them to verify the plumbing runs, then stop. Permuted-MNIST does have one legitimate use for you: it scales to 100+ tasks cheaply, so it's usable for a first spawn-rate curve before you commit compute to CIFAR.

**The one you'll probably have to build**

Your consolidation logic makes assumptions no standard benchmark tests. Clustering, asymmetric merging, and outlier immunity all depend on the stream having a **known redundancy structure** — some groups of related tasks that should merge, and some genuine isolates that should be protected. Split-CIFAR-100 gives you 20 arbitrary class groupings with no ground truth about which ones ought to consolidate.

Two ways to get it:

**Semantic grouping of CIFAR-100.** The dataset has 20 built-in superclasses. Construct tasks so that several fall within one superclass (aquatic mammals, then fish, then reptiles — related, should merge) and others are semantically isolated. Now you have ground truth for whether your prototype clustering finds the right groups.

**A heterogeneous mixed stream.** The "5-datasets" benchmark used by HAT and DualPrompt — CIFAR-10, MNIST, SVHN, Fashion-MNIST, notMNIST — gives you genuinely non-overlapping domains. Extend it: several related fine-grained tasks that should cluster, plus one or two alien domains that should trigger outlier immunity. This is the cleanest way to test that your protection threshold does what you think.

Without a stream like this, you can measure that consolidation happened but not whether it made the *right* decisions.

**Ancestry threshold**

The question of how far apart two modules can be spawned and still merge cleanly needs a stream where you control the gap. Easiest done by interleaving: task A, then N unrelated tasks, then task A′ from the same domain. Sweep N. Constructible from CIFAR-100 superclasses.

**Access**

Avalanche ships CORe50, Split-CIFAR-100, Split-Tiny-ImageNet, CLEAR, Stream-51, and the MNIST variants as benchmark generators, and its `nc_benchmark` / `ni_benchmark` helpers will let you define the custom streams above without writing loaders. That's the main practical argument for building inside it.

**Compute note**

CORe50 NICv2-391 with 391 increments and a full ablation grid is not a laptop experiment. Plan the sweeps on Split-CIFAR-100, then run the long-stream confirmation on CORe50 once for the configurations that survive.

