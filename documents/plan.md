# **Autonomous Dynamic-Nesting Networks — Four-Stage Execution Plan**

Companion to the proposal. This document covers what gets built, where the code and data come from, what gets measured, and what decision each stage produces.

Assumed horizon: two semesters, roughly 32–36 working weeks. Each stage ends in a **gate** — a decision that determines whether the next stage proceeds as written or is revised.

---

## **Stage 0 — Setup (Weeks 1–2)**

Not a research stage. Do not skip it; every later stage assumes it is done.

### **Environment**

python 3.10  
torch 2.x \+ torchvision (CUDA build matching your GPU)  
avalanche-lib  
wandb            \# experiment tracking, free academic tier  
hydra-core       \# config management for the ablation grid  
scikit-learn     \# clustering, prototype utilities  
faiss-cpu        \# prototype nearest-neighbour search, needed from Stage 2

Pin versions in `requirements.txt` on day one. Avalanche's API has moved between releases and mid-project upgrades will cost you a week.

### **Code sources**

| Purpose | Repository |
| ----- | ----- |
| Framework, benchmarks, metrics | `ContinualAI/avalanche` |
| Rehearsal baselines (DER, DER++, ER) | `aimagelab/mammoth` |
| Class-IL expansion baselines (DER, FOSTER, MEMO) | `LAMDA-CL/PyCIL` |
| Closest competitor, expansion-rate comparison | `huiyiwang01/SEMA-CL` |
| Nested Learning reference (multimodal, for the non-trivial nesting variant) | `Yangl1nFeng/MoNet` |
| Task-free blurry streams | `naver-ai/i-Blurry` |

Verify each URL before relying on it. PyCIL matters more than it looks: Avalanche is weak on expansion-based methods, and PyCIL is where working DER/FOSTER/MEMO implementations live.

### **Data**

| Dataset | Source | Notes |
| ----- | ----- | ----- |
| CIFAR-100 | torchvision, auto-download | Primary workhorse. \~170 MB |
| Tiny-ImageNet | Stanford CS231n release | \~240 MB |
| MNIST / Fashion-MNIST / SVHN / notMNIST | torchvision | Debugging \+ 5-datasets stream |
| CORe50 | `vlomonaco.github.io/core50` | Download the **pre-extracted feature** version first; full images are \~10 GB |
| CLEAR | `clear-benchmark.github.io` | Only if Stage 3 has compute headroom |

### **Compute budget**

Establish this in week 1, because the ablation grid in Stage 3 is sized by it.

Reference cost: ResNet-18 on Split-CIFAR-100, 20 tasks, single run ≈ 1–2 GPU-hours on a T4. Five seeds × a 12-configuration grid ≈ 60–240 GPU-hours. That is the bulk of your compute and it must be planned, not discovered.

If you are on free tier (Kaggle ≈ 30 GPU-hrs/week on P100/T4, Colab similar), you have roughly 120 GPU-hrs/month across two accounts. Workable, but it means Stage 3's grid runs on **reduced ResNet-18 at 32×32 with 3 seeds**, and full 5-seed runs happen only for the final configurations. Secure university lab GPU access in week 1 if it exists.

### **Repository conventions**

Public from day one. `configs/` with Hydra, `src/` for the system, `experiments/` for run scripts, `results/` for CSV outputs, everything seeded and logged to W\&B. Every number in the final report should be regenerable from one command.

**Gate 0:** a baseline EWC run on Split-CIFAR-100 completes end to end in Avalanche and logs to W\&B. Nothing else is built until this works.

---

## **Stage 1 — The Deflationary Control (Weeks 3–10)**

**Purpose.** Determine whether multi-timescale nesting is an architecture or a learning-rate schedule. This stage can kill the project cheaply, which is why it comes first.

### **1.1 Resolve the update-frequency specification (Weeks 3–4)**

The blocking open question from proposal §7.3. Implement all four candidates as swappable strategies behind one interface:

* **A — Periodic stepping.** Level *i* accumulates gradients and applies its optimizer step every `k_i` steps.  
* **B — Per-layer learning rates.** Level *i* trains at `lr × r_i`. Implemented as a param-group config.  
* **C — EMA coupling.** Fast level trains normally; slower levels are exponential moving averages of it at rates `τ_i`. The CLS-ER mechanism, generalized to three levels.  
* **D — Nested objectives.** Level *i+1* optimizes over the gradient stream of level *i* rather than over the data directly. The only architecturally non-trivial option; MoNet's DMSM block is a working reference implementation.

Build all four. They share the same `NestedLearner` module and differ only in the update rule.

### **1.2 The control experiment (Weeks 5–8)**

Sequential training, no spawning, no routing, no freezing. This is the only configuration in which forgetting is free to occur, and therefore the only one in which reducing it means anything.

**Benchmark:** Split-CIFAR-100, 20 tasks × 5 classes, class-IL, ResNet-18 (reduced, 32×32 input variant used by Mammoth and PyCIL).

**Arms:**

| Arm | Description |
| ----- | ----- |
| Flat | Standard ResNet-18, uniform LR |
| Flat \+ LLRD | Per-layer learning-rate decay matched to nesting ratios — **the control that decides the project** |
| Nested-A / B / C / D | Each update-frequency implementation |
| Nested-2 level | Depth ablation |
| Nested-3, reversed | Fast-deep instead of slow-deep |

**Matched across all arms:** parameter count, total gradient steps, effective learning rate, wall-clock, data order, seeds.

**Measured:** average accuracy, forgetting, per-layer CKA drift between task checkpoints, linear-probe accuracy on old tasks from each level's activations, gradient cosine between consecutive tasks per level.

The probe and CKA measurements matter as much as the accuracy. Your claim is that slow levels drift less while staying useful. Accuracy alone cannot show that, and published work has found that representations often retain task information even when accuracy collapses.

### **1.3 Baseline reproduction (Weeks 5–10, parallel)**

While the control runs, reproduce and log on the same benchmark: EWC, LwF, ER, DER++, GDumb, and a parameter-matched single network with a byte-matched replay buffer. These come almost free from Avalanche and Mammoth and give you the context column for every later table.

**Deliverables:** control-experiment report, four working nesting implementations, baseline table, a decision on which update rule to carry forward.

**Gate 1\.** Does any nesting variant beat Flat \+ LLRD on forgetting at matched everything?

* **Yes** → carry the winning variant into Stage 2\.  
* **No** → the mechanism is a schedule. Do not proceed to the full system as written. Two options: pivot to reporting this as a negative result with the probe analysis as the substance (a legitimate FYP outcome, and honest), or switch the primary claim to plasticity-driven spawn reduction, which does not depend on nesting. Decide with your supervisor within a week of the result.

---

## **Stage 2 — Core System (Weeks 11–20)**

**Purpose.** Build the spawning, routing, and control-plane machinery. No consolidation. Correctness before scale.

Build order matters — the abstain gate comes before anything that depends on it.

### **2.1 Control plane (Weeks 11–12)**

A plain Python class. No gradients anywhere in it.

Per module: prototype set (list of centroids), running covariance (optional, for Mahalanobis), firing count, running competence score, spawn generation, last-fired timestamp, anchor-set handle.

Queries it must answer: nearest prototype to an input, pairwise prototype distances, prototype-to-nearest-neighbour distance per module (for outlier immunity), ancestry distance between two modules, whether a module is stale, whether the main learner is saturated.

Use FAISS for nearest-prototype once module count exceeds \~20; a linear scan is fine below that but becomes the inference bottleneck on long streams.

### **2.2 Routing with abstain (Weeks 12–13)**

Distance to nearest prototype; route if within threshold, else abstain to the main learner. Calibrate the threshold on a held-out slice rather than hand-tuning it.

Instrument from the start: log every routing decision with the ground-truth task ID (available in benchmarks even though the system does not use it) so routing accuracy and misroute rate are measurable later without re-running.

### **2.3 Spawn trigger and warm-start cloning (Weeks 13–15)**

Trigger: competence plateau on the main learner under sustained unrouted volume. Implement plateau detection with an explicit window and tolerance, both exposed as config.

On spawn: deep-copy the main learner, register it, compute its prototype from its recent training data, initialize counters, record spawn generation. The main learner continues unchanged.

**Guard against pathological spawning.** Add a minimum-samples-since-last-spawn floor. Without it, an early misconfiguration produces a module per batch and you will lose days diagnosing it.

### **2.4 Plastic offspring, anchor sets, re-anchoring (Weeks 15–17)**

Routed batches train their assigned module. Each module holds an anchor set (start at 50 samples, class-balanced, reservoir-sampled from its own routed data) replayed alongside incoming batches at a fixed ratio.

Re-anchoring: after every *N* training blocks, recompute prototypes from held-out anchor samples.

### **2.5 Integration testing (Weeks 17–20)**

**Split-MNIST first**, 5 tasks, MLP. This is a plumbing test only — it verifies that spawning fires, routing routes, anchors replay, and nothing crashes. Expect trivially good numbers and read nothing into them.

**Then Split-CIFAR-100**, 20 tasks. First real spawn-rate reading.

Then the ablation the proposal requires: **plastic versus frozen offspring**, everything else held constant. This isolates plasticity's contribution to spawn reduction from nesting's, and it must be run before Stage 3 or the headline result is uninterpretable.

**Gate 2\.** The system runs 20 tasks end to end, spawn count is a sensible number (not 1, not 20), routing accuracy above 70% at task 20, and the plastic-versus-frozen delta is measured.

---

## **Stage 3 — The Primary Experiment (Weeks 21–28)**

**Purpose.** Produce the headline result: spawn rate as a function of nesting depth, against every relevant baseline, over long streams.

### **3.1 Long-stream curves (Weeks 21–24)**

Run in this order, cheapest first:

1. **Permuted-MNIST, 100 tasks.** Cheap, and the only way to get a 100-point spawn-rate curve on a student budget. Its weaknesses as an accuracy benchmark do not disqualify it as an expansion-rate probe, but state that limitation explicitly in the write-up.  
2. **Split-CIFAR-100, 20 tasks.** The comparable benchmark. Full ablation grid runs here.  
3. **Split-Tiny-ImageNet, 40 tasks.** Sequence-length confirmation.  
4. **CORe50 NICv2-391.** Run **once**, for the two or three configurations that survive. Use pre-extracted features. This is the long-stream confirmation, not an exploration surface.

### **3.2 The ablation grid (Weeks 22–26)**

On Split-CIFAR-100:

* Nesting depth: 1, 2, 3 levels — the headline curve  
* Frequency ratios: (1,4,16) versus (1,16,256)  
* Ordering: slow-deep versus fast-deep  
* Plasticity: plastic versus frozen  
* Routing: prototype versus learned keys; single versus multi-centroid  
* Abstain threshold: three settings spanning conservative to permissive

Three seeds during exploration, five seeds for the final configurations. Every run logged with its full config hash.

### **3.3 Expansion-rate baselines (Weeks 24–27)**

The comparison that decides whether you have a paper. These are not retention baselines — they are expansion baselines, and one of them is your real competitor.

* **PNN** — one column per task, the linear-growth reference  
* **DEN** — selective retraining with growth  
* **CN-DPM** — Dirichlet-process expansion  
* **SEMA** — already reports sub-linear expansion; the comparison that matters most

SEMA runs adapters on a frozen pretrained ViT while you run from-scratch learners, so the comparison is not apples to apples. Resolve it one of two ways: port SEMA's expansion policy into your backbone (preferred — isolates the policy from the backbone), or report both settings and discuss the gap honestly. Budget two weeks for this; it is the most likely schedule overrun in the project.

### **3.4 Task-free validation (Weeks 26–28)**

**i-Blurry** on CIFAR-100. This is where autonomous spawning is actually tested — on clean Split-CIFAR the boundaries are trivially detectable and the trigger proves nothing. Report spawn rate and routing accuracy under blurred boundaries alongside the clean-boundary numbers.

**Gate 3\.** Does nesting depth reduce spawn rate, and does the effect hold against SEMA? A negative here is still a publishable result, but it changes what Stage 4 writes up.

---

## **Stage 4 — Dream Phase, Efficiency, Write-up (Weeks 29–36)**

### **4.1 Build the evaluation stream (Week 29\)**

No standard benchmark tests whether consolidation makes the *right* decisions. Build one.

**Semantic CIFAR-100 stream.** Use the dataset's 20 built-in superclasses. Construct a task sequence where several consecutive tasks fall inside one superclass (aquatic mammals → fish → reptiles: related, should cluster and merge) and one or two are semantically isolated (should trigger outlier immunity). You now have ground truth for clustering correctness and protection correctness, neither of which Split-CIFAR-100 provides.

**Ancestry-gap stream.** Task A, then *N* unrelated tasks, then task A′ from A's domain. Sweep *N* ∈ {2, 5, 10, 15}. This measures how far ancestry can diverge before weight merging degrades, which determines when the distillation fallback is needed.

Both are constructible with Avalanche's `nc_benchmark` helper without writing dataset loaders.

### **4.2 Consolidation operators (Weeks 29–32)**

Compare on the semantic stream:

* Asymmetric merge (slow levels merged, fast retained as per-domain heads) — **your contribution**  
* Uniform weight merge  
* TIES-merging (handles sign conflicts; cite Yadav et al.)  
* Anchor-set distillation  
* Generative distillation  
* No consolidation

Measured per operator: memory saved, post-consolidation retention on pooled anchor sets, **mean inter-prototype distance before and after** (routing resolution lost), **time-to-saturation of merged versus fresh modules** (does consolidation secretly raise spawn rate), and **level-wise drift in merged modules** (is the nesting functional or only structural).

The last two are the checks that the Dream Phase is not working against the thesis. Do not skip them because the accuracy numbers look fine.

Also verify clustering and protection correctness against the stream's ground truth: did it merge what should merge, and protect what should be protected?

### **4.3 Compute and memory mitigations (Weeks 31–33)**

Measured, not assumed. Each mitigation gets a before/after number.

**Inference compute.** Routing cost grows with module count. Mitigations: FAISS indexing over prototypes instead of linear scan; confidence-gated bypass, where the main learner answers directly when its own predictive entropy is low and the router is skipped entirely. Measure inference FLOPs and latency as a function of module count, with and without each.

**Memory.** Anchor-set size sweep (25 / 50 / 100 samples per module) against retention — find the knee. Total system footprint reported as: module parameters \+ anchor bytes \+ prototype table, compared against the byte-matched replay baseline from Stage 1\. This is the honest accounting; parameter count alone understates your cost.

**Training compute.** Mixed precision, gradient accumulation for the periodic-stepping variant, consolidation overhead measured as wall-clock per Dream Phase and amortized per task.

Report a single combined figure: **cost per unit of retention**, plotted against the baselines. Zero forgetting at unbounded cost is trivially achievable, so the interesting axis is what you paid.

### **4.4 Write-up (Weeks 33–36)**

* Full results, five seeds, variance reported throughout  
* Class-IL versus task-IL stated explicitly on every table  
* Negative results reported, including whichever gates failed  
* Related work positioned honestly against SEMA, MADE-IT, HAM, CP-MoE, HOPE. Your novelty is narrow and specific: nested learners as the expanding unit, expansion rate as a function of module capacity rather than routing policy, and level-asymmetric consolidation. Claim those and cite the rest.  
* Target CoLLAs. Public repository with one-command reproduction.

---

## **Schedule summary**

| Stage | Weeks | Output | Gate |
| ----- | ----- | ----- | ----- |
| 0 — Setup | 1–2 | Environment, compute plan, repo | Baseline run completes |
| 1 — Control | 3–10 | Nesting implementations, baselines | Does nesting beat LLRD? |
| 2 — System | 11–20 | Full architecture, no consolidation | 20 tasks, routing above 70% |
| 3 — Primary | 21–28 | Spawn-rate curves vs. all baselines | Does depth reduce spawn rate vs. SEMA? |
| 4 — Dream \+ write-up | 29–36 | Consolidation comparison, cost accounting, paper | Submission |

## **Slippage plan**

If you are behind at week 20, cut in this order:

1. CORe50 and CLEAR (keep Split-CIFAR-100 and Permuted-MNIST)  
2. The ancestry-gap sweep (report the ancestry question as open)  
3. Generative distillation (anchor-set distillation covers the comparison)  
4. Stage 4.3 mitigations beyond the basic footprint accounting

Do not cut: the Stage 1 control, the plastic-versus-frozen ablation, the SEMA comparison, or the time-to-saturation check. Those four are what make the result interpretable.

