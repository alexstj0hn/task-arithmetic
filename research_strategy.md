# NeurIPS Research Strategy: Generative AI for Protein Design

**Target:** NeurIPS 2026 (deadline ~May 2026) or ICML 2026 / ICLR 2027  
**Researcher profile:** Solo, strong ML + compbio, academic compute, no wet lab  
**Date:** February 2026

---

## STEP 1 — MAP THE FIELD

### 1.1 Dominant Paradigms

**Protein Language Models (PLMs):** ESM-2 (Meta, 2022) remains the workhorse embedding model. ESM-3 and ESM-C are recent multimodal successors. ProGen3, xTrimoPGLM push scale. Key finding from ProteinGym v1.3 (May 2025): **scaling PLMs beyond 1–4B parameters does not improve zero-shot fitness prediction.** Multimodal models combining MSAs + structure (VenusREM, S3F-MSA) consistently outperform even the largest sequence-only PLMs.

**Diffusion models for proteins:** RFdiffusion (Baker lab), Chroma (Generate Biomedicines), FrameDiff, FoldFlow, and Genie2 dominate backbone generation. DiffAb and dyMEAN target antibodies specifically. RFantibody and JAM-2 are the newest antibody diffusion models.

**Inverse folding:** ProteinMPNN remains the default. ESM-IF and LM-Design offer alternatives. All are evaluated primarily by sequence recovery — a metric with weak experimental correlation.

**De novo antibody design:** Chai-2 reported 16% experimental hit rate for de novo antibodies (NeurIPS 2025). RFantibody (Baker lab) and JAM-2 (Nabla Bio) are competitive. This subfield went from "5 years away" to "working today" in 2025.

**Few-shot fitness prediction:** PRIMO (Dec 2025) uses in-context learning + test-time training. Metalic (ICLR 2025) introduced meta-learning over DMS tasks. FSFP (Nat Comms 2024) combines meta-transfer learning with learning-to-rank. EVOLVEpro (Science 2024) uses few-shot ridge regression on PLM embeddings for iterative directed evolution.

**Structure prediction:** AlphaFold3 and OpenFold-Multimer for complexes. ESMFold for fast single-chain. Pearl (Genesis) for protein-ligand cofolding. This area is dominated by industry labs and is largely saturated for a solo academic.

### 1.2 Over-Explored Areas

- **Yet another PLM variant with marginal ProteinGym improvement.** The leaderboard has 90+ baselines. Incremental gains are insufficient for NeurIPS.
- **Unconditional protein backbone generation.** Multiple diffusion models already exist; evaluation remains the bottleneck, not generation.
- **Single-property fitness prediction.** Zero-shot Spearman correlation on ProteinGym is the most commoditized metric in the field.
- **Inverse folding on CATH benchmarks.** Sequence recovery improvements are saturating and the metric itself is problematic.
- **Structure prediction.** Cannot compete with DeepMind/Meta resources.

### 1.3 Under-Explored but Important Areas

- **Multi-property design with formal guarantees.** Everyone does generate-then-filter. Almost nobody integrates multiple objectives into the generative process with provable trade-off control.
- **Calibration and uncertainty in protein fitness models.** Greenman et al. (PLoS Comp Bio 2025) benchmarked UQ methods and found no single method dominates; uncertainty-based BO often fails to beat greedy sampling. This is an open wound.
- **Cross-assay transfer learning.** DMS assays measure different properties of different proteins. Learning transferable structure across these tasks is nascent (PRIMO and Metalic are first attempts, both have serious data leakage and evaluation issues).
- **Model merging and task arithmetic for PLMs.** Completely unexplored. A major technique in NLP/vision with zero application to protein models.
- **Evaluation of the full generative pipeline.** The generate → score → filter pipeline has compounding, unmeasured errors. No benchmark captures this.
- **Developability as a design objective (not a filter).** Current pipelines generate candidates optimized for binding/structure then post-hoc filter for developability. Nobody integrates it into generation.
- **Distribution shift between natural evolution and therapeutic design.** PLMs are trained on evolutionary data; therapeutic proteins (engineered antibodies, fusion proteins) occupy a different distribution. Failure modes are uncharacterized.

### 1.4 Contradictions in Literature

1. **PLM likelihood ≠ fitness.** Everyone uses masked marginal likelihood as a proxy for fitness, but Yale researchers (2025) demonstrated that in-context patterns (repeated motifs) can produce anomalously high likelihoods regardless of biological plausibility. This fundamentally undermines zero-shot fitness assumptions.

2. **Scaling doesn't help.** ProteinGym v1.3 shows diminishing returns past 1–4B parameters for sequence-only models, yet the field continues to build larger PLMs. The useful inductive bias appears to come from MSAs and structure, not from scale.

3. **Sequence recovery is the wrong metric.** Inverse folding papers report 40–60% recovery as if it were success, but high recovery doesn't predict experimental viability. Some of the best experimentally-validated designs have low sequence recovery.

4. **UQ doesn't help Bayesian optimization.** Greenman et al. showed that sophisticated uncertainty estimates frequently fail to improve optimization over greedy approaches — suggesting our UQ methods are miscalibrated in ways that compound rather than correct during optimization.

5. **Circular evaluation in generative pipelines.** RFdiffusion generates backbones → ProteinMPNN designs sequences → AlphaFold2 predicts structure → compare to target. This is a closed loop of computational tools validating each other with no ground truth.

### 1.5 Methodological Weaknesses

- **No standard for evaluating multi-property design.** There is no ProteinGym equivalent for multi-objective problems.
- **Epistasis is poorly modelled.** PLMs with independent masked positions cannot capture higher-order mutational interactions critical for multi-mutant design.
- **Generative models produce mode-averaged structures.** They cannot represent conformational ensembles, allosteric effects, or flexibility.
- **Assay noise is ignored.** DMS measurements have variable quality (different dynamic ranges, different noise levels, different biases). Models treat all assays as equally reliable.
- **Train/test contamination is rampant.** PRIMO (Dec 2025) explicitly showed that Metalic's claimed performance was inflated by train-test overlap. This is likely endemic across many published results.

### 1.6 Dataset Landscape

| Dataset | Size | Scope | Status |
|---------|------|-------|--------|
| ProteinGym v1.3 | 217 DMS assays, ~2.7M mutations | Fitness prediction | Standard benchmark |
| FLIP | 3 assays, curated splits | Few-shot fitness | Limited scope |
| SAbDab | ~7,000 antibody structures | Antibody structure | Well-maintained |
| OAS | >1B antibody sequences | Antibody sequences | Unpaired limitation |
| SKEMPI v2 | ~7,000 binding ΔΔG measurements | Protein-protein binding | Small |
| MaveDB | Hundreds of DMS assays | Raw assay repository | Not curated |
| Therapeutic Antibody Profiler (TAP) | Developability rules | Antibody properties | Rule-based, no ML data |
| ATLAS | Antibody-antigen structures | Complex structures | Growing |

---

## STEP 2 — IDENTIFY RESEARCH GAPS

### High-Value Gaps (ordered by estimated impact)

**G1. No one has applied task arithmetic / model merging to protein language models.**  
Task arithmetic (Ilharco et al. 2023) and its extensions (TIES, DARE, TATR) have transformed multi-task composition in NLP and vision. For proteins, where different properties (stability, binding, expression, immunogenicity) are distinct tasks, the potential is enormous. Fine-tune ESM-2 on stability data, binding data, expression data separately → extract task vectors → compose arithmetically for multi-property models. **Nobody has tried this.** This is shocking given how straightforward the transfer is.

**G2. The full generative-predictive pipeline has unmeasured compounding errors.**  
When you chain diffusion backbone generation → inverse folding → structure re-prediction → property filtering, each step introduces systematic bias that propagates. The correlation between "predicted designability" and actual experimental success is unknown at scale. No benchmark exists for end-to-end pipeline evaluation.

**G3. Assay-specific biases corrupt fitness prediction benchmarks.**  
Different DMS assays use different selection pressures, enrichment protocols, and dynamic ranges. A mutation scored as "fit" in a growth assay may score differently in a binding assay for the same protein. Models trained/evaluated on these assays implicitly encode assay-specific biases that nobody disentangles from true biological signal.

**G4. Conformal prediction hasn't been applied to the multi-property design setting.**  
Fannjiang et al. (2022) addressed feedback covariate shift for single-property design. CalPro (Jan 2026) tackled structural uncertainty. But multi-objective design — where you need coverage guarantees across MULTIPLE properties simultaneously — remains unsolved. The statistical tools exist (Learn-then-Test, conformal risk control for sets) but haven't been connected to protein design.

**G5. PLM representations are topologically misaligned with fitness landscapes.**  
ESM-2 embeds sequences in a space shaped by evolutionary co-occurrence, not by functional fitness. The topology of this embedding space (connectivity, curvature, local geometry) differs from the topology of the actual fitness landscape. This misalignment is the underlying cause of many prediction failures, but nobody has systematically characterized or corrected it.

**G6. Missing benchmark for multi-property antibody design.**  
De novo antibody design requires simultaneously optimizing binding affinity, specificity, stability, expression, immunogenicity, and polyspecificity. There is no standardized benchmark that evaluates models across all these axes jointly. Without this, the field cannot distinguish genuine progress from cherry-picked results.

**G7. Cross-assay generalization is evaluated with contaminated splits.**  
Both Metalic and FSFP used data splits with protein-level overlap between train and test. PRIMO (Dec 2025) demonstrated this leads to inflated performance. The field needs rigorous cross-assay, cross-protein evaluation protocols, and the consequences of current contamination are underappreciated.

**G8. No theory for when evolutionary likelihood correlates with functional fitness.**  
Everyone assumes PLM likelihoods are informative for fitness. Sometimes they are (Spearman ~0.5–0.7). Sometimes they fail catastrophically (Spearman ~0.0–0.1). There is no theoretical framework predicting WHEN the correlation holds or fails, which would be enormously useful for practitioners deciding whether to trust zero-shot predictions.

**G9. Diffusion models for proteins don't provide meaningful uncertainty.**  
Protein diffusion models generate diverse samples, but the diversity of these samples doesn't correspond to meaningful uncertainty about design success. There's no calibration between "diffusion model thinks this region is uncertain" and "this region is actually difficult to design."

**G10. Developability prediction uses disconnected, non-transferable features.**  
Aggregation propensity, charge patches, hydrophobicity, viscosity — these are computed with separate tools, on separate scales, with separate failure modes. No unified representation captures developability as a learnable, multi-dimensional property. SAbDab + TAP rules + scattered experimental data exist but haven't been unified into an ML-ready dataset.

**G11. Test-time training for proteins lacks a principled approach.**  
PRIMO showed test-time training helps, but does it ad hoc. The theory of when and why TTT works for proteins (vs. when it overfits on few shots) is completely absent.

**G12. Multi-mutant fitness prediction remains far harder than single-mutant.**  
Most benchmarks focus on single-site mutations. Combinatorial (multi-mutant) fitness landscapes are exponentially larger and exhibit strong epistasis. Extrapolating from single-mutant to multi-mutant data is a major unsolved problem that limits practical protein engineering.

**G13. Sequence-structure-function triality is treated as a pipeline, not a joint model.**  
Models predict structure from sequence, then predict function from structure. The reverse paths (what sequence gives a function? what structure enables a function?) are poorly integrated. A unified generative model over the sequence-structure-function space would be transformative but computationally challenging.

**G14. Optimal transport hasn't been applied to protein fitness landscape alignment.**  
Different assays produce different fitness landscapes for different proteins. Aligning these landscapes (to enable transfer) is a natural application of optimal transport, but OT methods haven't been brought to this problem.

---

## STEP 3 — GENERATE RESEARCH DIRECTIONS

### Direction 1: Task Arithmetic for Protein Property Composition

**Problem:** How can we compose multiple protein properties (stability, binding, expression) by arithmetic operations on PLM weight vectors, without joint multi-task training?

**Why it matters:** Multi-property optimization is the central practical challenge in protein engineering. Current approaches require either expensive multi-task training or post-hoc Pareto filtering. Task arithmetic could enable zero-cost composition of arbitrary property combinations.

**Why now:** Task arithmetic is mature in NLP (2023–2025), PLMs like ESM-2 are standardized, ProteinGym provides property-specific DMS assays for evaluation, and PRIMO/Metalic have shown property-specific fine-tuning works.

**Novelty vs existing:** Task arithmetic and model merging are completely unexplored for protein models. No prior work exists. The closest is multi-task training of PLMs, which is computationally expensive and inflexible.

**Required datasets/tools:** ProteinGym DMS assays (stability, binding, expression subsets), ESM-2 (650M), LoRA fine-tuning infrastructure, standard GPU cluster.

**Difficulty:** 5/10  
**Time to strong paper:** 8–10 weeks  
**NeurIPS acceptance likelihood:** 40–50%

---

### Direction 2: Conformal Risk Control for Multi-Objective Protein Design

**Problem:** How do we provide finite-sample statistical guarantees on the false discovery rate when screening protein designs across multiple properties simultaneously?

**Why it matters:** Current pipelines generate thousands of candidates with no principled control over how many will fail. Wet lab validation is expensive. Controlling FDR across multiple properties (not just one) would transform design campaigns.

**Why now:** Angelopoulos et al.'s Learn-then-Test framework (2022), Fannjiang's feedback covariate shift work (2022), and CalPro (Jan 2026) provide the theoretical machinery. Multi-property DMS data exists in ProteinGym. The gap is the connection.

**Novelty:** Multi-property conformal risk control for protein design. Extends Fannjiang's single-property work to the multi-objective setting with formal guarantees on joint coverage.

**Required:** ProteinGym, FLIP, SKEMPI, Python (conformal prediction libraries), existing fitness predictors as base models.

**Difficulty:** 7/10  
**Time:** 10–12 weeks  
**NeurIPS likelihood:** 35–45%

---

### Direction 3: The Fitness-Embedding Misalignment Problem

**Problem:** When and why do PLM embedding geometries fail to reflect fitness landscape topology? Can we systematically diagnose and correct this?

**Why it matters:** Every downstream application of PLM embeddings (fitness prediction, guided generation, active learning) assumes that embedding proximity reflects functional similarity. When this fails, everything downstream fails silently.

**Why now:** Persistent homology / TDA tools are mature. ProteinGym v1.3 has 217 assays to systematically test misalignment patterns. The ProteinGym scaling wall result (big PLMs aren't better) demands explanation.

**Novelty:** First systematic topological analysis of PLM representation spaces vs. fitness landscapes. First correction method based on manifold alignment.

**Required:** ProteinGym, ESM-2 embeddings, TDA libraries (giotto-tda, ripser), alignment methods.

**Difficulty:** 8/10  
**Time:** 12–16 weeks  
**NeurIPS likelihood:** 30–40%

---

### Direction 4: Benchmarking the Generative-Predictive Pipeline Gap

**Problem:** What is the actual correlation between computational designability metrics and experimental protein success, and where in the generate→score→filter pipeline do errors compound?

**Why it matters:** The field is building on a foundation of circular validation. Quantifying where errors compound would redirect optimization efforts and calibrate practitioner expectations.

**Why now:** Enough experimental data exists from recent de novo design campaigns (Baker lab, Chai-2, Nabla Bio) to begin retrospective analysis. The community is ready for honest benchmarking (NeurIPS 2025 had explicit emphasis on reproducibility).

**Novelty:** First systematic analysis of error propagation through the protein design pipeline. Could serve as a "Dataset & Benchmarks" track submission.

**Required:** Published experimental results from design campaigns, published design pipelines (RFdiffusion + ProteinMPNN + AF2), public databases. Heavy curation work.

**Difficulty:** 6/10  
**Time:** 12–14 weeks  
**NeurIPS likelihood:** 25–35% (if main track), 50–60% (if D&B track)

---

### Direction 5: Optimal Transport for Cross-Assay Fitness Transfer

**Problem:** Can we use optimal transport to align fitness landscapes across different DMS assays, enabling transfer learning between proteins and properties?

**Why it matters:** Most protein engineering campaigns have <100 measurements. Transferring knowledge from the 217 ProteinGym assays to a new target protein could dramatically reduce experimental burden.

**Why now:** OT methods (Sinkhorn, neural OT) are computationally tractable. PRIMO/Metalic showed cross-assay learning is possible but used ad hoc methods. OT provides a principled geometric framework.

**Novelty:** First application of optimal transport to protein fitness landscape alignment. Principled framework vs. ad hoc meta-learning.

**Required:** ProteinGym, POT library, ESM-2 embeddings.

**Difficulty:** 7/10  
**Time:** 10–12 weeks  
**NeurIPS likelihood:** 30–40%

---

### Direction 6: Property-Conditioned Guidance for Protein Diffusion

**Problem:** Can we guide protein diffusion models to sample from the Pareto frontier of multiple properties simultaneously, without retraining the diffusion model?

**Why it matters:** Current diffusion models generate structures optimized for a single objective. Real design requires multi-objective Pareto-optimal sampling.

**Why now:** Classifier-free guidance and compositional diffusion are mature (2024–2025). Protein diffusion models (RFdiffusion, Chroma) have public weights. Multi-objective guidance for images exists.

**Novelty:** Pareto-optimal guidance for protein diffusion. Extends compositional diffusion to the multi-objective protein setting.

**Required:** FrameDiff or Chroma (open-source), property predictors, significant GPU compute for diffusion inference.

**Difficulty:** 8/10  
**Time:** 14–16 weeks  
**NeurIPS likelihood:** 35–45%

---

### Direction 7: Disentangling Assay Bias from Biological Signal in DMS Data

**Problem:** DMS assay measurements confound true biological fitness with assay-specific technical biases (selection pressure, enrichment noise, dynamic range effects). Can we learn to separate them?

**Why it matters:** If we could disentangle assay noise from biology, every downstream fitness predictor would improve. It would also enable more principled cross-assay comparison.

**Why now:** ProteinGym has multiple assays for the same protein (different labs, different assay types), providing natural controlled experiments. Causal disentanglement methods have matured (2023–2025).

**Novelty:** First principled approach to DMS assay bias correction using multiple measurements of the same protein.

**Required:** ProteinGym metadata (assay types, selection pressures), MaveDB raw counts where available.

**Difficulty:** 7/10  
**Time:** 10–12 weeks  
**NeurIPS likelihood:** 30–40%

---

### Direction 8: When Does Evolution Predict Function? A Theory of PLM-Fitness Correlation

**Problem:** Under what conditions does the likelihood assigned by an evolutionary model (PLM) correlate with functional fitness? Can we derive conditions under which this assumption holds or fails?

**Why it matters:** This is the foundational assumption of zero-shot fitness prediction. Understanding when it holds would tell practitioners when to trust PLM predictions and when to invest in supervised data.

**Why now:** Enough empirical evidence exists (217 assays in ProteinGym) to ground theoretical analysis. The PLM scaling wall result demands theoretical explanation.

**Novelty:** First theoretical framework characterizing the PLM-fitness correlation. Connects population genetics theory (nearly neutral theory, stabilizing selection) to representation learning.

**Required:** ProteinGym, population genetics theory, mathematical analysis.

**Difficulty:** 9/10  
**Time:** 16–20 weeks  
**NeurIPS likelihood:** 35–50% (high variance — could be a NeurIPS oral or a reject)

---

### Direction 9: Test-Time Training Dynamics for Protein Adaptation

**Problem:** PRIMO showed TTT helps protein fitness prediction but with no understanding of when/why. Can we characterize the loss landscape dynamics during TTT and predict when it helps vs. overfits?

**Why it matters:** TTT is the most promising direction for few-shot protein prediction, but without understanding its failure modes, practitioners can't use it reliably.

**Why now:** PRIMO (Dec 2025) provides the baseline system. The test-time training theory from vision (2024–2025) can be adapted.

**Novelty:** First analysis of TTT dynamics for protein fitness prediction. Practical guidelines for when TTT helps.

**Required:** ProteinGym, PRIMO codebase (if released), ESM-2.

**Difficulty:** 6/10  
**Time:** 8–10 weeks  
**NeurIPS likelihood:** 25–35%

---

### Direction 10: Epistasis-Aware Protein Language Models via Higher-Order Masking

**Problem:** Standard MLM training uses single-token masking, which cannot capture epistatic (higher-order) interactions between mutations. Can modified training objectives capture epistasis?

**Why it matters:** Multi-mutant design is the goal of protein engineering, but epistasis makes multi-mutant effects non-additive and unpredictable from single-mutant data.

**Why now:** Multi-mutant DMS data exists (albeit limited). Higher-order masking strategies have been explored in NLP (span masking) but not adapted for proteins with epistasis in mind.

**Novelty:** Epistasis-aware pre-training objectives for PLMs.

**Required:** ProteinGym multi-mutant data, ESM-2 fine-tuning, combinatorially complete DMS datasets.

**Difficulty:** 7/10  
**Time:** 12–14 weeks  
**NeurIPS likelihood:** 30–40%

---

### Direction 11: Unified Developability Representations via Multi-Task Contrastive Learning

**Problem:** Developability properties (aggregation, viscosity, clearance, expression, immunogenicity) are currently predicted by separate tools. Can we learn a unified representation?

**Why it matters:** Integrated developability assessment is the bottleneck between computational design and clinical candidates. Fragmented tools with incompatible scales prevent optimization.

**Why now:** SAbDab has grown large enough. Therapeutic antibody data from literature can be curated. Contrastive learning methods are mature.

**Novelty:** First unified embedding for antibody developability. First multi-task developability prediction model.

**Required:** SAbDab, TAP, scattered literature data, significant curation effort.

**Difficulty:** 7/10  
**Time:** 12–16 weeks  
**NeurIPS likelihood:** 25–35%

---

### Direction 12: Selective Prediction for Protein Fitness — Knowing When You Don't Know

**Problem:** Rather than predicting fitness for ALL mutations and hoping the model is right, can we build models that selectively abstain when uncertain and only predict when confident?

**Why it matters:** In practice, a model that says "I don't know" on 40% of mutations but is 95% reliable on the remaining 60% is more useful than a model that's 70% reliable on everything.

**Why now:** Selective prediction theory is mature (Geifman & El-Yaniv, RECO framework). Greenman et al. (2025) showed UQ doesn't help BO — but maybe selective prediction is the right framing instead.

**Novelty:** First application of selective prediction / learn-to-defer to protein fitness prediction. Reframes the UQ problem from "estimate uncertainty" to "know when to abstain."

**Required:** ProteinGym, existing fitness predictors, selective prediction infrastructure.

**Difficulty:** 5/10  
**Time:** 8–10 weeks  
**NeurIPS likelihood:** 30–40%

---

### Direction 13: Data Contamination Audit for Protein Fitness Benchmarks

**Problem:** PRIMO showed Metalic's performance was inflated by train-test overlap. How widespread is this problem across the 90+ models on the ProteinGym leaderboard?

**Why it matters:** If many published results are contaminated, the field's understanding of progress is distorted. A systematic audit would recalibrate the entire leaderboard.

**Why now:** PRIMO demonstrated the problem exists. The community needs a comprehensive audit.

**Novelty:** First systematic contamination audit of protein fitness prediction benchmarks.

**Required:** ProteinGym, training data specifications for major models, sequence identity analysis tools (MMseqs2).

**Difficulty:** 5/10  
**Time:** 6–8 weeks  
**NeurIPS likelihood:** 30% (main), 50% (D&B track)

---

### Direction 14: Foundation Model for Protein Dynamics via Flow Matching on MD Trajectories

**Problem:** All current protein models capture static snapshots. Can we build a flow-matching model over conformational ensembles?

**Why it matters:** Protein function depends on dynamics, not just the equilibrium structure.

**Why now:** AlphaFlow (ICML 2024) and Structure Language Models (ICLR 2025) are beginning this. Flow matching is more compute-efficient than diffusion.

**Novelty:** Limited — this is becoming crowded quickly.

**Required:** MD trajectory datasets, significant GPU compute.

**Difficulty:** 9/10  
**Time:** 16+ weeks  
**NeurIPS likelihood:** 25–35%

---

### Direction 15: Representation Surgery for Multi-Task PLM Merging

**Problem:** When merging property-specific PLMs, task vectors interfere because they share representation space. Can we apply representation surgery (Yang et al., ICML 2024) or trust-region methods to eliminate interference?

**Why it matters:** Direct extension of Direction 1, addressing the likely failure mode of naive task arithmetic for proteins.

**Why now:** TATR (ICLR 2025) and Representation Surgery (ICML 2024) provide the tools. No protein application exists.

**Novelty:** First application of interference-aware merging to PLMs. Addresses limitations that Direction 1 would reveal.

**Required:** Same as Direction 1 plus interference analysis tools.

**Difficulty:** 6/10  
**Time:** 10–12 weeks  
**NeurIPS likelihood:** 35–45%

---

### Additional Directions (16–20, briefer)

**16. Active property elicitation via natural language conditioning.** PRIMO encoded DMS assays as tokens; what if you encode assay descriptions in natural language? Enables conditioning diffusion/prediction on free-text property descriptions. *Difficulty: 7, Time: 12 weeks, NeurIPS: 25%*

**17. Protein fitness prediction as a ranking problem.** FSFP used ListMLE. Explore modern learning-to-rank methods (differentiable sorting, optimal transport ranking) for fitness. *Difficulty: 5, Time: 8 weeks, NeurIPS: 20%*

**18. Causal inference for mutation effects.** Treat DMS data as observational data with confounders (assay effects, epistasis, selection bias). Apply causal inference to estimate direct mutation effects. *Difficulty: 8, Time: 14 weeks, NeurIPS: 30%*

**19. Self-play for protein sequence optimization.** Use MCTS or AlphaZero-style search over the mutation landscape, with a PLM as the value function. *Difficulty: 7, Time: 12 weeks, NeurIPS: 25%*

**20. Reward-model alignment for protein generators.** Apply DPO or RLHF-style alignment to protein diffusion models, using DMS data as preference labels. *Difficulty: 8, Time: 14 weeks, NeurIPS: 35%*

---

## STEP 4 — SELECT BEST 5

### Selection Criteria Applied

For a solo researcher targeting NeurIPS 2026 with academic compute:
- **Novelty weight: HIGH** — NeurIPS demands clear differentiation
- **Feasibility weight: HIGH** — must be executable in ~10 weeks
- **Risk tolerance: MEDIUM** — need a clear MVP path
- **Compute constraint: STRICT** — must work with 1–4 A100s

**Selected:** Directions 1, 2, 5, 12, 15

---

### CANDIDATE A: Task Arithmetic for Protein Property Composition

**Title:** *"Property Vectors: Composing Protein Fitness Predictors via Task Arithmetic on Language Model Weights"*

**Core hypothesis:** Fine-tuning a PLM on different protein properties produces task vectors that can be composed arithmetically, enabling multi-property fitness prediction without multi-task training. Biological properties are sufficiently disentangled in PLM weight space to permit additive composition.

**Proposed method:**
1. Start with ESM-2 (650M).
2. Fine-tune with LoRA on property-specific subsets of ProteinGym: stability assays, binding assays, expression/abundance assays, enzymatic activity assays.
3. Extract task vectors: τ_stability = θ_stability - θ_pretrained, etc.
4. Compose: θ_multi = θ_pretrained + α_1·τ_stability + α_2·τ_binding + ...
5. Evaluate on held-out multi-property proteins.
6. Compare against: (a) vanilla ESM-2, (b) multi-task fine-tuned ESM-2, (c) PRIMO, (d) independent single-property models.
7. Analyze: task vector similarity/interference, which properties compose well vs. conflict, scaling coefficient sensitivity.
8. Extend with TIES-Merging and DARE to mitigate interference.

**Experiment plan:**
- Exp 1: Property-specific fine-tuning (do LoRA-tuned models beat ESM-2 zero-shot on their respective properties?). Establish that task vectors contain useful property information.
- Exp 2: Pairwise composition (stability + binding, stability + expression, etc.). Measure multi-task performance vs. single-task baselines.
- Exp 3: Full composition (all properties simultaneously). Compare task arithmetic vs. multi-task training.
- Exp 4: Negation (subtract stability task vector → does model lose stability-prediction ability while retaining others?). Demonstrates interpretability.
- Exp 5: Transfer to held-out proteins not in any training assay. Test generalization.
- Exp 6: Analysis of task vector geometry (cosine similarities, singular value structure, interference patterns). Provides scientific insight beyond the method.

**Expected results if hypothesis true:** Multi-property models via task arithmetic achieve within 5–10% of multi-task-trained models, while requiring no joint training. Negation cleanly removes targeted properties. Some property pairs compose better than others (prediction: stability + expression compose well; binding + specificity conflict).

**What would make reviewers excited:**
- Completely novel application of a well-understood technique to an important domain
- The negative results are also interesting (which properties interfere tells us about PLM representations)
- Immediate practical impact (any protein engineer can compose property models)
- Clean experimental design with strong baselines
- Connects two major research communities (model merging + protein design)

**What would kill the paper:**
- If task vectors are so entangled that composition never works (all properties interfere catastrophically)
- If the improvement over simple ensembling is negligible (task arithmetic = weighted averaging of outputs in disguise)
- If the experimental setup is too simple and reviewers want more sophisticated merging methods
- If concurrent work appears (currently nobody is working on this, but timing matters)

**Minimal viable version:** Exp 1 + Exp 2 + Exp 6 only. Show that property-specific fine-tuning produces useful task vectors, that pairwise composition works for at least 3/6 property pairs, and provide the geometric analysis. This could be done in 5–6 weeks.

---

### CANDIDATE B: Conformal Risk Control for Multi-Objective Protein Design

**Title:** *"Design with Guarantees: Multi-Objective Conformal Risk Control for Protein Engineering"*

**Core hypothesis:** By extending conformal risk control to the multi-property setting, we can provide finite-sample guarantees on the joint false discovery rate across multiple protein design objectives, and these guarantees remain tight enough to be practically useful.

**Proposed method:**
1. Train property-specific predictors (or use existing ones) for stability, binding, expression.
2. Calibrate each predictor using conformal prediction with domain-specific nonconformity scores.
3. Extend Learn-then-Test (Angelopoulos et al., 2022) to jointly control FDR across properties.
4. Handle feedback covariate shift (designed proteins ≠ training distribution) using Fannjiang's weighting scheme.
5. Demonstrate: given a protein library, select a subset with guaranteed α-level FDR for ALL properties simultaneously.

**Experiment plan:**
- Exp 1: Single-property calibration on ProteinGym (show existing predictors are miscalibrated).
- Exp 2: Multi-property joint calibration on proteins with multiple assay measurements.
- Exp 3: Simulated design campaign — generate candidates, apply joint conformal screening, measure actual FDR vs. guaranteed FDR.
- Exp 4: Compare selection efficiency (number of candidates passing) vs. naive thresholding.
- Exp 5: Sensitivity to calibration set size (how many labeled examples needed for useful guarantees?).

**Expected results:** Current predictors are systematically overconfident. Joint conformal screening reduces experimental failures by 30–50% while retaining 60–80% of the best candidates. Guarantees hold empirically across diverse protein families.

**Reviewers would love:** Rigorous statistical framework applied to practical protein engineering. Theoretically grounded. Practically impactful.

**What would kill it:** If guarantees are so conservative they're useless (retain 5% of candidates). If calibration requires more labeled data than a typical assay produces.

**MVP:** Exp 1 + Exp 2 + Exp 3. 7–8 weeks.

---

### CANDIDATE C: Optimal Transport for Cross-Assay Fitness Transfer

**Title:** *"Wasserstein Alignment of Protein Fitness Landscapes for Cross-Assay Transfer Learning"*

**Core hypothesis:** Fitness landscapes from different DMS assays can be geometrically aligned using optimal transport in PLM embedding space, enabling knowledge transfer to new proteins with minimal labeled data.

**Proposed method:**
1. Embed protein variants from multiple DMS assays using ESM-2.
2. Compute empirical fitness distributions in embedding space for each assay.
3. Learn Monge maps between source and target fitness landscapes using neural OT.
4. At test time: map labeled source-assay variants into the target assay's landscape to provide pseudo-labels.
5. Fine-tune a predictor using these pseudo-labels + any available target-assay labels.

**Experiment plan:**
- Exp 1: Pairwise alignment quality between all ProteinGym assay pairs (what determines alignment difficulty?).
- Exp 2: Few-shot transfer (varying N from 0 to 64) using OT alignment vs. PRIMO vs. Metalic vs. zero-shot.
- Exp 3: Cross-property transfer (can a stability landscape inform binding prediction for the same protein?).
- Exp 4: Ablation on OT components (Sinkhorn vs. neural OT, different cost functions).
- Exp 5: Theoretical analysis of when alignment works (shared structural constraints = better alignment).

**Reviewers would love:** Elegant geometric framework, principled alternative to meta-learning.

**What would kill it:** If neural OT adds too much complexity for marginal gains over simple fine-tuning. If fitness landscapes are too dissimilar to align.

**MVP:** Exp 1 + Exp 2. 8–9 weeks.

---

### CANDIDATE D: Selective Prediction for Protein Fitness

**Title:** *"I Don't Know Is the Best Prediction: Selective Fitness Prediction with Coverage Guarantees"*

**Core hypothesis:** By learning when to abstain from prediction, we can dramatically improve reliability of protein fitness models at the cost of reduced coverage — and for practical protein engineering, this trade-off is almost always worthwhile.

**Proposed method:**
1. Train standard fitness predictors on ProteinGym.
2. Learn a "confidence function" g(x) that estimates whether the predictor's output for variant x is reliable.
3. At desired error rate α, learn a threshold on g(x): predict when g(x) > threshold, abstain otherwise.
4. Provide conformal coverage guarantees on the selective predictions.
5. Compare against: standard prediction (no abstention), ensembles, MC dropout, conformal sets.

**Experiment plan:**
- Exp 1: Measure reliability-coverage trade-off across all 217 ProteinGym assays.
- Exp 2: Characterize which mutations the model should abstain on (buried vs. surface, conserved vs. variable positions).
- Exp 3: Simulate directed evolution with selective prediction (does abstaining on uncertain mutations improve campaign efficiency?).
- Exp 4: Compare confidence functions (learned vs. heuristic, e.g., PLM entropy, MSA depth).
- Exp 5: Transfer: can a confidence function learned on training assays generalize to held-out assays?

**Reviewers would love:** Practical framing, clean story, addresses the real problem (models don't know when they're wrong).

**What would kill it:** If reviewers see this as "just" conformal prediction applied to proteins (incremental). Need strong biological insights about WHAT drives abstention.

**MVP:** Exp 1 + Exp 2 + Exp 4. 6–7 weeks.

---

### CANDIDATE E: Representation Surgery for Multi-Task PLM Merging

**Title:** *"Merging Protein Experts: Interference-Aware Weight Composition for Multi-Property Language Models"*

**Core hypothesis:** Naive task arithmetic for PLMs suffers from task interference, but representation surgery (projecting out conflicting directions) and trust-region constraints can recover multi-task performance.

**Proposed method:**
1. Fine-tune ESM-2 on property-specific tasks (same setup as Candidate A).
2. Diagnose interference: compute cosine similarity between task vectors, identify conflicting layers/heads.
3. Apply representation surgery: project task vectors into orthogonal subspaces to eliminate interference.
4. Apply TATR: restrict merging to trust regions where interference is low.
5. Evaluate multi-property performance and compare to Candidate A baselines.

**Experiment plan:** Same as Candidate A, but with interference diagnosis and correction as the central contribution. Additional experiments on which layers/components carry which properties.

**Reviewers would love:** More technically sophisticated than plain task arithmetic. Interpretable analysis of which representations encode which properties.

**What would kill it:** Depends on Candidate A as a prerequisite — if plain task arithmetic already works well, the value of surgery is reduced.

**MVP:** Combined with Candidate A as a single paper. Task arithmetic + interference analysis + surgery.

---

## STEP 5 — PICK ONE WINNER

### Winner: CANDIDATE A — Task Arithmetic for Protein Property Composition

**Rationale for selection:**

1. **Highest novelty density.** Task arithmetic for PLMs is genuinely unexplored. Every other candidate has closer prior art (conformal prediction for proteins exists; OT for bio exists; selective prediction exists). This has ZERO prior work.

2. **Risk/reward profile.** If it works → strong positive result, immediate impact, clear story. If it fails → the failure modes are themselves a publishable contribution (understanding WHY protein properties interfere in weight space reveals PLM representations). There is no silent failure mode.

3. **Execution speed.** A strong version is achievable in 8 weeks. The experimental design is clean and well-defined. No custom model architecture needed — it's fine-tuning + arithmetic.

4. **Compute friendly.** LoRA fine-tuning ESM-2-650M is feasible on 1–2 A100s. No diffusion model inference, no large-scale pre-training.

5. **Connects two large communities.** Model merging researchers and protein ML researchers. This broadens reviewer appeal at NeurIPS.

6. **Natural extension to Candidate E.** If plain task arithmetic has interference issues, the paper can incorporate surgical merging as a remedy, making the contribution deeper.

---

### Full Research Plan

#### Phase 1: Property-Specific Fine-Tuning (Weeks 1–2)

**Objective:** Establish that LoRA fine-tuning ESM-2 on property-specific DMS data produces useful, specialized models.

**Tasks:**
1. Download and preprocess ProteinGym v1.3 substitution benchmark.
2. Categorize all 217 assays by property type using ProteinGym metadata:
   - Stability (thermostability, ΔΔG-like assays): ~40–50 assays
   - Binding (protein-protein, protein-ligand): ~30–40 assays
   - Expression/Abundance (fluorescence, solubility, expression): ~30–40 assays
   - Enzymatic activity: ~30–40 assays
3. Design clean train/test splits ensuring NO protein-level overlap (learn from PRIMO's critique of Metalic).
4. LoRA fine-tune ESM-2-650M on each property category:
   - Rank-based loss (ListMLE, following FSFP)
   - LoRA rank = 8–16 (sweep)
   - Training on property-category aggregated data
5. Evaluate single-property performance vs. zero-shot ESM-2 and supervised baselines.

**Deliverable:** 4 property-specific LoRA-adapted models with demonstrated improvement over zero-shot.

#### Phase 2: Task Vector Extraction & Analysis (Week 3)

**Objective:** Extract and characterize property task vectors.

**Tasks:**
1. Compute task vectors: τ_property = θ_finetuned - θ_pretrained (in LoRA parameter space).
2. Analyze task vector geometry:
   - Pairwise cosine similarity between property vectors
   - Singular value decomposition of the task vector matrix
   - Layer-wise analysis (which layers encode which properties?)
   - Attention head specialization (do different heads specialize?)
3. Visualize with UMAP/t-SNE: how do task vectors relate in weight space?
4. Compute interference metrics from TIES/DARE literature.

**Deliverable:** Comprehensive geometric analysis of property encoding in PLM weight space. This is already a valuable scientific contribution regardless of whether composition works.

#### Phase 3: Composition Experiments (Weeks 4–5)

**Objective:** Test task arithmetic for multi-property fitness prediction.

**Tasks:**
1. Implement task arithmetic composition: θ_multi = θ_pre + Σ α_i · τ_i
2. Pairwise composition experiments (6 pairs from 4 properties):
   - Grid search over scaling coefficients α
   - Evaluate multi-task performance: does the composed model retain both properties?
3. Full composition (all 4 properties):
   - Compare against: (a) zero-shot ESM-2, (b) multi-task trained ESM-2, (c) independent single-property models, (d) simple output ensembling
4. Negation experiments:
   - θ_neg = θ_stability - α·τ_binding
   - Verify: stability prediction retained, binding prediction degraded
5. Implement and compare advanced merging:
   - TIES-Merging (trim, elect sign, merge)
   - DARE (drop and rescale)
   - TATR (trust region)
   - Representation surgery if naive methods show interference

**Deliverable:** Main results table showing multi-property performance across methods.

#### Phase 4: Generalization & Transfer (Week 6)

**Objective:** Test whether composed models generalize to unseen proteins.

**Tasks:**
1. Leave-out protein families and evaluate transfer.
2. Test on FLIP benchmark (independent dataset).
3. Vary calibration: how many labeled examples per property are needed for good task vectors?
4. Test on multi-mutant DMS data (do composed models handle epistasis better?).

**Deliverable:** Generalization results showing practical utility beyond training distribution.

#### Phase 5: Ablations & Analysis (Week 7)

**Objective:** Thorough ablation study.

**Tasks:**
1. Ablate LoRA rank (4, 8, 16, 32).
2. Ablate fine-tuning data size (10%, 25%, 50%, 100% of property data).
3. Ablate base model (ESM-2-150M, ESM-2-650M, ESM-2-3B if feasible).
4. Ablate merging method (simple addition vs. TIES vs. DARE vs. TATR).
5. Ablate loss function (regression MSE vs. ranking ListMLE vs. contrastive).
6. Analyze per-assay results: which assays/proteins benefit most from composition?

**Deliverable:** Comprehensive ablation tables + per-assay analysis.

#### Phase 6: Writing & Submission (Weeks 7–8)

**Objective:** Write camera-ready quality paper.

**Structure:**
1. Introduction: Multi-property protein design is the goal; current approaches are either expensive (multi-task training) or post-hoc (generate-then-filter). Task arithmetic offers a third way.
2. Background: Task arithmetic in NLP/vision + protein fitness prediction. Connect the two.
3. Method: Property-specific fine-tuning → task vector extraction → arithmetic composition. Clean formulation.
4. Experiments: Single-property baselines → pairwise composition → full composition → negation → generalization → ablations.
5. Analysis: Task vector geometry. Interference patterns. Which properties are compatible? Why?
6. Discussion: Implications for practical protein engineering. Connection to PLM representation theory. Limitations.

---

### 8-Week Execution Roadmap

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 1 | Data prep, splits, infrastructure | Clean ProteinGym splits, training pipeline |
| 2 | Property-specific fine-tuning | 4 LoRA-adapted models with baseline results |
| 3 | Task vector extraction & geometric analysis | Interference analysis, visualizations |
| 4 | Pairwise composition experiments | 6 pairwise results + scaling coefficient analysis |
| 5 | Full composition + negation + advanced merging | Main results table, comparison with all baselines |
| 6 | Generalization, transfer, multi-mutant | Transfer results, FLIP evaluation |
| 7 | Ablations + paper draft (intro, method, experiments) | Complete ablation tables, 60% of paper |
| 8 | Complete paper, figures, supplementary | Camera-ready paper |

---

### Datasets

| Dataset | Role in Paper | Access |
|---------|---------------|--------|
| ProteinGym v1.3 (substitutions) | Primary benchmark | GitHub, CC-BY |
| ProteinGym v1.3 (indels) | Extension experiment | GitHub, CC-BY |
| FLIP | Independent validation | GitHub |
| ESM-2-650M weights | Base model | Hugging Face |
| SKEMPI v2 | Binding validation | Public |

### Baseline Models to Beat

1. **ESM-2 zero-shot** (masked marginal likelihood) — naive baseline
2. **ESM-2 + LoRA single-property** — single-property fine-tuned models applied independently
3. **ESM-2 multi-task trained** — jointly trained on all properties (upper bound for training-based approaches)
4. **Output ensembling** — average predictions of single-property models (cheap composition baseline)
5. **PRIMO** (if code released) — state-of-art few-shot method
6. **Metalic** — meta-learning baseline (use PRIMO-clean splits)
7. **Tranception** — strong zero-shot baseline

### Figures That Would Appear in the Paper

1. **Figure 1: Overview.** Schematic: PLM → property-specific fine-tuning → task vector extraction → arithmetic composition. Clean, one-page figure.

2. **Figure 2: Task vector geometry.** (a) Cosine similarity matrix between property task vectors. (b) Singular value spectrum. (c) Layer-wise analysis showing which layers encode which properties.

3. **Figure 3: Main results.** Bar chart showing Spearman correlation for: zero-shot, single-property, multi-task trained, output ensemble, task arithmetic, TIES, DARE, TATR. Across all property pairs.

4. **Figure 4: Negation.** Performance on Property A vs. Property B as you interpolate from +τ_A to -τ_A. Should show smooth property control.

5. **Figure 5: Scaling coefficient sensitivity.** Heatmap of multi-task performance vs. α_stability and α_binding for a representative pair.

6. **Figure 6: Per-assay analysis.** Scatter plot of improvement from task arithmetic vs. task vector interference for each assay. Shows when composition works and when it fails.

7. **Figure 7: Ablation summary.** Panel showing sensitivity to LoRA rank, data size, base model size, and merging method.

### Risks + Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Properties interfere catastrophically | Medium | High | (1) This is itself a publishable negative result. (2) TIES/DARE/TATR explicitly designed to fix this. (3) Representation surgery as backup. |
| Task arithmetic = ensembling in disguise | Low-Medium | High | Ablation comparing task arithmetic vs. output ensembling. If they're equivalent, analyze WHY (this is interesting). |
| Concurrent work appears | Low | Medium | Speed of execution is the mitigation. 8-week timeline → submit well before the space fills. |
| ProteinGym splits have subtle biases | Medium | Medium | Use PRIMO's clean splitting protocol. Report results on multiple splitting strategies. |
| LoRA fine-tuning doesn't work well | Low | Medium | Fall back to full fine-tuning or adapter methods. ESM-2-650M is fine-tunable on A100. |
| Reviewers want wet-lab validation | Medium | Medium | Acknowledge as limitation. Emphasize computational contribution. NeurIPS is a methods venue. |

### What Makes This NeurIPS-Worthy

1. **Novelty is unambiguous.** Zero prior work on task arithmetic for protein language models. The connection between two major fields is natural but unexplored.

2. **The analysis is scientifically valuable regardless of the sign of results.** Understanding how properties are encoded in PLM weight space — which layers, which heads, how they interact — advances our understanding of protein representation learning. This is true whether composition works perfectly, partially, or not at all.

3. **Immediate practical utility.** Any protein engineer with access to property-labeled data can fine-tune, extract task vectors, and compose. No new architecture, no expensive training, no special infrastructure.

4. **Clean experimental design.** Well-defined baselines, comprehensive ablations, rigorous splits (learning from PRIMO's critique of Metalic). Reviewers appreciate thoroughness.

5. **Multiple related contributions in one paper.** Method (task arithmetic for PLMs) + analysis (property encoding geometry) + practical insight (which properties compose). Sufficient density for a NeurIPS main conference paper.

6. **Extends to broader impact.** The framework generalizes to any protein property, including developability attributes, immunogenicity, and pharmacokinetics. Opens a research programme, not just a single result.

---

## APPENDIX: Honest Assessment of Alternative Paths

If the task arithmetic direction proves too thin (unlikely but possible), the fallback strategy is:

**Fallback A:** Combine Candidates A + E into a single paper: "Task arithmetic + interference diagnosis + surgical correction" gives three contributions in one.

**Fallback B:** Pivot to Candidate D (selective prediction). Fastest execution path (6–7 weeks to MVP), lower NeurIPS ceiling but safer.

**Fallback C:** If NeurIPS main track feels too ambitious, target the MLSB or GenBio workshops at NeurIPS (lower bar, same venue, good signal for follow-up).

**Venue alternatives if NeurIPS doesn't work out:** ICML 2026 (Jan 2027 deadline), ICLR 2027 (Sep 2026 deadline), Nature Methods (longer format, needs experimental validation pathway), Bioinformatics.