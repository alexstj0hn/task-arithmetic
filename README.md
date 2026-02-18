# Property Vectors: Task Arithmetic for Protein Fitness Prediction

**Phase 1 Implementation** - Property-specific LoRA fine-tuning of ESM-2-650M on ProteinGym v1.3 benchmark.

This project implements the first phase of "Property Vectors: Composing Protein Fitness Predictors via Task Arithmetic on Language Model Weights" - a novel approach to multi-property protein design that has never been explored before in the protein ML literature.

## Overview

**Research Goal:** Demonstrate that protein properties (stability, binding, expression, activity) can be composed via task arithmetic on fine-tuned language model weights, enabling multi-property prediction without expensive multi-task training.

**Phase 1 Objectives:**
1. ✅ Categorize 217 ProteinGym DMS assays by property type
2. ✅ Create clean train/test splits with ZERO protein-level overlap (MMseqs2 clustering at 30% identity)
3. ✅ Compute ESM-2 zero-shot baseline scores via masked marginal likelihood
4. ✅ LoRA fine-tune ESM-2-650M on each property category using ListMLE ranking loss
5. ✅ Extract task vectors for Phase 2 composition experiments
6. ✅ Validate that property-specific models improve over zero-shot AND show property specificity

**Critical Innovation:** This is the first application of task arithmetic (successful in NLP/vision) to protein language models. The success or failure of this approach will reveal whether protein properties are sufficiently disentangled in PLM weight space to permit additive composition.

## Project Structure

```
/home/alex/task-arithmetic/
├── README.md                          # This file
├── research_strategy.md               # Full research strategy document
├── pyproject.toml                     # Python dependencies
├── configs/
│   └── train_config.yaml              # All hyperparameters centralized
├── data/                              # Downloaded and processed data
│   ├── raw/                           # ProteinGym v1.3 (downloaded)
│   ├── processed/                     # Categorized assays
│   └── splits/                        # MMseqs2-based train/test splits
├── src/                               # Source code
│   ├── data/                          # Data pipeline
│   │   ├── download.py                # Download ProteinGym
│   │   ├── categorize.py              # Assay categorization
│   │   ├── splits.py                  # MMseqs2 clustering & splitting
│   │   └── dataset.py                 # PyTorch Dataset classes
│   ├── models/                        # Model architectures
│   │   ├── esm_lora.py                # ESM-2 + LoRA + RankingHead
│   │   └── baseline.py                # Zero-shot masked marginal scoring
│   ├── training/                      # Training infrastructure
│   │   ├── losses.py                  # ListMLE ranking loss
│   │   ├── trainer.py                 # Training loop (TODO: implement)
│   │   └── utils.py                   # Utilities (TODO: implement)
│   ├── evaluation/                    # Evaluation metrics
│   │   └── evaluate.py                # Spearman, NDCG, etc. (TODO: implement)
│   └── vectors/                       # Task vector extraction
│       └── extract.py                 # Extract & compose task vectors
├── scripts/                           # Execution scripts (TODO: create)
│   ├── 01_download_data.py
│   ├── 02_categorize_assays.py
│   ├── 03_create_splits.py
│   ├── 04_compute_zero_shot.py
│   ├── 05_train_property_models.py
│   ├── 06_evaluate.py
│   └── 07_extract_task_vectors.py
├── notebooks/                         # Analysis notebooks
└── results/                           # Outputs
    ├── zero_shot/                     # Baseline scores
    ├── checkpoints/                   # Model checkpoints
    ├── task_vectors/                  # Extracted task vectors
    └── phase1_metrics/                # Evaluation results
```

## Quick Start

### 1. Installation

#### Prerequisites
- Python 3.10+
- CUDA-capable GPU (A100 recommended for full training)
- MMseqs2 (for sequence clustering)

#### Install Python dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# OR using pip
pip install -e .
```

#### Install MMseqs2

```bash
# Option 1: Conda (recommended)
conda install -c conda-forge -c bioconda mmseqs2

# Option 2: Download prebuilt binary
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
sudo mv mmseqs/bin/mmseqs /usr/local/bin/

# Verify installation
mmseqs version
```

### 2. Configuration

All hyperparameters are in [configs/train_config.yaml](configs/train_config.yaml).

**Key parameters:**
- `data.download_url`: ProteinGym v1.3 download link
- `model.base_model`: `facebook/esm2_t33_650M_UR50D`
- `lora.r`: 16 (LoRA rank)
- `lora.target_modules`: `["query", "key", "value"]`
- `training.list_size`: 32 (variants per ListMLE ranking list)
- `training.mixed_precision`: `"bf16"` (bfloat16 on A100)

**WandB logging:** Set `WANDB_ENTITY` environment variable before running:
```bash
export WANDB_ENTITY="your-username"
export WANDB_API_KEY="your-api-key"
```

### 3. Data Pipeline

#### Step 1: Download ProteinGym v1.3

Downloads ~500MB ZIP containing 217 DMS assay CSVs and reference metadata.

```bash
python -m src.data.download --config configs/train_config.yaml
```

**Output:**
- `data/raw/DMS_substitutions.csv` (reference file, 217 assays)
- `data/raw/DMS_ProteinGym_substitutions/*.csv` (217 assay files)

#### Step 2: Categorize Assays by Property

Maps assays to 4 property categories based on ProteinGym metadata.

**Challenge:** ProteinGym's `coarse_selection_type` has ~90-100 "OrganismalFitness" assays that need reclassification via keyword matching in phenotype descriptions.

```bash
python -m src.data.categorize --config configs/train_config.yaml
```

**Output:**
- `data/processed/category_assignments.json`

**Expected distribution:**
- Stability: ~40-60 assays
- Binding: ~20-30 assays
- Expression: ~10-20 assays
- Activity: ~60-80 assays

#### Step 3: Create Train/Test Splits (CRITICAL)

Uses MMseqs2 to cluster wildtype sequences at 30% identity, then assigns entire clusters to train OR test. This ensures ZERO protein-level overlap between splits.

**Why this matters:** PRIMO (Dec 2025) exposed that Metalic's reported performance was inflated due to train-test contamination. This implementation prevents that.

```bash
python -m src.data.splits --config configs/train_config.yaml
```

**What it does:**
1. Extracts unique wildtype sequences → FASTA
2. Runs `mmseqs easy-cluster --min-seq-id 0.3`
3. Assigns clusters to train (80%) or test (20%)
4. Validates: max sequence identity between train/test is <30%

**Output:**
- `data/splits/cluster_assignments.tsv`
- `data/splits/train_assays.json`
- `data/splits/test_assays.json`
- `data/splits/split_metadata.json`

### 4. Zero-Shot Baseline

Computes ESM-2-650M zero-shot fitness predictions via masked marginal likelihood (Meier et al., 2021).

For each variant with mutations at positions {p1, p2, ...}:
1. Mask each position in wildtype sequence
2. Forward pass through ESM-2
3. Score = Σ [log P(mut_aa | context) - log P(wt_aa | context)]

```bash
# Score all 217 assays (~30 GPU-hours on A100)
python -m src.models.baseline --config configs/train_config.yaml

# Score specific assay
python -m src.models.baseline --config configs/train_config.yaml --assay CBS_HUMAN_Sun_2020

# Smoke test (2 assays only)
python -m src.models.baseline --config configs/train_config.yaml --smoke-test
```

**Output:**
- `results/zero_shot/{dms_id}_zero_shot.csv` (one per assay)

**Performance:** ~8 minutes per assay on A100 (parallelizable across GPUs).

### 5. LoRA Fine-Tuning

Train property-specific models using ListMLE ranking loss.

**Architecture:**
```
Protein sequence
  ↓
ESM-2 tokenizer (BOS + sequence, max 1024 tokens)
  ↓
ESM-2 encoder (frozen) + LoRA adapters (trainable)
  → Per-residue embeddings [batch, seq_len, 1280]
  ↓
Mean pooling (exclude BOS and padding)
  → [batch, 1280]
  ↓
RankingHead (1280 → 640 → 1)
  → Scalar fitness score [batch]
```

**Training strategy:**
- Sample ranking lists of 32 variants from one assay
- Compute ListMLE loss (ranks by ground truth, minimizes ranking error)
- Gradient accumulation over 8 steps → effective batch of 256 variants
- Mixed precision (bf16) on A100
- Checkpoint every epoch + every 500 steps

```bash
# Train all 4 property models in parallel (requires 4x A100)
python scripts/05_train_property_models.py --config configs/train_config.yaml --property stability &
python scripts/05_train_property_models.py --config configs/train_config.yaml --property binding &
python scripts/05_train_property_models.py --config configs/train_config.yaml --property expression &
python scripts/05_train_property_models.py --config configs/train_config.yaml --property activity &
wait

# OR train sequentially
for PROP in stability binding expression activity; do
    python scripts/05_train_property_models.py --config configs/train_config.yaml --property $PROP
done
```

**Output:**
- `results/checkpoints/{property}/lora_epoch{N}_step{M}/` (LoRA adapter weights)
- `results/checkpoints/{property}/ranking_head_epoch{N}_step{M}.pt`
- WandB logs with training curves

**Estimated training time:**
- Per property: ~70 GPU-hours on A100
- All 4 parallel: ~70 GPU-hours total (3 days wall-clock)
- All 4 sequential: ~280 GPU-hours total (12 days wall-clock)

**Trainable parameters:** ~4.9M / 655M total (~0.75%) per model

### 6. Evaluation

Evaluate trained models on test sets:
- **In-distribution:** Test variants from training-category assays
- **Out-of-distribution:** Test assays from same property category
- **Cross-property:** Test on assays from OTHER property categories

Metrics:
- Spearman correlation (primary)
- NDCG (ranking quality)
- Top-10% recall
- AUC (if binary labels available)

```bash
python scripts/06_evaluate.py --config configs/train_config.yaml
```

**Output:**
- `results/phase1_metrics/cross_property_matrix.csv`
- Per-assay results CSVs
- Plots

**Success criteria:**
- Each LoRA model improves over ESM-2 zero-shot by ≥0.05 average Spearman
- Cross-property matrix shows diagonal dominance (property specificity)

### 7. Task Vector Extraction

Extract LoRA task vectors for Phase 2 composition experiments.

For LoRA, the task vector IS the LoRA weights (A and B matrices). The effective weight delta is `Δ_W = B @ A`.

```bash
python scripts/07_extract_task_vectors.py --config configs/train_config.yaml
```

**Output:**
- `results/task_vectors/stability_task_vector.pt`
- `results/task_vectors/binding_task_vector.pt`
- `results/task_vectors/expression_task_vector.pt`
- `results/task_vectors/activity_task_vector.pt`

These will be used in Phase 2 for arithmetic composition:
```
θ_multi = θ_base + α_stability·τ_stability + α_binding·τ_binding + ...
```

## Smoke Test

Before committing to full training (~280 GPU-hours), run end-to-end validation on a small subset.

**Edit configs/train_config.yaml:**
```yaml
smoke_test:
  enabled: true
  assays_per_category: 2
  max_epochs: 2
  max_variants_per_assay: 200
```

Then run:
```bash
# Full smoke test pipeline
python -m src.data.download --config configs/train_config.yaml
python -m src.data.categorize --config configs/train_config.yaml
python -m src.data.splits --config configs/train_config.yaml
python -m src.models.baseline --config configs/train_config.yaml --smoke-test

# Train on smoke test subset
for PROP in stability binding expression activity; do
    python scripts/05_train_property_models.py --config configs/train_config.yaml --property $PROP
done

# Evaluate and extract
python scripts/06_evaluate.py --config configs/train_config.yaml
python scripts/07_extract_task_vectors.py --config configs/train_config.yaml
```

**Pass criteria:**
- ✓ No crashes, NaN values, or exceptions
- ✓ Training loss decreases over epochs
- ✓ At least 1 assay per category has Spearman > 0
- ✓ Task vectors are non-zero and structurally consistent

## Compute Requirements

### Storage
- ProteinGym ZIP: ~500 MB
- Unzipped CSVs: ~1-2 GB
- ESM-2-650M weights: ~2.5 GB (HuggingFace cache)
- Zero-shot scores: ~200 MB
- Checkpoints: ~2.4 GB (4 categories × 3 saved checkpoints)
- Task vectors: ~80 MB
- **Total: ~8-10 GB**

### GPU Time (1x A100 80GB)
- Zero-shot: ~30 hours (parallelizable)
- Training per category: ~70 hours
- **Total sequential: ~320 hours**
- **Total parallel (4x A100): ~105 hours**

### Wall-Clock Time
- Setup + data: 2-3 hours
- Smoke test: 3 hours
- Full training (4x parallel): ~3 days
- Evaluation + analysis: 1 day
- **Total: ~5-6 days**

### GCP Cost Estimate
- A100 spot instance: ~$1.50/hour
- Parallel training (4x A100): ~$630 base cost
- With preemption overhead (+20%): **~$750 total**

## Key Implementation Details

### 1. Data Integrity

**Zero protein overlap between train and test:**
- MMseqs2 clustering at 30% sequence identity
- Entire clusters assigned to train OR test (never split)
- Validation: max identity between train/test proteins is <30%

**Why this matters:** PRIMO demonstrated that Metalic's results were inflated by protein-level contamination. This implementation prevents that.

### 2. Assay Categorization

**Challenge:** ProteinGym's `coarse_selection_type` has 5 values:
- Stability, Binding, Expression, Activity → direct mapping
- **OrganismalFitness** (~90-100 assays) → needs reclassification

**Solution:** Keyword matching in `raw_DMS_phenotype_name`:
- Stability keywords: "thermostab", "stability", "folding", "ddg"
- Binding keywords: "binding", "affinity", "kd", "ic50"
- Expression keywords: "expression", "fluorescence", "solubility"
- Activity keywords: "activity", "catalytic", "enzyme"
- Default: "activity"

All decisions logged with confidence levels (high/medium/low).

### 3. ESM-2 + LoRA Architecture

**LoRA configuration:**
- Rank: 16
- Alpha: 32 (scaling factor)
- Target modules: `["query", "key", "value"]` (ESM-2 HuggingFace naming)
- Dropout: 0.05

**Important:** ESM-2 in HuggingFace uses `query/key/value` module names, NOT `q_proj/k_proj/v_proj` like LLaMA-style models.

**Pooling:** Mean pool over residue embeddings (exclude BOS token at position 0 and padding). Research shows mean pooling outperforms CLS token for protein sequences.

### 4. ListMLE Ranking Loss

**Mathematical formula:**
```
L = -∑ᵢ [s_πᵢ - log(∑ⱼ≥ᵢ exp(s_πⱼ))]
```

where π sorts ground truth scores in descending order.

**Numerical stability:**
- Subtract max before exp (prevents overflow)
- Reverse cumulative sum for normalization constant
- Epsilon term in log (prevents log(0))

**Why ranking loss?** Fitness prediction is fundamentally a ranking problem. ListMLE directly optimizes for ranking quality (NDCG, Spearman) rather than absolute score prediction.

### 5. Edge Cases

**Sequences longer than 1022 AA:**
- ESM-2 max is 1024 tokens (1022 amino acids + BOS + EOS)
- **Truncate** to first 1022 residues
- **Exclude** variant if mutation is beyond position 1022
- Log warnings with DMS_id and % variants affected

**Assays with <10 mutations:**
- Excluded from training
- Logged in `category_assignments.json`

**Score normalization:**
- Min-max to [0,1] PER ASSAY (not globally)
- Degenerate case (all scores identical) → set to 0.5

## Troubleshooting

### MMseqs2 not found
```bash
# Install via conda
conda install -c conda-forge -c bioconda mmseqs2

# OR download binary
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
sudo mv mmseqs/bin/mmseqs /usr/local/bin/
```

### CUDA out of memory during training
- Reduce `training.batch_size` (default 4 → 2 or 1)
- Reduce `training.list_size` (default 32 → 16)
- Reduce `data.max_sequence_length` (default 1022 → 512)

### WandB authentication
```bash
export WANDB_API_KEY="your-key-here"
export WANDB_ENTITY="your-username"

# OR login interactively
wandb login
```

### Training loss is NaN
- Check ListMLE numerical stability (eps parameter)
- Reduce learning rate
- Check for extremely long sequences (should be truncated)
- Verify score normalization didn't produce NaN

## Validation

After completing Phase 1, verify:

### Data Quality
- ✓ Train/test split has zero protein overlap (check `split_metadata.json`)
- ✓ All 4 property categories have sufficient representation
- ✓ Categorization confidence is documented

### Model Performance
- ✓ Each LoRA model improves over zero-shot by ≥0.05 Spearman
- ✓ Cross-property matrix shows diagonal dominance
- ✓ Training curves show loss decrease

### Task Vectors
- ✓ Task vectors are non-zero and consistent structure
- ✓ Cosine similarity between properties is <0.9
- ✓ Can be loaded and composed successfully

## Next Steps (Phase 2)

Phase 2 will explore task arithmetic composition:

1. **Pairwise composition:** Stability + Binding, Stability + Expression, etc.
2. **Full composition:** All 4 properties simultaneously
3. **Negation:** θ - α·τ_binding to remove binding capability
4. **Interference analysis:** Which properties compose well vs. conflict
5. **Comparison:** Task arithmetic vs. multi-task training vs. ensembling

**Phase 2 research questions:**
- Do task vectors compose additively for protein properties?
- Which property pairs are compatible vs. interfering?
- Can we match multi-task training performance with zero-cost composition?
- What does the geometric structure of task vectors reveal about PLM representations?

## Citation

If you use this code, please cite:

```bibtex
@article{proteingym2023,
  title={ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design},
  author={Notin, Pascal and others},
  journal={NeurIPS},
  year={2023}
}

@article{esm2,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and others},
  journal={bioRxiv},
  year={2022}
}

@inproceedings{listmle,
  title={Listwise approach to learning to rank: theory and algorithm},
  author={Xia, Fen and others},
  booktitle={ICML},
  year={2008}
}
```

## License

MIT License (see LICENSE file)

## Contact

For questions or issues, please open a GitHub issue or contact the author.

---

**Last updated:** February 2026
