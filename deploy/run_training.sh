#!/bin/bash
# ============================================================
# Main training orchestrator.
#
# Runs the full Phase 1 pipeline end-to-end. Each step is
# idempotent — safe to re-run after preemption. Completed
# steps are detected and skipped.
#
# Usage:
#   bash deploy/run_training.sh                    # full run
#   bash deploy/run_training.sh --smoke-test       # quick validation
#   bash deploy/run_training.sh --property binding  # train one property
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/configs/train_config.yaml"

# Parse args
SMOKE_TEST=""
PROPERTY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test) SMOKE_TEST="--smoke-test"; shift ;;
        --property) PROPERTY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$REPO_DIR"

echo "============================================================"
echo "  Protein Task Vectors — Phase 1 Training Pipeline"
echo "============================================================"
echo "  Config: $CONFIG"
echo "  Smoke test: ${SMOKE_TEST:-no}"
echo "  Property: ${PROPERTY:-all}"
echo "  Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""

# --- Step 0: Sync from GCS (resume after preemption) ---
echo "=== Step 0: Sync from GCS ==="
bash deploy/sync_from_gcs.sh || echo "  GCS sync failed, continuing with local state"
echo ""

# --- Step 1: Download data ---
echo "=== Step 1: Download ProteinGym data ==="
if [ -d "data/raw/DMS_ProteinGym_substitutions" ]; then
    echo "  Data already downloaded, skipping."
else
    python -m src.data.download --config "$CONFIG"
fi
echo ""

# --- Step 2: Categorize assays ---
echo "=== Step 2: Categorize assays ==="
if [ -f "data/processed/category_assignments.json" ]; then
    echo "  Already categorized, skipping."
else
    python -m src.data.categorize --config "$CONFIG"
fi
echo ""

# --- Step 3: Create splits ---
echo "=== Step 3: Create train/test splits ==="
if [ -f "data/splits/train_assays.json" ]; then
    echo "  Splits already created, skipping."
else
    python -m src.data.splits --config "$CONFIG"
fi
echo ""

# --- Step 4: Zero-shot baseline ---
echo "=== Step 4: Zero-shot baseline scoring ==="
python scripts/04_zero_shot.py --config "$CONFIG" $SMOKE_TEST || {
    echo "  WARNING: Zero-shot scoring failed (non-fatal)"
}
echo ""

# --- Step 5: Train property models ---
echo "=== Step 5: Train property models ==="
if [ -n "$PROPERTY" ]; then
    PROPERTIES=("$PROPERTY")
else
    PROPERTIES=("stability" "binding" "expression" "activity")
fi

for PROP in "${PROPERTIES[@]}"; do
    echo ""
    echo "--- Training: $PROP ---"

    python scripts/05_train_property_models.py \
        --config "$CONFIG" \
        --property "$PROP" \
        --resume \
        $SMOKE_TEST

    # Sync to GCS after each property
    bash deploy/sync_to_gcs.sh || echo "  GCS sync failed for $PROP (non-fatal)"
done
echo ""

# --- Step 6: Evaluate models ---
echo "=== Step 6: Evaluation ==="

# Per-property evaluation
for PROP in "${PROPERTIES[@]}"; do
    echo "--- Evaluating: $PROP ---"
    python scripts/06_evaluate.py \
        --config "$CONFIG" \
        --property "$PROP" \
        $SMOKE_TEST || echo "  WARNING: Evaluation failed for $PROP"
done

# Cross-property matrix (skip if only training one property)
if [ -z "$PROPERTY" ]; then
    echo "--- Cross-property evaluation matrix ---"
    python scripts/06_evaluate.py \
        --config "$CONFIG" \
        --cross-property \
        $SMOKE_TEST || echo "  WARNING: Cross-property evaluation failed"
fi
echo ""

# --- Step 7: Extract task vectors ---
echo "=== Step 7: Extract task vectors ==="
python scripts/07_extract_vectors.py --config "$CONFIG" || {
    echo "  WARNING: Task vector extraction failed"
}
echo ""

# --- Step 8: Final sync ---
echo "=== Step 8: Final GCS sync ==="
bash deploy/sync_to_gcs.sh || echo "  Final GCS sync failed"

# --- Optional: Notification ---
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
if [ -n "$SLACK_WEBHOOK" ]; then
    SUMMARY="Phase 1 pipeline complete for ${PROPERTY:-all properties}."
    if [ -f "results/phase1_metrics/cross_property_matrix.csv" ]; then
        SUMMARY="$SUMMARY Cross-property matrix saved."
    fi
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"$SUMMARY\"}" \
        "$SLACK_WEBHOOK" || true
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""
echo "Results:"
echo "  Zero-shot:        results/zero_shot/"
echo "  Checkpoints:      results/checkpoints/"
echo "  Metrics:          results/phase1_metrics/"
echo "  Task vectors:     results/task_vectors/"
echo "  Status:           results/status.json"
