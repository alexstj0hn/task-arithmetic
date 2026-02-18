#!/bin/bash
# ============================================================
# Pull latest state from GCS (for resuming after preemption).
#
# This is the first thing run_training.sh calls — it restores
# checkpoints and results from the last successful sync.
#
# Usage:
#   bash deploy/sync_from_gcs.sh
#   GCS_BUCKET=gs://my-bucket bash deploy/sync_from_gcs.sh
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET="${GCS_BUCKET:-gs://protein-task-vectors}"

echo "Syncing from $BUCKET..."

# Check if bucket exists
if ! gsutil ls "$BUCKET" &>/dev/null; then
    echo "Bucket $BUCKET does not exist or is not accessible. Starting fresh."
    exit 0
fi

# Pull checkpoints
echo "  checkpoints..."
mkdir -p "$REPO_DIR/results/checkpoints"
gsutil -m rsync -r "$BUCKET/checkpoints/" "$REPO_DIR/results/checkpoints/" 2>&1 | tail -3 || true

# Pull zero-shot results
echo "  zero_shot..."
mkdir -p "$REPO_DIR/results/zero_shot"
gsutil -m rsync -r "$BUCKET/results/zero_shot/" "$REPO_DIR/results/zero_shot/" 2>&1 | tail -3 || true

# Pull evaluation results
echo "  phase1_metrics..."
mkdir -p "$REPO_DIR/results/phase1_metrics"
gsutil -m rsync -r "$BUCKET/results/phase1_metrics/" "$REPO_DIR/results/phase1_metrics/" 2>&1 | tail -3 || true

# Pull task vectors
echo "  task_vectors..."
mkdir -p "$REPO_DIR/results/task_vectors"
gsutil -m rsync -r "$BUCKET/results/task_vectors/" "$REPO_DIR/results/task_vectors/" 2>&1 | tail -3 || true

# Pull processed data
echo "  data..."
mkdir -p "$REPO_DIR/data/processed" "$REPO_DIR/data/splits"
gsutil -m rsync -r "$BUCKET/data/processed/" "$REPO_DIR/data/processed/" 2>&1 | tail -3 || true
gsutil -m rsync -r "$BUCKET/data/splits/" "$REPO_DIR/data/splits/" 2>&1 | tail -3 || true

# Pull status
gsutil cp "$BUCKET/status.json" "$REPO_DIR/results/status.json" 2>/dev/null || true

echo "Sync from GCS complete."
