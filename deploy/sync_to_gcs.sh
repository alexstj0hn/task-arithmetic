#!/bin/bash
# ============================================================
# Push local checkpoints and results to GCS.
#
# Called automatically after each training epoch, and can be
# run manually at any time.
#
# Usage:
#   bash deploy/sync_to_gcs.sh
#   GCS_BUCKET=gs://my-bucket bash deploy/sync_to_gcs.sh
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET="${GCS_BUCKET:-gs://protein-task-vectors}"

echo "Syncing to $BUCKET..."

# Sync checkpoints (the big one — includes optimizer state)
if [ -d "$REPO_DIR/results/checkpoints" ]; then
    echo "  checkpoints..."
    gsutil -m rsync -r "$REPO_DIR/results/checkpoints/" "$BUCKET/checkpoints/" 2>&1 | tail -3
fi

# Sync zero-shot results
if [ -d "$REPO_DIR/results/zero_shot" ]; then
    echo "  zero_shot..."
    gsutil -m rsync -r "$REPO_DIR/results/zero_shot/" "$BUCKET/results/zero_shot/" 2>&1 | tail -3
fi

# Sync evaluation results
if [ -d "$REPO_DIR/results/phase1_metrics" ]; then
    echo "  phase1_metrics..."
    gsutil -m rsync -r "$REPO_DIR/results/phase1_metrics/" "$BUCKET/results/phase1_metrics/" 2>&1 | tail -3
fi

# Sync task vectors
if [ -d "$REPO_DIR/results/task_vectors" ]; then
    echo "  task_vectors..."
    gsutil -m rsync -r "$REPO_DIR/results/task_vectors/" "$BUCKET/results/task_vectors/" 2>&1 | tail -3
fi

# Sync status file
if [ -f "$REPO_DIR/results/status.json" ]; then
    gsutil cp "$REPO_DIR/results/status.json" "$BUCKET/status.json" 2>/dev/null || true
fi

# Sync processed data (upload once, small)
if [ -d "$REPO_DIR/data/processed" ]; then
    echo "  data/processed..."
    gsutil -m rsync -r "$REPO_DIR/data/processed/" "$BUCKET/data/processed/" 2>&1 | tail -3
fi
if [ -d "$REPO_DIR/data/splits" ]; then
    echo "  data/splits..."
    gsutil -m rsync -r "$REPO_DIR/data/splits/" "$BUCKET/data/splits/" 2>&1 | tail -3
fi

echo "Sync complete."
