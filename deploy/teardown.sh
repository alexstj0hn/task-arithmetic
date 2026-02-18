#!/bin/bash
# ============================================================
# Tear down the GCP training VM.
#
# Syncs state to GCS before stopping. The persistent data disk
# is preserved (not deleted) for later resumption.
#
# Usage:
#   bash deploy/teardown.sh          # stop VM (can restart later)
#   bash deploy/teardown.sh --delete  # delete VM (disk preserved)
# ============================================================
set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-protein-training}"
DELETE=false

if [[ "${1:-}" == "--delete" ]]; then
    DELETE=true
fi

echo "=== Teardown ==="
echo "  VM: $VM_NAME"
echo "  Zone: $ZONE"
echo "  Action: $([ "$DELETE" = true ] && echo 'DELETE' || echo 'STOP')"
echo ""

# --- Step 1: Sync to GCS (run on VM if accessible) ---
echo "Syncing results to GCS..."
gcloud compute ssh "$VM_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --command="cd /mnt/data/task-arithmetic && bash deploy/sync_to_gcs.sh" 2>/dev/null || {
    echo "  Could not sync (VM may already be stopped)"
}

# --- Step 2: Stop or delete VM ---
if [ "$DELETE" = true ]; then
    echo "Deleting VM $VM_NAME..."
    gcloud compute instances delete "$VM_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --quiet
    echo "VM deleted."
else
    echo "Stopping VM $VM_NAME..."
    gcloud compute instances stop "$VM_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT"
    echo "VM stopped."
fi

echo ""
echo "Persistent disk preserved: protein-data-disk"
echo ""
echo "To restart later:"
echo "  gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "To delete everything:"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE --project=$PROJECT"
echo "  gcloud compute disks delete protein-data-disk --zone=$ZONE --project=$PROJECT"
