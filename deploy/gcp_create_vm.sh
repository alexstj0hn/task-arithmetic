#!/bin/bash
# ============================================================
# Create a GCP spot VM with A100 GPU for training.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GCP project with Compute Engine API enabled
#   - A100 quota in the chosen zone
#
# Usage:
#   GCP_PROJECT=my-project ./deploy/gcp_create_vm.sh
#   GCP_ZONE=europe-west4-a GCP_PROJECT=my-project ./deploy/gcp_create_vm.sh
# ============================================================
set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-protein-training}"
DISK_NAME="${DISK_NAME:-protein-data-disk}"
DISK_SIZE="${DISK_SIZE:-200GB}"
MACHINE_TYPE="a2-highgpu-1g"  # 1x A100 80GB
BOOT_DISK_SIZE="100GB"

echo "=== GCP VM Creation ==="
echo "  Project:  $PROJECT"
echo "  Zone:     $ZONE"
echo "  VM:       $VM_NAME"
echo "  Machine:  $MACHINE_TYPE (spot)"
echo "  Data disk: $DISK_NAME ($DISK_SIZE)"
echo ""

# --- Step 1: Create persistent data disk (survives VM deletion) ---
if gcloud compute disks describe "$DISK_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" &>/dev/null; then
    echo "Persistent disk '$DISK_NAME' already exists."
else
    echo "Creating persistent SSD: $DISK_NAME ($DISK_SIZE)..."
    gcloud compute disks create "$DISK_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --size="$DISK_SIZE" \
        --type=pd-ssd
    echo "Disk created."
fi

# --- Step 2: Create GCS bucket for checkpoints ---
BUCKET="${GCS_BUCKET:-gs://protein-task-vectors}"
if gsutil ls "$BUCKET" &>/dev/null; then
    echo "GCS bucket '$BUCKET' already exists."
else
    echo "Creating GCS bucket: $BUCKET..."
    gsutil mb -p "$PROJECT" -l "${ZONE%-*}" "$BUCKET"
    echo "Bucket created."
fi

# --- Step 3: Create spot VM ---
echo ""
echo "Creating spot VM: $VM_NAME..."
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --disk="name=$DISK_NAME,device-name=data-disk,mode=rw,auto-delete=no" \
    --scopes=storage-full \
    --metadata="install-nvidia-driver=True"

echo ""
echo "=== VM Created ==="
echo "SSH:   gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT"
echo "Setup: gcloud compute ssh $VM_NAME --zone=$ZONE --command='bash /path/to/deploy/setup_vm.sh'"
echo ""
echo "Next steps:"
echo "  1. SSH into the VM"
echo "  2. Clone the repo"
echo "  3. Run deploy/setup_vm.sh"
echo "  4. Run deploy/run_training.sh"
