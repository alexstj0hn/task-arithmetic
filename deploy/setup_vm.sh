#!/bin/bash
# ============================================================
# First-boot setup script for the GCP training VM.
#
# Run this after creating the VM and SSHing in:
#   bash deploy/setup_vm.sh
#
# What it does:
#   1. Mounts the persistent data disk at /mnt/data
#   2. Installs MMseqs2
#   3. Installs Python project dependencies
#   4. Configures WandB (if WANDB_API_KEY is set)
#   5. Symlinks data/results dirs to persistent storage
# ============================================================
set -euo pipefail

DISK_DEVICE="${DISK_DEVICE:-/dev/disk/by-id/google-data-disk}"
MOUNT_POINT="/mnt/data"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== VM Setup ==="
echo "  Repo: $REPO_DIR"
echo "  Mount: $MOUNT_POINT"
echo ""

# --- Step 1: Mount persistent disk ---
echo "--- Mounting persistent disk ---"
sudo mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
    echo "Already mounted at $MOUNT_POINT"
else
    # Format if not already formatted
    if ! sudo blkid "$DISK_DEVICE" &>/dev/null; then
        echo "Formatting disk..."
        sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0 "$DISK_DEVICE"
    fi

    sudo mount -o discard,defaults "$DISK_DEVICE" "$MOUNT_POINT"
    sudo chmod 777 "$MOUNT_POINT"

    # Add to fstab for auto-mount on restart
    if ! grep -q "$MOUNT_POINT" /etc/fstab; then
        echo "$DISK_DEVICE $MOUNT_POINT ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
    fi

    echo "Mounted $DISK_DEVICE at $MOUNT_POINT"
fi

# --- Step 2: Install system dependencies ---
echo ""
echo "--- Installing system dependencies ---"

# MMseqs2
if command -v mmseqs &>/dev/null; then
    echo "MMseqs2 already installed: $(mmseqs version)"
else
    echo "Installing MMseqs2..."
    cd /tmp
    wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
    tar xzf mmseqs-linux-avx2.tar.gz
    sudo cp mmseqs/bin/mmseqs /usr/local/bin/
    rm -rf mmseqs mmseqs-linux-avx2.tar.gz
    echo "MMseqs2 installed: $(mmseqs version)"
fi

# --- Step 3: Install Python dependencies ---
echo ""
echo "--- Installing Python dependencies ---"
cd "$REPO_DIR"
pip install -e . 2>&1 | tail -5
echo "Python dependencies installed."

# --- Step 4: Configure WandB ---
echo ""
echo "--- Configuring WandB ---"
if [ -n "${WANDB_API_KEY:-}" ]; then
    python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"
    echo "WandB configured."
else
    echo "WANDB_API_KEY not set. Set it to enable experiment tracking:"
    echo "  export WANDB_API_KEY=your-key-here"
fi

# --- Step 5: Set up persistent storage symlinks ---
echo ""
echo "--- Setting up persistent storage ---"

# Create directories on persistent disk
mkdir -p "$MOUNT_POINT/data/raw"
mkdir -p "$MOUNT_POINT/data/processed"
mkdir -p "$MOUNT_POINT/data/splits"
mkdir -p "$MOUNT_POINT/results/checkpoints"
mkdir -p "$MOUNT_POINT/results/zero_shot"
mkdir -p "$MOUNT_POINT/results/phase1_metrics"
mkdir -p "$MOUNT_POINT/results/task_vectors"

# Symlink data and results dirs
# If they're already real dirs with data, move contents first
for dir in data results; do
    src="$REPO_DIR/$dir"
    dst="$MOUNT_POINT/$dir"

    if [ -L "$src" ]; then
        echo "  $dir: already symlinked"
    elif [ -d "$src" ]; then
        # Move existing contents to persistent disk
        echo "  $dir: moving existing data to persistent disk..."
        cp -rn "$src"/* "$dst"/ 2>/dev/null || true
        rm -rf "$src"
        ln -s "$dst" "$src"
        echo "  $dir: symlinked to $dst"
    else
        ln -s "$dst" "$src"
        echo "  $dir: symlinked to $dst"
    fi
done

# --- Step 6: Verify GPU ---
echo ""
echo "--- GPU Check ---"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'CUDA: {torch.version.cuda}')
    print(f'bfloat16: {torch.cuda.is_bf16_supported()}')
else:
    print('WARNING: No GPU detected!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next: run the training pipeline:"
echo "  cd $REPO_DIR"
echo "  bash deploy/run_training.sh"
echo "  # or for smoke test:"
echo "  bash deploy/run_training.sh --smoke-test"
