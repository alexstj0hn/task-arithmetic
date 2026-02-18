# GCP Quickstart: From Zero to Training

Complete step-by-step instructions for someone who has never used Google Cloud.

## Prerequisites

- A Google account (Gmail works)
- A credit card (for GCP signup — you get $300 free credits)
- ~30 minutes for initial setup (plus 24-48h wait for GPU quota approval)

---

## Part 1: GCP Account Setup (do once)

### 1.1 Create GCP Account

1. Go to https://cloud.google.com
2. Click "Get started for free"
3. Sign in with Google account
4. Enter billing info → you get **$300 free credits for 90 days**
5. You'll land at console.cloud.google.com

### 1.2 Create a Project

1. Top bar → project dropdown → "New Project"
2. Name: `protein-vectors`
3. Click "Create"
4. **Write down the Project ID** (shown below the name, like `protein-vectors-412305`)

### 1.3 Enable Required APIs

In the Cloud Console:

1. Go to **APIs & Services** → **Library**
2. Search "Compute Engine API" → click → **Enable**
3. Search "Cloud Storage" → click → **Enable** (may already be enabled)

### 1.4 Request GPU Quota

New accounts have 0 GPU quota. You must request it.

1. Go to **IAM & Admin** → **Quotas & System Limits**
2. In the filter bar, type: `NVIDIA A100 GPUs`
3. Check the box for `us-central1` region
4. Click **"Edit Quotas"** at the top
5. Set new limit to: `1`
6. Justification: "Machine learning research - protein fitness prediction with ESM-2"
7. Submit

**This takes 24-48 hours to approve.** You'll get an email.

> **Can't wait?** Request `NVIDIA T4 GPUs` instead — these are usually
> auto-approved within minutes. T4 is slower but works. See the "Using a
> T4 instead" section at the bottom.

---

## Part 2: Install gcloud CLI (do once, on your local machine)

### Linux/WSL:

```bash
curl https://sdk.cloud.google.com | bash
```

When prompted:
- "Modify profile to update your $PATH?" → Yes
- Restart your terminal, then:

```bash
gcloud init
```

This opens a browser:
- Sign in with your Google account
- Select your project (`protein-vectors`)
- Default region: pick `us-central1-a`

### Verify:

```bash
gcloud config list
```

Should show your project ID and zone.

---

## Part 3: Get Your Code to a Git Repo

The VM needs to download your code. Easiest way: push to GitHub.

```bash
cd ~/task-arithmetic

# Initialize git
git init
git add .
git commit -m "Phase 1: full training pipeline"

# Create a NEW repo on github.com (don't add README/gitignore)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/task-arithmetic.git
git branch -M main
git push -u origin main
```

If you prefer SSH: `git remote add origin git@github.com:YOUR_USERNAME/task-arithmetic.git`

---

## Part 4: Create the VM (once your GPU quota is approved)

Set your project ID and run one command:

```bash
export GCP_PROJECT="protein-vectors-412305"  # <-- YOUR project ID
export GCP_ZONE="us-central1-a"

cd ~/task-arithmetic
bash deploy/gcp_create_vm.sh
```

This creates:
- An A100 spot VM (~$3.67/hr — stops automatically when preempted)
- A 200GB persistent SSD (survives VM deletion)
- A GCS bucket for checkpoint backup

**Expected output:**
```
=== VM Created ===
SSH: gcloud compute ssh protein-training --zone=us-central1-a
```

---

## Part 5: Set Up the VM

### 5.1 SSH into the VM

```bash
gcloud compute ssh protein-training --zone=us-central1-a --project=$GCP_PROJECT
```

First time: it will generate SSH keys automatically. Say yes to everything.

### 5.2 Clone your repo on the VM

Once SSH'd in:

```bash
git clone https://github.com/YOUR_USERNAME/task-arithmetic.git /mnt/data/task-arithmetic
cd /mnt/data/task-arithmetic
```

### 5.3 Run setup

```bash
# Optional: set WandB key for experiment tracking
# Get your key from https://wandb.ai/authorize
export WANDB_API_KEY="your-key-here"

# Run setup
bash deploy/setup_vm.sh
```

This installs everything and mounts the persistent disk. Takes ~5 minutes.

**Check the GPU is detected:**
```
GPU: NVIDIA A100-SXM4-80GB
Memory: 80.0 GB
bfloat16: True
```

---

## Part 6: Run Training

### Smoke test first (recommended, ~10 minutes)

```bash
cd /mnt/data/task-arithmetic
bash deploy/run_training.sh --smoke-test
```

This runs the full pipeline on a tiny data subset to verify everything works.

### Full training

```bash
bash deploy/run_training.sh
```

This will take **6-12 hours** for all 4 properties. The script:
1. Downloads ProteinGym data (~500MB)
2. Runs zero-shot baseline on all assays
3. Trains stability → binding → expression → activity models
4. Computes cross-property evaluation matrix
5. Extracts task vectors
6. Syncs everything to GCS after each step

### Train one property at a time

```bash
bash deploy/run_training.sh --property stability
```

---

## Part 7: Dealing with Preemption

Spot VMs can be killed by Google at any time. When that happens:

1. The VM stops (your persistent disk is safe)
2. Restart it:

```bash
gcloud compute instances start protein-training --zone=us-central1-a --project=$GCP_PROJECT
```

3. SSH back in:

```bash
gcloud compute ssh protein-training --zone=us-central1-a --project=$GCP_PROJECT
```

4. Resume training (it picks up from the last checkpoint):

```bash
cd /mnt/data/task-arithmetic
bash deploy/run_training.sh
```

The `--resume` flag is automatic — the scripts detect existing checkpoints.

---

## Part 8: Get Your Results

### Option A: Download from GCS (best)

Results are automatically synced to GCS. From your local machine:

```bash
# Download everything
gsutil -m cp -r gs://protein-task-vectors/results/ ~/task-arithmetic/results_from_gcp/

# Or just the key result:
gsutil cp gs://protein-task-vectors/results/phase1_metrics/cross_property_matrix.csv .
```

### Option B: Copy directly from VM

```bash
gcloud compute scp --recurse \
    protein-training:/mnt/data/task-arithmetic/results/ \
    ~/task-arithmetic/results_from_gcp/ \
    --zone=us-central1-a --project=$GCP_PROJECT
```

---

## Part 9: Shut Down (stop billing!)

**IMPORTANT: Stop or delete the VM when you're done. You're billed while it runs.**

```bash
# Stop VM (can restart later, disk preserved)
export GCP_PROJECT="protein-vectors-412305"
bash deploy/teardown.sh

# Or delete VM entirely (disk still preserved)
bash deploy/teardown.sh --delete
```

### Check what's running / costing money

Go to https://console.cloud.google.com/compute/instances
- If any VMs show "Running", they're costing money
- Persistent disks cost ~$0.17/GB/month ($34/month for 200GB) even when VM is stopped
- Delete the disk when completely done:

```bash
gcloud compute disks delete protein-data-disk --zone=us-central1-a --project=$GCP_PROJECT
```

---

## Cost Estimates

| Resource | Cost | Duration | Total |
|----------|------|----------|-------|
| A100 spot VM | ~$3.67/hr | ~8-12 hours | ~$30-44 |
| 200GB persistent SSD | $0.17/GB/mo | ~1 month | ~$34 |
| GCS storage | $0.02/GB/mo | ~10GB | ~$0.20 |
| **Total** | | | **~$65-80** |

Fits well within the $300 free credit.

---

## Troubleshooting

### "Quota GPUS_ALL_REGIONS exceeded"
Your GPU quota request hasn't been approved yet. Check your email or go to
IAM → Quotas to check status. Use a T4 in the meantime (see below).

### "The resource is not ready" / VM won't create
Spot A100s aren't available in the zone. Try a different zone:
```bash
GCP_ZONE=us-east1-b bash deploy/gcp_create_vm.sh
```

### "Permission denied" on SSH
```bash
gcloud compute config-ssh --project=$GCP_PROJECT
```

### VM was preempted during training
Normal! Just restart and re-run. See Part 7.

### "No module named 'peft'" or similar import error
Run `pip install -e .` on the VM to reinstall dependencies.

---

## Using a T4 Instead of A100

If you can't get A100 quota, edit the create script to use a T4:

```bash
# In deploy/gcp_create_vm.sh, change:
MACHINE_TYPE="n1-standard-8"  # instead of a2-highgpu-1g

# And change the accelerator line to:
--accelerator=count=1,type=nvidia-tesla-t4
```

T4 has 16GB VRAM (vs A100's 80GB). You may need to reduce batch size:
- In `configs/train_config.yaml`, change `training.batch_size: 2` (from 4)
- Change `training.list_size: 16` (from 32)
- Training will be ~4x slower but will work
