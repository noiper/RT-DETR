#!/bin/bash -l
#SBATCH --job-name=rtdetr_visdrone
#SBATCH --partition=opengpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=0-06:55:00
#SBATCH --output=rtdetr_visdrone_%j.out
#SBATCH --error=rtdetr_visdrone_%j.err

set -e

echo "======================================================"
echo "Job starting on host:  $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "======================================================"

# --- 1. Environment Setup ---
echo "Initializing Conda for this shell session..."
eval "$(/pkg/anaconda3/2023.07/bin/conda shell.bash hook)"

echo "Activating Conda environment 'rtdetr'..."
conda activate rtdetr
echo "Conda environment activated."

# --- 2. Diagnostics ---
echo "Running nvidia-smi"
nvidia-smi
echo "Visible Devices set by Slurm: $CUDA_VISIBLE_DEVICES"

# --- 3. Dataset Copy to /scratch (fast local storage) ---
SOURCE_DATASET_DIR="/home/fengshus/dataset/VisDrone"  # UPDATE THIS PATH
LOCAL_DATASET_DIR="/scratch/$USER/VisDrone"

echo "======================================================"
echo "Copying VisDrone dataset to /scratch for fast I/O..."
echo "Source: $SOURCE_DATASET_DIR"
echo "Destination: $LOCAL_DATASET_DIR"
echo "======================================================"

mkdir -p "$LOCAL_DATASET_DIR"
rsync -av --progress "$SOURCE_DATASET_DIR/" "$LOCAL_DATASET_DIR/"

echo "Dataset copy completed at: $(date)"
du -sh "$LOCAL_DATASET_DIR"

# --- 4. Create symlink so config paths work ---
# The config expects ../dataset/VisDrone relative to the config file
# Config is at:  rtdetrv2_pytorch/configs/dataset/visdrone_vid_detection.yml
# It references:  ../dataset/VisDrone
# Which resolves to: rtdetrv2_pytorch/dataset/VisDrone

SYMLINK_TARGET="rtdetrv2_pytorch/dataset/VisDrone"
mkdir -p "rtdetrv2_pytorch/dataset"

# Remove old symlink/dir if exists
rm -rf "$SYMLINK_TARGET"

# Create symlink pointing to /scratch
ln -s "$LOCAL_DATASET_DIR" "$SYMLINK_TARGET"

echo "Created symlink: $SYMLINK_TARGET -> $LOCAL_DATASET_DIR"

# --- 5. Logic for Resuming Training ---
CONFIG_FILE="rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_visdrone_v4.yml" # Change this for different configs
OUTPUT_DIR=$(grep 'output_dir: ' $CONFIG_FILE | awk '{print $2}')
RESUME_ARG=""

echo "CONFIG FILE IS $CONFIG_FILE"
echo "OUTPUT DIR IS $OUTPUT_DIR"

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    # First priority: resume from best_model if it exists
    BEST_MODEL="$OUTPUT_DIR/best_model.pth"
    
    if [ -f "$BEST_MODEL" ]; then
        echo "Found best model: $BEST_MODEL"
        RESUME_ARG="-r $BEST_MODEL"
    else
        # Fallback:  resume from latest checkpoint
        LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "*.pth" 2>/dev/null | sort -V | tail -n 1)
        
        if [ -f "$LATEST_CHECKPOINT" ]; then
            echo "Found latest checkpoint: $LATEST_CHECKPOINT"
            RESUME_ARG="-r $LATEST_CHECKPOINT"
        else
            echo "No checkpoint found. Starting training from scratch."
        fi
    fi
else
    echo "Output directory does not exist.  Starting training from scratch."
fi

# --- 6. Run Multi-GPU Training ---
NUM_GPUS=$SLURM_NTASKS

echo "======================================================"
echo "Starting training with $NUM_GPUS GPUs..."
echo "Configuration: $CONFIG_FILE"
echo "Resume argument:  $RESUME_ARG"
echo "======================================================"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
echo "Using MASTER_ADDR=$MASTER_ADDR and MASTER_PORT=$MASTER_PORT"

torchrun --nproc_per_node=$NUM_GPUS --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT rtdetrv2_pytorch/tools/train.py \
    --use-amp \
    --seed 0 \
    -c $CONFIG_FILE \
    $RESUME_ARG

echo "======================================================"
echo "Training completed at: $(date)"
echo "======================================================"

# --- 7. Cleanup ---
echo "Cleaning up /scratch and symlink..."
rm -rf "$LOCAL_DATASET_DIR"
rm -f "$SYMLINK_TARGET"

echo "======================================================"
echo "Job finished at: $(date)"
echo "======================================================"