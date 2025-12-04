#!/bin/bash

# ============================================================
# Generic Two-Environment Script Runner
# Usage: ./run_two_envs.sh
# ============================================================

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

# Environment 1
ENV1="florence2_env"
SCRIPT1="./Florence-2-Vision-Language-Model/demo.py"
ARGS1="--image ./Florence-2-Vision-Language-Model/rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_47.png 
       --depth ./Florence-2-Vision-Language-Model/rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_47_depth.png
       --text_prompt 'Coca-Cola can on the table'
       --checkpoint-dir ./Florence-2-Vision-Language-Model/checkpoints
       --PCD_dir ./Florence-2-Vision-Language-Model/PCD/target.ply"
       

       

# Environment 2
ENV2="GraspGen"
SCRIPT2="./GraspGen/grasp_generator.py"
ARGS2="--mode object 
       --sample_data_dir ./Florence-2-Vision-Language-Model/PCD 
       --gripper_config ./GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml"
       

# Intermediate data path
DATA_DIR="./pipeline_output"

# ============================================================
# Pipeline Execution
# ============================================================

# Create output directory
mkdir -p $DATA_DIR

echo "========================================"
echo "Two-Environment Pipeline"
echo "========================================"
echo "Environment 1: $ENV1"
echo "Script 1:      $SCRIPT1"
echo "Environment 2: $ENV2"
echo "Script 2:      $SCRIPT2"
echo "Data dir:      $DATA_DIR"
echo "========================================"

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# ============================================================
# Stage 1
# ============================================================
echo ""
echo "[1/2] Running in $ENV1..."
conda activate $ENV1

# Check environment
if [ "$CONDA_DEFAULT_ENV" != "$ENV1" ]; then
    echo "❌ Failed to activate $ENV1"
    exit 1
fi

echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo "Command: python $SCRIPT1 $ARGS1"
echo ""

# Run script 1
eval python $SCRIPT1 $ARGS1

if [ $? -ne 0 ]; then
    echo "❌ Script 1 failed!"
    exit 1
fi

echo "✓ Stage 1 completed!"

# ============================================================
# Stage 2
# ============================================================
echo ""
echo "[2/2] Running in $ENV2..."
conda activate $ENV2

# Check environment
if [ "$CONDA_DEFAULT_ENV" != "$ENV2" ]; then
    echo "❌ Failed to activate $ENV2"
    exit 1
fi

echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo "Command: python $SCRIPT2 $ARGS2"
echo ""

# Run script 2
eval python $SCRIPT2 $ARGS2

if [ $? -ne 0 ]; then
    echo "❌ Script 2 failed!"
    exit 1
fi

echo "✓ Stage 2 completed!"

# ============================================================
# Done
# ============================================================
echo ""
echo "========================================"
echo "✓ All stages completed successfully!"
echo "========================================"
echo "Results saved in: $DATA_DIR"