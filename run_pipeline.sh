#!/bin/bash
#
# Multi-GPU BindCraft Launcher for RunPod
# Edit the CONFIG section below, then just run: ./run_bindbot.sh
#

set -euo pipefail


#############################################
### CONFIG - EDIT THESE SETTINGS
#############################################

# Required: Path to your settings file (target configuration)
SETTINGS_FILE="settings/settings_target/PDL1.json"

# Optional: Path to filters and advanced settings
FILTERS_FILE="settings/settings_filters/default_filters.json"
ADVANCED_FILE="settings/settings_advanced/default_4stage_multimer.json"

# Total number of designs to generate (will be split across GPUs)
TOTAL_DESIGNS=10

# Additional options (set to "true" to enable)
NO_PYROSETTA=true
VERBOSE=true

#############################################
### Extra PIPELINE FEATURES
#############################################

# Post-filtering: Apply additional biochemical/structural filters to accepted designs
ENABLE_POST_FILTERING=true
POST_FILTER_CONFIG="settings/settings_post_filter/post_filter_config.json"

# Multi-state validation: Test binders against multiple target conformations
ENABLE_MULTI_STATE=true
MULTI_STATE_CONFIG="settings/settings_validation/multi_state_config.json"

#############################################
### END CONFIG - Don't edit below this line
#############################################

BINDCRAFT_DIR="/workspace/BindBot"
WORKSPACE_DIR="/workspace"
LOG_FILE="$WORKSPACE_DIR/run.log"


#############################################
### LOGGING & SETUP
#############################################

mkdir -p "$WORKSPACE_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [MULTI-GPU LAUNCHER] $(date) ==="


#############################################
### PATH NORMALIZATION
#############################################

# Make paths absolute if they're relative
if [[ "$SETTINGS_FILE" != /* ]]; then
    SETTINGS_FILE="$BINDCRAFT_DIR/$SETTINGS_FILE"
fi
if [[ "$FILTERS_FILE" != /* ]]; then
    FILTERS_FILE="$BINDCRAFT_DIR/$FILTERS_FILE"
fi
if [[ "$ADVANCED_FILE" != /* ]]; then
    ADVANCED_FILE="$BINDCRAFT_DIR/$ADVANCED_FILE"
fi

# Verify files exist
for file in "$SETTINGS_FILE" "$FILTERS_FILE" "$ADVANCED_FILE"; do
    if [ ! -f "$file" ]; then
        echo "[ERROR] File not found: $file"
        exit 1
    fi
done

echo "[INFO] Settings file: $SETTINGS_FILE"
echo "[INFO] Filters file: $FILTERS_FILE"
echo "[INFO] Advanced file: $ADVANCED_FILE"
echo "[INFO] Total designs to generate: $TOTAL_DESIGNS"


#############################################
### BUILD EXTRA ARGUMENTS
#############################################

EXTRA_ARGS=()
if [ "$NO_PYROSETTA" = true ]; then
    EXTRA_ARGS+=("--no-pyrosetta")
    echo "[INFO] Running without PyRosetta"
fi
if [ "$VERBOSE" = true ]; then
    EXTRA_ARGS+=("--verbose")
    echo "[INFO] Verbose mode enabled"
fi


#############################################
### SOURCE LIBRARY MODULES
#############################################

# Source GPU orchestration module
source "$BINDCRAFT_DIR/lib/gpu_launcher.sh"

# Source post-processing module
source "$BINDCRAFT_DIR/lib/post_processing.sh"


#############################################
### GPU DETECTION
#############################################

detect_gpus


#############################################
### CONDA ENVIRONMENT ACTIVATION
#############################################

echo "[STEP] Activating BindCraft conda environment..."
source /miniforge3/etc/profile.d/conda.sh
conda activate BindCraft || {
    echo "[ERROR] Failed to activate BindCraft environment"
    exit 1
}


#############################################
### JAX/XLA OPTIMIZATION SETTINGS
#############################################

echo "[OPTIMIZATION] Configuring JAX persistent compilation cache..."

JAX_CACHE_DIR="$WORKSPACE_DIR/jax_compilation_cache"
mkdir -p "$JAX_CACHE_DIR"

export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

# Optional harmless-ish XLA flag
export XLA_FLAGS="--xla_gpu_enable_fast_min_max=true"

echo "[OPTIMIZATION] JAX cache: $JAX_CACHE_DIR"


#############################################
### PREPARE CONFIGURATION FILES
#############################################

echo "[STEP] Preparing configuration files for each GPU instance..."
TEMP_DIR="$WORKSPACE_DIR/multi_gpu_temp"
mkdir -p "$TEMP_DIR"

# Read the original settings file
ORIGINAL_SETTINGS=$(cat "$SETTINGS_FILE")
BINDER_NAME=$(echo "$ORIGINAL_SETTINGS" | python -c "import sys, json; print(json.load(sys.stdin)['binder_name'])" 2>/dev/null || echo "binder")
BASE_DESIGN_PATH=$(echo "$ORIGINAL_SETTINGS" | python -c "import sys, json; print(json.load(sys.stdin).get('design_path', '/workspace/outputs'))" 2>/dev/null || echo "/workspace/outputs")


#############################################
### EXECUTION: SINGLE GPU MODE
#############################################

if [ "$NUM_GPUS" -eq 1 ]; then
    # Run BindCraft on single GPU
    run_single_gpu

    # Run post-processing pipeline features (sequential)
    run_post_processing "$BASE_DESIGN_PATH"

    echo ""
    echo "=== [FINISHED] $(date) ==="
    exit 0
fi


#############################################
### EXECUTION: MULTI-GPU MODE
#############################################

# Launch GPU workers
launch_gpu_workers

# Monitor workers and stop when target reached
monitor_and_stop_workers

# Wait for all workers to terminate and collect exit codes
wait_for_workers
FAILED=$?

# Merge results from all GPUs
merge_gpu_results

# Run post-filtering on merged results (sequential)
if run_post_filter_multi_gpu "$MERGED_DIR"; then
    echo "[SUCCESS] Post-filtering completed"
fi

# Run multi-state validation in parallel across GPUs
if run_multi_state_validation_parallel "$MERGED_DIR" "$NUM_GPUS"; then
    echo "[SUCCESS] Multi-state validation completed"
else
    FAILED=$((FAILED + $?))
fi

# Clean up GPU-specific directories
cleanup_gpu_directories

# Print final summary
print_gpu_summary "$FAILED"

echo ""
echo "=== [FINISHED] $(date) ==="
