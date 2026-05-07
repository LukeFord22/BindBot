#!/bin/bash
#
# Multi-GPU BindCraft Launcher for RunPod
# Edit the CONFIG section below, then just run: ./run_bindcraft.sh
#

set -e

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
### NEW PIPELINE FEATURES
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
### PIPELINE FUNCTIONS
#############################################


############## post-processing pipeline ##############

run_post_processing() {
    local output_dir="$1"

    echo ""
    echo "=== [POST-PROCESSING] Running pipeline enhancements ==="
    echo "[INFO] Pipeline flow: BindCraft → Post-filter → Multi-state validation"
    echo ""

    # STEP 1: Post-filtering (if enabled)
    local post_filter_success=false
    if [ "$ENABLE_POST_FILTERING" = true ]; then
        echo "[STEP 1/2] Running post-filtering on accepted designs..."

        local filter_config="$BINDCRAFT_DIR/$POST_FILTER_CONFIG"

        if [ ! -f "$filter_config" ]; then
            echo "[WARN] Post-filter config not found: $filter_config"
            echo "[WARN] Skipping post-filtering"
        else
            python "$BINDCRAFT_DIR/extras/post_filter_designs.py" \
                --config "$filter_config" \
                --input "$output_dir" \
                --output "$output_dir" || {
                echo "[ERROR] Post-filtering failed"
                echo "[WARN] Multi-state validation will use original BindCraft designs"
            }

            if [ -f "$output_dir/filtered_designs.csv" ]; then
                num_filtered=$(wc -l < "$output_dir/filtered_designs.csv")
                num_filtered=$((num_filtered - 1))  # Subtract header
                echo "[SUCCESS] Post-filtering complete: $num_filtered designs passed"
                post_filter_success=true
            fi
        fi
    else
        echo "[STEP 1/2] Post-filtering disabled, skipping..."
    fi

    # STEP 2: Multi-state validation (if enabled)
    # IMPORTANT: Reads from filtered_designs.csv if post-filter was successful,
    # otherwise falls back to accepted_mpnn_full_stats.csv
    if [ "$ENABLE_MULTI_STATE" = true ]; then
        echo ""
        if [ "$post_filter_success" = true ]; then
            echo "[STEP 2/2] Running multi-state validation on POST-FILTERED designs..."
            echo "[INFO] Input source: filtered_designs.csv (post-filter output)"
        else
            echo "[STEP 2/2] Running multi-state validation on ORIGINAL accepted designs..."
            echo "[INFO] Input source: accepted_mpnn_full_stats.csv (BindCraft output)"
        fi

        local validation_config="$BINDCRAFT_DIR/$MULTI_STATE_CONFIG"

        if [ ! -f "$validation_config" ]; then
            echo "[WARN] Multi-state validation config not found: $validation_config"
            echo "[WARN] Skipping multi-state validation"
        else
            python "$BINDCRAFT_DIR/extras/multi_state_validate.py" \
                --config "$validation_config" \
                --input "$output_dir" \
                --output "$output_dir" || {
                echo "[ERROR] Multi-state validation failed"
            }

            if [ -f "$output_dir/multi_state_scores.csv" ]; then
                num_validated=$(wc -l < "$output_dir/multi_state_scores.csv")
                num_validated=$((num_validated - 1))  # Subtract header
                echo "[SUCCESS] Multi-state validation complete: $num_validated designs tested"
            fi
        fi
    else
        echo ""
        echo "[STEP 2/2] Multi-state validation disabled, skipping..."
    fi

    echo ""
    echo "[POST-PROCESSING] Complete"
    echo ""
}


############## Logging & setup ##############

mkdir -p "$WORKSPACE_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [MULTI-GPU LAUNCHER] $(date) ==="

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

# Build extra arguments
EXTRA_ARGS=()
if [ "$NO_PYROSETTA" = true ]; then
    EXTRA_ARGS+=("--no-pyrosetta")
    echo "[INFO] Running without PyRosetta"
fi
if [ "$VERBOSE" = true ]; then
    EXTRA_ARGS+=("--verbose")
    echo "[INFO] Verbose mode enabled"
fi

# Detect number of GPUs
echo "[STEP] Detecting available GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. This script requires NVIDIA GPUs."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[INFO] Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected!"
    exit 1
fi

# Display GPU information
echo "[INFO] GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Calculate designs per GPU
DESIGNS_PER_GPU=$((TOTAL_DESIGNS / NUM_GPUS))
REMAINDER=$((TOTAL_DESIGNS % NUM_GPUS))

echo "[INFO] Designs per GPU: $DESIGNS_PER_GPU (base)"
if [ "$REMAINDER" -gt 0 ]; then
    echo "[INFO] First $REMAINDER GPU(s) will handle 1 extra design"
fi

# Activate conda environment
echo "[STEP] Activating BindCraft conda environment..."
source /miniforge3/etc/profile.d/conda.sh
conda activate BindCraft || {
    echo "[ERROR] Failed to activate BindCraft environment"
    exit 1
}

############## JAX/XLA Optimization Settings ##############

echo "[OPTIMIZATION] Configuring JAX persistent compilation cache..."

JAX_CACHE_DIR="$WORKSPACE_DIR/jax_compilation_cache"
mkdir -p "$JAX_CACHE_DIR"

export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

# Optional harmless-ish XLA flag
export XLA_FLAGS="--xla_gpu_enable_fast_min_max=true"

echo "[OPTIMIZATION] JAX cache: $JAX_CACHE_DIR"

# Create modified settings files for each GPU
echo "[STEP] Preparing configuration files for each GPU instance..."
TEMP_DIR="$WORKSPACE_DIR/multi_gpu_temp"
mkdir -p "$TEMP_DIR"

# Read the original settings file
ORIGINAL_SETTINGS=$(cat "$SETTINGS_FILE")
BINDER_NAME=$(echo "$ORIGINAL_SETTINGS" | python -c "import sys, json; print(json.load(sys.stdin)['binder_name'])" 2>/dev/null || echo "binder")
BASE_DESIGN_PATH=$(echo "$ORIGINAL_SETTINGS" | python -c "import sys, json; print(json.load(sys.stdin).get('design_path', '/workspace/outputs'))" 2>/dev/null || echo "/workspace/outputs")


############## Handling for single GPU #############

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "[INFO] Single GPU detected - running standard BindCraft without splitting"

    # Create modified settings file with TOTAL_DESIGNS
    SINGLE_GPU_SETTINGS="$TEMP_DIR/settings_single_gpu.json"

    python -c "
import json

with open('$SETTINGS_FILE', 'r') as f:
    settings = json.load(f)

settings['number_of_final_designs'] = $TOTAL_DESIGNS

with open('$SINGLE_GPU_SETTINGS', 'w') as f:
    json.dump(settings, f, indent=4)
" || {
        echo "[ERROR] Failed to create settings file"
        exit 1
    }

    cd "$BINDCRAFT_DIR"
    python bindcraft.py \
        --settings "$SINGLE_GPU_SETTINGS" \
        --filters "$FILTERS_FILE" \
        --advanced "$ADVANCED_FILE" \
        "${EXTRA_ARGS[@]}" || {
        echo "[ERROR] BindCraft execution failed"
        exit 1
    }

    echo ""
    echo "=== [SUCCESS] BindCraft completed ==="
    echo "[INFO] Output directory: $BASE_DESIGN_PATH"

    if [ -d "$BASE_DESIGN_PATH/Accepted" ]; then
        num_accepted=$(find "$BASE_DESIGN_PATH/Accepted" -name "*.pdb" 2>/dev/null | wc -l || echo 0)
        echo "[INFO] Accepted designs: $num_accepted"
    fi

    # Run post-processing pipeline features
    run_post_processing "$BASE_DESIGN_PATH"

    echo ""
    echo "=== [FINISHED] $(date) ==="
    exit 0
fi


############# Multi-GPU path: Launch BindCraft instances for each GPU #############

PIDS=()
echo "[STEP] Launching BindCraft instances across $NUM_GPUS GPU(s)..."

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Calculate designs for this GPU
    if [ "$gpu_id" -lt "$REMAINDER" ]; then
        GPU_DESIGNS=$((DESIGNS_PER_GPU + 1))
    else
        GPU_DESIGNS=$DESIGNS_PER_GPU
    fi

    # Skip if no designs allocated
    if [ "$GPU_DESIGNS" -eq 0 ]; then
        continue
    fi

    # Create GPU-specific output directory
    GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
    mkdir -p "$GPU_OUTPUT_DIR"

    # Create modified settings file for this GPU
    GPU_SETTINGS_FILE="$TEMP_DIR/settings_gpu${gpu_id}.json"

    # Modify the settings JSON: update design_path and number_of_final_designs
    python -c "
import json
import sys

with open('$SETTINGS_FILE', 'r') as f:
    settings = json.load(f)

settings['design_path'] = '$GPU_OUTPUT_DIR'
settings['number_of_final_designs'] = $GPU_DESIGNS
settings['binder_name'] = '${BINDER_NAME}_gpu${gpu_id}'

with open('$GPU_SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=4)
" || {
        echo "[ERROR] Failed to create settings file for GPU $gpu_id"
        exit 1
    }

    # Set up GPU-specific log file
    GPU_LOG="$WORKSPACE_DIR/bindcraft_gpu${gpu_id}.log"

    echo "[INFO] Launching BindCraft on GPU $gpu_id: $GPU_DESIGNS designs -> $GPU_OUTPUT_DIR"

    # Launch BindCraft with CUDA_VISIBLE_DEVICES set to this GPU
    cd "$BINDCRAFT_DIR"
    CUDA_VISIBLE_DEVICES=$gpu_id python bindcraft.py \
        --settings "$GPU_SETTINGS_FILE" \
        --filters "$FILTERS_FILE" \
        --advanced "$ADVANCED_FILE" \
        "${EXTRA_ARGS[@]}" \
        > "$GPU_LOG" 2>&1 &

    # Store the PID
    PIDS+=($!)
    echo "[INFO] GPU $gpu_id: PID ${PIDS[-1]}, Log: $GPU_LOG"
done

echo ""
echo "[INFO] All $NUM_GPUS BindCraft instances launched"
echo "[INFO] Process IDs: ${PIDS[*]}"
echo ""
echo "[MONITORING] Waiting for all processes to complete..."
echo "[MONITORING] Monitor individual GPU logs at: $WORKSPACE_DIR/bindcraft_gpu*.log"
echo ""

# Monitor processes and launch parallel post-processing
FAILED=0
POST_PROCESS_PIDS=()

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu_id=$i

    echo "[MONITORING] Waiting for GPU $gpu_id (PID $pid)..."
    if wait "$pid"; then
        echo "[SUCCESS] GPU $gpu_id completed successfully"

        # Launch post-processing for this GPU in the background
        GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
        if [ -d "$GPU_OUTPUT_DIR" ]; then
            echo "[POST-PROCESS] Starting parallel post-processing for GPU $gpu_id output..."

            # Run post-processing in background
            (
                run_post_processing "$GPU_OUTPUT_DIR"
                echo "[POST-PROCESS] GPU $gpu_id post-processing complete"
            ) > "$WORKSPACE_DIR/post_process_gpu${gpu_id}.log" 2>&1 &

            # Store post-processing PID
            POST_PROCESS_PIDS+=($!)
            echo "[POST-PROCESS] GPU $gpu_id: Post-processing PID ${POST_PROCESS_PIDS[-1]}"
        fi
    else
        exit_code=$?
        echo "[ERROR] GPU $gpu_id failed with exit code $exit_code"
        echo "[ERROR] Check log: $WORKSPACE_DIR/bindcraft_gpu${gpu_id}.log"
        FAILED=$((FAILED + 1))
    fi
done

# Wait for all post-processing to complete
if [ ${#POST_PROCESS_PIDS[@]} -gt 0 ]; then
    echo ""
    echo "[POST-PROCESS] Waiting for all post-processing jobs to complete..."
    POST_PROCESS_FAILED=0

    for i in "${!POST_PROCESS_PIDS[@]}"; do
        pp_pid=${POST_PROCESS_PIDS[$i]}
        gpu_id=$i

        echo "[POST-PROCESS] Waiting for GPU $gpu_id post-processing (PID $pp_pid)..."
        if wait "$pp_pid"; then
            echo "[POST-PROCESS] GPU $gpu_id post-processing succeeded"
        else
            echo "[POST-PROCESS] GPU $gpu_id post-processing failed"
            echo "[POST-PROCESS] Check log: $WORKSPACE_DIR/post_process_gpu${gpu_id}.log"
            POST_PROCESS_FAILED=$((POST_PROCESS_FAILED + 1))
        fi
    done

    if [ "$POST_PROCESS_FAILED" -gt 0 ]; then
        echo "[WARN] $POST_PROCESS_FAILED post-processing job(s) failed"
        echo "[WARN] Continuing with merge, but some post-processed data may be missing"
    else
        echo "[SUCCESS] All post-processing jobs completed successfully"
    fi
fi

echo ""
echo "=== [SUMMARY] ==="
if [ "$FAILED" -eq 0 ]; then
    echo "[SUCCESS] All $NUM_GPUS BindCraft instances completed successfully!"
    echo ""
    echo "[INFO] Output directories before merge:"
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
        if [ -d "$GPU_OUTPUT_DIR" ]; then
            num_accepted=$(find "$GPU_OUTPUT_DIR/Accepted" -name "*.pdb" 2>/dev/null | wc -l || echo 0)
            echo "  GPU $gpu_id: $GPU_OUTPUT_DIR ($num_accepted accepted designs)"
        fi
    done

    # Automatic merge and cleanup
    echo ""
    echo "=== [AUTO-MERGE] Merging results from all GPUs ==="

    # Create merged output directory
    MERGED_DIR="$BASE_DESIGN_PATH"
    mkdir -p "$MERGED_DIR"

    # Merge subdirectories
    for subdir in Accepted Rejected Trajectory; do
        echo "[MERGE] Processing $subdir..."
        MERGED_SUBDIR="$MERGED_DIR/$subdir"
        mkdir -p "$MERGED_SUBDIR"

        file_count=0
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
            gpu_subdir="$GPU_OUTPUT_DIR/$subdir"
            if [ -d "$gpu_subdir" ]; then
                gpu_files=$(find "$gpu_subdir" -type f 2>/dev/null | wc -l)
                if [ "$gpu_files" -gt 0 ]; then
                    cp -r "$gpu_subdir"/* "$MERGED_SUBDIR/" 2>/dev/null || true
                    file_count=$((file_count + gpu_files))
                fi
            fi
        done
        echo "[MERGE] Total files in merged $subdir: $file_count"
    done

    # Merge CSV files (including post-processing outputs)
    echo "[MERGE] Merging CSV statistics..."
    for csv_name in filter_pass_fail.csv trajectory_stats.csv accepted_mpnn_full_stats.csv rejected_mpnn_full_stats.csv filtered_designs.csv multi_state_scores.csv; do
        MERGED_CSV="$MERGED_DIR/$csv_name"
        first_file=true

        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
            gpu_csv="$GPU_OUTPUT_DIR/$csv_name"
            if [ -f "$gpu_csv" ]; then
                if $first_file; then
                    cp "$gpu_csv" "$MERGED_CSV"
                    first_file=false
                else
                    tail -n +2 "$gpu_csv" >> "$MERGED_CSV"
                fi
            fi
        done

        if [ -f "$MERGED_CSV" ]; then
            echo "[MERGE]   Merged: $csv_name"
        fi
    done

    # Clean up GPU-specific directories
    echo "[CLEANUP] Removing GPU-specific directories..."
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
        if [ -d "$GPU_OUTPUT_DIR" ]; then
            echo "[CLEANUP] Removing $GPU_OUTPUT_DIR"
            rm -rf "$GPU_OUTPUT_DIR"
        fi
    done

    echo ""
    echo "=== [MERGE COMPLETE] ==="
    echo "[SUCCESS] All results merged into: $MERGED_DIR"
    echo ""
    echo "Final counts:"
    for subdir in Accepted Rejected Trajectory; do
        merged_subdir="$MERGED_DIR/$subdir"
        if [ -d "$merged_subdir" ]; then
            count=$(find "$merged_subdir" -name "*.pdb" 2>/dev/null | wc -l || echo 0)
            echo "  $subdir: $count PDB files"
        fi
    done

    # Post-processing already completed in parallel for each GPU
    echo ""
    echo "[INFO] Post-processing was completed in parallel for each GPU's output"
    echo "[INFO] Post-processed results (filtered_designs.csv, multi_state_scores.csv) have been merged"

else
    echo "[ERROR] $FAILED out of $NUM_GPUS instances failed"
    echo "[ERROR] Check individual log files for details"
    echo "[ERROR] Skipping merge due to failures"
    exit 1
fi

echo ""
echo "=== [FINISHED] $(date) ==="
