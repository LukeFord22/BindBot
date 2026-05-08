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

export JAX_ENABLE_COMPILATION_CACHE=1
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


############# Multi-GPU path: Launch BindCraft workers with dynamic work queue #############

echo "[STEP] Launching BindCraft instances across $NUM_GPUS GPU(s) with WORK QUEUE MODE..."
echo "[INFO] Dynamic load balancing: All GPUs write to shared directory and race to complete target"
echo "[INFO] Fast GPUs automatically do more trajectories - no GPU will lag behind!"
echo ""

# Create shared output directory (all GPUs write here)
SHARED_OUTPUT_DIR="$BASE_DESIGN_PATH"
mkdir -p "$SHARED_OUTPUT_DIR"

# Create shared settings file pointing to unified output
SHARED_SETTINGS_FILE="$TEMP_DIR/settings_shared.json"
python -c "
import json
with open('$SETTINGS_FILE', 'r') as f:
    settings = json.load(f)

settings['design_path'] = '$SHARED_OUTPUT_DIR'
settings['number_of_final_designs'] = $TOTAL_DESIGNS

with open('$SHARED_SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
" || {
    echo "[ERROR] Failed to create shared settings file"
    exit 1
}

echo "[SETTINGS] Shared output directory: $SHARED_OUTPUT_DIR"
echo "[SETTINGS] Target designs: $TOTAL_DESIGNS"
echo ""

# Launch BindCraft on each GPU (all writing to shared directory)
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_LOG="$WORKSPACE_DIR/bindcraft_gpu${gpu_id}.log"

    echo "[LAUNCH] Starting BindCraft on GPU $gpu_id (log: $GPU_LOG)"

    # Launch BindCraft in background
    # BindCraft will keep running until check_accepted_designs() sees enough designs in shared Accepted/ folder
    (
        cd "$BINDCRAFT_DIR"
        CUDA_VISIBLE_DEVICES=$gpu_id python bindcraft.py \
            --settings "$SHARED_SETTINGS_FILE" \
            --filters "$FILTERS_FILE" \
            --advanced "$ADVANCED_FILE" \
            2>&1 | sed "s/^/[GPU $gpu_id] /"
    ) > "$GPU_LOG" 2>&1 &

    PIDS+=($!)
    echo "[LAUNCH] GPU $gpu_id: PID ${PIDS[-1]}"
done

echo ""
echo "[INFO] All $NUM_GPUS BindCraft instances launched"
echo "[INFO] Process PIDs: ${PIDS[*]}"
echo ""
echo "[MONITORING] All GPUs racing to complete $TOTAL_DESIGNS designs..."
echo "[MONITORING] Monitor individual GPU logs: $WORKSPACE_DIR/bindcraft_gpu*.log"
echo "[MONITORING] Monitor shared output: $SHARED_OUTPUT_DIR"
echo ""

# Periodic monitoring loop (show progress every 30 seconds)
MONITOR_INTERVAL=30
while true; do
    # Check if any workers are still running
    STILL_RUNNING=false
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            STILL_RUNNING=true
            break
        fi
    done

    # If all workers finished, break
    if [ "$STILL_RUNNING" = false ]; then
        echo ""
        echo "[MONITORING] All GPU processes have finished"
        break
    fi

    # Show current progress by counting designs in shared Accepted folder
    if [ -d "$SHARED_OUTPUT_DIR/Accepted" ]; then
        CURRENT_ACCEPTED=$(find "$SHARED_OUTPUT_DIR/Accepted" -name "*.pdb" -type f 2>/dev/null | wc -l)
    else
        CURRENT_ACCEPTED=0
    fi

    echo "[PROGRESS] Accepted designs: $CURRENT_ACCEPTED / $TOTAL_DESIGNS"

    sleep $MONITOR_INTERVAL
done

echo ""
echo "[MONITORING] Waiting for all GPU processes to fully terminate..."

# Wait for all workers and collect exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu_id=$i

    echo "[MONITORING] Waiting for GPU $gpu_id (PID $pid)..."
    if wait "$pid"; then
        echo "[SUCCESS] GPU $gpu_id completed successfully"
    else
        exit_code=$?
        echo "[ERROR] GPU $gpu_id failed with exit code $exit_code"
        echo "[ERROR] Check log: $WORKSPACE_DIR/bindcraft_gpu${gpu_id}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

# Parallel post-processing across GPUs (split design list)
echo "[POST-PROCESS] Running parallel post-processing across $NUM_GPUS GPU(s)..."
echo "[POST-PROCESS] Splitting design list for parallel processing..."

# First run post-filter sequentially (it's fast, just filters CSV)
if [ "$ENABLE_POST_FILTERING" = true ]; then
    echo "[POST-FILTER] Running post-filtering on all designs..."

    filter_config="$BINDCRAFT_DIR/$POST_FILTER_CONFIG"
    if [ ! -f "$filter_config" ]; then
        echo "[WARN] Post-filter config not found: $filter_config"
    else
        python "$BINDCRAFT_DIR/extras/post_filter_designs.py" \
            --config "$filter_config" \
            --input "$SHARED_OUTPUT_DIR" \
            --output "$SHARED_OUTPUT_DIR" || {
            echo "[ERROR] Post-filtering failed"
        }

        if [ -f "$SHARED_OUTPUT_DIR/filtered_designs.csv" ]; then
            num_filtered=$(tail -n +2 "$SHARED_OUTPUT_DIR/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
            echo "[SUCCESS] Post-filtering complete: $num_filtered designs passed"
        fi
    fi
fi

# Multi-state validation in parallel across GPUs
if [ "$ENABLE_MULTI_STATE" = true ]; then
    echo "[MULTI-STATE] Splitting validation workload across $NUM_GPUS GPU(s)..."

    validation_config="$BINDCRAFT_DIR/$MULTI_STATE_CONFIG"
    if [ ! -f "$validation_config" ]; then
        echo "[WARN] Multi-state validation config not found: $validation_config"
    else
        # Determine input CSV (filtered or original)
        if [ -f "$SHARED_OUTPUT_DIR/filtered_designs.csv" ]; then
            INPUT_CSV="$SHARED_OUTPUT_DIR/filtered_designs.csv"
            echo "[INFO] Validating post-filtered designs"
        elif [ -f "$SHARED_OUTPUT_DIR/final_design_stats.csv" ]; then
            INPUT_CSV="$SHARED_OUTPUT_DIR/final_design_stats.csv"
            echo "[INFO] Validating BindCraft final designs"
        else
            echo "[WARN] No design CSV found, skipping multi-state validation"
            INPUT_CSV=""
        fi

        if [ -n "$INPUT_CSV" ]; then
            # Split CSV for parallel processing
            TOTAL_DESIGNS_TO_VALIDATE=$(tail -n +2 "$INPUT_CSV" 2>/dev/null | wc -l || echo 0)
            DESIGNS_PER_GPU=$((TOTAL_DESIGNS_TO_VALIDATE / NUM_GPUS))
            REMAINDER=$((TOTAL_DESIGNS_TO_VALIDATE % NUM_GPUS))

            echo "[INFO] Splitting $TOTAL_DESIGNS_TO_VALIDATE designs across $NUM_GPUS GPUs"

            # Create split CSV files
            VALIDATION_PIDS=()
            START_LINE=2  # Skip header

            for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
                # Calculate lines for this GPU
                if [ "$gpu_id" -lt "$REMAINDER" ]; then
                    GPU_DESIGNS=$((DESIGNS_PER_GPU + 1))
                else
                    GPU_DESIGNS=$DESIGNS_PER_GPU
                fi

                if [ "$GPU_DESIGNS" -eq 0 ]; then
                    continue
                fi

                END_LINE=$((START_LINE + GPU_DESIGNS - 1))

                # Create GPU-specific CSV split
                GPU_CSV="$TEMP_DIR/designs_gpu${gpu_id}.csv"
                head -1 "$INPUT_CSV" > "$GPU_CSV"  # Header
                sed -n "${START_LINE},${END_LINE}p" "$INPUT_CSV" >> "$GPU_CSV"

                # Create temporary output directory for this GPU
                GPU_VALIDATION_DIR="$TEMP_DIR/validation_gpu${gpu_id}"
                mkdir -p "$GPU_VALIDATION_DIR/Accepted"

                # Copy this GPU's PDB files to temp directory
                while IFS=, read -r design_name rest; do
                    if [ -n "$design_name" ] && [ "$design_name" != "Design" ]; then
                        # Find and copy PDB file
                        if [ -f "$SHARED_OUTPUT_DIR/Accepted/${design_name}.pdb" ]; then
                            cp "$SHARED_OUTPUT_DIR/Accepted/${design_name}.pdb" "$GPU_VALIDATION_DIR/Accepted/"
                        fi
                    fi
                done < <(tail -n +2 "$GPU_CSV")

                # Copy CSV to temp dir
                cp "$GPU_CSV" "$GPU_VALIDATION_DIR/$(basename $INPUT_CSV)"

                echo "[LAUNCH] GPU $gpu_id: Validating designs $START_LINE-$END_LINE ($GPU_DESIGNS designs)"

                # Launch validation in background on this GPU
                (
                    CUDA_VISIBLE_DEVICES=$gpu_id python "$BINDCRAFT_DIR/extras/multi_state_validate.py" \
                        --config "$validation_config" \
                        --input "$GPU_VALIDATION_DIR" \
                        --output "$GPU_VALIDATION_DIR" \
                        2>&1 | sed "s/^/[GPU $gpu_id VALIDATION] /"
                ) > "$WORKSPACE_DIR/validation_gpu${gpu_id}.log" 2>&1 &

                VALIDATION_PIDS+=($!)
                START_LINE=$((END_LINE + 1))
            done

            # Wait for all validation jobs
            echo "[MONITORING] Waiting for all validation jobs to complete..."
            VALIDATION_FAILED=0
            for i in "${!VALIDATION_PIDS[@]}"; do
                pid=${VALIDATION_PIDS[$i]}
                gpu_id=$i

                if wait "$pid"; then
                    echo "[SUCCESS] GPU $gpu_id validation completed"
                else
                    echo "[ERROR] GPU $gpu_id validation failed"
                    VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
                fi
            done

            # Merge validation results
            if [ "$VALIDATION_FAILED" -eq 0 ]; then
                echo "[MERGE] Merging validation results from all GPUs..."
                MERGED_CSV="$SHARED_OUTPUT_DIR/multi_state_scores.csv"

                first_file=true
                for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
                    GPU_VALIDATION_CSV="$TEMP_DIR/validation_gpu${gpu_id}/multi_state_scores.csv"
                    if [ -f "$GPU_VALIDATION_CSV" ]; then
                        if $first_file; then
                            cp "$GPU_VALIDATION_CSV" "$MERGED_CSV"
                            first_file=false
                        else
                            tail -n +2 "$GPU_VALIDATION_CSV" >> "$MERGED_CSV"
                        fi
                    fi
                done

                if [ -f "$MERGED_CSV" ]; then
                    num_validated=$(tail -n +2 "$MERGED_CSV" 2>/dev/null | wc -l || echo 0)
                    echo "[SUCCESS] Multi-state validation complete: $num_validated designs validated"
                fi
            else
                echo "[ERROR] $VALIDATION_FAILED validation job(s) failed"
                FAILED=$((FAILED + VALIDATION_FAILED))
            fi
        fi
    fi
fi

echo ""
echo "=== [SUMMARY] ==="
if [ "$FAILED" -eq 0 ]; then
    echo "[SUCCESS] All $NUM_GPUS worker instances completed successfully!"
    echo ""
    echo "[INFO] Shared output directory: $SHARED_OUTPUT_DIR"

    if [ -d "$SHARED_OUTPUT_DIR/Accepted" ]; then
        num_accepted=$(find "$SHARED_OUTPUT_DIR/Accepted" -name "*.pdb" 2>/dev/null | wc -l || echo 0)
        echo "[INFO] Total accepted designs: $num_accepted"
    fi

    echo ""
    echo "[INFO] Work queue mode: All GPUs wrote to shared directory (no merge needed)"
    echo "[INFO] Post-processing has been completed on the unified output"

    if [ -f "$SHARED_OUTPUT_DIR/filtered_designs.csv" ]; then
        num_filtered=$(tail -n +2 "$SHARED_OUTPUT_DIR/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
        echo "[INFO] Filtered designs: $num_filtered"
    fi

    if [ -f "$SHARED_OUTPUT_DIR/multi_state_scores.csv" ]; then
        num_validated=$(tail -n +2 "$SHARED_OUTPUT_DIR/multi_state_scores.csv" 2>/dev/null | wc -l || echo 0)
        echo "[INFO] Multi-state validated designs: $num_validated"
    fi

else
    echo "[ERROR] $FAILED out of $NUM_GPUS instances failed"
    echo "[ERROR] Check individual log files for details"
    echo "[ERROR] Skipping merge due to failures"
    exit 1
fi

echo ""
echo "=== [FINISHED] $(date) ==="
