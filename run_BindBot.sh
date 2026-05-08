#!/bin/bash
#
# Multi-GPU BindCraft Launcher for RunPod
# Edit the CONFIG section below, then just run: ./run_bindcraft.sh
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


############# Multi-GPU path: Hybrid approach with shared counter #############

echo "[STEP] Launching BindCraft instances across $NUM_GPUS GPU(s) with HYBRID WORK QUEUE..."
echo "[INFO] Each GPU writes to separate directory (no race conditions)"
echo "[INFO] Shared counter enables dynamic load balancing (fast GPUs do more work)"
echo "[INFO] Wrapper monitors all GPU outputs and stops when total target reached"
echo ""

# Create shared counter directory
COUNTER_DIR="$WORKSPACE_DIR/gpu_sync"
mkdir -p "$COUNTER_DIR"
echo "0" > "$COUNTER_DIR/total_accepted.txt"
echo "running" > "$COUNTER_DIR/status.txt"

# Launch BindCraft on each GPU with separate output directory
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Create GPU-specific output directory
    GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
    mkdir -p "$GPU_OUTPUT_DIR"

    # Create GPU-specific settings file with high target (will be stopped by wrapper)
    GPU_SETTINGS_FILE="$TEMP_DIR/settings_gpu${gpu_id}.json"
    python -c "
import json
with open('$SETTINGS_FILE', 'r') as f:
    settings = json.load(f)

settings['design_path'] = '$GPU_OUTPUT_DIR'
settings['number_of_final_designs'] = 999999  # Unlimited - wrapper will stop it
settings['binder_name'] = '${BINDER_NAME}_gpu${gpu_id}'

with open('$GPU_SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
" || {
        echo "[ERROR] Failed to create settings for GPU $gpu_id"
        exit 1
    }

    GPU_LOG="$WORKSPACE_DIR/bindcraft_gpu${gpu_id}.log"
    echo "[LAUNCH] GPU $gpu_id: Output -> $GPU_OUTPUT_DIR"

    # Launch BindCraft in background
    (
        cd "$BINDCRAFT_DIR"
        CUDA_VISIBLE_DEVICES=$gpu_id python bindcraft.py \
            --settings "$GPU_SETTINGS_FILE" \
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
echo "[MONITORING] Wrapper monitoring all GPUs with shared counter..."
echo "[MONITORING] Individual GPU logs: $WORKSPACE_DIR/bindcraft_gpu*.log"
echo ""

# Monitoring loop with shared counter
MONITOR_INTERVAL=10  # Check every 10 seconds
while true; do
    # Check if any workers are still running
    STILL_RUNNING=false
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            STILL_RUNNING=true
            break
        fi
    done

    # Count total accepted designs across all GPU directories
    TOTAL_ACCEPTED=0
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
        if [ -d "$GPU_OUTPUT_DIR/Accepted" ]; then
            GPU_ACCEPTED=$(find "$GPU_OUTPUT_DIR/Accepted" -name "*.pdb" -type f 2>/dev/null | wc -l)
            TOTAL_ACCEPTED=$((TOTAL_ACCEPTED + GPU_ACCEPTED))
        fi
    done

    # Update shared counter (with file locking)
    (
        flock -x 200
        echo "$TOTAL_ACCEPTED" > "$COUNTER_DIR/total_accepted.txt"
    ) 200>"$COUNTER_DIR/counter.lock"

    echo "[PROGRESS] Accepted designs: $TOTAL_ACCEPTED / $TOTAL_DESIGNS (across all GPUs)"

    # If target reached, signal all GPUs to stop
    if [ "$TOTAL_ACCEPTED" -ge "$TOTAL_DESIGNS" ]; then
        echo ""
        echo "[TARGET REACHED] $TOTAL_ACCEPTED designs accepted - stopping all GPUs..."
        echo "stopped" > "$COUNTER_DIR/status.txt"

        # Send SIGTERM to all running BindCraft processes
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "[STOP] Sending SIGTERM to PID $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done

        # Wait a bit for graceful shutdown
        sleep 20

        # Force kill any remaining processes
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "[STOP] Force killing PID $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done

        break
    fi

    # If all processes finished naturally, break
    if [ "$STILL_RUNNING" = false ]; then
        echo ""
        echo "[MONITORING] All GPU processes have finished"
        break
    fi

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
echo "=== [MERGE] Merging results from all GPUs ==="

# Create merged output directory
MERGED_DIR="$BASE_DESIGN_PATH"
mkdir -p "$MERGED_DIR"

# Merge subdirectories (PDB files)
for subdir in Accepted Rejected Trajectory; do
    echo "[MERGE] Merging $subdir..."
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

# Merge CSV files
echo "[MERGE] Merging CSV statistics..."
for csv_name in trajectory_stats.csv mpnn_design_stats.csv final_design_stats.csv; do
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

echo "[MERGE] Merge complete!"
echo ""

# Post-filtering (fast, sequential is fine)
if [ "$ENABLE_POST_FILTERING" = true ]; then
    echo "[POST-FILTER] Running post-filtering on merged designs..."

    filter_config="$BINDCRAFT_DIR/$POST_FILTER_CONFIG"
    if [ ! -f "$filter_config" ]; then
        echo "[WARN] Post-filter config not found: $filter_config"
    else
        python "$BINDCRAFT_DIR/extras/post_filter_designs.py" \
            --config "$filter_config" \
            --input "$MERGED_DIR" \
            --output "$MERGED_DIR" || {
            echo "[ERROR] Post-filtering failed"
        }

        if [ -f "$MERGED_DIR/filtered_designs.csv" ]; then
            num_filtered=$(tail -n +2 "$MERGED_DIR/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
            echo "[SUCCESS] Post-filtering complete: $num_filtered designs passed"
        fi
    fi
fi

# Multi-state validation in PARALLEL across GPUs
if [ "$ENABLE_MULTI_STATE" = true ]; then
    echo "[MULTI-STATE] Splitting validation workload across $NUM_GPUS GPU(s)..."

    validation_config="$BINDCRAFT_DIR/$MULTI_STATE_CONFIG"
    if [ ! -f "$validation_config" ]; then
        echo "[WARN] Multi-state validation config not found: $validation_config"
    else
        # Determine input CSV (filtered or original)
        if [ -f "$MERGED_DIR/filtered_designs.csv" ]; then
            INPUT_CSV="$MERGED_DIR/filtered_designs.csv"
            echo "[INFO] Validating post-filtered designs"
        elif [ -f "$MERGED_DIR/final_design_stats.csv" ]; then
            INPUT_CSV="$MERGED_DIR/final_design_stats.csv"
            echo "[INFO] Validating BindCraft final designs"
        else
            echo "[WARN] No design CSV found, skipping multi-state validation"
            INPUT_CSV=""
        fi

        if [ -n "$INPUT_CSV" ]; then
            # Split CSV for parallel processing
            TOTAL_DESIGNS_TO_VALIDATE=$(tail -n +2 "$INPUT_CSV" 2>/dev/null | wc -l || echo 0)
            VAL_DESIGNS_PER_GPU=$((TOTAL_DESIGNS_TO_VALIDATE / NUM_GPUS))
            VAL_REMAINDER=$((TOTAL_DESIGNS_TO_VALIDATE % NUM_GPUS))

            echo "[INFO] Splitting $TOTAL_DESIGNS_TO_VALIDATE designs across $NUM_GPUS GPUs"

            # Create split CSV files
            VALIDATION_PIDS=()
            START_LINE=2  # Skip header

            for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
                # Calculate lines for this GPU
                if [ "$gpu_id" -lt "$VAL_REMAINDER" ]; then
                    GPU_VAL_DESIGNS=$((VAL_DESIGNS_PER_GPU + 1))
                else
                    GPU_VAL_DESIGNS=$VAL_DESIGNS_PER_GPU
                fi

                if [ "$GPU_VAL_DESIGNS" -eq 0 ]; then
                    continue
                fi

                END_LINE=$((START_LINE + GPU_VAL_DESIGNS - 1))

                # Create GPU-specific CSV split
                GPU_CSV="$TEMP_DIR/designs_gpu${gpu_id}.csv"
                head -1 "$INPUT_CSV" > "$GPU_CSV"  # Header
                sed -n "${START_LINE},${END_LINE}p" "$INPUT_CSV" >> "$GPU_CSV"

                # Create temporary output directory for this GPU
                GPU_VALIDATION_DIR="$TEMP_DIR/validation_gpu${gpu_id}"
                mkdir -p "$GPU_VALIDATION_DIR/Accepted"

                # Copy this GPU's PDB files to temp directory using Python (safer CSV parsing)
                python3 -c "
import csv
import shutil
from pathlib import Path

csv_file = '$GPU_CSV'
merged_dir = Path('$MERGED_DIR')
val_dir = Path('$GPU_VALIDATION_DIR')

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        design_name = row.get('Design', row.get('design_name', row.get('design', '')))
        if design_name:
            pdb_file = merged_dir / 'Accepted' / f'{design_name}.pdb'
            if pdb_file.exists():
                shutil.copy(pdb_file, val_dir / 'Accepted')
"

                # Copy CSV to temp dir
                cp "$GPU_CSV" "$GPU_VALIDATION_DIR/$(basename $INPUT_CSV)"

                echo "[LAUNCH] GPU $gpu_id: Validating $GPU_VAL_DESIGNS designs"

                # Launch validation in background on this GPU
                (
                    CUDA_VISIBLE_DEVICES=$gpu_id python "$BINDCRAFT_DIR/extras/multi_state_validate.py" \
                        --config "$validation_config" \
                        --input "$GPU_VALIDATION_DIR" \
                        --output "$GPU_VALIDATION_DIR" \
                        2>&1 | sed "s/^/[GPU $gpu_id VAL] /"
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
                MERGED_VAL_CSV="$MERGED_DIR/multi_state_scores.csv"

                first_file=true
                for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
                    GPU_VALIDATION_CSV="$TEMP_DIR/validation_gpu${gpu_id}/multi_state_scores.csv"
                    if [ -f "$GPU_VALIDATION_CSV" ]; then
                        if $first_file; then
                            cp "$GPU_VALIDATION_CSV" "$MERGED_VAL_CSV"
                            first_file=false
                        else
                            tail -n +2 "$GPU_VALIDATION_CSV" >> "$MERGED_VAL_CSV"
                        fi
                    fi
                done

                if [ -f "$MERGED_VAL_CSV" ]; then
                    num_validated=$(tail -n +2 "$MERGED_VAL_CSV" 2>/dev/null | wc -l || echo 0)
                    echo "[SUCCESS] Multi-state validation complete: $num_validated designs validated"
                fi
            else
                echo "[ERROR] $VALIDATION_FAILED validation job(s) failed"
                FAILED=$((FAILED + VALIDATION_FAILED))
            fi
        fi
    fi
fi

# Clean up GPU-specific directories
echo "[CLEANUP] Removing GPU-specific directories..."
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
    if [ -d "$GPU_OUTPUT_DIR" ]; then
        rm -rf "$GPU_OUTPUT_DIR"
    fi
done

echo ""
echo "=== [SUMMARY] ==="
if [ "$FAILED" -eq 0 ]; then
    echo "[SUCCESS] All $NUM_GPUS GPU instances completed successfully!"
    echo ""
    echo "[INFO] Merged output directory: $MERGED_DIR"

    if [ -d "$MERGED_DIR/Accepted" ]; then
        num_accepted=$(find "$MERGED_DIR/Accepted" -name "*.pdb" -type f 2>/dev/null | wc -l)
        echo "[INFO] Total accepted designs: $num_accepted"
    fi

    echo ""
    echo "[INFO] Hybrid work queue mode: Each GPU had separate directory (no race conditions)"
    echo "[INFO] Results merged and post-processed in parallel"

    if [ -f "$MERGED_DIR/filtered_designs.csv" ]; then
        num_filtered=$(tail -n +2 "$MERGED_DIR/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
        echo "[INFO] Filtered designs: $num_filtered"
    fi

    if [ -f "$MERGED_DIR/multi_state_scores.csv" ]; then
        num_validated=$(tail -n +2 "$MERGED_DIR/multi_state_scores.csv" 2>/dev/null | wc -l || echo 0)
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
