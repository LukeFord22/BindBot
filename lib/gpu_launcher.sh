#!/bin/bash
#
# GPU Launcher Module
# Handles GPU detection, worker launching, monitoring, and result merging
#

#############################################
### GPU DETECTION
#############################################

detect_gpus() {
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
}


#############################################
### SINGLE GPU EXECUTION
#############################################

run_single_gpu() {
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
}


#############################################
### MULTI-GPU WORKER LAUNCHING
#############################################

launch_gpu_workers() {
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
                "${EXTRA_ARGS[@]}" \
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
}


#############################################
### MONITORING & STOPPING LOGIC
#############################################

monitor_and_stop_workers() {
    local MONITOR_INTERVAL=10  # Check every 10 seconds
    TARGET_REACHED=false  # Flag to track if we intentionally stopped workers

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
            TARGET_REACHED=true

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
}


#############################################
### WAIT FOR WORKERS & COLLECT EXIT CODES
#############################################

wait_for_workers() {
    FAILED=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        gpu_id=$i

        echo "[MONITORING] Waiting for GPU $gpu_id (PID $pid)..."
        if wait "$pid"; then
            echo "[SUCCESS] GPU $gpu_id completed successfully"
        else
            exit_code=$?
            # Exit codes 143 (SIGTERM) and 137 (SIGKILL) are success if we intentionally stopped workers
            if [ "$TARGET_REACHED" = true ] && { [ "$exit_code" -eq 143 ] || [ "$exit_code" -eq 137 ]; }; then
                echo "[SUCCESS] GPU $gpu_id stopped intentionally (target reached)"
            else
                echo "[ERROR] GPU $gpu_id failed with exit code $exit_code"
                echo "[ERROR] Check log: $WORKSPACE_DIR/bindcraft_gpu${gpu_id}.log"
                FAILED=$((FAILED + 1))
            fi
        fi
    done

    return $FAILED
}


#############################################
### RESULT MERGING
#############################################

merge_gpu_results() {
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
}


#############################################
### CLEANUP GPU DIRECTORIES
#############################################

cleanup_gpu_directories() {
    echo "[CLEANUP] Removing GPU-specific directories..."
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_OUTPUT_DIR="${BASE_DESIGN_PATH}_gpu${gpu_id}"
        if [ -d "$GPU_OUTPUT_DIR" ]; then
            rm -rf "$GPU_OUTPUT_DIR"
        fi
    done
}


#############################################
### PRINT FINAL SUMMARY
#############################################

print_gpu_summary() {
    local failed_count=$1

    echo ""
    echo "=== [SUMMARY] ==="
    if [ "$failed_count" -eq 0 ]; then
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
        echo "[ERROR] $failed_count out of $NUM_GPUS instances failed"
        echo "[ERROR] Check individual log files for details"
        echo "[ERROR] Skipping merge due to failures"
        exit 1
    fi
}
