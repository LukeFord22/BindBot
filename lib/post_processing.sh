#!/bin/bash
#
# Post-Processing Module
# Handles post-filtering, multi-state validation, and result aggregation
#

#############################################
### POST-FILTERING (SEQUENTIAL)
#############################################

run_post_filter() {
    local output_dir="$1"

    echo "[POST-FILTER] Running post-filtering on accepted designs..."

    # Use POST_FILTER_CONFIG as-is if absolute, otherwise prepend BINDCRAFT_DIR
    local filter_config="$POST_FILTER_CONFIG"
    if [[ "$filter_config" != /* ]]; then
        filter_config="$BINDCRAFT_DIR/$POST_FILTER_CONFIG"
    fi

    if [ ! -f "$filter_config" ]; then
        echo "[WARN] Post-filter config not found: $filter_config"
        return 1
    fi

    python "$BINDCRAFT_DIR/extras/post_filter_designs.py" \
        --config "$filter_config" \
        --input "$output_dir" \
        --output "$output_dir" || {
        echo "[ERROR] Post-filtering failed"
        return 1
    }

    if [ -f "$output_dir/filtered_designs.csv" ]; then
        num_filtered=$(tail -n +2 "$output_dir/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
        echo "[SUCCESS] Post-filtering complete: $num_filtered designs passed"
        return 0
    fi

    return 1
}


#############################################
### MULTI-STATE VALIDATION (SEQUENTIAL)
#############################################

run_multi_state_validation_sequential() {
    local output_dir="$1"
    local post_filter_success="$2"

    echo ""
    if [ "$post_filter_success" = "true" ]; then
        echo "[STEP 2/2] Running multi-state validation on POST-FILTERED designs..."
        echo "[INFO] Input source: filtered_designs.csv (post-filter output)"
    else
        echo "[STEP 2/2] Running multi-state validation on ORIGINAL accepted designs..."
        echo "[INFO] Input source: accepted_mpnn_full_stats.csv (BindCraft output)"
    fi

    # Use MULTI_STATE_CONFIG as-is if absolute, otherwise prepend BINDCRAFT_DIR
    local validation_config="$MULTI_STATE_CONFIG"
    if [[ "$validation_config" != /* ]]; then
        validation_config="$BINDCRAFT_DIR/$MULTI_STATE_CONFIG"
    fi

    if [ ! -f "$validation_config" ]; then
        echo "[WARN] Multi-state validation config not found: $validation_config"
        echo "[WARN] Skipping multi-state validation"
        return 1
    fi

    PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u "$BINDCRAFT_DIR/extras/multi_state_validate.py" \
        --config "$validation_config" \
        --input "$output_dir" \
        --output "$output_dir" || {
        echo "[ERROR] Multi-state validation failed"
        return 1
    }

    if [ -f "$output_dir/multi_state_scores.csv" ]; then
        num_validated=$(wc -l < "$output_dir/multi_state_scores.csv")
        num_validated=$((num_validated - 1))  # Subtract header
        echo "[SUCCESS] Multi-state validation complete: $num_validated designs tested"
        return 0
    fi

    return 1
}


#############################################
### POST-PROCESSING ORCHESTRATOR (SINGLE GPU)
#############################################

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

        if run_post_filter "$output_dir"; then
            post_filter_success=true
        else
            echo "[WARN] Multi-state validation will use original BindCraft designs"
        fi
    else
        echo "[STEP 1/2] Post-filtering disabled, skipping..."
    fi

    # STEP 2: Multi-state validation (if enabled)
    # IMPORTANT: Reads from filtered_designs.csv if post-filter was successful,
    # otherwise falls back to accepted_mpnn_full_stats.csv
    if [ "$ENABLE_MULTI_STATE" = true ]; then
        run_multi_state_validation_sequential "$output_dir" "$post_filter_success"
    else
        echo ""
        echo "[STEP 2/2] Multi-state validation disabled, skipping..."
    fi

    echo ""
    echo "[POST-PROCESSING] Complete"
    echo ""
}


#############################################
### MULTI-GPU POST-FILTERING
#############################################

run_post_filter_multi_gpu() {
    local merged_dir="$1"

    if [ "$ENABLE_POST_FILTERING" != true ]; then
        return 1
    fi

    echo "[POST-FILTER] Running post-filtering on merged designs..."

    # Use POST_FILTER_CONFIG as-is if absolute, otherwise prepend BINDCRAFT_DIR
    local filter_config="$POST_FILTER_CONFIG"
    if [[ "$filter_config" != /* ]]; then
        filter_config="$BINDCRAFT_DIR/$POST_FILTER_CONFIG"
    fi

    if [ ! -f "$filter_config" ]; then
        echo "[WARN] Post-filter config not found: $filter_config"
        return 1
    fi

    python "$BINDCRAFT_DIR/extras/post_filter_designs.py" \
        --config "$filter_config" \
        --input "$merged_dir" \
        --output "$merged_dir" || {
        echo "[ERROR] Post-filtering failed"
        return 1
    }

    if [ -f "$merged_dir/filtered_designs.csv" ]; then
        num_filtered=$(tail -n +2 "$merged_dir/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
        echo "[SUCCESS] Post-filtering complete: $num_filtered designs passed"
        return 0
    fi

    return 1
}


#############################################
### SPLIT CSV FOR PARALLEL VALIDATION
#############################################

split_designs_for_validation() {
    local input_csv="$1"
    local num_gpus="$2"
    local merged_dir="$3"

    TOTAL_DESIGNS_TO_VALIDATE=$(tail -n +2 "$input_csv" 2>/dev/null | wc -l || echo 0)
    VAL_DESIGNS_PER_GPU=$((TOTAL_DESIGNS_TO_VALIDATE / num_gpus))
    VAL_REMAINDER=$((TOTAL_DESIGNS_TO_VALIDATE % num_gpus))

    echo "[INFO] Splitting $TOTAL_DESIGNS_TO_VALIDATE designs across $num_gpus GPUs"

    START_LINE=2  # Skip header

    for gpu_id in $(seq 0 $((num_gpus - 1))); do
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
        head -1 "$input_csv" > "$GPU_CSV"  # Header
        sed -n "${START_LINE},${END_LINE}p" "$input_csv" >> "$GPU_CSV"

        # Create temporary output directory for this GPU
        GPU_VALIDATION_DIR="$TEMP_DIR/validation_gpu${gpu_id}"
        mkdir -p "$GPU_VALIDATION_DIR/Accepted"
        mkdir -p "$GPU_VALIDATION_DIR/MultiStateValidation/generated_positive_states"

        # Copy any pre-generated positive states from merged directory to GPU worker directory
        MERGED_GEN_STATES="$merged_dir/MultiStateValidation/generated_positive_states"
        if [ -d "$MERGED_GEN_STATES" ]; then
            cp "$MERGED_GEN_STATES/"*.pdb "$GPU_VALIDATION_DIR/MultiStateValidation/generated_positive_states/" 2>/dev/null && \
                echo "[GPU $gpu_id] Copied generated positive states" || true
        fi

        # Copy this GPU's PDB files to temp directory using Python (safer CSV parsing)
        python3 -c "
import csv
import shutil
from pathlib import Path
import sys

csv_file = '$GPU_CSV'
merged_dir = Path('$merged_dir')
val_dir = Path('$GPU_VALIDATION_DIR')

copied = 0
missing = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Try different column name variations
        design_name = row.get('design') or row.get('Design') or row.get('design_name') or row.get('Design_Name')
        if not design_name:
            continue

        # Strategy 1: Exact match
        pdb_file = (merged_dir / 'Accepted' / f'{design_name}.pdb')
        if pdb_file.exists():
            shutil.copy(pdb_file, val_dir / 'Accepted')
            copied += 1
            continue

        # File not found
        missing.append(design_name)

if missing:
    print(f'[GPU $gpu_id WARN] {len(missing)} PDB files not found in {merged_dir}/Accepted:', file=sys.stderr)
    for name in missing[:5]:  # Show first 5
        print(f'  - {name}', file=sys.stderr)
    if len(missing) > 5:
        print(f'  ... and {len(missing) - 5} more', file=sys.stderr)

print(f'[GPU $gpu_id INFO] Copied {copied}/{copied + len(missing)} PDB files', file=sys.stderr)
"

        # Copy CSV to temp dir
        cp "$GPU_CSV" "$GPU_VALIDATION_DIR/$(basename $input_csv)"

        START_LINE=$((END_LINE + 1))
    done
}


#############################################
### LAUNCH PARALLEL VALIDATION WORKERS
#############################################

launch_validation_workers() {
    local num_gpus="$1"
    local validation_config="$2"

    VALIDATION_PIDS=()
    VALIDATION_GPU_IDS=()  # Track actual GPU IDs corresponding to PIDs

    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        GPU_VALIDATION_DIR="$TEMP_DIR/validation_gpu${gpu_id}"

        # Skip if no work for this GPU
        if [ ! -d "$GPU_VALIDATION_DIR" ]; then
            continue
        fi

        GPU_CSV="$TEMP_DIR/designs_gpu${gpu_id}.csv"
        if [ ! -f "$GPU_CSV" ]; then
            continue
        fi

        GPU_VAL_DESIGNS=$(tail -n +2 "$GPU_CSV" 2>/dev/null | wc -l || echo 0)
        if [ "$GPU_VAL_DESIGNS" -eq 0 ]; then
            continue
        fi

        echo "[LAUNCH] GPU $gpu_id: Validating $GPU_VAL_DESIGNS designs"

        # Launch validation in background on this GPU
        # Use script command to force unbuffered output to log file
        script -f -c "CUDA_VISIBLE_DEVICES=$gpu_id PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u '$BINDCRAFT_DIR/extras/multi_state_validate.py' --config '$validation_config' --input '$GPU_VALIDATION_DIR' --output '$GPU_VALIDATION_DIR' 2>&1 | stdbuf -oL -eL sed -u 's/^/[GPU $gpu_id VAL] /'" "$WORKSPACE_DIR/validation_gpu${gpu_id}.log" > /dev/null 2>&1 &

        VALIDATION_PIDS+=($!)
        VALIDATION_GPU_IDS+=($gpu_id)  # Store the actual GPU ID
    done
}


#############################################
### WAIT FOR VALIDATION & MERGE RESULTS
#############################################

wait_and_merge_validation() {
    local num_gpus="$1"
    local merged_dir="$2"

    # Wait for all validation jobs
    echo "[MONITORING] Waiting for all validation jobs to complete..."
    VALIDATION_FAILED=0
    for i in "${!VALIDATION_PIDS[@]}"; do
        pid=${VALIDATION_PIDS[$i]}
        gpu_id=${VALIDATION_GPU_IDS[$i]}  # Use actual GPU ID, not array index

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
        MERGED_VAL_CSV="$merged_dir/multi_state_scores.csv"
        MERGED_VAL_DIR="$merged_dir/MultiStateValidation"
        mkdir -p "$MERGED_VAL_DIR"

        first_file=true
        # Iterate over actual GPU IDs that had validation work
        for gpu_id in "${VALIDATION_GPU_IDS[@]}"; do
            GPU_VALIDATION_CSV="$TEMP_DIR/validation_gpu${gpu_id}/multi_state_scores.csv"
            if [ -f "$GPU_VALIDATION_CSV" ]; then
                if $first_file; then
                    cp "$GPU_VALIDATION_CSV" "$MERGED_VAL_CSV"
                    first_file=false
                else
                    tail -n +2 "$GPU_VALIDATION_CSV" >> "$MERGED_VAL_CSV"
                fi
            fi

            # Merge complex PDB files from MultiStateValidation directory
            GPU_VAL_DIR="$TEMP_DIR/validation_gpu${gpu_id}/MultiStateValidation"
            if [ -d "$GPU_VAL_DIR" ]; then
                cp -r "$GPU_VAL_DIR/"*.pdb "$MERGED_VAL_DIR/" 2>/dev/null || true
            fi
        done

        if [ -f "$MERGED_VAL_CSV" ]; then
            num_validated=$(tail -n +2 "$MERGED_VAL_CSV" 2>/dev/null | wc -l || echo 0)
            num_complexes=$(ls -1 "$MERGED_VAL_DIR"/*.pdb 2>/dev/null | wc -l || echo 0)
            echo "[SUCCESS] Multi-state validation complete: $num_validated designs validated"
            echo "[SUCCESS] Merged $num_complexes complex PDB files to $MERGED_VAL_DIR"
        fi

        return 0
    else
        echo "[ERROR] $VALIDATION_FAILED validation job(s) failed"
        return $VALIDATION_FAILED
    fi
}


#############################################
### MULTI-GPU PARALLEL VALIDATION ORCHESTRATOR
#############################################

run_multi_state_validation_parallel() {
    local merged_dir="$1"
    local num_gpus="$2"

    if [ "$ENABLE_MULTI_STATE" != true ]; then
        return 0
    fi

    echo "[MULTI-STATE] Splitting validation workload across $num_gpus GPU(s)..."

    # Use MULTI_STATE_CONFIG as-is if absolute, otherwise prepend BINDCRAFT_DIR
    local validation_config="$MULTI_STATE_CONFIG"
    if [[ "$validation_config" != /* ]]; then
        validation_config="$BINDCRAFT_DIR/$MULTI_STATE_CONFIG"
    fi

    if [ ! -f "$validation_config" ]; then
        echo "[WARN] Multi-state validation config not found: $validation_config"
        return 1
    fi

    # Determine input CSV (filtered or original)
    if [ -f "$merged_dir/filtered_designs.csv" ]; then
        INPUT_CSV="$merged_dir/filtered_designs.csv"
        echo "[INFO] Validating post-filtered designs"
    elif [ -f "$merged_dir/final_design_stats.csv" ]; then
        INPUT_CSV="$merged_dir/final_design_stats.csv"
        echo "[INFO] Validating BindCraft final designs"
    else
        echo "[WARN] No design CSV found, skipping multi-state validation"
        return 1
    fi

    # Split CSV for parallel processing
    split_designs_for_validation "$INPUT_CSV" "$num_gpus" "$merged_dir"

    # Launch validation workers
    launch_validation_workers "$num_gpus" "$validation_config"

    # Wait and merge results
    wait_and_merge_validation "$num_gpus" "$merged_dir"
}
