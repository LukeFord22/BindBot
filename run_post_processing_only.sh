#!/bin/bash
#
# Standalone Post-Processing Script
# Runs post-filtering and/or multi-state validation on existing BindCraft results
# Usage: ./run_post_processing_only.sh [OPTIONS]
#

set -euo pipefail


#############################################
### CONFIG - EDIT THESE SETTINGS
#############################################

# Required: Path to your BindCraft output directory
OUTPUT_DIR="/workspace/outputs/your_design_here"

# Post-filtering: Apply additional biochemical/structural filters to accepted designs
ENABLE_POST_FILTERING=true
POST_FILTER_CONFIG="settings/settings_post_filter/post_filter_config.json"

# Multi-state validation: Test binders against multiple target conformations
ENABLE_MULTI_STATE=true
MULTI_STATE_CONFIG="settings/settings_validation/multi_state_config.json"

# GPU settings (only used for multi-state validation if parallel mode enabled)
PARALLEL_MODE=false  # Set to true to use multiple GPUs for validation
NUM_GPUS=1           # Number of GPUs to use (only if PARALLEL_MODE=true)

#############################################
### END CONFIG
#############################################

BINDCRAFT_DIR="/workspace/BindBot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$OUTPUT_DIR/post_processing.log"


#############################################
### ARGUMENT PARSING
#############################################

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Standalone post-processing script for BindCraft results.

OPTIONS:
    -i, --input DIR           Input directory (BindCraft output) [required]
    -p, --post-filter         Enable post-filtering
    -m, --multi-state         Enable multi-state validation
    --post-filter-config PATH Config file for post-filtering
    --multi-state-config PATH Config file for multi-state validation
    --parallel                Enable parallel multi-state validation (multi-GPU)
    --num-gpus N              Number of GPUs to use for parallel validation
    -h, --help                Show this help message

EXAMPLES:
    # Run both post-filtering and multi-state validation
    ./run_post_processing_only.sh -i /workspace/outputs/PDL1_designs -p -m

    # Run only post-filtering
    ./run_post_processing_only.sh -i /workspace/outputs/PDL1_designs -p

    # Run only multi-state validation with 4 GPUs in parallel
    ./run_post_processing_only.sh -i /workspace/outputs/PDL1_designs -m --parallel --num-gpus 4

NOTES:
    - Input directory must contain BindCraft output (final_design_stats.csv or Accepted/ folder)
    - Post-filtering reads from final_design_stats.csv or accepted_mpnn_full_stats.csv
    - Multi-state validation reads from filtered_designs.csv (if post-filter ran) or original designs
    - Results are saved in the same input directory

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--post-filter)
            ENABLE_POST_FILTERING=true
            shift
            ;;
        -m|--multi-state)
            ENABLE_MULTI_STATE=true
            shift
            ;;
        --post-filter-config)
            POST_FILTER_CONFIG="$2"
            shift 2
            ;;
        --multi-state-config)
            MULTI_STATE_CONFIG="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done


#############################################
### VALIDATION
#############################################

# Verify output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[ERROR] Output directory not found: $OUTPUT_DIR"
    echo "[HINT] Use -i or --input to specify the BindCraft output directory"
    exit 1
fi

# Verify at least one processing step is enabled
if [ "$ENABLE_POST_FILTERING" != true ] && [ "$ENABLE_MULTI_STATE" != true ]; then
    echo "[ERROR] No processing steps enabled. Use -p for post-filtering or -m for multi-state validation"
    show_help
    exit 1
fi

# Make OUTPUT_DIR absolute
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# Setup logging
mkdir -p "$OUTPUT_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [POST-PROCESSING STANDALONE] $(date) ==="
echo "[INFO] Input directory: $OUTPUT_DIR"
echo "[INFO] Post-filtering: $ENABLE_POST_FILTERING"
echo "[INFO] Multi-state validation: $ENABLE_MULTI_STATE"
echo "[INFO] Parallel mode: $PARALLEL_MODE"
if [ "$PARALLEL_MODE" = true ]; then
    echo "[INFO] Number of GPUs: $NUM_GPUS"
fi
echo ""


#############################################
### PATH NORMALIZATION
#############################################

# Make config paths absolute if they're relative
if [[ "$POST_FILTER_CONFIG" != /* ]]; then
    POST_FILTER_CONFIG="$BINDCRAFT_DIR/$POST_FILTER_CONFIG"
fi
if [[ "$MULTI_STATE_CONFIG" != /* ]]; then
    MULTI_STATE_CONFIG="$BINDCRAFT_DIR/$MULTI_STATE_CONFIG"
fi

# Verify config files exist (only if feature is enabled)
if [ "$ENABLE_POST_FILTERING" = true ] && [ ! -f "$POST_FILTER_CONFIG" ]; then
    echo "[ERROR] Post-filter config not found: $POST_FILTER_CONFIG"
    exit 1
fi
if [ "$ENABLE_MULTI_STATE" = true ] && [ ! -f "$MULTI_STATE_CONFIG" ]; then
    echo "[ERROR] Multi-state config not found: $MULTI_STATE_CONFIG"
    exit 1
fi


#############################################
### CONDA ENVIRONMENT ACTIVATION
#############################################

echo "[STEP] Activating BindCraft conda environment..."
# Temporarily disable nounset to avoid conda activation script issues
set +u
if [ -f "/miniforge3/etc/profile.d/conda.sh" ]; then
    source /miniforge3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "[ERROR] Could not find conda initialization script"
    exit 1
fi

conda activate BindCraft || {
    echo "[ERROR] Failed to activate BindCraft environment"
    exit 1
}
# Re-enable nounset after conda activation
set -u
echo "[SUCCESS] Environment activated"
echo ""


#############################################
### SOURCE LIBRARY MODULES
#############################################

# Set required variables for library modules
WORKSPACE_DIR="$(dirname "$OUTPUT_DIR")"
TEMP_DIR="$OUTPUT_DIR/post_processing_temp"
MERGED_DIR="$OUTPUT_DIR"  # For compatibility with multi-GPU functions
mkdir -p "$TEMP_DIR"

# Source post-processing module
if [ -f "$BINDCRAFT_DIR/lib/post_processing.sh" ]; then
    source "$BINDCRAFT_DIR/lib/post_processing.sh"
else
    echo "[ERROR] Post-processing library not found: $BINDCRAFT_DIR/lib/post_processing.sh"
    exit 1
fi


#############################################
### VERIFY INPUT DATA EXISTS
#############################################

echo "[STEP] Verifying input data..."

# Check for accepted designs
if [ ! -d "$OUTPUT_DIR/Accepted" ] || [ -z "$(ls -A "$OUTPUT_DIR/Accepted" 2>/dev/null)" ]; then
    echo "[WARN] No designs found in $OUTPUT_DIR/Accepted/"
fi

# Check for design CSV files
HAS_CSV=false
if [ -f "$OUTPUT_DIR/final_design_stats.csv" ]; then
    echo "[FOUND] final_design_stats.csv"
    HAS_CSV=true
elif [ -f "$OUTPUT_DIR/accepted_mpnn_full_stats.csv" ]; then
    echo "[FOUND] accepted_mpnn_full_stats.csv"
    HAS_CSV=true
elif [ -f "$OUTPUT_DIR/mpnn_design_stats.csv" ]; then
    echo "[FOUND] mpnn_design_stats.csv"
    HAS_CSV=true
fi

if [ "$HAS_CSV" = false ]; then
    echo "[ERROR] No design CSV files found in $OUTPUT_DIR"
    echo "[HINT] Expected one of: final_design_stats.csv, accepted_mpnn_full_stats.csv, mpnn_design_stats.csv"
    exit 1
fi

echo "[SUCCESS] Input data verified"
echo ""


#############################################
### EXECUTION: POST-FILTERING
#############################################

POST_FILTER_SUCCESS=false

if [ "$ENABLE_POST_FILTERING" = true ]; then
    echo "=== [STEP 1/2] POST-FILTERING ==="
    echo ""

    if run_post_filter "$OUTPUT_DIR"; then
        POST_FILTER_SUCCESS=true
        echo "[SUCCESS] Post-filtering completed"
    else
        echo "[WARN] Post-filtering failed or no designs passed"
        echo "[INFO] Multi-state validation will use original designs"
    fi
    echo ""
fi


#############################################
### EXECUTION: MULTI-STATE VALIDATION
#############################################

if [ "$ENABLE_MULTI_STATE" = true ]; then
    echo "=== [STEP 2/2] MULTI-STATE VALIDATION ==="
    echo ""

    if [ "$PARALLEL_MODE" = true ] && [ "$NUM_GPUS" -gt 1 ]; then
        echo "[MODE] Running parallel validation across $NUM_GPUS GPUs"

        # Check GPU availability
        if ! command -v nvidia-smi &> /dev/null; then
            echo "[ERROR] nvidia-smi not found. Cannot use parallel mode."
            exit 1
        fi

        AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
            echo "[WARN] Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
            NUM_GPUS=$AVAILABLE_GPUS
        fi

        run_multi_state_validation_parallel "$OUTPUT_DIR" "$NUM_GPUS"
    else
        echo "[MODE] Running sequential validation on single GPU"
        run_multi_state_validation_sequential "$OUTPUT_DIR" "$POST_FILTER_SUCCESS"
    fi

    echo ""
fi


#############################################
### CLEANUP & SUMMARY
#############################################

echo "=== [CLEANUP] ==="
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "[CLEANUP] Removed temporary files"
fi
echo ""

echo "=== [SUMMARY] ==="
echo "[OUTPUT] All results saved to: $OUTPUT_DIR"
echo ""

if [ "$ENABLE_POST_FILTERING" = true ]; then
    if [ -f "$OUTPUT_DIR/filtered_designs.csv" ]; then
        NUM_FILTERED=$(tail -n +2 "$OUTPUT_DIR/filtered_designs.csv" 2>/dev/null | wc -l || echo 0)
        echo "[RESULT] Post-filtering: $NUM_FILTERED designs passed"
    else
        echo "[RESULT] Post-filtering: No output generated"
    fi
fi

if [ "$ENABLE_MULTI_STATE" = true ]; then
    if [ -f "$OUTPUT_DIR/multi_state_scores.csv" ]; then
        NUM_VALIDATED=$(tail -n +2 "$OUTPUT_DIR/multi_state_scores.csv" 2>/dev/null | wc -l || echo 0)
        echo "[RESULT] Multi-state validation: $NUM_VALIDATED designs validated"
    else
        echo "[RESULT] Multi-state validation: No output generated"
    fi
fi

echo ""
echo "=== [FINISHED] $(date) ==="
echo "[LOG] Full log saved to: $LOG_FILE"
