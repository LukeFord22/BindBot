# BindBot

Multi-GPU launcher for [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) optimized for RunPod deployment.

## What This Is

This repository takes the FreeBindCraft pipeline (a modified version of [BindCraft](https://github.com/martinpacesa/BindCraft)) and adds:

- **Multi-GPU parallelization** - Automatically detects and distributes work across all available GPUs
- **RunPod optimization** - Pre-configured for RunPod server environments
- **Simplified launcher** - Edit a config section and run, no command-line arguments needed
- **Auto-merge results** - Automatically combines outputs from all GPUs when complete

## Quick Start

1. **Edit the config** in `run_bindBot.sh`:
   ```bash
   SETTINGS_FILE="settings_target/PDL1.json"
   TOTAL_DESIGNS=100
   ```

2. **Run it**:
   ```bash
   ./run_bindBot.sh
   ```

The script automatically:
- Detects available GPUs (1, 2, 4, 8, etc.)
- Splits the workload evenly
- Runs parallel BindCraft instances
- Merges results when complete
- Cleans up temporary files

## What We Modified

**Base**: [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) (BindCraft v1.52 with optional PyRosetta bypass)

**Our additions**:
- `run_bindBot.sh` - Multi-GPU launcher with auto-merge
- Modified settings files - Removed duplicate config, fixed paths for containers
- `USAGE.md` - Simple usage guide

**Unchanged**: All BindCraft pipeline code, AlphaFold, MPNN, filters, etc.

## Documentation

- **Quick start**: `USAGE.md`
- **Full multi-GPU details**: `MULTI_GPU_README.md`
- **Original BindCraft docs**: [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft)
- **FreeBindCraft details**: See `technical_overview/` directory

## Requirements

- NVIDIA GPU(s) with CUDA support
- Docker (for RunPod deployment)
- See Dockerfile for full environment details

## Example Usage

### Single Target (PDL1)
```bash
# Edit run_bindBot.sh:
SETTINGS_FILE="settings_target/PDL1.json"
TOTAL_DESIGNS=500

# Run
./run_bindBot.sh
```

### Multiple GPUs
No special configuration needed - automatically detected:
- 1 GPU: Runs 500 designs on 1 GPU
- 4 GPUs: Runs 125 designs per GPU (500 total)
- 8 GPUs: Runs 62-63 designs per GPU (500 total)

Results are automatically merged into a single output directory.

## Output

Results in `/workspace/outputs/` with:
- `Accepted/` - Designs that passed filters
- `Rejected/` - Designs that failed filters
- `Trajectory/` - Design trajectories and plots
- CSV files with statistics

## Credits

- **BindCraft**: Martin Pacesa et al. - [Original paper](https://www.biorxiv.org/content/10.1101/2024.09.30.615802)
- **FreeBindCraft**: PyRosetta-optional fork by cytokineking
- **BindBot**: Multi-GPU launcher modifications

## License

Same as FreeBindCraft/BindCraft - See LICENSE file
