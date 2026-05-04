# BindBot Multi-GPU Launcher - Quick Start

Simple multi-GPU launcher for running BindCraft on RunPod servers.

## Quick Start

1. **Edit the config section** in [run_BindBot.sh](run_BindBot.sh:14-25):
   ```bash
   # Required: Path to your settings file (target configuration)
   SETTINGS_FILE="settings_target/PDL1.json"

   # Optional: Path to filters and advanced settings
   FILTERS_FILE="settings_filters/default_filters.json"
   ADVANCED_FILE="settings_advanced/default_4stage_multimer.json"

   # Total number of designs to generate (will be split across GPUs)
   TOTAL_DESIGNS=100

   # Additional options (set to "true" to enable)
   NO_PYROSETTA=false
   VERBOSE=false
   ```

2. **Run it**:
   ```bash
   ./run_BindBot.sh
   ```

That's it! The script will:
- Auto-detect available GPUs
- Split work evenly across all GPUs
- Run BindCraft instances in parallel
- Auto-merge results when done
- Clean up temporary files

## Settings Files

### Target Settings (`settings_target/*.json`)

**Important**: Do NOT include `number_of_final_designs` in your settings files - use `TOTAL_DESIGNS` in the script instead.

Example `settings_target/PDL1.json`:
```json
{
    "design_path": "/workspace/outputs/PDL1_designs",
    "binder_name": "PDL1",
    "starting_pdb": "/app/Inputs/PDL1.pdb",
    "chains": "A",
    "target_hotspot_residues": "56",
    "lengths": [65, 150]
}
```

**Required fields:**
- `design_path` - Output directory (e.g., `/workspace/outputs/ProjectName`)
- `binder_name` - Name for your binder protein
- `starting_pdb` - Path to target PDB file (use `/app/Inputs/filename.pdb`)
- `chains` - Target chain(s) (e.g., `"A"` or `"A,B"`)
- `lengths` - Min/max binder length (e.g., `[65, 150]` for miniprotein, `[8, 25]` for peptide)

**Optional fields:**
- `target_hotspot_residues` - Specific residues to target (e.g., `"56"` or `"A56,B78"`)

### File Paths

All paths in settings files should use the **container paths**:
- PDB files: `/app/Inputs/your_file.pdb`
- Output: `/workspace/outputs/your_project_name`
- Filters: `/app/settings_filters/filter_name.json`
- Advanced: `/app/settings_advanced/advanced_name.json`

## Available Presets

### Filter Presets (`settings_filters/`)
- `default_filters.json` - Standard filtering
- `relaxed_filters.json` - More permissive filtering
- `peptide_filters.json` - Optimized for peptides
- `no_filters.json` - No filtering (accept all)

### Advanced Presets (`settings_advanced/`)
- `default_4stage_multimer.json` - Standard 4-stage design
- `default_4stage_multimer_mpnn.json` - With MPNN sequence design
- `default_4stage_multimer_flexible.json` - Flexible backbone
- `peptide_3stage_multimer.json` - For peptide binders (8-30 aa)
- `betasheet_4stage_multimer.json` - Beta-sheet scaffolds

## Examples

### Example 1: Simple PDL1 Binder (100 designs)

Edit `run_BindBot.sh`:
```bash
SETTINGS_FILE="settings_target/PDL1.json"
FILTERS_FILE="settings_filters/default_filters.json"
ADVANCED_FILE="settings_advanced/default_4stage_multimer.json"
TOTAL_DESIGNS=100
```

Run: `./run_BindBot.sh`

### Example 2: IgG Fc Binder (500 designs, relaxed filters)

Edit `run_BindBot.sh`:
```bash
SETTINGS_FILE="settings_target/IgG_Fc.json"
FILTERS_FILE="settings_filters/relaxed_filters.json"
ADVANCED_FILE="settings_advanced/default_4stage_multimer_mpnn.json"
TOTAL_DESIGNS=500
```

Run: `./run_BindBot.sh`

### Example 3: Fast run without PyRosetta

Edit `run_BindBot.sh`:
```bash
SETTINGS_FILE="settings_target/PDL1.json"
TOTAL_DESIGNS=100
NO_PYROSETTA=true  # Skip relaxation for speed
VERBOSE=true       # Show detailed progress
```

Run: `./run_BindBot.sh`

## GPU Distribution

The script automatically distributes work across available GPUs:

| GPUs | Total Designs | Per GPU |
|------|---------------|---------|
| 1    | 100           | 100     |
| 2    | 100           | 50, 50  |
| 4    | 100           | 25 each |
| 4    | 500           | 125 each |
| 8    | 1000          | 125 each |

## Monitoring

### Check Progress

```bash
# Watch all GPU logs
tail -f /workspace/bindcraft_gpu*.log

# Check specific GPU
tail -f /workspace/bindcraft_gpu0.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Check Disk Space

```bash
df -h /workspace
du -sh /workspace/outputs/*
```

## Output Structure

After completion, results are in the `design_path` specified in your settings:

```
/workspace/outputs/PDL1_designs/
├── Accepted/              # Designs that passed filters
│   ├── design_001.pdb
│   ├── design_002.pdb
│   └── ...
├── Rejected/              # Designs that failed filters
├── Trajectory/            # Design trajectories
│   ├── Plots/
│   └── Structures/
├── accepted_mpnn_full_stats.csv    # Statistics for accepted designs
├── rejected_mpnn_full_stats.csv    # Statistics for rejected designs
├── trajectory_stats.csv            # Per-trajectory statistics
└── filter_pass_fail.csv           # Filter performance
```

## Troubleshooting

### Issue: "File not found" error

**Solution**: Check that paths in your settings JSON use `/app/` prefix:
```json
"starting_pdb": "/app/Inputs/PDL1.pdb"  ✓ Correct
"starting_pdb": "./Inputs/PDL1.pdb"     ✗ Wrong
```

### Issue: GPU not being used

**Solution**: Check GPU visibility:
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

### Issue: Out of memory

**Solution**: Reduce `TOTAL_DESIGNS` or set `NO_PYROSETTA=true`

### Issue: One GPU fails

**Solution**: Check the specific GPU log:
```bash
cat /workspace/bindcraft_gpu<N>.log
```

## Performance Tips

### For Speed (No Relaxation)
```bash
NO_PYROSETTA=true
```
~50% faster, but designs won't be relaxed

### For Quality (Flexible Backbone)
```bash
ADVANCED_FILE="settings_advanced/default_4stage_multimer_flexible.json"
```
Better quality, but slower

### For Peptides (8-30 aa)
```bash
ADVANCED_FILE="settings_advanced/peptide_3stage_multimer.json"
FILTERS_FILE="settings_filters/peptide_filters.json"
```
In settings JSON: `"lengths": [8, 25]`

## Adding Your Own Target

1. Place your PDB file in the `Inputs/` directory
2. Create a new settings file:

```json
{
    "design_path": "/workspace/outputs/MyTarget_designs",
    "binder_name": "MyTarget",
    "starting_pdb": "/app/Inputs/my_target.pdb",
    "chains": "A",
    "target_hotspot_residues": "100,150,200",
    "lengths": [65, 150]
}
```

3. Update `run_BindBot.sh`:
```bash
SETTINGS_FILE="settings_target/MyTarget.json"
TOTAL_DESIGNS=500
```

4. Run: `./run_BindBot.sh`

## Need Help?

- Check logs: `/workspace/*.log`
- Verify GPUs: `nvidia-smi`
- Test settings: `python /app/bindcraft.py --help`
- See full documentation: `MULTI_GPU_README.md`
