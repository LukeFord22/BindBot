#!/usr/bin/env python3
"""
Integration Script for Custom AF2 Losses

This script patches BindCraft to support custom AF2 losses via configuration.
It modifies bindcraft.py to load and apply custom losses during binder design.

Usage:
    # Apply integration (modifies bindcraft.py)
    python integrate_custom_losses.py --apply

    # Revert integration (restores original)
    python integrate_custom_losses.py --revert

    # Check if integration is applied
    python integrate_custom_losses.py --check
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path


BINDCRAFT_PY = Path("/app/bindcraft.py") if Path("/app/bindcraft.py").exists() else Path("bindcraft.py")
BACKUP_PATH = BINDCRAFT_PY.with_suffix('.py.backup')


# Code to insert into bindcraft.py
INTEGRATION_MARKER_START = "### CUSTOM AF2 LOSSES INTEGRATION START ###"
INTEGRATION_MARKER_END = "### CUSTOM AF2 LOSSES INTEGRATION END ###"

INTEGRATION_CODE = f"""
{INTEGRATION_MARKER_START}
# Load and apply custom AF2 losses if config file exists
custom_losses_config_path = args.custom_losses_config if hasattr(args, 'custom_losses_config') else None
if custom_losses_config_path is None:
    # Try default location
    custom_losses_config_path = os.path.join(os.path.dirname(__file__), 'settings_losses', 'custom_af2_losses.json')

if custom_losses_config_path and os.path.exists(custom_losses_config_path):
    try:
        with open(custom_losses_config_path, 'r') as f:
            custom_losses_config = json.load(f)

        # Import custom losses module
        from functions.custom_af2_losses import apply_custom_losses_from_config

        # Apply custom losses to AF model
        af_model = apply_custom_losses_from_config(af_model, custom_losses_config)

        print(f"[INFO] Custom AF2 losses loaded from: {{custom_losses_config_path}}")
    except Exception as e:
        print(f"[WARN] Failed to load custom losses: {{e}}")
        print("[WARN] Continuing with standard losses only")
{INTEGRATION_MARKER_END}
"""


def check_integration():
    """Check if integration is already applied"""
    if not BINDCRAFT_PY.exists():
        print(f"[ERROR] {BINDCRAFT_PY} not found")
        return False

    with open(BINDCRAFT_PY, 'r') as f:
        content = f.read()

    return INTEGRATION_MARKER_START in content


def find_insertion_point():
    """Find the best place to insert custom loss code"""
    with open(BINDCRAFT_PY, 'r') as f:
        lines = f.readlines()

    # Look for where AF model is created/initialized
    # Typical patterns:
    # - af_model = mk_afdesign_model(...)
    # - af_model.prep_inputs(...)

    for idx, line in enumerate(lines):
        # Insert after af_model is created and prepared
        if 'af_model.prep_inputs' in line and 'design' in line.lower():
            # Find the end of this statement (could be multi-line)
            insert_idx = idx + 1
            while insert_idx < len(lines) and lines[insert_idx].strip().startswith('rm_'):
                insert_idx += 1
            return insert_idx

    # Fallback: look for first design loop
    for idx, line in enumerate(lines):
        if 'for seed in' in line or 'range(target_settings["number_of_final_designs"])' in line:
            return idx

    print("[ERROR] Could not find suitable insertion point")
    return None


def apply_integration():
    """Apply integration to bindcraft.py"""
    if check_integration():
        print("[INFO] Integration already applied")
        return True

    print("[INFO] Backing up original bindcraft.py...")
    shutil.copy(BINDCRAFT_PY, BACKUP_PATH)

    print("[INFO] Finding insertion point...")
    insert_idx = find_insertion_point()

    if insert_idx is None:
        print("[ERROR] Could not determine where to insert custom loss code")
        print("[ERROR] Manual integration required")
        return False

    print(f"[INFO] Inserting custom loss code at line {insert_idx}...")

    with open(BINDCRAFT_PY, 'r') as f:
        lines = f.readlines()

    # Insert the integration code
    lines.insert(insert_idx, INTEGRATION_CODE + '\n')

    with open(BINDCRAFT_PY, 'w') as f:
        f.writelines(lines)

    print(f"[SUCCESS] Integration applied successfully")
    print(f"[INFO] Backup saved to: {BACKUP_PATH}")
    print(f"[INFO] Custom losses can now be configured in settings_losses/custom_af2_losses.json")

    return True


def revert_integration():
    """Revert integration (restore from backup)"""
    if not check_integration():
        print("[INFO] Integration not currently applied")
        return True

    if not BACKUP_PATH.exists():
        print("[ERROR] Backup file not found, cannot revert automatically")
        print("[ERROR] Please restore bindcraft.py manually")
        return False

    print("[INFO] Restoring original bindcraft.py from backup...")
    shutil.copy(BACKUP_PATH, BINDCRAFT_PY)

    print("[SUCCESS] Integration reverted successfully")
    return True


def generate_integration_instructions():
    """Generate manual integration instructions"""
    instructions = """
=== MANUAL INTEGRATION INSTRUCTIONS ===

To manually integrate custom AF2 losses into BindCraft:

1. Open bindcraft.py in your editor

2. Locate where the AF2 model is initialized and prepared. Look for lines like:

   af_model = mk_afdesign_model(...)
   af_model.prep_inputs(...)

3. After the af_model.prep_inputs() call, add the following code:

---CODE START---
""" + INTEGRATION_CODE + """
---CODE END---

4. Save bindcraft.py

5. Create/edit settings_losses/custom_af2_losses.json to enable desired losses

6. Run BindCraft normally. Custom losses will be automatically applied if config exists.

=== ALTERNATIVE: Use as standalone patch ===

You can also add --custom-losses-config argument to bindcraft.py argparse:

parser.add_argument('--custom-losses-config', type=str, default=None,
                   help='Path to custom AF2 losses configuration JSON')

Then load and apply in your design loop:

if args.custom_losses_config:
    from functions.custom_af2_losses import apply_custom_losses_from_config
    with open(args.custom_losses_config) as f:
        custom_config = json.load(f)
    af_model = apply_custom_losses_from_config(af_model, custom_config)

========================================
"""
    return instructions


def main():
    parser = argparse.ArgumentParser(description='Integrate custom AF2 losses into BindCraft')
    parser.add_argument('--apply', action='store_true', help='Apply integration')
    parser.add_argument('--revert', action='store_true', help='Revert integration')
    parser.add_argument('--check', action='store_true', help='Check integration status')
    parser.add_argument('--manual-instructions', action='store_true', help='Show manual integration instructions')

    args = parser.parse_args()

    if args.check or (not args.apply and not args.revert and not args.manual_instructions):
        # Default action: check status
        if check_integration():
            print("[INFO] Custom AF2 losses integration: APPLIED")
            print(f"[INFO] Configure losses in: settings_losses/custom_af2_losses.json")
        else:
            print("[INFO] Custom AF2 losses integration: NOT APPLIED")
            print(f"[INFO] Run with --apply to integrate, or --manual-instructions for manual setup")
        sys.exit(0)

    if args.manual_instructions:
        print(generate_integration_instructions())
        sys.exit(0)

    if args.apply:
        success = apply_integration()
        sys.exit(0 if success else 1)

    if args.revert:
        success = revert_integration()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
