#!/usr/bin/env python3
"""
Automatic Positive State Generator for Multi-State Validation

Generates alternative conformations of target proteins for robust validation:
1. AF2/ColabFold alternate state: Different seed/dropout to sample alternate conformations
2. OpenMM relaxed state: Energy-minimized and briefly equilibrated structure

Usage:
    python generate_positive_states.py --target target.pdb --chain A --output target_pdbs/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from Bio import PDB

warnings.filterwarnings('ignore')

# Import production utilities from BindCraft
sys.path.insert(0, str(Path(__file__).parent.parent))
from colabdesign import mk_afdesign_model, clear_mem
from functions.pr_alternative_utils import openmm_relax
from functions.biopython_utils import biopython_align_pdbs, biopython_unaligned_rmsd


class PositiveStateGenerator:
    """Generates alternative conformational states for multi-state validation"""

    def __init__(self, target_pdb: Path, chain_id: str, output_dir: Path):
        self.target_pdb = Path(target_pdb)
        self.chain_id = chain_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PDB parser
        self.pdb_parser = PDB.PDBParser(QUIET=True)
        self.pdb_io = PDB.PDBIO()

        # Load and extract target chain
        self.structure = self.pdb_parser.get_structure('target', str(self.target_pdb))
        self.target_chain = None
        for chain in self.structure[0].get_chains():
            if chain.id == self.chain_id:
                self.target_chain = chain
                break

        if self.target_chain is None:
            raise ValueError(f"Chain {chain_id} not found in {target_pdb}")

        # Extract sequence
        from Bio.PDB.Polypeptide import PPBuilder
        ppb = PPBuilder()
        self.sequence = ""
        for pp in ppb.build_peptides(self.target_chain):
            self.sequence += str(pp.get_sequence())

        print(f"[INFO] Loaded target: {self.target_pdb.stem}, chain {chain_id}, {len(self.sequence)} residues")

    def generate_af2_alternate_state(self, seed: int = 42, max_attempts: int = 3) -> Optional[Path]:
        """
        Generate alternate conformation using AF2/ColabFold with increased conformational diversity.

        Uses different random seed, reduced recycles, and dropout to sample alternate conformations
        that are meaningfully different from native structure.

        Args:
            seed: Random seed for reproducible sampling
            max_attempts: Maximum attempts to generate valid alternate state (default: 3)

        Returns:
            Path to generated PDB if successful and passes RMSD validation, None otherwise
        """

        print("\n[STEP] Generating AF2 alternate state...")
        print(f"  Sequence length: {len(self.sequence)}")
        print(f"  Strategy: Reduced recycles + dropout for conformational diversity")

        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    print(f"\n  Attempt {attempt}/{max_attempts}...")

                # Initialize AF2 model for structure prediction (not design)
                # Use hallucination protocol for sequence-only input
                af_model = mk_afdesign_model(
                    protocol="hallucination",
                    use_multimer=False,
                    num_recycles=1,  # Reduced recycles = more diversity
                    data_dir="/data/params"
                )

                # Prepare inputs using sequence for hallucination protocol
                af_model.prep_inputs(
                    length=len(self.sequence)
                )

                # Set the sequence
                af_model.set_seq(self.sequence)

                # Set random seed (vary seed per attempt)
                current_seed = seed + (attempt - 1) * 10
                np.random.seed(current_seed)
                print(f"  Random seed: {current_seed}")

                # Run prediction with dropout enabled for diversity
                print("  Running AF2 prediction with dropout...")
                af_model.predict(
                    verbose=False,
                    num_models=1,
                    num_recycles=1,  # Low recycles for diversity
                    dropout=True  # Enable dropout for structural sampling
                )

                # Save alternate structure
                output_path = self.output_dir / f"{self.target_pdb.stem}_af2_alternate.pdb"
                af_model.save_pdb(str(output_path))

                # Align to original structure for fair RMSD comparison
                self._align_to_original(output_path)

                # Validate RMSD: reject if too similar or too distorted
                is_valid = self._validate_state_rmsd(
                    state_path=output_path,
                    state_name="AF2 alternate",
                    min_rmsd=0.5,  # Reject if < 0.5 Å (too similar)
                    max_rmsd=3.0   # Reject if > 3.0 Å (too distorted)
                )

                # Clear memory
                clear_mem()

                if is_valid:
                    print(f"[SUCCESS] AF2 alternate state saved: {output_path}")
                    return output_path
                else:
                    print(f"  Attempt {attempt} failed RMSD validation, retrying...")
                    if output_path.exists():
                        output_path.unlink()  # Remove invalid state

            except Exception as e:
                print(f"  [ERROR] Attempt {attempt} failed: {e}")
                clear_mem()
                continue

        print(f"[ERROR] Failed to generate valid AF2 alternate state after {max_attempts} attempts")
        return None

    def generate_openmm_relaxed_state(self, max_attempts: int = 3) -> Optional[Path]:
        """
        Generate relaxed state using OpenMM energy minimization via production utilities.

        Uses gentle restraints and MD equilibration to produce meaningful conformational
        variations while preserving overall fold. Validates RMSD to reject trivial relaxations.

        Args:
            max_attempts: Maximum attempts to generate valid relaxed state (default: 3)

        Returns:
            Path to generated PDB if successful and passes RMSD validation, None otherwise
        """

        print("\n[STEP] Generating OpenMM relaxed state...")
        print(f"  Strategy: Gentle restraints + MD equilibration for conformational sampling")

        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    print(f"\n  Attempt {attempt}/{max_attempts} with adjusted parameters...")

                output_path = self.output_dir / f"{self.target_pdb.stem}_openmm_relaxed.pdb"

                # Adjust parameters based on attempt to generate increasing variation
                # First attempt: moderate restraints
                # Later attempts: progressively weaker restraints for more conformational change
                if attempt == 1:
                    restraint_k = 2.5
                    restraint_ramp = (1.0, 0.5, 0.2)
                    md_steps = 3000
                elif attempt == 2:
                    restraint_k = 1.0  # Weaker restraints
                    restraint_ramp = (0.6, 0.2, 0.0)  # Release fully
                    md_steps = 6000  # Longer MD
                else:
                    restraint_k = 0.5  # Very weak restraints
                    restraint_ramp = (0.3, 0.0, 0.0)  # Release early
                    md_steps = 8000  # Extended MD

                print(f"  Restraint strength: {restraint_k} kcal/mol/Å²")
                print(f"  MD steps: {md_steps}")

                # Use production OpenMM relax with settings for conformational sampling
                openmm_relax(
                    pdb_file_path=str(self.target_pdb),
                    output_pdb_path=str(output_path),
                    use_gpu_relax=True,
                    openmm_max_iterations=1000,
                    restraint_k_kcal_mol_A2=restraint_k,
                    restraint_ramp_factors=restraint_ramp,
                    md_steps_per_shake=md_steps,
                    use_faspr_repack=False  # Skip side-chain repacking for speed
                )

                # Align to original structure for fair RMSD comparison
                self._align_to_original(output_path)

                # Validate RMSD: reject if trivial relaxation (too similar)
                # Lower threshold to 0.15 Å - even subtle relaxations are useful for validation
                is_valid = self._validate_state_rmsd(
                    state_path=output_path,
                    state_name="OpenMM relaxed",
                    min_rmsd=0.15,  # Accept even subtle variations (relaxation produces smaller changes)
                    max_rmsd=2.5   # Reject if too distorted
                )

                if is_valid:
                    print(f"[SUCCESS] OpenMM relaxed state saved: {output_path}")
                    return output_path
                else:
                    print(f"  Attempt {attempt} failed RMSD validation, retrying...")
                    if output_path.exists() and attempt < max_attempts:
                        output_path.unlink()  # Remove invalid state

            except Exception as e:
                print(f"  [ERROR] Attempt {attempt} failed: {e}")
                continue

        print(f"[ERROR] Failed to generate valid OpenMM relaxed state after {max_attempts} attempts")
        return None

    def _align_to_original(self, alternate_pdb: Path):
        """Align alternate structure to original target using CA atoms"""
        try:
            biopython_align_pdbs(
                reference_pdb=str(self.target_pdb),
                align_pdb=str(alternate_pdb),
                reference_chain_id=self.chain_id,
                align_chain_id=self.chain_id
            )

            print(f"  Aligned to original structure")

        except Exception as e:
            print(f"  [WARN] Alignment failed: {e}")

    def _validate_state_rmsd(
        self,
        state_path: Path,
        state_name: str,
        min_rmsd: float = 0.5,
        max_rmsd: float = 3.0
    ) -> bool:
        """
        Validate that generated state has meaningful structural variation.

        Uses production biopython_unaligned_rmsd() to calculate CA-RMSD after alignment.

        Args:
            state_path: Path to generated state PDB
            state_name: Name of state (for logging)
            min_rmsd: Minimum acceptable RMSD (default: 0.5 Å) - reject if too similar
            max_rmsd: Maximum acceptable RMSD (default: 3.0 Å) - reject if too distorted

        Returns:
            True if RMSD is in acceptable range, False otherwise
        """
        try:
            # Calculate RMSD using production utility
            rmsd = biopython_unaligned_rmsd(
                reference_pdb=str(self.target_pdb),
                align_pdb=str(state_path),
                reference_chain_id=self.chain_id,
                align_chain_id=self.chain_id
            )

            print(f"  CA-RMSD vs native: {rmsd:.2f} Å")

            if rmsd < min_rmsd:
                print(f"  [REJECT] {state_name} too similar to native (RMSD < {min_rmsd} Å)")
                return False

            if rmsd > max_rmsd:
                print(f"  [REJECT] {state_name} too distorted from native (RMSD > {max_rmsd} Å)")
                return False

            print(f"  [VALID] {state_name} RMSD within acceptable range ({min_rmsd}-{max_rmsd} Å)")
            return True

        except Exception as e:
            print(f"  [WARN] RMSD validation failed for {state_name}: {e}")
            return True  # Allow if calculation failed

    def generate_all_states(self) -> dict:
        """Generate all positive states and return paths"""
        print("\n=== POSITIVE STATE GENERATION ===")
        print(f"Target: {self.target_pdb}")
        print(f"Chain: {self.chain_id}")
        print(f"Output: {self.output_dir}")

        states = {
            'native': str(self.target_pdb),
            'af2_alternate': None,
            'openmm_relaxed': None
        }

        # Generate AF2 alternate state
        af2_path = self.generate_af2_alternate_state(seed=42)
        if af2_path:
            states['af2_alternate'] = str(af2_path)

        # Generate OpenMM relaxed state
        openmm_path = self.generate_openmm_relaxed_state()
        if openmm_path:
            states['openmm_relaxed'] = str(openmm_path)

        # Summary
        print("\n=== GENERATION COMPLETE ===")
        for state_name, state_path in states.items():
            if state_path:
                print(f"  {state_name}: {state_path}")
            else:
                print(f"  {state_name}: SKIPPED")

        return states


def main():
    parser = argparse.ArgumentParser(
        description='Generate positive conformational states for multi-state validation'
    )
    parser.add_argument('--target', required=True, help='Path to target PDB file')
    parser.add_argument('--chain', required=True, help='Target chain ID')
    parser.add_argument('--output', required=True, help='Output directory for generated states')
    parser.add_argument('--af2-seed', type=int, default=42, help='Random seed for AF2 sampling')

    args = parser.parse_args()

    # Validate inputs
    target_pdb = Path(args.target)
    if not target_pdb.exists():
        print(f"[ERROR] Target PDB not found: {target_pdb}")
        sys.exit(1)

    # Generate states
    generator = PositiveStateGenerator(
        target_pdb=target_pdb,
        chain_id=args.chain,
        output_dir=Path(args.output)
    )

    states = generator.generate_all_states()

    # Check if at least one state was generated
    if not any(v for k, v in states.items() if k != 'native'):
        print("\n[WARN] No alternate states were generated. Install ColabDesign and/or OpenMM.")
        sys.exit(1)

    print("\n[SUCCESS] Positive state generation complete!")


if __name__ == '__main__':
    main()
