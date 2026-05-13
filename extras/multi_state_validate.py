#!/usr/bin/env python3
"""
Multi-State AF2 Validation Wrapper for BindCraft

Tests accepted binder designs against multiple target PDB states to ensure
robust binding across conformational variations, mutations, or relaxed states.

For each binder-target pair, this script:
1. Runs AF2 prediction for each target state
2. Collects metrics: binder pLDDT, interface pAE, complex pLDDT, contacts, RMSD, clashes
3. Ranks/rejects based on worst-case or average performance across states

Usage:
    python multi_state_validate.py --config settings_validation/multi_state_config.json --input /workspace/outputs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from Bio import PDB
import warnings
warnings.filterwarnings('ignore')

from colabdesign import mk_afdesign_model, clear_mem

# Import utility functions from multi_state_utils
import multi_state_utils

# Import structural alignment utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from functions.biopython_utils import biopython_align_pdbs



class MultiStateValidator:
    """Validates binder designs against multiple target states"""

    def __init__(self, config_path: str, input_dir: str):
        """Initialize validator with config and input directory"""
        self.config = self.load_config(config_path)
        self.input_dir = Path(input_dir)
        self.accepted_dir = self.input_dir / "Accepted"
        self.validation_dir = self.input_dir / "MultiStateValidation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PDB parser
        self.pdb_parser = PDB.PDBParser(QUIET=True)
        self.pdb_io = PDB.PDBIO()

        # Load target states (separated into positive and negative)
        self.target_states = self.load_target_states()
        self.positive_states = [s for s in self.target_states if s.get('validation_type') == 'positive']
        self.negative_states = [s for s in self.target_states if s.get('validation_type') == 'negative']

        print(f"[INFO] Loaded {len(self.positive_states)} positive states, {len(self.negative_states)} negative states")

        # Initialize AF2 model if requested
        self.af_model = None
        self.init_af2_model()

        self.results = []

    def load_config(self, config_path: str) -> Dict:
        """Load validation configuration from JSON"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def load_target_states(self) -> List[Dict]:
        """Load all target PDB states from config, auto-generating positive states if needed"""
        target_states = []

        # Check if auto-generation is enabled
        auto_generate_positive = self.config.get('auto_generate_positive_states', True)
        native_target_pdb = self.config.get('native_target_pdb')
        native_target_chain = self.config.get('native_target_chain', 'A')

        # Auto-generate positive states if enabled and native target provided
        if auto_generate_positive and native_target_pdb:
            print("[INFO] Auto-generating positive conformational states...")
            generated_states = self._generate_positive_states(native_target_pdb, native_target_chain)
            target_states.extend(generated_states)

        # Load additional states from config
        for state_config in self.config.get('target_states', []):
            state_name = state_config['name']
            pdb_path = Path(state_config['pdb_path'])

            if not pdb_path.exists():
                print(f"[WARN] Target state PDB not found: {pdb_path}")
                continue

            target_states.append({
                'name': state_name,
                'pdb_path': pdb_path,
                'chain_id': state_config.get('chain_id', 'A'),
                'description': state_config.get('description', ''),
                'validation_type': state_config.get('validation_type', 'positive')  # Default to positive
            })

        return target_states

    def _generate_positive_states(self, native_pdb: str, chain_id: str) -> List[Dict]:
        """
        Generate positive conformational states automatically (AF2 alternate + OpenMM relaxed).

        Does NOT include the native state - user can add it manually if desired.
        """
        from pathlib import Path as P
        import subprocess

        generated_states = []
        native_path = P(native_pdb)

        if not native_path.exists():
            print(f"[WARN] Native target PDB not found: {native_pdb}")
            return generated_states

        # Output directory for generated states
        output_dir = self.validation_dir / "generated_positive_states"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to generate alternate states using the generator script
        generator_script = P(__file__).parent / "generate_positive_states.py"

        if generator_script.exists():
            try:
                print(f"  Running positive state generator...")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(generator_script),
                        "--target", str(native_path),
                        "--chain", chain_id,
                        "--output", str(output_dir)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )

                if result.returncode == 0:
                    # Check for generated files
                    af2_alt = output_dir / f"{native_path.stem}_af2_alternate.pdb"
                    openmm_relax = output_dir / f"{native_path.stem}_openmm_relaxed.pdb"

                    if af2_alt.exists():
                        generated_states.append({
                            'name': 'af2_alternate',
                            'pdb_path': af2_alt,
                            'chain_id': chain_id,
                            'description': 'AF2-sampled alternate conformation',
                            'validation_type': 'positive'
                        })
                        print(f"  [SUCCESS] Generated AF2 alternate state")

                    if openmm_relax.exists():
                        generated_states.append({
                            'name': 'openmm_relaxed',
                            'pdb_path': openmm_relax,
                            'chain_id': chain_id,
                            'description': 'OpenMM energy-minimized and relaxed',
                            'validation_type': 'positive'
                        })
                        print(f"  [SUCCESS] Generated OpenMM relaxed state")

                else:
                    print(f"  [WARN] Positive state generation failed:")
                    print(f"    {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"  [WARN] Positive state generation timed out")
            except Exception as e:
                print(f"  [WARN] Positive state generation error: {e}")
        else:
            print(f"  [WARN] Generator script not found: {generator_script}")

        return generated_states

    def init_af2_model(self):
        """Initialize AlphaFold2 model for predictions"""
        try:
            print("[INFO] Initializing AlphaFold2 model...")
            self.af_model = mk_afdesign_model(
                protocol="binder",
                use_multimer=True,
                num_recycles=3,
                data_dir="/data/params"  # Assumes params are in standard location
            )
            print("[SUCCESS] AF2 model initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize AF2 model: {e}")
            self.af_model = None

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """Run multi-state validation on all accepted designs"""
        print("[INFO] Starting multi-state validation...")

        if len(self.target_states) == 0:
            print("[ERROR] No target states loaded. Check configuration.")
            return None, None

        # Load accepted designs
        accepted_csv = self.input_dir / "filtered_designs.csv"
        if accepted_csv is None:
            print("[ERROR] Could not find accepted designs CSV")
            return None, None

        df_accepted = pd.read_csv(accepted_csv)
        print(f"[INFO] Found {len(df_accepted)} accepted designs to validate")

        # Process each design
        for idx, row in df_accepted.iterrows():
            design_name = row.get('design', row.get('binder_name', f'design_{idx}'))
            binder_pdb = self.find_pdb_file(design_name)

            if binder_pdb is None:
                print(f"[WARN] Could not find PDB for {design_name}, skipping")
                continue

            print(f"\n[INFO] Validating {design_name} against {len(self.target_states)} states...")

            # Validate against all states
            state_results = self.validate_design(design_name, binder_pdb, row)

            # Aggregate results
            aggregated = multi_state_utils.aggregate_state_results(design_name, state_results, row, self.config)
            self.results.append(aggregated)

        # Convert to DataFrame
        df_results = pd.DataFrame(self.results)

        # Apply scoring and ranking
        df_results = multi_state_utils.apply_multistate_filters(df_results, self.config)

        # Generate report
        report = self.generate_report(df_results)

        print(f"\n[SUMMARY] Multi-state validation complete:")
        print(f"  Total designs: {len(df_results)}")
        print(f"  Hard failures: {len(df_results[df_results['hard_failure'] == True])}")
        print(f"  Designs passing hard filters: {len(df_results[df_results['hard_failure'] == False])}")
        if len(df_results) > 0:
            print(f"  Rank score range: {df_results['composite_rank_score'].min():.1f} - {df_results['composite_rank_score'].max():.1f}")
            print(f"  Mean rank score: {df_results['composite_rank_score'].mean():.1f}")

        return df_results, report

    def find_pdb_file(self, design_name: str) -> Optional[Path]:
        """Find PDB file for a given design"""
        candidates = [
            self.accepted_dir / f"{design_name}.pdb",
            self.accepted_dir / f"{design_name}_relaxed.pdb",
            self.accepted_dir / f"{design_name}_final.pdb",
        ]

        for pdb_path in candidates:
            if pdb_path.exists():
                return pdb_path

        # Search for partial match
        if self.accepted_dir.exists():
            for pdb_file in self.accepted_dir.glob("*.pdb"):
                if design_name in pdb_file.stem:
                    return pdb_file

        return None

    def validate_design(self, design_name: str, binder_pdb: Path, original_metrics: pd.Series) -> List[Dict]:
        """Validate a single design against all target states"""
        state_results = []

        for target_state in self.target_states:
            validation_type = target_state.get('validation_type', 'positive')
            print(f"  Testing {validation_type} state: {target_state['name']}...")

            result = {
                'design': design_name,
                'state_name': target_state['name'],
                'state_description': target_state['description'],
                'validation_type': validation_type
            }

            # Create complex structure (binder + target state)
            complex_pdb = self.create_complex(binder_pdb, target_state)

            if complex_pdb is None:
                print(f"    [ERROR] Failed to create complex for {target_state['name']}")
                result['validation_status'] = 'FAILED'
                state_results.append(result)
                continue

            # Run AF2 prediction if available
            if self.af_model is not None and self.config.get('run_af2_prediction', True):
                af2_metrics = self.run_af2_prediction(complex_pdb, design_name, target_state['name'])
                result.update(af2_metrics)
            else:
                # Use structural analysis only
                result.update(multi_state_utils.analyze_structure(self.pdb_parser, complex_pdb))

            # Calculate interface metrics
            interface_metrics = multi_state_utils.calculate_interface_metrics(complex_pdb)
            result.update(interface_metrics)

            # Calculate RMSD if reference structure available
            if 'reference_complex' in original_metrics:
                rmsd = multi_state_utils.calculate_rmsd(complex_pdb, original_metrics['reference_complex'])
                result['complex_rmsd'] = rmsd

            result['validation_status'] = 'SUCCESS'
            state_results.append(result)

        return state_results

    def create_complex(self, binder_pdb: Path, target_state: Dict) -> Optional[Path]:
        """
        Create a complex structure from binder and target state.

        Supports both single-chain and multi-chain targets (e.g., oligomers, multimers).
        If align_target_states is enabled and reference_target_pdb is provided,
        aligns target state to reference frame before combining with binder.
        """
        try:
            # Load binder
            binder_structure = self.pdb_parser.get_structure('binder', str(binder_pdb))
            binder_chain = list(binder_structure[0].get_chains())[0]

            # Load target
            target_structure = self.pdb_parser.get_structure('target', str(target_state['pdb_path']))
            target_chain_ids = target_state['chain_id']

            # Handle both single chain (string) and multi-chain (list) targets
            if isinstance(target_chain_ids, str):
                target_chain_ids = [target_chain_ids]

            # Extract target chains
            target_chains = []
            for chain in target_structure[0].get_chains():
                if chain.id in target_chain_ids:
                    target_chains.append(chain)

            if len(target_chains) == 0:
                print(f"    [ERROR] No target chains found from {target_chain_ids}")
                return None

            if len(target_chains) != len(target_chain_ids):
                print(f"    [WARN] Expected {len(target_chain_ids)} chains, found {len(target_chains)}")

            # ALIGNMENT: Align target to reference if configured
            # Note: For multi-chain targets, align using first chain as reference
            if self.config.get('align_target_states', False):
                reference_pdb = self.config.get('reference_target_pdb')
                if reference_pdb:
                    # Save target state temporarily
                    temp_target_path = self.validation_dir / f"temp_{target_state['name']}.pdb"
                    self.pdb_io.set_structure(target_structure)
                    self.pdb_io.save(str(temp_target_path))

                    # Use production alignment utility from biopython_utils
                    # Align using first chain as reference
                    biopython_align_pdbs(
                        reference_pdb=reference_pdb,
                        align_pdb=str(temp_target_path),
                        reference_chain_id=target_chain_ids[0],
                        align_chain_id=target_chain_ids[0]
                    )

                    # Reload aligned target
                    target_structure = self.pdb_parser.get_structure('target_aligned', str(temp_target_path))
                    target_chains = []
                    for chain in target_structure[0].get_chains():
                        if chain.id in target_chain_ids:
                            target_chains.append(chain)

                    print(f"    [INFO] Aligned {target_state['name']} to reference structure")

                    # Clean up temp file
                    temp_target_path.unlink()

            # Create new structure with binder + target chains
            complex_structure = PDB.Structure.Structure('complex')
            complex_model = PDB.Model.Model(0)
            complex_structure.add(complex_model)

            # Add target chains first (keep original IDs or rename to A, B, C, D...)
            # For multi-chain targets, preserve original chain IDs for proper structure
            for target_chain in target_chains:
                complex_model.add(target_chain.copy())

            # Add binder chain last (rename to avoid collision)
            # Use next available chain ID after target chains
            used_chain_ids = set(chain.id for chain in target_chains)
            binder_chain_id = 'Z'  # Default to Z for binder
            for potential_id in 'XYZWVU':
                if potential_id not in used_chain_ids:
                    binder_chain_id = potential_id
                    break

            binder_chain.id = binder_chain_id
            complex_model.add(binder_chain.copy())

            # Save complex
            complex_pdb_path = self.validation_dir / f"{binder_pdb.stem}_{target_state['name']}_complex.pdb"
            self.pdb_io.set_structure(complex_structure)
            self.pdb_io.save(str(complex_pdb_path))

            print(f"    [INFO] Created complex: {len(target_chains)} target chain(s) + binder (chain {binder_chain_id})")

            return complex_pdb_path

        except Exception as e:
            print(f"    [ERROR] Failed to create complex: {e}")
            return None

    def run_af2_prediction(self, complex_pdb: Path, design_name: str, state_name: str) -> Dict:
        """Run AlphaFold2 prediction on the complex (supports multi-chain targets)"""
        metrics = {}

        try:
            # Load sequences from complex
            structure = self.pdb_parser.get_structure('complex', str(complex_pdb))
            chains = list(structure[0].get_chains())

            # VALIDATION: Need at least 2 chains (target + binder)
            if len(chains) < 2:
                error_msg = f'Expected at least 2 chains, found {len(chains)}'
                print(f"    [ERROR] {error_msg}")
                return {'af2_error': error_msg, 'validation_status': 'FAILED'}

            # Identify binder chain (last chain = binder)
            binder_chain = chains[-1]
            target_chains = chains[:-1]

            # Extract sequences
            target_seqs = [multi_state_utils.get_sequence(tc) for tc in target_chains]
            binder_seq = multi_state_utils.get_sequence(binder_chain)

            target_total_len = sum(len(seq) for seq in target_seqs)

            # VALIDATION: Check sequence lengths are reasonable
            if len(binder_seq) < 10 or len(binder_seq) > 500:
                print(f"    [WARN] Unusual binder length: {len(binder_seq)} residues")
            if target_total_len < 20:
                print(f"    [WARN] Very short target: {target_total_len} residues total")

            # Build chain string for AF2 (all chains comma-separated)
            chain_ids = [c.id for c in chains]
            chain_string = ",".join(chain_ids)

            # Prepare AF2 model
            self.af_model.prep_inputs(
                pdb_filename=str(complex_pdb),
                chain=chain_string,
                binder_len=len(binder_seq)
            )

            # Run prediction
            self.af_model.predict(verbose=False)

            # Extract metrics
            af2_outputs = self.af_model.aux

            # Binder pLDDT (last binder_len residues)
            binder_plddt = np.mean(af2_outputs['plddt'][target_total_len:])
            metrics['binder_plddt'] = float(binder_plddt)

            # Complex pLDDT
            complex_plddt = np.mean(af2_outputs['plddt'])
            metrics['complex_plddt'] = float(complex_plddt)

            # Interface pAE (binder to all target chains)
            pae_matrix = af2_outputs['pae']
            target_idx = list(range(target_total_len))
            binder_idx = list(range(target_total_len, target_total_len + len(binder_seq)))

            # Average pAE from binder to target
            interface_pae = np.mean([pae_matrix[b, t] for b in binder_idx for t in target_idx])
            metrics['interface_pae'] = float(interface_pae)

            # pTM score
            metrics['ptm'] = float(af2_outputs.get('ptm', 0))

            # i pTM score (interface pTM)
            metrics['iptm'] = float(af2_outputs.get('iptm', 0))

            num_target_chains = len(target_chains)
            print(f"    pLDDT: {binder_plddt:.1f}, iPAE: {interface_pae:.2f}, ipTM: {metrics['iptm']:.3f} ({num_target_chains} target chains)")

            # Clear memory
            clear_mem()

        except Exception as e:
            print(f"    [ERROR] AF2 prediction failed: {e}")
            metrics['af2_error'] = str(e)

        return metrics

    def generate_report(self, df_results: pd.DataFrame) -> Dict:
        """Generate summary report with positive/negative breakdown and ranking statistics"""
        report = {
            'total_designs': len(df_results),
            'hard_failures': len(df_results[df_results['hard_failure'] == True]),
            'designs_passing_hard_filters': len(df_results[df_results['hard_failure'] == False]),
            'num_positive_states': len(self.positive_states),
            'num_negative_states': len(self.negative_states),
            'positive_state_names': [s['name'] for s in self.positive_states],
            'negative_state_names': [s['name'] for s in self.negative_states] if self.negative_states else []
        }

        # Ranking score statistics
        if 'composite_rank_score' in df_results.columns:
            rank_scores = pd.to_numeric(df_results['composite_rank_score'], errors='coerce').dropna()
            if len(rank_scores) > 0:
                report['ranking_statistics'] = {
                    'mean_score': float(rank_scores.mean()),
                    'median_score': float(rank_scores.median()),
                    'std_score': float(rank_scores.std()),
                    'min_score': float(rank_scores.min()),
                    'max_score': float(rank_scores.max()),
                    'top_10_percentile': float(rank_scores.quantile(0.9)) if len(rank_scores) >= 10 else float(rank_scores.max())
                }

        # Positive state performance
        positive_metrics = {}
        for metric in ['positive_mean_score', 'positive_worst_score', 'positive_consistency']:
            if metric in df_results.columns:
                valid_data = pd.to_numeric(df_results[metric], errors='coerce').dropna()
                if len(valid_data) > 0:
                    positive_metrics[metric] = {
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max())
                    }
        report['positive_state_performance'] = positive_metrics

        # Negative state performance (specificity)
        negative_metrics = {}
        for metric in ['negative_mean_specificity', 'negative_worst_specificity', 'specificity_margin']:
            if metric in df_results.columns:
                valid_data = pd.to_numeric(df_results[metric], errors='coerce').dropna()
                if len(valid_data) > 0:
                    negative_metrics[metric] = {
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max())
                    }
        report['negative_state_performance'] = negative_metrics if negative_metrics else None

        # Catastrophic failures breakdown
        catastrophic_count = df_results['has_catastrophic_failure'].sum() if 'has_catastrophic_failure' in df_results.columns else 0
        report['designs_with_catastrophic_failures'] = int(catastrophic_count)

        # Top designs summary
        if len(df_results) > 0:
            top_n = min(10, len(df_results))
            top_designs = df_results.nlargest(top_n, 'composite_rank_score')[['design', 'composite_rank_score', 'positive_mean_score', 'negative_mean_specificity']].to_dict('records')
            report['top_designs'] = top_designs

        return report


def main():
    parser = argparse.ArgumentParser(description='Multi-state validation for BindCraft designs')
    parser.add_argument('--config', required=True, help='Path to multi-state validation config JSON')
    parser.add_argument('--input', required=True, help='Path to BindCraft output directory')
    parser.add_argument('--output', default=None, help='Output directory (default: same as input)')

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output) if args.output else Path(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    validator = MultiStateValidator(args.config, args.input)
    df_results, report = validator.run()

    if df_results is None:
        print("[ERROR] Validation failed")
        sys.exit(1)

    # Save results
    results_csv = output_dir / "multi_state_scores.csv"
    report_json = output_dir / "multi_state_report.json"

    df_results.to_csv(results_csv, index=False)

    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Results saved:")
    print(f"  Scores: {results_csv}")
    print(f"  Report: {report_json}")


if __name__ == '__main__':
    main()
