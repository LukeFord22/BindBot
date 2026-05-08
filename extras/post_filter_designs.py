#!/usr/bin/env python3
"""
Post-Filtering Script for BindCraft Accepted Designs

This script applies additional biochemical and structural filters to designs
that have already been accepted by BindCraft's standard filters.

Filters include:
- Protease cleavage motif detection
- Exposed Lys/Arg analysis
- Histidine count and clustering
- Flexible/disordered loop detection
- Low pLDDT patches
- Poor interface metrics (pAE, contacts)
- Aggregation risk prediction
- Charge patch analysis

Usage:
    python post_filter_designs.py --config settings_filters/post_filter_config.json --input /workspace/outputs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import re
from collections import Counter
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import NeighborSearch
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Import production-ready DSSP utilities from BindBot
sys.path.insert(0, str(Path(__file__).parent.parent))
from functions.biopython_utils import safe_dssp_calculation


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences"""
    if len(seq1) != len(seq2):
        # Align via simple global alignment for different lengths
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max(len(seq1), len(seq2))

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1) if len(seq1) > 0 else 0.0


class PostDesignFilter:
    """Main class for post-filtering BindCraft designs"""

    # =========================================================================
    # Run Methods & Data loading
    # =========================================================================


    def __init__(self, config_path: str, input_dir: str):
        """Initialize filter with config and input directory"""
        self.config = self.load_config(config_path)
        self.input_dir = Path(input_dir)
        self.accepted_dir = self.input_dir / "Accepted"
        self.results = []

        # Initialize PDB parser
        self.pdb_parser = PDB.PDBParser(QUIET=True)

        # Get DSSP path from config (with fallback to 'mkdssp')
        self.dssp_path = self.config.get('dssp_path', 'mkdssp')

        # Caching for performance (design_name -> cached data)
        # Note: DSSP caching is handled globally in biopython_utils
        self.exposure_cache = {}  # (design_name, res_idx) -> float exposure
        self.residue_map_cache = {}  # design_name -> list of standard residues
        self.interface_cache = {}  # design_name -> set of residue indices


    def load_config(self, config_path: str) -> Dict:
        """Load filter configuration from JSON"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config


    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run all filters and return results"""
        print("[INFO] Starting post-filtering of accepted designs...")

        # Load accepted designs CSV
        accepted_csv = self.find_accepted_csv()
        if accepted_csv is None:
            print("[ERROR] Could not find accepted designs CSV")
            return None, None, None

        df_accepted = pd.read_csv(accepted_csv)
        print(f"[INFO] Found {len(df_accepted)} accepted designs to filter")

        # Process each design
        for idx, row in df_accepted.iterrows():
            design_name = row.get('design', row.get('binder_name', f'design_{idx}'))
            pdb_path = self.find_pdb_file(design_name)

            if pdb_path is None:
                print(f"[WARN] Could not find PDB for {design_name}, skipping")
                continue

            print(f"[INFO] Filtering {design_name}...")

            # Run all filters
            filter_results = self.filter_design(design_name, pdb_path, row)
            filter_results['design'] = design_name
            filter_results['original_rank'] = idx + 1

            # Copy original metrics
            for col in row.index:
                if col not in filter_results:
                    filter_results[col] = row[col]

            self.results.append(filter_results)

        # Convert to DataFrame
        df_results = pd.DataFrame(self.results)

        # Add composite scores and normalization
        df_results = self.add_composite_scores(df_results)

        # Cluster by sequence identity (remove redundancy)
        df_results = self.cluster_by_sequence_identity(df_results)

        # Separate passed and rejected
        df_passed = df_results[df_results['post_filter_pass'] == True].copy()
        df_rejected = df_results[df_results['post_filter_pass'] == False].copy()

        # Generate report
        report = self.generate_report(df_results, df_passed, df_rejected)

        # Print sequence clustering summary
        if 'sequence_cluster_id' in df_results.columns:
            n_clusters = df_results['sequence_cluster_id'].nunique()
            n_redundant = df_results['redundancy_flag'].sum()
            if n_redundant > 0:
                print(f"  Sequence diversity: {n_clusters} clusters, {n_redundant} redundant designs")

        print(f"\n[SUMMARY] Post-filtering complete:")
        print(f"  Total designs: {len(df_results)}")
        print(f"  Passed: {len(df_passed)}")
        print(f"  Rejected: {len(df_rejected)}")

        return df_passed, df_rejected, report


    def find_accepted_csv(self) -> Optional[Path]:
        """Find the accepted designs CSV file"""
        # BindCraft outputs final_design_stats.csv or mpnn_design_stats.csv
        candidates = [
            self.input_dir / "final_design_stats.csv",    # BindCraft final designs (primary)
            self.input_dir / "mpnn_design_stats.csv"      # BindCraft MPNN stats (fallback)
        ]

        for csv_path in candidates:
            if csv_path.exists():
                print(f"[INFO] Found accepted designs CSV: {csv_path}")
                return csv_path

        print(f"[ERROR] No CSV file found. Expected {self.input_dir}/final_design_stats.csv")
        return None


    def find_pdb_file(self, design_name: str) -> Optional[Path]:
        """Find PDB file for a given design"""
        # Try common naming patterns
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


    def filter_design(self, design_name: str, pdb_path: Path, original_metrics: pd.Series) -> Dict:
        """Apply all filters to a single design"""
        results = {
            'post_filter_pass': True,
            'rejection_reasons': []
        }

        # Load structure
        try:
            structure = self.pdb_parser.get_structure(design_name, str(pdb_path))
            model = structure[0]
        except Exception as e:
            print(f"[ERROR] Failed to parse PDB {pdb_path}: {e}")
            results['post_filter_pass'] = False
            results['rejection_reasons'].append(f"PDB_PARSE_ERROR: {e}")
            return results

        # Extract binder chain (assume it's the designed chain)
        binder_chain = self.get_binder_chain(model, original_metrics)
        if binder_chain is None:
            results['post_filter_pass'] = False
            results['rejection_reasons'].append("Could not identify binder chain")
            return results

        # Get sequence
        sequence = self.get_sequence(binder_chain)
        results['sequence'] = sequence
        results['binder_length'] = len(sequence)

        # Precompute exposure map once
        self.compute_exposure_map(design_name, binder_chain, model, pdb_path)

        # Run individual filter checks (now using cached exposure data)
        self.check_protease_motifs(sequence, binder_chain, pdb_path, design_name, results)
        self.check_exposed_lysine_arginine(binder_chain, pdb_path, design_name, results)
        self.check_histidine_content(sequence, binder_chain, pdb_path, design_name, results)
        self.check_flexible_loops(binder_chain, pdb_path, design_name, results)
        self.check_plddt_patches(binder_chain, original_metrics, results)
        self.check_interface_quality(original_metrics, results)
        self.check_aggregation_risk(sequence, binder_chain, pdb_path, design_name, results)
        self.check_charge_patches(binder_chain, pdb_path, design_name, results)

        # Advanced filters (optional, config-driven)
        self.check_buried_polar_residues(binder_chain, pdb_path, design_name, results)
        self.estimate_self_binding_risk(binder_chain, pdb_path, results)
        self.assess_packing_quality(binder_chain, pdb_path, results)

        # Determine overall pass/fail
        if len(results['rejection_reasons']) > 0:
            results['post_filter_pass'] = False

        # Convert rejection reasons to string
        results['rejection_reasons'] = '; '.join(results['rejection_reasons'])

        return results


    def get_binder_chain(self, model, original_metrics: pd.Series):
        """
        Identify the binder chain using multiple heuristics:
        1. Metadata from original metrics
        2. Chain length (binder typically shorter)
        3. pLDDT from B-factors (if available)
        4. Fallback to first chain
        """
        chains = list(model.get_chains())
        if len(chains) == 0:
            return None

        if len(chains) == 1:
            return chains[0]

        # 1. Try metadata
        if 'binder_chain' in original_metrics:
            chain_id = original_metrics['binder_chain']
            for chain in chains:
                if chain.id == chain_id:
                    return chain

        # 2. Use chain length heuristic
        # Binder is typically shorter than target
        chain_lengths = []
        for chain in chains:
            residues = [r for r in chain.get_residues() if r.id[0] == ' ']
            chain_lengths.append((chain, len(residues)))

        chain_lengths.sort(key=lambda x: x[1])

        # If there's a clear size difference, assume shorter is binder
        if len(chain_lengths) >= 2:
            shortest_len = chain_lengths[0][1]
            second_len = chain_lengths[1][1]

            # If shortest is <70% of second, it's likely the binder
            if shortest_len < 0.7 * second_len:
                return chain_lengths[0][0]

        # 3. Try pLDDT heuristic (designed chain often has higher pLDDT)
        chain_plddt = []
        for chain in chains:
            residues = [r for r in chain.get_residues() if r.id[0] == ' ']
            plddt_values = []
            for r in residues:
                for atom in r.get_atoms():
                    plddt_values.append(atom.get_bfactor())
                    break

            if plddt_values:
                avg_plddt = np.mean(plddt_values)
                chain_plddt.append((chain, avg_plddt))

        if chain_plddt:
            # Higher pLDDT might indicate designed chain
            chain_plddt.sort(key=lambda x: x[1], reverse=True)
            # But only use this if pLDDT difference is significant
            if len(chain_plddt) >= 2:
                best_plddt = chain_plddt[0][1]
                second_plddt = chain_plddt[1][1]
                if best_plddt > second_plddt + 5.0:  # >5 pLDDT units difference
                    return chain_plddt[0][0]

        # 4. Fallback: shortest chain (reasonable default)
        if chain_lengths:
            return chain_lengths[0][0]

        # Final fallback
        return chains[0]


    def get_sequence(self, chain) -> str:
        """Extract amino acid sequence from chain using Biopython's Polypeptide"""
        from Bio.PDB.Polypeptide import PPBuilder

        # Use Bio.PDB's built-in sequence extraction
        ppb = PPBuilder()
        sequence = ""
        for pp in ppb.build_peptides(chain):
            sequence += str(pp.get_sequence())

        return sequence

    # =========================================================================
    # CACHING METHODS
    # =========================================================================

    def get_standard_residues_cached(self, design_name: str, chain) -> List:
        """
        Get list of standard residues for a chain with caching.

        Returns list of residues with id[0] == ' ' (standard residues only)
        """
        cache_key = f"{design_name}_{chain.id}"

        if cache_key in self.residue_map_cache:
            return self.residue_map_cache[cache_key]

        residues = [r for r in chain.get_residues() if r.id[0] == ' ']
        self.residue_map_cache[cache_key] = residues
        return residues

    def compute_exposure_map(self, design_name: str, chain, model, pdb_path: Path) -> Dict[int, float]:
        """
        Compute exposure for all residues in chain with caching.

        Uses DSSP for solvent accessibility calculation via safe_dssp_calculation
        from biopython_utils (production-ready with retry logic and global caching).

        Returns: Dict mapping residue index (0-indexed) to exposure (0-1)
        """
        cache_key = f"{design_name}_{chain.id}"

        # Build exposure map
        exposure_map = {}
        residues = self.get_standard_residues_cached(design_name, chain)

        # Use production-ready DSSP calculation from biopython_utils
        # This handles retries, fallback executables, and global caching
        dssp = safe_dssp_calculation(model, str(pdb_path), self.dssp_path)

        if dssp is not None:
            # Use DSSP relative accessibility for all residues
            for idx, residue in enumerate(residues):
                dssp_key = (chain.id, residue.id)
                if dssp_key in dssp:
                    exposure_map[idx] = dssp[dssp_key][3]  # Relative accessibility
                else:
                    exposure_map[idx] = None

        # Cache individual residue exposures
        for idx, exposure in exposure_map.items():
            self.exposure_cache[(cache_key, idx)] = exposure

        return exposure_map

    def get_cached_exposure(self, design_name: str, chain_id: str, res_idx: int) -> Optional[float]:
        """
        Get cached exposure for a specific residue.

        Returns exposure (0-1) or None if not cached
        """
        cache_key = (f"{design_name}_{chain_id}", res_idx)
        return self.exposure_cache.get(cache_key)

    # =========================================================================
    # CORE HELPER METHODS - Exposure, Interface, Clustering
    # =========================================================================

    def identify_interface_residues(self, binder_chain, model, pdb_path: Path,
                                    design_name: str, distance_cutoff: float = 6.0) -> Set[int]:
        """
        Identify interface residues (binder residues near target).

        Interface = any binder residue with atoms within distance_cutoff of target atoms.
        Uses distance cutoff of 5-8Å (default 6Å for moderate stringency).

        Returns: Set of residue indices (0-indexed)
        """
        # Check cache
        if design_name in self.interface_cache:
            return self.interface_cache[design_name]

        interface_residues = set()

        try:
            chains = list(model.get_chains())
            if len(chains) < 2:
                # No target chain - no interface
                return interface_residues

            # Identify target chain (not binder)
            target_chain = None
            for chain in chains:
                if chain.id != binder_chain.id:
                    target_chain = chain
                    break

            if target_chain is None:
                return interface_residues

            # Get all atoms
            binder_atoms = list(binder_chain.get_atoms())
            target_atoms = list(target_chain.get_atoms())

            # Build neighbor search
            ns = NeighborSearch(target_atoms)

            # Find binder residues near target
            binder_residues = list(binder_chain.get_residues())
            for idx, residue in enumerate(binder_residues):
                if residue.id[0] != ' ':
                    continue

                # Check if any atom is near target
                for atom in residue.get_atoms():
                    nearby = ns.search(atom.coord, distance_cutoff, 'A')
                    if len(nearby) > 0:
                        interface_residues.add(idx)
                        break

            # Cache result
            self.interface_cache[design_name] = interface_residues

        except Exception as e:
            print(f"    [WARN] Interface identification failed: {e}")

        return interface_residues

    def cluster_spatial_points_robust(self, coords: np.ndarray, eps: float, min_samples: int = 1) -> List[List[int]]:
        """
        Spatial clustering using DBSCAN.

        Returns list of clusters, where each cluster is a list of point indices.

        Args:
            coords: Nx3 array of coordinates
            eps: distance threshold for neighbors
            min_samples: minimum cluster size (DBSCAN parameter)
        """
        if len(coords) == 0:
            return []

        if len(coords) == 1:
            return [[0]]

        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(coords)

        # Group by cluster label
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise point - treat as singleton
                clusters[f'noise_{idx}'] = [idx]
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

        return list(clusters.values())

    def get_largest_cluster_size(self, coords: List, distance_threshold: float) -> int:
        """Get size of largest spatial cluster (wrapper for robust clustering)"""
        if len(coords) == 0:
            return 0

        coords_array = np.array(coords)
        clusters = self.cluster_spatial_points_robust(coords_array, distance_threshold, min_samples=1)

        if not clusters:
            return 0

        return max(len(cluster) for cluster in clusters)

    def get_residue_functional_coord(self, residue, aa_code: str):
        """Return the chemically relevant coordinate for residue clustering."""
        def centroid(atom_names):
            coords = [residue[a].coord for a in atom_names if a in residue]
            if coords:
                return np.mean(coords, axis=0)
            return None

        # Charged residues
        if aa_code == "K" and "NZ" in residue:
            return residue["NZ"].coord

        if aa_code == "R":
            coord = centroid(["CZ", "NH1", "NH2", "NE"])
            if coord is not None:
                return coord

        if aa_code == "D":
            coord = centroid(["OD1", "OD2"])
            if coord is not None:
                return coord

        if aa_code == "E":
            coord = centroid(["OE1", "OE2"])
            if coord is not None:
                return coord

        # Histidine imidazole ring
        if aa_code == "H":
            coord = centroid(["CG", "ND1", "CD2", "CE1", "NE2"])
            if coord is not None:
                return coord

        # Aromatics / hydrophobics
        if aa_code == "F":
            coord = centroid(["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
            if coord is not None:
                return coord

        if aa_code == "Y":
            coord = centroid(["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"])
            if coord is not None:
                return coord

        if aa_code == "W":
            coord = centroid(["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"])
            if coord is not None:
                return coord

        # Generic sidechain centroid
        sidechain_atoms = [
            atom.coord for atom in residue.get_atoms()
            if atom.get_id() not in ["N", "CA", "C", "O"]
        ]
        if sidechain_atoms:
            return np.mean(sidechain_atoms, axis=0)

        # Final fallback
        if "CA" in residue:
            return residue["CA"].coord

        return None

    # =========================================================================
    # INDIVIDUAL FILTER CHECKS
    # =========================================================================

    def check_protease_motifs(self, sequence: str, binder_chain, pdb_path: Path,
                              design_name: str, results: Dict):
        """Check for EXPOSED protease cleavage motifs using structural context"""
        if not self.config.get('check_protease_motifs', True):
            return

        protease_motifs = self.config.get('protease_motifs', [
            'KR', 'RR', 'RXXR', 'DE', 'LL', 'FF'
        ])

        try:
            exposed_motifs = []
            exposure_threshold = self.config.get('protease_exposure_threshold', 0.3)

            for motif in protease_motifs:
                pattern = motif.replace('X', '.')
                for match in re.finditer(pattern, sequence):
                    pos = match.start()
                    motif_length = len(match.group())

                    # Check accessibility of motif residues using cached exposure
                    motif_exposed = False
                    for i in range(motif_length):
                        residue_idx = pos + i
                        # Use cached exposure
                        accessibility = self.get_cached_exposure(design_name, binder_chain.id, residue_idx)
                        if accessibility is not None and accessibility > exposure_threshold:
                            motif_exposed = True
                            break

                    if motif_exposed:
                        exposed_motifs.append(f"{motif}@{pos}(exposed)")

            results['exposed_protease_motifs'] = len(exposed_motifs)
            results['protease_motifs_list'] = ','.join(exposed_motifs) if exposed_motifs else 'None'

            max_allowed = self.config.get('max_exposed_protease_motifs', 1)
            if len(exposed_motifs) > max_allowed:
                results['rejection_reasons'].append(
                    f"Exposed protease motifs ({len(exposed_motifs)} > {max_allowed})"
                )

        except Exception as e:
            print(f"[WARN] Protease motif check failed: {e}")
            results['exposed_protease_motifs'] = 'N/A'

    def check_exposed_lysine_arginine(self, chain, pdb_path: Path, design_name: str, results: Dict):
        """Check for excessive exposed Lys/Arg residues"""
        if not self.config.get('check_exposed_kr', True):
            return

        try:
            total_kr = 0
            exposed_kr = 0
            exposure_threshold = self.config.get('kr_exposure_threshold', 0.25)

            # Use cached residue list and exposure
            residues = self.get_standard_residues_cached(design_name, chain)

            for idx, residue in enumerate(residues):
                res_name = residue.get_resname()
                if res_name in ['LYS', 'ARG']:
                    total_kr += 1

                    # Get cached accessibility
                    accessibility = self.get_cached_exposure(design_name, chain.id, idx)
                    if accessibility is not None and accessibility > exposure_threshold:
                        exposed_kr += 1

            results['total_kr'] = total_kr
            results['exposed_kr'] = exposed_kr
            results['exposed_kr_fraction'] = exposed_kr / len(self.get_sequence(chain)) if len(self.get_sequence(chain)) > 0 else 0

            max_fraction = self.config.get('max_exposed_kr_fraction', 0.15)
            if results['exposed_kr_fraction'] > max_fraction:
                results['rejection_reasons'].append(
                    f"Excessive exposed K/R ({results['exposed_kr_fraction']:.2%} > {max_fraction:.2%})"
                )

        except Exception as e:
            print(f"[WARN] Exposed K/R check failed: {e}")
            results['exposed_kr'] = 'N/A'
            results['exposed_kr_fraction'] = 'N/A'

    def check_histidine_content(self, sequence: str, chain, pdb_path: Path,
                                design_name: str, results: Dict):
        """
        Check histidine SPATIAL clustering using robust clustering.
        Interface histidines may be functional - weighted differently.
        """
        if not self.config.get('check_histidine', True):
            return

        his_count = sequence.count('H')
        his_fraction = his_count / len(sequence) if len(sequence) > 0 else 0

        results['histidine_count'] = his_count
        results['histidine_fraction'] = his_fraction

        if his_count == 0:
            results['his_spatial_cluster_size'] = 0
            results['his_scaffold_cluster_size'] = 0
            return

        try:
            # Get interface residues
            model = chain.get_parent()
            interface_residues = self.identify_interface_residues(
                chain, model, pdb_path, design_name
            )

            # Find histidine positions and coordinates
            his_positions = [i for i, aa in enumerate(sequence) if aa == 'H']
            residues = self.get_standard_residues_cached(design_name, chain)

            his_coords_all = []
            his_coords_scaffold = []  # Non-interface only

            for pos in his_positions:
                if pos < len(residues):
                    residue = residues[pos]
                    # Use imidazole ring centroid for histidine clustering
                    coord = self.get_residue_functional_coord(residue, 'H')
                    if coord is not None:
                        his_coords_all.append(coord)

                        # Track scaffold (non-interface) separately
                        if pos not in interface_residues:
                            his_coords_scaffold.append(coord)

            if len(his_coords_all) < 2:
                results['his_spatial_cluster_size'] = his_count
                results['his_scaffold_cluster_size'] = len(his_coords_scaffold)
                return

            # Use robust clustering
            cluster_distance = self.config.get('histidine_cluster_distance', 10.0)

            # All histidines
            max_cluster_size = self.get_largest_cluster_size(his_coords_all, cluster_distance)
            results['his_spatial_cluster_size'] = max_cluster_size

            # Scaffold histidines only (more concerning if clustered)
            max_scaffold_cluster = self.get_largest_cluster_size(his_coords_scaffold, cluster_distance)
            results['his_scaffold_cluster_size'] = max_scaffold_cluster

            # Rejection based on SCAFFOLD clustering (interface His may be intentional)
            max_allowed_cluster = self.config.get('max_histidine_spatial_cluster', 3)
            if max_scaffold_cluster >= max_allowed_cluster:
                results['rejection_reasons'].append(
                    f"Histidine scaffold cluster ({max_scaffold_cluster} His within {cluster_distance}Å, non-interface)"
                )

        except Exception as e:
            print(f"[WARN] Histidine spatial clustering failed: {e}")
            results['his_spatial_cluster_size'] = 'N/A'
            results['his_scaffold_cluster_size'] = 'N/A'

    def check_flexible_loops(self, chain, pdb_path: Path, design_name: str, results: Dict):
        """Detect long flexible or disordered loops"""
        if not self.config.get('check_flexible_loops', True):
            return

        try:
            # Use production-ready DSSP calculation from biopython_utils
            model = chain.get_parent()
            dssp = safe_dssp_calculation(model, str(pdb_path), self.dssp_path)

            if dssp is None:
                results['max_loop_length'] = 'N/A'
                return

            # Track consecutive coil residues
            max_coil_length = 0
            current_coil_length = 0

            residues = self.get_standard_residues_cached(design_name, chain)
            for residue in residues:
                dssp_key = (chain.id, residue.id)
                if dssp_key in dssp:
                    ss = dssp[dssp_key][2]  # Secondary structure
                    if ss in ['-', 'T', 'S']:  # Coil, turn, bend
                        current_coil_length += 1
                        max_coil_length = max(max_coil_length, current_coil_length)
                    else:
                        current_coil_length = 0

            results['max_loop_length'] = max_coil_length

            max_allowed = self.config.get('max_loop_length', 15)
            if max_coil_length > max_allowed:
                results['rejection_reasons'].append(
                    f"Long flexible loop detected ({max_coil_length} > {max_allowed} residues)"
                )

        except Exception as e:
            print(f"[WARN] Loop check failed: {e}")
            results['max_loop_length'] = 'N/A'

    def check_plddt_patches(self, chain, original_metrics: pd.Series, results: Dict):
        """Check for low pLDDT patches in the structure"""
        if not self.config.get('check_plddt_patches', True):
            return

        # Try to get pLDDT from B-factors (common in AF2 outputs)
        plddt_values = []
        for residue in chain.get_residues():
            if residue.id[0] == ' ':
                for atom in residue.get_atoms():
                    plddt_values.append(atom.get_bfactor())
                    break  # Only need one atom per residue

        if not plddt_values:
            results['min_plddt'] = 'N/A'
            results['low_plddt_patch_length'] = 'N/A'
            return

        plddt_values = np.array(plddt_values)
        results['min_plddt'] = float(np.min(plddt_values))
        results['mean_plddt'] = float(np.mean(plddt_values))

        # Find consecutive low pLDDT patches
        low_plddt_threshold = self.config.get('low_plddt_threshold', 70)
        max_patch_length = 0
        current_patch_length = 0

        for plddt in plddt_values:
            if plddt < low_plddt_threshold:
                current_patch_length += 1
                max_patch_length = max(max_patch_length, current_patch_length)
            else:
                current_patch_length = 0

        results['low_plddt_patch_length'] = max_patch_length

        max_allowed_patch = self.config.get('max_low_plddt_patch_length', 8)
        if max_patch_length > max_allowed_patch:
            results['rejection_reasons'].append(
                f"Low pLDDT patch detected ({max_patch_length} residues < {low_plddt_threshold})"
            )

    def check_interface_quality(self, original_metrics: pd.Series, results: Dict):
        """Check interface quality metrics from original BindCraft output"""
        if not self.config.get('check_interface_quality', True):
            return

        # Check interface pAE
        if 'i_pae' in original_metrics or 'interface_pae' in original_metrics:
            i_pae = original_metrics.get('i_pae', original_metrics.get('interface_pae'))
            results['interface_pae'] = i_pae

            max_pae = self.config.get('max_interface_pae', 8.0)
            if i_pae > max_pae:
                results['rejection_reasons'].append(
                    f"Poor interface pAE ({i_pae:.2f} > {max_pae})"
                )

        # Check interface contacts
        if 'i_con' in original_metrics or 'interface_contacts' in original_metrics:
            i_con = original_metrics.get('i_con', original_metrics.get('interface_contacts'))
            results['interface_contacts'] = i_con

            min_contacts = self.config.get('min_interface_contacts', 20)
            if i_con < min_contacts:
                results['rejection_reasons'].append(
                    f"Insufficient interface contacts ({i_con} < {min_contacts})"
                )

    def check_aggregation_risk(self, sequence: str, chain, pdb_path: Path,
                               design_name: str, results: Dict):
        """
        Predict aggregation via EXPOSED hydrophobic PATCHES.
        Interface hydrophobics are EXCLUDED - they're functional.
        Uses spatial patch analysis, not just global fraction.
        """
        if not self.config.get('check_aggregation', True):
            return

        try:
            model = chain.get_parent()

            # Get interface residues and cached residue list
            interface_residues = self.identify_interface_residues(
                chain, model, pdb_path, design_name
            )
            residues = self.get_standard_residues_cached(design_name, chain)

            hydrophobic_aas = set('AILMFVWY')
            exposed_hydrophobic_coords_all = []
            exposed_hydrophobic_coords_scaffold = []  # Non-interface only
            exposed_count_all = 0
            exposed_count_scaffold = 0
            total_hydrophobic = 0
            exposed_hydrophobic_area = 0.0

            for idx, residue in enumerate(residues):
                aa_code = sequence[idx] if idx < len(sequence) else 'X'

                if aa_code in hydrophobic_aas:
                    total_hydrophobic += 1

                    # Get cached exposure
                    exposure = self.get_cached_exposure(design_name, chain.id, idx)

                    if exposure is not None and exposure > 0.25:
                        exposed_count_all += 1
                        exposed_hydrophobic_area += exposure

                        # Use sidechain-aware functional coordinate
                        coord = self.get_residue_functional_coord(residue, aa_code)
                        if coord is not None:
                            exposed_hydrophobic_coords_all.append(coord)

                            # Track scaffold separately (exclude interface)
                            if idx not in interface_residues:
                                exposed_hydrophobic_coords_scaffold.append(coord)
                                exposed_count_scaffold += 1

            # Calculate fractions
            exposed_fraction_all = (exposed_count_all / total_hydrophobic if total_hydrophobic > 0 else 0)
            exposed_fraction_scaffold = (exposed_count_scaffold / total_hydrophobic if total_hydrophobic > 0 else 0)

            results['exposed_hydrophobic_fraction'] = exposed_fraction_all
            results['exposed_hydrophobic_scaffold_fraction'] = exposed_fraction_scaffold
            results['exposed_hydrophobic_area'] = exposed_hydrophobic_area

            # PATCH ANALYSIS - spatial clustering of exposed hydrophobics
            patch_distance = self.config.get('hydrophobic_patch_distance', 10.0)

            # All exposed hydrophobics
            largest_patch_all = self.get_largest_cluster_size(exposed_hydrophobic_coords_all, patch_distance)
            results['largest_hydrophobic_patch'] = largest_patch_all

            # Scaffold only (more concerning for aggregation)
            largest_patch_scaffold = self.get_largest_cluster_size(exposed_hydrophobic_coords_scaffold, patch_distance)
            results['largest_hydrophobic_scaffold_patch'] = largest_patch_scaffold

            # Count patches
            if len(exposed_hydrophobic_coords_scaffold) > 0:
                clusters = self.cluster_spatial_points_robust(
                    np.array(exposed_hydrophobic_coords_scaffold), patch_distance
                )
                results['hydrophobic_patch_count'] = len([c for c in clusters if len(c) >= 3])
            else:
                results['hydrophobic_patch_count'] = 0

            # Rejection based on SCAFFOLD patches (not interface)
            max_allowed_patch = self.config.get('max_hydrophobic_patch_size', 8)
            if largest_patch_scaffold > max_allowed_patch:
                results['rejection_reasons'].append(
                    f"Large hydrophobic scaffold patch ({largest_patch_scaffold} residues, non-interface)"
                )

            # Also check global scaffold fraction
            max_allowed_fraction = self.config.get('max_exposed_hydrophobic_scaffold_fraction', 0.4)
            if exposed_fraction_scaffold > max_allowed_fraction:
                results['rejection_reasons'].append(
                    f"High scaffold hydrophobic exposure ({exposed_fraction_scaffold:.1%} > {max_allowed_fraction:.1%})"
                )


        except Exception as e:
            print(f"[WARN] Aggregation check failed: {e}")
            results['exposed_hydrophobic_fraction'] = 'N/A'
            results['exposed_hydrophobic_area'] = 'N/A'

    def check_charge_patches(self, chain, pdb_path: Path, design_name: str, results: Dict):
        """
        Detect SPATIAL charge patches using robust clustering.
        Interface charges may be functional - analyze scaffold separately.
        """
        if not self.config.get('check_charge_patches', True):
            return

        sequence = self.get_sequence(chain)

        try:
            model = chain.get_parent()

            # Get interface residues
            interface_residues = self.identify_interface_residues(
                chain, model, pdb_path, design_name
            )

            positive_aas = set('KR')
            negative_aas = set('DE')

            residues = self.get_standard_residues_cached(design_name, chain)

            # Separate interface and scaffold charged residues
            positive_coords_all = []
            positive_coords_scaffold = []
            negative_coords_all = []
            negative_coords_scaffold = []

            for idx, residue in enumerate(residues):
                if idx >= len(sequence):
                    continue

                aa = sequence[idx]
                is_interface = idx in interface_residues

                # Use sidechain-aware functional coordinate for charged residues
                if aa in positive_aas or aa in negative_aas:
                    coord = self.get_residue_functional_coord(residue, aa)
                    if coord is None:
                        continue

                    if aa in positive_aas:
                        positive_coords_all.append(coord)
                        if not is_interface:
                            positive_coords_scaffold.append(coord)
                    elif aa in negative_aas:
                        negative_coords_all.append(coord)
                        if not is_interface:
                            negative_coords_scaffold.append(coord)

            cluster_distance = self.config.get('charge_cluster_distance', 12.0)

            # Analyze ALL charged residues
            max_positive_all = self.get_largest_cluster_size(positive_coords_all, cluster_distance)
            max_negative_all = self.get_largest_cluster_size(negative_coords_all, cluster_distance)

            results['max_positive_spatial_cluster'] = max_positive_all
            results['max_negative_spatial_cluster'] = max_negative_all

            # Analyze SCAFFOLD only (more concerning)
            max_positive_scaffold = self.get_largest_cluster_size(positive_coords_scaffold, cluster_distance)
            max_negative_scaffold = self.get_largest_cluster_size(negative_coords_scaffold, cluster_distance)

            results['max_positive_scaffold_cluster'] = max_positive_scaffold
            results['max_negative_scaffold_cluster'] = max_negative_scaffold

            # Store interface/scaffold counts
            results['interface_residue_count'] = len(interface_residues)
            results['scaffold_residue_count'] = len(residues) - len(interface_residues)

            # Rejection based on SCAFFOLD clusters (interface may be intentional)
            max_allowed_cluster = self.config.get('max_charge_spatial_cluster', 5)

            if max_positive_scaffold > max_allowed_cluster:
                results['rejection_reasons'].append(
                    f"Large positive scaffold charge cluster ({max_positive_scaffold} K/R within {cluster_distance}Å, non-interface)"
                )
            if max_negative_scaffold > max_allowed_cluster:
                results['rejection_reasons'].append(
                    f"Large negative scaffold charge cluster ({max_negative_scaffold} D/E within {cluster_distance}Å, non-interface)"
                )

        except Exception as e:
            print(f"[WARN] Charge patch analysis failed: {e}")
            results['max_positive_spatial_cluster'] = 'N/A'
            results['max_negative_spatial_cluster'] = 'N/A'


    def check_buried_polar_residues(self, chain, pdb_path: Path, design_name: str, results: Dict):
        """
        Identify buried polar residues that may be unsatisfied.
        Buried polar atoms without H-bond partners are destabilizing.
        """
        if not self.config.get('check_buried_polars', False):
            return

        try:
            model = chain.get_parent()
            residues = self.get_standard_residues_cached(design_name, chain)
            sequence = self.get_sequence(chain)

            polar_aas = set('DEHKNQRST')
            buried_polar_count = 0
            total_polar = 0

            for idx, residue in enumerate(residues):
                if idx >= len(sequence):
                    continue

                aa = sequence[idx]
                if aa not in polar_aas:
                    continue

                total_polar += 1

                # Check if buried using cached exposure
                exposure = self.get_cached_exposure(design_name, chain.id, idx)
                if exposure is not None and exposure < 0.15:  # Buried
                    # Count nearby polar atoms (simple H-bond proxy)
                    if 'CA' in residue:
                        ca_coord = residue['CA'].coord
                        all_residues = [r for r in residues if r.id[0] == ' ']
                        nearby_polar = 0

                        for other in all_residues:
                            if other == residue:
                                continue

                            other_idx = all_residues.index(other)
                            if other_idx < len(sequence) and sequence[other_idx] in polar_aas:
                                if 'CA' in other:
                                    dist = np.linalg.norm(ca_coord - other['CA'].coord)
                                    if dist < 6.0:  # Potential H-bond distance
                                        nearby_polar += 1

                        # Buried polar without nearby polars = potential problem
                        if nearby_polar < 2:
                            buried_polar_count += 1

            results['buried_unsatisfied_polar_count'] = buried_polar_count

            max_allowed = self.config.get('max_buried_unsatisfied_polars', 3)
            if buried_polar_count > max_allowed:
                results['rejection_reasons'].append(
                    f"Buried unsatisfied polars ({buried_polar_count} > {max_allowed})"
                )

        except Exception as e:
            print(f"[WARN] Buried polar check failed: {e}")
            results['buried_unsatisfied_polar_count'] = 'N/A'

    def estimate_self_binding_risk(self, chain, pdb_path: Path, results: Dict):
        """
        Lightweight heuristic for self-association/oligomerization risk.
        Looks for large exposed sticky patches that could promote self-binding.
        """
        if not self.config.get('check_self_binding_risk', False):
            return

        try:
            # Combine hydrophobic and charge information
            hydrophobic_patch = results.get('largest_hydrophobic_scaffold_patch', 0)
            charge_patch_pos = results.get('max_positive_scaffold_cluster', 0)
            charge_patch_neg = results.get('max_negative_scaffold_cluster', 0)

            # Simple heuristic: large sticky patches = self-binding risk
            risk_score = 0.0

            # Large hydrophobic patches
            if hydrophobic_patch > 6:
                risk_score += (hydrophobic_patch - 6) * 0.1

            # Large charge patches
            max_charge_patch = max(charge_patch_pos, charge_patch_neg)
            if max_charge_patch > 4:
                risk_score += (max_charge_patch - 4) * 0.15

            # Check for symmetric exposed surfaces (very rough heuristic)
            # If binder has both large pos and neg patches, risk of charge-driven oligomerization
            if charge_patch_pos > 3 and charge_patch_neg > 3:
                risk_score += 0.3

            results['self_binding_risk_score'] = min(1.0, risk_score)
            results['oligomerization_risk'] = 'HIGH' if risk_score > 0.5 else ('MEDIUM' if risk_score > 0.25 else 'LOW')

            max_risk = self.config.get('max_self_binding_risk', 0.6)
            if risk_score > max_risk:
                results['rejection_reasons'].append(
                    f"High self-binding risk (score={risk_score:.2f})"
                )

        except Exception as e:
            print(f"[WARN] Self-binding risk assessment failed: {e}")
            results['self_binding_risk_score'] = 'N/A'

    def assess_packing_quality(self, chain, pdb_path: Path, results: Dict):
        """
        Assess local packing density.
        Poorly packed regions are destabilizing and aggregation-prone.
        """
        if not self.config.get('check_packing_quality', False):
            return

        try:
            residues = list(chain.get_residues())
            model = chain.get_parent()

            packing_scores = []

            for residue in residues:
                if residue.id[0] != ' ' or 'CA' not in residue:
                    continue

                ca_coord = residue['CA'].coord

                # Count nearby CA atoms
                all_ca = [r['CA'] for r in residues if r.id[0] == ' ' and 'CA' in r]
                neighbors = 0

                for ca in all_ca:
                    dist = np.linalg.norm(ca_coord - ca.coord)
                    if 0 < dist < 10.0:  # Exclude self, count within 10Å
                        neighbors += 1

                packing_scores.append(neighbors)

            if packing_scores:
                avg_packing = np.mean(packing_scores)
                min_packing = np.min(packing_scores)

                results['average_packing_density'] = avg_packing
                results['min_packing_density'] = min_packing

                # Poor packing threshold
                min_allowed = self.config.get('min_packing_density', 8)
                if min_packing < min_allowed:
                    results['rejection_reasons'].append(
                        f"Poor local packing (min={min_packing:.1f} neighbors)"
                    )

        except Exception as e:
            print(f"[WARN] Packing quality assessment failed: {e}")
            results['average_packing_density'] = 'N/A'


    def add_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add DEVELOPABILITY score (0-100, higher is better).

        Focuses on protease resistance, aggregation risk, manufacturability.
        Deliberately EXCLUDES BindCraft metrics (pLDDT, iPAE, contacts) from heavy weighting.
        """
        if len(df) == 0:
            return df

        # Get scoring weights from config
        weights_config = self.config.get('scoring_weights', {})

        # Define developability metrics to normalize (lower values are better for all these)
        developability_metrics = [
            'exposed_protease_motifs',
            'exposed_kr_fraction',
            'histidine_fraction',
            'his_scaffold_cluster_size',
            'max_loop_length',
            'low_plddt_patch_length',
            'largest_hydrophobic_scaffold_patch',
            'exposed_hydrophobic_scaffold_fraction',
            'hydrophobic_patch_count',
            'max_positive_scaffold_cluster',
            'max_negative_scaffold_cluster',
            'buried_unsatisfied_polar_count',
            'self_binding_risk_score',
            'average_packing_density',  # Higher is better
            'min_packing_density',       # Higher is better
        ]

        # Normalize each metric to 0-1 scale (higher is better after normalization)
        normalized_scores = {}

        for metric in developability_metrics:
            if metric not in df.columns:
                continue

            # Get valid numeric data
            valid_data = pd.to_numeric(df[metric], errors='coerce')

            if valid_data.isna().all():
                continue

            min_val = valid_data.min()
            max_val = valid_data.max()

            if max_val == min_val:
                normalized = pd.Series([0.5] * len(df), index=df.index)
            else:
                # Normalize to 0-1
                normalized = (valid_data - min_val) / (max_val - min_val)

                # For most metrics, lower is better, so invert
                if metric not in ['average_packing_density', 'min_packing_density']:
                    normalized = 1.0 - normalized

            # Fill NaN with neutral value
            normalized = normalized.fillna(0.5)
            normalized_scores[f'{metric}_normalized'] = normalized

        # Add normalized scores to dataframe
        for col, values in normalized_scores.items():
            df[col] = values

        # Handle sequence redundancy penalty
        if 'redundancy_flag' in df.columns:
            redundancy_penalty = df['redundancy_flag'].apply(lambda x: 0.0 if x else 1.0)
            normalized_scores['redundancy_normalized'] = redundancy_penalty
            df['redundancy_normalized'] = redundancy_penalty

        # Compute weighted developability score (0-100)
        total_score = pd.Series([0.0] * len(df), index=df.index)
        total_weight = 0.0

        for metric, weight in weights_config.items():
            # Map config keys to normalized column names
            if metric == 'sequence_redundancy_penalty':
                normalized_col = 'redundancy_normalized'
            else:
                normalized_col = f'{metric}_normalized'

            if normalized_col in df.columns:
                total_score += df[normalized_col] * weight
                total_weight += weight

        # Normalize to 0-100 scale
        if total_weight > 0:
            df['developability_score'] = (total_score / total_weight) * 100
        else:
            df['developability_score'] = 50.0  # Neutral

        # Legacy alias for compatibility
        df['composite_robustness_score'] = df['developability_score']

        # Percentile ranks
        df['percentile_rank'] = df['developability_score'].rank(pct=True) * 100

        return df

    def cluster_by_sequence_identity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster designs by sequence identity to remove redundancy.
        Keeps best representative from each cluster.
        """
        if len(df) == 0:
            return df

        threshold = self.config.get('sequence_identity_threshold', 0.85)
        enable_clustering = self.config.get('enable_sequence_clustering', True)

        if not enable_clustering:
            df['sequence_cluster_id'] = list(range(len(df)))
            df['cluster_representative'] = True
            df['redundancy_flag'] = False
            return df

        # Build similarity matrix
        sequences = df['sequence'].tolist()
        n = len(sequences)

        # Simple greedy clustering
        cluster_ids = [-1] * n
        representatives = [False] * n
        current_cluster = 0

        for i in range(n):
            if cluster_ids[i] != -1:
                continue  # Already assigned

            # Start new cluster
            cluster_ids[i] = current_cluster
            representatives[i] = True

            # Find similar sequences
            for j in range(i + 1, n):
                if cluster_ids[j] != -1:
                    continue

                identity = compute_sequence_identity(sequences[i], sequences[j])
                if identity >= threshold:
                    cluster_ids[j] = current_cluster

            current_cluster += 1

        df['sequence_cluster_id'] = cluster_ids
        df['cluster_representative'] = representatives
        df['redundancy_flag'] = [not rep for rep in representatives]

        return df

    def generate_report(self, df_all: pd.DataFrame, df_passed: pd.DataFrame, df_rejected: pd.DataFrame) -> Dict:
        """Generate summary report of filtering results"""
        report = {
            'total_designs': len(df_all),
            'passed': len(df_passed),
            'rejected': len(df_rejected),
            'pass_rate': len(df_passed) / len(df_all) if len(df_all) > 0 else 0,
            'rejection_breakdown': {},
            'filter_statistics': {}
        }

        # Count rejection reasons
        if len(df_rejected) > 0:
            all_reasons = []
            for reasons_str in df_rejected['rejection_reasons']:
                if isinstance(reasons_str, str) and reasons_str:
                    all_reasons.extend([r.split('(')[0].strip() for r in reasons_str.split(';')])

            reason_counts = Counter(all_reasons)
            report['rejection_breakdown'] = dict(reason_counts)

        # Calculate statistics for current metrics
        numeric_cols = [
            # Basic metrics
            'binder_length',
            'exposed_protease_motifs',
            'exposed_kr_fraction',
            'total_kr',
            'exposed_kr',

            # Histidine clustering (structure-aware)
            'histidine_count',
            'histidine_fraction',
            'his_spatial_cluster_size',
            'his_scaffold_cluster_size',

            # Structure quality
            'max_loop_length',
            'min_plddt',
            'mean_plddt',
            'low_plddt_patch_length',

            # Interface metrics
            'interface_pae',
            'interface_contacts',
            'interface_residue_count',
            'scaffold_residue_count',

            # Aggregation risk (structure-aware)
            'exposed_hydrophobic_fraction',
            'exposed_hydrophobic_scaffold_fraction',
            'exposed_hydrophobic_area',
            'largest_hydrophobic_patch',
            'largest_hydrophobic_scaffold_patch',
            'hydrophobic_patch_count',

            # Charge patches (structure-aware)
            'max_positive_spatial_cluster',
            'max_negative_spatial_cluster',
            'max_positive_scaffold_cluster',
            'max_negative_scaffold_cluster',

            # Advanced filters
            'buried_unsatisfied_polar_count',
            'self_binding_risk_score',
            'average_packing_density',
            'min_packing_density',

            # Composite scores
            'composite_robustness_score',
            'percentile_rank'
        ]

        for col in numeric_cols:
            if col in df_all.columns:
                valid_data = pd.to_numeric(df_all[col], errors='coerce').dropna()
                if len(valid_data) > 0:
                    report['filter_statistics'][col] = {
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max())
                    }

        return report


def main():
    parser = argparse.ArgumentParser(description='Post-filter BindCraft accepted designs')
    parser.add_argument('--config', required=True, help='Path to filter configuration JSON')
    parser.add_argument('--input', required=True, help='Path to BindCraft output directory')
    parser.add_argument('--output', default=None, help='Output directory (default: same as input)')

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output) if args.output else Path(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run filtering
    filter_obj = PostDesignFilter(args.config, args.input)
    df_passed, df_rejected, report = filter_obj.run()

    if df_passed is None:
        print("[ERROR] Filtering failed")
        sys.exit(1)

    # Save results
    passed_csv = output_dir / "filtered_designs.csv"
    rejected_csv = output_dir / "rejected_designs.csv"
    report_json = output_dir / "filter_report.json"

    df_passed.to_csv(passed_csv, index=False)
    df_rejected.to_csv(rejected_csv, index=False)

    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Results saved:")
    print(f"  Passed designs: {passed_csv}")
    print(f"  Rejected designs: {rejected_csv}")
    print(f"  Report: {report_json}")


if __name__ == '__main__':
    main()
