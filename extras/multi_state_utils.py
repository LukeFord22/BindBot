#!/usr/bin/env python3
"""
Utility functions for multi-state validation scoring and analysis.

These functions are used by MultiStateValidator to score positive/negative
validation states and aggregate results.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from Bio import PDB

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from functions.biopython_utils import (
    calculate_clash_score,
    hotspot_residues,
    biopython_unaligned_rmsd
)


def analyze_structure(pdb_parser: PDB.PDBParser, complex_pdb: Path) -> Dict:
    """Analyze structure without AF2 prediction (fallback)"""
    metrics = {}

    try:
        structure = pdb_parser.get_structure('complex', str(complex_pdb))
        model = structure[0]
        chains = list(model.get_chains())

        if len(chains) < 2:
            return {'error': 'Insufficient chains in complex'}

        # Use B-factors as pLDDT proxy if available
        binder_chain = chains[1]
        plddt_values = []
        for residue in binder_chain.get_residues():
            if residue.id[0] == ' ':
                for atom in residue.get_atoms():
                    plddt_values.append(atom.get_bfactor())
                    break

        if plddt_values:
            metrics['binder_plddt'] = float(np.mean(plddt_values))

    except Exception as e:
        metrics['error'] = str(e)

    return metrics


def calculate_interface_metrics(complex_pdb: Path, binder_chain: str = 'B') -> Dict:
    """
    Calculate interface contacts and clashes using production utilities.

    Returns interface residue positions for consistency checking across states.

    Args:
        complex_pdb: Path to complex PDB file
        binder_chain: Chain ID of the binder (default 'B')
    """
    metrics = {}

    try:
        # Use production-ready hotspot_residues for interface contact counting
        # This function returns dict of {residue_position: aa_code} for interface residues
        interface_residues = hotspot_residues(str(complex_pdb), binder_chain=binder_chain, atom_distance_cutoff=4.0)
        metrics['interface_contacts'] = len(interface_residues)

        # Store interface residue positions for consistency checking
        # Convert keys to sorted list of integers
        metrics['interface_residue_positions'] = sorted([int(pos) for pos in interface_residues.keys()])

        # Use production-ready clash detection from biopython_utils
        # Calculate clashes at interface (non-CA atoms between different chains)
        clash_count = calculate_clash_score(str(complex_pdb), threshold=2.4, only_ca=False)
        metrics['interface_clashes'] = clash_count

    except Exception as e:
        print(f"    [WARN] Interface metrics failed: {e}")

    return metrics


def calculate_rmsd(complex_pdb: Path, reference_pdb: Path) -> float:
    """Calculate RMSD between complex and reference using production utility"""
    try:
        # Use production-ready RMSD calculation from biopython_utils
        # Assumes both structures have chain A (target) and chain B (binder)
        rmsd = biopython_unaligned_rmsd(
            str(reference_pdb),
            str(complex_pdb),
            reference_chain_id='A',
            align_chain_id='A'
        )
        return float(rmsd)

    except Exception as e:
        print(f"    [WARN] RMSD calculation failed: {e}")
        return np.nan


def get_sequence(chain) -> str:
    """Extract amino acid sequence from chain using Biopython's Polypeptide"""
    from Bio.PDB.Polypeptide import PPBuilder

    # Use Bio.PDB's built-in sequence extraction
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(chain):
        sequence += str(pp.get_sequence())

    return sequence


def check_interface_consistency(positive_results: List[Dict],
                                consistency_threshold: float = 0.5) -> Tuple[bool, Dict]:
    """
    Check if binder engages same interface residues across positive states.

    Computes Jaccard similarity between interface residue sets.
    Low consistency suggests binder binds different epitopes across states.

    Args:
        positive_results: List of validation results for positive states
        consistency_threshold: Minimum mean Jaccard similarity (0-1)

    Returns:
        (is_consistent, consistency_metrics_dict)
    """
    from itertools import combinations

    # Extract interface residue sets from each state
    interface_sets = []
    for result in positive_results:
        if 'interface_residue_positions' in result and result.get('validation_status') == 'SUCCESS':
            interface_sets.append(set(result['interface_residue_positions']))

    if len(interface_sets) < 2:
        return True, {
            'reason': 'Not enough states to compare',
            'num_states': len(interface_sets),
            'is_consistent': True
        }

    # Calculate Jaccard similarity between all pairs
    similarities = []
    for set1, set2 in combinations(interface_sets, 2):
        if len(set1) == 0 or len(set2) == 0:
            similarities.append(0.0)
        else:
            # Jaccard index: intersection / union
            jaccard = len(set1 & set2) / len(set1 | set2)
            similarities.append(jaccard)

    mean_similarity = float(np.mean(similarities))
    min_similarity = float(np.min(similarities))
    is_consistent = mean_similarity >= consistency_threshold

    return is_consistent, {
        'mean_jaccard_similarity': mean_similarity,
        'min_jaccard_similarity': min_similarity,
        'is_consistent': is_consistent,
        'num_states_compared': len(interface_sets),
        'threshold': consistency_threshold
    }


def check_catastrophic_failure(result: Dict, config: Dict) -> Tuple[bool, str]:
    """
    Check if a state result represents a catastrophic failure.

    Returns: (is_catastrophic, failure_reason)
    """
    catastrophic_def = config.get('catastrophic_failure_definition', {})

    # Check binder pLDDT
    binder_plddt = result.get('binder_plddt', 100)
    if not np.isnan(binder_plddt) and binder_plddt < catastrophic_def.get('binder_plddt_below', 70):
        return True, f"binder_plddt={binder_plddt:.1f} < {catastrophic_def.get('binder_plddt_below')}"

    # Check interface pAE
    interface_pae = result.get('interface_pae', 0)
    if not np.isnan(interface_pae) and interface_pae > catastrophic_def.get('interface_pae_above', 20):
        return True, f"interface_pae={interface_pae:.2f} > {catastrophic_def.get('interface_pae_above')}"

    # Check interface contacts
    if result.get('interface_contacts', 100) < catastrophic_def.get('interface_contacts_below', 5):
        return True, f"interface_contacts={result.get('interface_contacts')} < {catastrophic_def.get('interface_contacts_below')}"

    # Check ipTM - skip if NaN (not available)
    iptm = result.get('iptm', 1.0)
    if not np.isnan(iptm) and iptm < catastrophic_def.get('iptm_below', 0.35):
        return True, f"iptm={iptm:.3f} < {catastrophic_def.get('iptm_below')}"

    # Check clashes
    if result.get('interface_clashes', 0) > catastrophic_def.get('major_clashes_above', 0):
        return True, f"interface_clashes={result.get('interface_clashes')} > {catastrophic_def.get('major_clashes_above')}"

    return False, ""


def compute_positive_state_score(result: Dict, config: Dict) -> float:
    """
    Compute normalized score (0-1) for a positive state result.
    Higher is better.
    """
    score = 0.0
    weights_sum = 0.0

    # Binder pLDDT (weight: 0.3)
    if 'binder_plddt' in result and not np.isnan(result['binder_plddt']):
        plddt = result['binder_plddt']
        score += (plddt / 100.0) * 0.3
        weights_sum += 0.3

    # Interface pAE (weight: 0.25, inverted - lower is better)
    if 'interface_pae' in result and not np.isnan(result['interface_pae']):
        ipae = result['interface_pae']
        normalized_ipae = max(0, 1 - (ipae / 20.0))
        score += normalized_ipae * 0.25
        weights_sum += 0.25

    # ipTM (weight: 0.2) - skip if NaN
    if 'iptm' in result and not np.isnan(result['iptm']):
        iptm = result['iptm']
        score += iptm * 0.2
        weights_sum += 0.2

    # Interface contacts from i_cmap (weight: 0.15) - prioritize AF2's predicted contacts
    if 'interface_contact_count' in result and not np.isnan(result.get('interface_contact_count', np.nan)):
        contacts = result['interface_contact_count']
        # Expect 10-30 high-confidence contacts for good binder
        preferred_contacts = config.get('positive_validation', {}).get('scoring', {}).get('preferred_interface_contacts', 20)
        normalized_contacts = min(1.0, contacts / preferred_contacts)
        score += normalized_contacts * 0.15
        weights_sum += 0.15
    # Fallback to geometric contacts if i_cmap not available
    elif 'interface_contacts' in result:
        contacts = result['interface_contacts']
        preferred_contacts = config.get('positive_validation', {}).get('scoring', {}).get('preferred_interface_contacts', 15)
        normalized_contacts = min(1.0, contacts / preferred_contacts)
        score += normalized_contacts * 0.15
        weights_sum += 0.15

    # Interface contact probability (weight: 0.1) - new metric from i_cmap
    if 'interface_contact_prob' in result and not np.isnan(result.get('interface_contact_prob', np.nan)):
        contact_prob = result['interface_contact_prob']
        # Mean probability > 0.3 indicates strong interface
        normalized_prob = min(1.0, contact_prob / 0.4)
        score += normalized_prob * 0.1
        weights_sum += 0.1

    # Clash penalty (weight: 0.05)
    if 'interface_clashes' in result:
        clashes = result['interface_clashes']
        clash_penalty = max(0, 1 - (clashes / 5.0))
        score += clash_penalty * 0.05
        weights_sum += 0.05

    # Normalize by actual weights used
    if weights_sum > 0:
        return score / weights_sum
    return 0.0


def compute_negative_state_score(result: Dict, config: Dict) -> float:
    """
    Compute normalized score (0-1) for a negative state result.
    Higher score = more specific (better rejection of off-target).
    Lower score = binding to off-target (bad).
    """
    score = 1.0  # Start at perfect specificity

    # Penalize high ipTM (indicates binding) - skip if NaN
    if 'iptm' in result and not np.isnan(result['iptm']):
        iptm = result['iptm']
        max_allowed_iptm = config.get('negative_validation', {}).get('scoring', {}).get('max_negative_iptm', 0.35)
        if iptm > max_allowed_iptm:
            penalty = (iptm - max_allowed_iptm) / (1.0 - max_allowed_iptm)
            score -= penalty * 0.4

    # Penalize low interface pAE (indicates confident binding) - skip if NaN
    if 'interface_pae' in result and not np.isnan(result['interface_pae']):
        ipae = result['interface_pae']
        min_allowed_ipae = config.get('negative_validation', {}).get('scoring', {}).get('min_negative_interface_pae', 15.0)
        if ipae < min_allowed_ipae:
            penalty = (min_allowed_ipae - ipae) / min_allowed_ipae
            score -= penalty * 0.3

    # Penalize high interface contact count from i_cmap (prioritize AF2's predictions)
    if 'interface_contact_count' in result and not np.isnan(result.get('interface_contact_count', np.nan)):
        contacts = result['interface_contact_count']
        # Should have < 8 high-confidence contacts to off-target
        max_allowed_contacts = config.get('negative_validation', {}).get('scoring', {}).get('max_negative_interface_contacts', 8)
        if contacts > max_allowed_contacts:
            penalty = min(1.0, (contacts - max_allowed_contacts) / 15.0)
            score -= penalty * 0.2
    # Fallback to geometric contacts
    elif 'interface_contacts' in result:
        contacts = result['interface_contacts']
        max_allowed_contacts = config.get('negative_validation', {}).get('scoring', {}).get('max_negative_interface_contacts', 6)
        if contacts > max_allowed_contacts:
            penalty = min(1.0, (contacts - max_allowed_contacts) / 10.0)
            score -= penalty * 0.2

    # Penalize high interface contact probability (new metric from i_cmap)
    if 'interface_contact_prob' in result and not np.isnan(result.get('interface_contact_prob', np.nan)):
        contact_prob = result['interface_contact_prob']
        # Mean probability should be < 0.2 for non-binder
        max_allowed_prob = 0.25
        if contact_prob > max_allowed_prob:
            penalty = (contact_prob - max_allowed_prob) / (1.0 - max_allowed_prob)
            score -= penalty * 0.1

    return max(0.0, score)


def aggregate_state_results(design_name: str, state_results: List[Dict], original_metrics: pd.Series, config: Dict) -> Dict:
    """Aggregate results across positive and negative states with scoring"""
    aggregated = {
        'design': design_name,
    }

    # Create score columns for each individual state (score matrix)
    # These columns show performance against each target state
    for result in state_results:
        state_name = result.get('state_name', 'unknown')
        validation_type = result.get('validation_type', 'positive')

        # Calculate score based on validation type
        # Check if AF2 prediction succeeded (has metrics we need)
        has_metrics = 'binder_plddt' in result or 'iptm' in result

        if validation_type == 'positive':
            # Positive state: binding quality score (0-1, higher = better binding)
            if has_metrics:
                score = compute_positive_state_score(result, config)
            else:
                score = 0.0  # AF2 prediction failed, no binding score
        else:
            # Negative state: specificity score (0-1, higher = better rejection of off-target)
            if has_metrics:
                score = compute_negative_state_score(result, config)
            else:
                score = 1.0  # AF2 prediction failed = assume no binding (perfect specificity)

        # Add column with state score
        aggregated[f'{state_name}_score'] = float(score)

    # Separate positive and negative results
    positive_results = [r for r in state_results if r.get('validation_type') == 'positive']
    negative_results = [r for r in state_results if r.get('validation_type') == 'negative']

    aggregated['num_positive_states'] = len(positive_results)
    aggregated['num_negative_states'] = len(negative_results)
    aggregated['num_positive_success'] = sum(1 for r in positive_results if r.get('validation_status') == 'SUCCESS')
    aggregated['num_negative_success'] = sum(1 for r in negative_results if r.get('validation_status') == 'SUCCESS')

    # Check for catastrophic failures in positive states
    catastrophic_failures = []
    for result in positive_results:
        is_catastrophic, reason = check_catastrophic_failure(result, config)
        if is_catastrophic:
            catastrophic_failures.append(f"{result.get('state_name')}: {reason}")

    aggregated['catastrophic_failures'] = catastrophic_failures
    aggregated['has_catastrophic_failure'] = len(catastrophic_failures) > 0

    # Check interface consistency across positive states
    consistency_threshold = config.get('interface_consistency_threshold', 0.5)
    is_consistent, consistency_metrics = check_interface_consistency(
        positive_results,
        consistency_threshold=consistency_threshold
    )
    aggregated['interface_consistency'] = consistency_metrics
    aggregated['interface_is_consistent'] = is_consistent

    if not is_consistent:
        print(f"    [WARN] Interface inconsistency detected: Jaccard={consistency_metrics.get('mean_jaccard_similarity', 0):.2f} < {consistency_threshold}")

    # Compute scores for each state
    positive_scores = [compute_positive_state_score(r, config) for r in positive_results if r.get('validation_status') == 'SUCCESS']
    negative_scores = [compute_negative_state_score(r, config) for r in negative_results if r.get('validation_status') == 'SUCCESS']

    # Positive state metrics
    if positive_scores:
        aggregated['positive_mean_score'] = float(np.mean(positive_scores))
        aggregated['positive_worst_score'] = float(np.min(positive_scores))
        aggregated['positive_best_score'] = float(np.max(positive_scores))
        aggregated['positive_score_std'] = float(np.std(positive_scores))
        aggregated['positive_consistency'] = 1.0 - min(1.0, aggregated['positive_score_std'])
    else:
        aggregated['positive_mean_score'] = 0.0
        aggregated['positive_worst_score'] = 0.0
        aggregated['positive_best_score'] = 0.0
        aggregated['positive_score_std'] = 0.0
        aggregated['positive_consistency'] = 0.0

    # Negative state metrics (specificity)
    if negative_scores:
        aggregated['negative_mean_specificity'] = float(np.mean(negative_scores))
        aggregated['negative_worst_specificity'] = float(np.min(negative_scores))
        aggregated['negative_best_specificity'] = float(np.max(negative_scores))
    else:
        aggregated['negative_mean_specificity'] = 1.0  # No negative states = assume specific
        aggregated['negative_worst_specificity'] = 1.0
        aggregated['negative_best_specificity'] = 1.0

    # Compute specificity margin (positive binding - negative binding)
    if positive_scores and negative_scores:
        aggregated['specificity_margin'] = aggregated['positive_mean_score'] - (1.0 - aggregated['negative_mean_specificity'])
    else:
        aggregated['specificity_margin'] = 0.0

    # Aggregate raw metrics across positive and negative states separately
    metric_keys = ['binder_plddt', 'complex_plddt', 'interface_pae', 'interface_contacts', 'interface_clashes', 'ptm', 'iptm']

    for metric in metric_keys:
        # Positive states
        pos_values = [r[metric] for r in positive_results if metric in r and isinstance(r[metric], (int, float))]
        if pos_values:
            aggregated[f'positive_{metric}_mean'] = float(np.mean(pos_values))
            aggregated[f'positive_{metric}_worst'] = float(np.min(pos_values)) if 'plddt' in metric or 'ptm' in metric or 'contacts' in metric else float(np.max(pos_values))

        # Negative states
        neg_values = [r[metric] for r in negative_results if metric in r and isinstance(r[metric], (int, float))]
        if neg_values:
            aggregated[f'negative_{metric}_mean'] = float(np.mean(neg_values))
            aggregated[f'negative_{metric}_worst'] = float(np.max(neg_values)) if 'plddt' in metric or 'ptm' in metric or 'contacts' in metric else float(np.min(neg_values))

    # Store per-state results as JSON string
    aggregated['state_details'] = json.dumps(state_results)

    return aggregated


def compute_composite_rank_score(row: pd.Series, config: Dict) -> float:
    """
    Compute final composite ranking score from individual state scores.
    Returns score 0-100 (higher is better).

    Simple 50/50 split:
    - 50% from average of all positive state scores (binding quality)
    - 50% from average of all negative state scores (specificity)
    """
    # Collect all state scores from the matrix
    positive_scores = []
    negative_scores = []

    for col in row.index:
        if col.endswith('_score'):
            # Determine if this is a positive or negative state
            # We can check the state name against the config or use naming convention
            state_name = col.replace('_score', '')

            # For now, assume negative states have known names (BSA, Streptavidin, etc.)
            # or we rely on the validation having stored this info
            # Simpler: use the aggregate metrics that were already calculated
            pass

    # Use the pre-calculated mean scores
    positive_mean = row.get('positive_mean_score', 0)
    negative_mean = row.get('negative_mean_specificity', 1.0)  # Default 1.0 if no negatives

    # Simple 50/50 weighted average
    composite_score = (positive_mean * 0.5) + (negative_mean * 0.5)

    return min(100, max(0, composite_score * 100))


def apply_multistate_filters(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply scoring and ranking system.

    No hard pass/fail filtering - only generates scores and rankings.
    Users can apply their own thresholds based on the score matrix.
    """
    # Compute composite rank scores
    df['composite_rank_score'] = df.apply(lambda row: compute_composite_rank_score(row, config), axis=1)

    # Sort by rank score (highest first)
    df = df.sort_values('composite_rank_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df
