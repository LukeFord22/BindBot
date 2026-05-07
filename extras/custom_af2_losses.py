"""
Custom AF2 Loss Functions for Enhanced Binder Design

This module provides additional loss terms that can be enabled via config to:
- Promote compact binders (radius of gyration)
- Encourage intrachain contacts (rigidity)
- Penalize low confidence regions
- Control terminal flexibility
- Manage loop flexibility

All losses are config-controlled and designed to complement (not dominate)
the existing binding/interface losses.
"""

import jax
import jax.numpy as jnp
import numpy as np
from alphafold.model import all_atom
from alphafold.common import residue_constants


def add_intrachain_contact_loss(self, weight=0.1, distance_threshold=8.0):
    """
    Add reward for intrachain contacts to promote rigid, well-packed binders.

    Args:
        weight: Loss weight (positive values reward contacts)
        distance_threshold: Maximum distance (Å) to count as contact
    """
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Binder only

        # Compute pairwise distances
        ca_i = ca[:, None, :]  # Shape: (L, 1, 3)
        ca_j = ca[None, :, :]  # Shape: (1, L, 3)
        dist = jnp.linalg.norm(ca_i - ca_j, axis=-1)  # Shape: (L, L)

        # Mask out diagonal and short-range (< 4 residues apart)
        mask = jnp.ones_like(dist)
        for i in range(len(ca)):
            for j in range(max(0, i-3), min(len(ca), i+4)):
                mask = mask.at[i, j].set(0.0)

        # Count contacts within threshold
        contacts = jnp.where(dist < distance_threshold, 1.0, 0.0) * mask
        contact_count = jnp.sum(contacts) / 2  # Divide by 2 for symmetry

        # Normalize by binder length (smaller binders naturally have fewer contacts)
        normalized_contacts = contact_count / (self._binder_len ** 1.5)

        # Return negative to reward more contacts
        return {"intra_contact": -normalized_contacts}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["intra_contact"] = weight


def add_low_plddt_penalty(self, weight=0.1, plddt_threshold=70):
    """
    Penalize regions with predicted low confidence (pLDDT).

    Args:
        weight: Loss weight
        plddt_threshold: pLDDT values below this trigger penalty
    """
    def loss_fn(inputs, outputs):
        # Get predicted pLDDT for binder
        plddt = outputs["predicted_lddt"]["logits"]

        # Convert logits to pLDDT scores (0-100)
        # AlphaFold pLDDT is calculated from binned probabilities
        # Simplified: use mean of logits as proxy
        binder_plddt = plddt[-self._binder_len:]

        # Apply sigmoid to get values in 0-1 range, then scale to 0-100
        plddt_scores = jax.nn.sigmoid(binder_plddt) * 100

        # Penalize residues below threshold
        penalty = jax.nn.relu(plddt_threshold - plddt_scores)
        mean_penalty = jnp.mean(penalty)

        return {"low_plddt": mean_penalty}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["low_plddt"] = weight


def add_terminal_looseness_penalty(self, weight=0.1, n_term_residues=3, c_term_residues=3, max_bfactor=50):
    """
    Penalize high B-factors (flexibility) at termini.

    Args:
        weight: Loss weight
        n_term_residues: Number of N-terminal residues to check
        c_term_residues: Number of C-terminal residues to check
        max_bfactor: B-factor threshold
    """
    def loss_fn(inputs, outputs):
        # Get predicted pLDDT as proxy for B-factor
        # Lower pLDDT = higher flexibility/B-factor
        plddt = outputs["predicted_lddt"]["logits"]
        binder_plddt = plddt[-self._binder_len:]

        # Convert to pseudo-B-factor (inverse relationship)
        # High pLDDT = low B-factor, low pLDDT = high B-factor
        bfactor_proxy = 100 - (jax.nn.sigmoid(binder_plddt) * 100)

        # Get terminal regions
        n_term_bf = bfactor_proxy[:n_term_residues]
        c_term_bf = bfactor_proxy[-c_term_residues:]

        # Penalize if above threshold
        n_penalty = jnp.mean(jax.nn.relu(n_term_bf - max_bfactor))
        c_penalty = jnp.mean(jax.nn.relu(c_term_bf - max_bfactor))

        total_penalty = (n_penalty + c_penalty) / 2

        return {"terminal_loose": total_penalty}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["terminal_loose"] = weight


def add_loop_flexibility_penalty(self, weight=0.1, target_secondary_structure=None):
    """
    Penalize excessive flexibility in loop regions.

    Args:
        weight: Loss weight
        target_secondary_structure: Optional mask for where loops are expected
    """
    def loss_fn(inputs, outputs):
        # Use distogram to identify structured vs unstructured regions
        dgram = outputs["distogram"]["logits"]

        # Get binder region only
        binder_start = self._target_len
        binder_end = self._target_len + self._binder_len

        binder_dgram = dgram[binder_start:binder_end, binder_start:binder_end]

        # Calculate contact propensity for each residue
        # Well-structured residues have more predicted contacts
        contact_threshold_bin = 10  # Corresponds to ~5-6Å
        contact_probs = jax.nn.softmax(binder_dgram, axis=-1)
        contacts_per_residue = jnp.sum(contact_probs[:, :, :contact_threshold_bin], axis=(1, 2))

        # Penalize residues with very few contacts (likely disordered loops)
        min_contacts_threshold = 0.5
        loop_penalty = jnp.mean(jax.nn.relu(min_contacts_threshold - contacts_per_residue))

        return {"loop_flex": loop_penalty}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["loop_flex"] = weight


def add_secondary_structure_bias(self, weight=0.1, prefer_helix=True):
    """
    Bias toward helical or sheet secondary structure.

    Args:
        weight: Loss weight
        prefer_helix: If True, bias toward helix; if False, bias toward sheet
    """
    def loss_fn(inputs, outputs):
        # Use distogram patterns to infer secondary structure
        dgram = outputs["distogram"]["logits"]

        binder_start = self._target_len
        binder_end = self._target_len + self._binder_len
        binder_dgram = dgram[binder_start:binder_end, binder_start:binder_end]

        if prefer_helix:
            # Helices have characteristic i, i+3/i+4 contacts
            offset_3 = jnp.diagonal(binder_dgram, offset=3, axis1=0, axis2=1)
            offset_4 = jnp.diagonal(binder_dgram, offset=4, axis1=0, axis2=1)

            # Get contact probability at ~5-6Å (typical for helix H-bonds)
            contact_probs_3 = jax.nn.softmax(offset_3, axis=-1)[:, 8:12].sum(axis=-1)
            contact_probs_4 = jax.nn.softmax(offset_4, axis=-1)[:, 8:12].sum(axis=-1)

            # Reward helix-like contacts (negative loss)
            helix_score = (jnp.mean(contact_probs_3) + jnp.mean(contact_probs_4)) / 2
            return {"ss_bias": -helix_score}
        else:
            # Sheets have characteristic i, i+2 contacts (parallel/antiparallel)
            offset_2 = jnp.diagonal(binder_dgram, offset=2, axis1=0, axis2=1)

            contact_probs_2 = jax.nn.softmax(offset_2, axis=-1)[:, 6:10].sum(axis=-1)
            sheet_score = jnp.mean(contact_probs_2)
            return {"ss_bias": -sheet_score}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["ss_bias"] = weight


def apply_custom_losses_from_config(af_model, config):
    """
    Apply custom AF2 losses based on configuration.

    Args:
        af_model: ColabDesign AF2 model instance
        config: Dictionary with custom loss settings

    Returns:
        af_model: Model with custom losses applied
    """
    print("[INFO] Applying custom AF2 losses from config...")

    # Radius of gyration (already exists in colabdesign_utils.py)
    if config.get("rg_loss", {}).get("enabled", False):
        from functions.colabdesign_utils import add_rg_loss
        weight = config["rg_loss"].get("weight", 0.1)
        add_rg_loss(af_model, weight=weight)
        print(f"  [+] Radius of gyration loss (weight={weight})")

    # Intrachain contacts
    if config.get("intrachain_contact_loss", {}).get("enabled", False):
        weight = config["intrachain_contact_loss"].get("weight", 0.1)
        distance_threshold = config["intrachain_contact_loss"].get("distance_threshold", 8.0)
        add_intrachain_contact_loss(af_model, weight=weight, distance_threshold=distance_threshold)
        print(f"  [+] Intrachain contact loss (weight={weight}, threshold={distance_threshold}Å)")

    # Low pLDDT penalty
    if config.get("low_plddt_penalty", {}).get("enabled", False):
        weight = config["low_plddt_penalty"].get("weight", 0.1)
        threshold = config["low_plddt_penalty"].get("plddt_threshold", 70)
        add_low_plddt_penalty(af_model, weight=weight, plddt_threshold=threshold)
        print(f"  [+] Low pLDDT penalty (weight={weight}, threshold={threshold})")

    # Terminal looseness penalty
    if config.get("terminal_looseness_penalty", {}).get("enabled", False):
        weight = config["terminal_looseness_penalty"].get("weight", 0.1)
        n_term = config["terminal_looseness_penalty"].get("n_term_residues", 3)
        c_term = config["terminal_looseness_penalty"].get("c_term_residues", 3)
        add_terminal_looseness_penalty(af_model, weight=weight,
                                      n_term_residues=n_term, c_term_residues=c_term)
        print(f"  [+] Terminal looseness penalty (weight={weight}, N={n_term}, C={c_term})")

    # Loop flexibility penalty
    if config.get("loop_flexibility_penalty", {}).get("enabled", False):
        weight = config["loop_flexibility_penalty"].get("weight", 0.1)
        add_loop_flexibility_penalty(af_model, weight=weight)
        print(f"  [+] Loop flexibility penalty (weight={weight})")

    # Secondary structure bias
    if config.get("secondary_structure_bias", {}).get("enabled", False):
        weight = config["secondary_structure_bias"].get("weight", 0.1)
        prefer_helix = config["secondary_structure_bias"].get("prefer_helix", True)
        add_secondary_structure_bias(af_model, weight=weight, prefer_helix=prefer_helix)
        ss_type = "helix" if prefer_helix else "sheet"
        print(f"  [+] Secondary structure bias toward {ss_type} (weight={weight})")

    # Helicity loss (already exists in colabdesign_utils.py)
    if config.get("helix_loss", {}).get("enabled", False):
        from functions.colabdesign_utils import add_helix_loss
        weight = config["helix_loss"].get("weight", 0.1)
        add_helix_loss(af_model, weight=weight)
        print(f"  [+] Helicity loss (weight={weight})")

    # Termini distance loss (already exists in colabdesign_utils.py)
    if config.get("termini_distance_loss", {}).get("enabled", False):
        from functions.colabdesign_utils import add_termini_distance_loss
        weight = config["termini_distance_loss"].get("weight", 0.1)
        threshold = config["termini_distance_loss"].get("threshold_distance", 7.0)
        add_termini_distance_loss(af_model, weight=weight, threshold_distance=threshold)
        print(f"  [+] Termini distance loss (weight={weight}, threshold={threshold}Å)")

    print("[INFO] Custom losses applied successfully")
    return af_model
