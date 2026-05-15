Parallelize Bindraft instances (Running Parallel Instances)

Integrate pDockQ2 or PI_Score. These metrics look specifically at the interface density and chemical complementarity (hydrophobicity, hydrogen bonds).

pDockQ2
https://gitlab.com/ElofssonLab/FoldDock

Use tools like MaSIF or PeSTo (surface fingerprinting) before running BindCraft to identify "druggable" patches.

PeSTo
https://github.com/LBM-EPFL/PeSTo

generate binder from known ppi complex

AF2 loss control

multi-state validation

Automatically pipe the top BindCraft designs through Chai-1 and Boltz-2.

filter for pH, protease, and temp selection

upgrade to make it protease-specific:

"protease_profiles": {
  "trypsin": ["K", "R"],
  "furin": ["R..R", "R.KR", "R.RR"],
  "chymotrypsin": ["F", "W", "Y", "L"],
  "proteinase_k": ["A", "F", "Y", "W", "L", "I", "V"]
}


ThermoMPNN-D
https://github.com/Kuhlman-Lab/ThermoMPNN-D/tree/main

PROPKA
https://github.com/jensengroup/propka

animation generation needs to be zoomed out

result dashboard

[DEBUG] AF2 output keys: ['aatype', 'atom_mask', 'atom_positions', 'cmap', 'grad', 'i_cmap', 'i_ptm', 'loss', 'losses', 'num_recycles', 'pae', 'plddt', 'prev', 'ptm', 'residue_index', 'seq', 'seq_pseudo', 'all', 'log']


Target: STAPHYLOCOCCAL ENTEROTOXIN B
3SEB chain A
Hotspots: [Y89,Y90]
Total trajectories:
Falied designs:

Binders passed by bindcraft:30
Binders passed by post filter:

$4.42/hr
$0.442/hr/gpu
A40 48GB 10x 
502GB RAM
96 (Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz)

Total combined runtime with container launch:(1 hour, 52 minutes, and 34 seconds)
cost break down: $8.27 ($0.276/binder)

Target 