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

$3.54/hr
A40 8x

10 min start