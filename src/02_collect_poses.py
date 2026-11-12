"""Step 2. 
Collect the 45 (exhaustiveness 32) poses (5 seeds x 9) into one array in dockprep SDF heavy-atom order, followed by the crystal pose (index 45). 
These poses get rescored in steps 7-9.
"""
import json
import numpy as np
from redock.common import INPUTS, N_HEAVY, REF, WORK, crystal_ligand, metis_ligand, models, perm_for, work

src = WORK if (WORK / "dock_results.json").exists() else REF
rows = json.load(open(src / ("dock_results.json")))
pose_dir = WORK / "vina_out"
assert pose_dir.exists(), "run 01_dock.py first (or see README for using the shipped poses)"
heavy = metis_ligand()
perm = perm_for((INPUTS / "erlotinib.pdbqt").read_text(), heavy)  # PDBQT atom order -> SDF heavy order
P, meta = [], []
for r in rows:
    if r["ex"] != 32:
        continue
    text = (pose_dir / f"pose_ex32_s{r['seed']}.pdbqt").read_text()
    for k, atoms in enumerate(models(text)):
        x = np.zeros((N_HEAVY, 3))
        for a, q in enumerate(atoms):
            x[perm[a]] = q
        P.append(x)
        meta.append(dict(seed=r["seed"], rank=k + 1, vina=r["scores"][k], rmsd=r["rmsds"][k]))
ref = crystal_ligand()
cx = np.array([list(ref.GetConformer().GetAtomPosition(i)) for i in range(ref.GetNumAtoms())])
match = ref.GetSubstructMatch(heavy)  # crystal atoms (template order) -> SDF heavy order
xc = np.zeros((N_HEAVY, 3))
for sdf_i, ref_i in enumerate(match):
    xc[sdf_i] = cx[ref_i]
P.append(xc)
meta.append(dict(seed=0, rank=0, vina=None, rmsd=0.0, crystal=True))
np.save(work("poses_xyz.npy"), np.array(P))
json.dump(meta, open(work("poses_meta.json"), "w"))
print(len(P), "poses saved; crystal pose last (index 45)")
