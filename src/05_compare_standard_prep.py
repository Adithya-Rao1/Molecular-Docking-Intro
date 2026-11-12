"""Step 5. 
Dock with {dockprep, Open Babel} receptor x {dockprep, Meeko} ligand, exhaustiveness 32, seeds 1-5. 
Also scores the crystal pose in place for the two receptors.
Ccompare with data/results/standard_prep_comparison.json.
"""
import argparse, json
import numpy as np
from rdkit import Chem
from vina import Vina
from redock.common import (CPU, INPUTS, RmsdScorer, crystal_ligand, embedded_ligand, metis_ligand, perm_for, prep_ligand, read_box,
                           receptor_pdbqt, work)

ap = argparse.ArgumentParser(); ap.add_argument("--cpu", type=int, default=CPU); a = ap.parse_args()
center, size = read_box(); ref = crystal_ligand(); scorer = RmsdScorer(ref)
m = embedded_ligand(7); meeko_txt = prep_ligand(m); meeko_heavy = Chem.RemoveHs(m)
metis_txt = (INPUTS / "erlotinib.pdbqt").read_text(); metis_heavy = metis_ligand()
cr = Chem.AddHs(ref, addCoords=True); cr_txt = prep_ligand(cr)
recs = {"metis_rec": str(INPUTS / "receptor.pdbqt"), "obabel_rec": receptor_pdbqt("rec_obabel")}
print("crystal-pose in-place score per receptor (kcal/mol), then after local minimisation:")
for rn, rp in recs.items():
    v = Vina(sf_name="vina", cpu=a.cpu, seed=1, verbosity=0); v.set_receptor(rp); v.set_ligand_from_string(cr_txt); v.compute_vina_maps(center=center, box_size=size)
    e0 = v.score()[0]; v.optimize(); print(" ", rn, "%.2f -> %.2f" % (e0, v.score()[0]))
conds = {"metis_rec+metis_lig": ("metis_rec", metis_txt, metis_heavy), "metis_rec+meeko_lig": ("metis_rec", meeko_txt, meeko_heavy),
         "obabel_rec+meeko_lig": ("obabel_rec", meeko_txt, meeko_heavy), "obabel_rec+metis_lig": ("obabel_rec", metis_txt, metis_heavy)}
res = {}
for name, (rn, txt, heavy) in conds.items():
    perm = perm_for(txt, heavy); rows = []
    for seed in range(1, 6):
        v = Vina(sf_name="vina", cpu=a.cpu, seed=seed, verbosity=0); v.set_receptor(recs[rn]); v.set_ligand_from_string(txt)
        v.compute_vina_maps(center=center, box_size=size); v.dock(exhaustiveness=32, n_poses=9)
        p = work("standard_prep", f"{name}_{seed}.pdbqt"); v.write_poses(str(p), n_poses=9, overwrite=True)
        r = scorer.of_poses(p.read_text(), perm, heavy); e = v.energies(n_poses=9)[:, 0].tolist(); rows.append((e[0], r[0], min(r)))
    arr = np.array(rows); res[name] = arr.tolist()
    print("%-24s top score %.2f+-%.2f  top-RMSD med %.2f (min %.2f max %.2f)  best-of-9 med %.2f  seeds top<2A: %d/5" % (
        name, arr[:, 0].mean(), arr[:, 0].std(), np.median(arr[:, 1]), arr[:, 1].min(), arr[:, 1].max(), np.median(arr[:, 2]), (arr[:, 1] < 2).sum()), flush=True)
work("standard_prep_comparison.json").write_text(json.dumps(res))
