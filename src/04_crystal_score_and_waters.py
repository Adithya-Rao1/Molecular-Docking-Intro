"""Step 4. 
Score the crystal pose in place and after local minimization, with and without crystal waters, then redock with each receptor (exhaustiveness 32, seeds 1-3). 
Uses standard preparation (Open Babel + Meeko).
Compare with logs/crystal_score_and_waters.log.
"""
import argparse
import numpy as np
from rdkit import Chem
from vina import Vina
from redock.common import (CPU, RmsdScorer, crystal_ligand, embedded_ligand, perm_for, prep_ligand, read_box, receptor_pdbqt)

ap = argparse.ArgumentParser(); ap.add_argument("--cpu", type=int, default=CPU); a = ap.parse_args()
center, size = read_box(); ref = crystal_ligand(); scorer = RmsdScorer(ref)
cr = Chem.AddHs(ref, addCoords=True); cr_txt = prep_ligand(cr); cr_heavy = Chem.RemoveHs(cr)
m = embedded_ligand(7); txt = prep_ligand(m); heavy = Chem.RemoveHs(m); perm = perm_for(txt, heavy)
for name, keep in (("dry", False), ("wet", True)):
    rp = receptor_pdbqt("rec_" + name, keep)
    v = Vina(sf_name="vina", cpu=a.cpu, seed=1, verbosity=0); v.set_receptor(rp); v.set_ligand_from_string(cr_txt); v.compute_vina_maps(center=center, box_size=size)
    e0 = v.score()[0]; v.optimize(); e1 = v.score()[0]
    print(f"[{name}] crystal pose in place {e0:.2f}, after local minimisation {e1:.2f}", flush=True)
    for seed in (1, 2, 3):
        v = Vina(sf_name="vina", cpu=a.cpu, seed=seed, verbosity=0); v.set_receptor(rp); v.set_ligand_from_string(txt); v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=32, n_poses=9)
        from redock.common import work
        p = work("diagnose", f"diag_{name}_{seed}.pdbqt"); v.write_poses(str(p), n_poses=9, overwrite=True)
        r = scorer.of_poses(p.read_text(), perm, heavy); e = v.energies(n_poses=9)[:, 0]
        print(f"[{name}] seed {seed} scores {np.round(e, 2).tolist()}\n          RMSD   {np.round(r, 1).tolist()}", flush=True)
