"""Step 3. 
Standard preparation (Open Babel receptor, Meeko ligand, no waters).

For seed 1 at exhaustiveness 32, reports: 
- score terms and geometry (hinge distances, centroid shift) of the top-ranked poses and of the near-crystal pose
- 35 A cube at exhaustiveness 64, exhaustiveness 128, and the Vinardo scoring function (seeds 1-3). 

Compare with logs/diagnose_top_pose.log.
"""
import argparse
import numpy as np
from rdkit import Chem
from vina import Vina
from redock.common import (CPU, RmsdScorer, atom_xyz, crystal_ligand, embedded_ligand, models, perm_for, prep_ligand,
                           read_box, receptor_pdbqt, work)

ap = argparse.ArgumentParser(); ap.add_argument("--cpu", type=int, default=CPU); a = ap.parse_args()
center, size = read_box()
ref = crystal_ligand(); scorer = RmsdScorer(ref)
rp = receptor_pdbqt("rec_dry")
MET_N = atom_xyz("MET", 769, "N"); THR_OG = atom_xyz("THR", 766, "OG1")
m = embedded_ligand(7); txt = prep_ligand(m); heavy = Chem.RemoveHs(m); perm = perm_for(txt, heavy)

def n1n3(mol):  # N3 is the quinazoline N bonded to the carbon that carries the aniline N; the other aromatic N is N1
    out = {}
    for at in mol.GetAtoms():
        if at.GetSymbol() == "N" and at.GetIsAromatic():
            c4 = [x for x in at.GetNeighbors() if any((y.GetSymbol() == "N" and not y.GetIsAromatic()) for y in x.GetNeighbors())]
            out["N3" if c4 else "N1"] = at.GetIdx()
    return out

rx = np.array([list(ref.GetConformer().GetAtomPosition(i)) for i in range(ref.GetNumAtoms())]); rn = n1n3(ref)
print("crystal: N1..Met769 N %.2f A | N3..Thr766 OG1 %.2f A | centroid %s" % (
    np.linalg.norm(rx[rn["N1"]] - MET_N), np.linalg.norm(rx[rn["N3"]] - THR_OG), np.round(rx.mean(0), 1).tolist()))
hn = n1n3(heavy)

def pose_xyz(text):
    for atoms in models(text):
        x = np.zeros((29, 3))
        for k, p in enumerate(atoms):
            x[perm[k]] = p
        yield x

def desc(x):
    return "N1..Met769N %.1f | N3..Thr766OG1 %.1f | centroid shift %.1f A | N1 shift %.1f A" % (
        np.linalg.norm(x[hn["N1"]] - MET_N), np.linalg.norm(x[hn["N3"]] - THR_OG),
        np.linalg.norm(x.mean(0) - rx.mean(0)), np.linalg.norm(x[hn["N1"]] - rx[rn["N1"]]))

def run(sf, box, ex, seed, tag):
    v = Vina(sf_name=sf, cpu=a.cpu, seed=seed, verbosity=0); v.set_receptor(rp); v.set_ligand_from_string(txt)
    v.compute_vina_maps(center=center, box_size=box); v.dock(exhaustiveness=ex, n_poses=9)
    p = work("diagnose", f"why_{tag}.pdbqt"); v.write_poses(str(p), n_poses=9, overwrite=True)
    return v, p, scorer.of_poses(p.read_text(), perm, heavy), v.energies(n_poses=9)

print("\n== Vina scoring, box as Metis wrote it, ex 32, seed 1 ==")
v, p, r, E = run("vina", size, 32, 1, "vina_s1"); xs = list(pose_xyz(p.read_text()))
for i in range(4):
    print(" pose %d score %.2f RMSD %.1f | %s | terms(total,inter,intra,torsion)=%s" % (i + 1, E[i][0], r[i], desc(xs[i]), np.round(E[i][:4], 2).tolist()))
j = int(np.argmin(r))
print(" near-crystal pose is #%d score %.2f RMSD %.1f | %s | terms=%s" % (j + 1, E[j][0], r[j], desc(xs[j]), np.round(E[j][:4], 2).tolist()))
for title, box, ex, sf, tagp in (("bigger box (35 A cube), ex 64", [35, 35, 35], 64, "vina", "big"),
                                 ("exhaustiveness 128, Metis box", size, 128, "vina", "ex128"),
                                 ("Vinardo scoring function (same package), Metis box, ex 32", size, 32, "vinardo", "vinardo")):
    print("\n== %s ==" % title)
    for seed in (1, 2, 3):
        v, p, r, E = run(sf, box, ex, seed, f"{tagp}_s{seed}"); j = int(np.argmin(r))
        print(" seed %d top score %.2f top RMSD %.1f | best RMSD %.1f at rank %d (score %.2f)" % (seed, E[0][0], r[0], r[j], j + 1, E[j][0]))
