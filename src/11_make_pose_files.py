"""Step 11a. 
Files for the PyMOL figures (work/pymol/): chain A of 1M17 (protein_A.pdb) and three ligand poses as SDF:
1.) crystal pose (index 45)
2.) Vina's top-ranked pose (index 0, 7.9 A RMSD) 
3.) near-native pose (index 38, 1.2 A, Vina rank 3)
"""
import numpy as np
from rdkit import Chem
from redock.common import PDB_1M17, REF, WORK, load_poses, metis_ligand, work

poses, meta = load_poses(WORK if (WORK / "poses_xyz.npy").exists() else REF)
work("pymol", "protein_A.pdb").write_text("".join(l for l in PDB_1M17.read_text().splitlines(True) if l.startswith("ATOM") and l[21] == "A") + "END\n")
for name, i in (("crystal", 45), ("vina_top", 0), ("near_native", 38)):
    m = Chem.Mol(metis_ligand()); m.RemoveAllConformers(); c = Chem.Conformer(29)
    for k in range(29):
        c.SetAtomPosition(k, poses[i][k].tolist())
    m.AddConformer(c); m.SetProp("_Name", name)
    w = Chem.SDWriter(str(work("pymol", f"{name}.sdf"))); w.write(m); w.close()
    print(name, "pose index", i, "meta", {k: meta[i][k] for k in ("seed", "rank", "rmsd")})
