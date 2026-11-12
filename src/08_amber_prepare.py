"""Step 8. 
Amber inputs shared by all poses (run inside the `rescoring` conda env: AmberTools).

  * ligand: hydrogens placed on the crystal pose (heavy atoms fixed, MMFF), GAFF2 atom types, AM1-BCC charges
    (antechamber), missing parameters (parmchk2)  ->  work/amber/lig_ref.mol2, lig.frcmod
  * protein: pdb4amber on the step-6 PDBFixer output (waters dropped), hydrogens removed (tleap re-adds them), and a TER
    inserted at the chain gap (HID293 -> LEU294, 4.6 A). tleap otherwise bonds across the gap  ->  work/amber/prot_noH.pdb
"""
import pathlib, subprocess, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from redock.common import INPUTS, PROTEIN_FIXED, REF, WORK, load_poses, work

A = work("amber", "x").parent
def sh(cmd):
    r = subprocess.run(cmd, shell=True, cwd=A, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{cmd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")

def ligand_h(heavy, x):
    """Add hydrogens to the heavy-atom ligand at coordinates x (MMFF-minimised, heavy atoms held fixed)."""
    m = Chem.AddHs(heavy, addCoords=True); c = m.GetConformer()
    for i in range(29):
        c.SetAtomPosition(i, x[i].tolist())
    f = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
    for i in range(29):
        f.AddFixedPoint(i)
    f.Minimize(maxIts=500)
    return m

if __name__ == "__main__":
    poses, _ = load_poses(WORK if (WORK / "poses_xyz.npy").exists() else REF)
    heavy = Chem.MolFromMolFile(str(INPUTS / "erlotinib.sdf"), removeHs=True)
    w = Chem.SDWriter(str(A / "lig_ref.sdf")); w.write(ligand_h(heavy, poses[-1])); w.close()
    sh("antechamber -i lig_ref.sdf -fi sdf -o lig_ref.mol2 -fo mol2 -c bcc -nc 0 -at gaff2 -rn LIG -pf y -dr no > ante.log 2>&1")
    sh("parmchk2 -i lig_ref.mol2 -f mol2 -o lig.frcmod -s gaff2")
    fixed = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else PROTEIN_FIXED
    if not fixed.exists():
        sys.exit(f"{fixed} not found")
    (A / "prot_fixed.pdb").write_text(fixed.read_text())
    sh("pdb4amber -i prot_fixed.pdb -o prot_amber.pdb --dry > pdb4amber.log 2>&1")
    # drop hydrogens; TER wherever consecutive residues are not bonded (C..N > 2 A)
    lines = [l for l in (A / "prot_amber.pdb").read_text().splitlines() if l.startswith("ATOM")]
    heavy_lines = [l for l in lines if l[76:78].strip() != "H"]
    out, prevC = [], None
    for l in heavy_lines:
        name, xyz = l[12:16].strip(), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        if name == "N" and prevC is not None and np.linalg.norm(xyz - prevC) > 2.0:
            out.append("TER")
        if name == "C":
            prevC = xyz
        out.append(l)
    out.append("TER"); out.append("END")
    (A / "prot_noH.pdb").write_text("\n".join(out) + "\n")
    print("prepared", A, "| TER lines:", out.count("TER"))
