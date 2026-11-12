"""
Shared helpers for the erlotinib / EGFR (PDB 1M17) redocking study.

Inputs are read from data/, and every script writes its own output to work/
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
REF = DATA / "results"  # reference results shipped with the package
WORK = ROOT / "work"  # everything the scripts regenerate

PDB_1M17 = DATA / "1M17.pdb"
PROTEIN_FIXED = INPUTS / "prot_fixed.pdb"  # PDBFixer output used for the published rescoring (see 06_prepare_protein.py)
ERLOTINIB_SMILES = "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC"
N_HEAVY = 29
CPU = 1  # Vina threads per run: one thread is deterministic; several threads can return different poses (see README)


def work(*parts) -> pathlib.Path:
    p = WORK.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_box(conf=INPUTS / "vina.conf"):
    """Search box (centre, size) in Angstrom from the Vina config."""
    cfg = {}
    for line in pathlib.Path(conf).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    center = [float(cfg[f"center_{a}"]) for a in "xyz"]
    size = [float(cfg[f"size_{a}"]) for a in "xyz"]
    return center, size


def template():
    return Chem.MolFromSmiles(ERLOTINIB_SMILES)


def crystal_ligand():
    """Co-crystal erlotinib (AQ4, chain A residue 999) with bond orders assigned from the SMILES."""
    pdb = Chem.MolFromPDBFile(str(INPUTS / "cocrystal_AQ4_A999.pdb"), removeHs=True)
    return AllChem.AssignBondOrdersFromTemplate(template(), pdb)


def metis_ligand():
    """Heavy-atom erlotinib from the dockprep SDF (29 atoms), in the atom order used for pose arrays."""
    lig = Chem.MolFromMolFile(str(INPUTS / "erlotinib.sdf"), removeHs=True)
    assert lig.GetNumAtoms() == N_HEAVY, lig.GetNumAtoms()
    return lig


def prep_ligand(mol_h) -> str:
    """Meeko PDBQT string for a hydrogenated RDKit molecule."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setup = MoleculePreparation().prepare(mol_h)[0]
    text, ok, err = PDBQTWriterLegacy.write_string(setup)
    assert ok, err
    return text


def embedded_ligand(seed=7):
    """ETKDG + MMFF erlotinib used for the 'standard preparation' (Meeko) ligand."""
    m = Chem.AddHs(template())
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    return m


def models(text):
    """Yield, for each MODEL in a PDBQT string, the heavy-atom coordinates in file order."""
    cur = None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith("ENDMDL"):
            yield cur
            cur = None
        elif cur is not None and line[:6] in ("ATOM  ", "HETATM"):
            if line.split()[-1] in ("H", "HD"):
                continue
            cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def perm_for(pdbqt_text, heavy):
    """Map PDBQT heavy-atom order -> atom order of `heavy` by matching input coordinates."""
    first = next(models("MODEL 1\n" + pdbqt_text + "\nENDMDL\n"))
    conf = heavy.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(heavy.GetNumAtoms())])
    perm = []
    for p in first:
        d = np.linalg.norm(xyz - np.array(p), axis=1)
        j = int(d.argmin())
        assert d[j] < 0.02, d[j]
        perm.append(j)
    assert sorted(perm) == list(range(len(perm))), "mapping is not a permutation"
    return perm


class RmsdScorer:
    """Symmetry-corrected heavy-atom RMSD to the crystal pose (spyrmsd)."""

    def __init__(self, ref_mol=None):
        from spyrmsd import molecule as sm

        self._sm = sm
        self.ref = sm.Molecule.from_rdkit(ref_mol if ref_mol is not None else crystal_ligand())

    def of_poses(self, text, perm, heavy):
        from spyrmsd import rmsd as sr

        out = []
        for atoms in models(text):
            xyz = np.zeros((len(perm), 3))
            for k, p in enumerate(atoms):
                xyz[perm[k]] = p
            mb = Chem.Mol(heavy)
            mb.RemoveAllConformers()
            c = Chem.Conformer(len(perm))
            for i in range(len(perm)):
                c.SetAtomPosition(i, xyz[i].tolist())
            mb.AddConformer(c)
            rb = self._sm.Molecule.from_rdkit(mb)
            out.append(
                float(
                    sr.symmrmsd(
                        self.ref.coordinates,
                        rb.coordinates,
                        self.ref.atomicnums,
                        rb.atomicnums,
                        self.ref.adjacency_matrix,
                        rb.adjacency_matrix,
                    )
                )
            )
        return out


def receptor_pdbqt(name, keep_water=False, pdb=PDB_1M17):
    """
    Standard receptor preparation: Open Babel, chain A, hydrogens at pH 7.4, rigid (-xr).

    Waters of chain A are kept only if `keep_water`. The ligand and other HETATM records are dropped.
    """
    import shutil
    import subprocess
    import sys

    raw = pathlib.Path(pdb).read_text().splitlines()
    lines = [
        l
        for l in raw
        if (l.startswith("ATOM") and l[21] == "A")
        or (keep_water and l.startswith("HETATM") and l[17:20] == "HOH" and l[21] == "A")
    ]
    src, dst = work("receptors", f"{name}.pdb"), work("receptors", f"{name}.pdbqt")
    src.write_text("\n".join(lines) + "\nEND\n")
    obabel = shutil.which("obabel") or str(pathlib.Path(sys.executable).parent / "obabel")
    subprocess.run(
        [obabel, "-ipdb", str(src), "-opdbqt", "-O", str(dst), "-xr", "-h", "-p", "7.4"],
        check=True,
        capture_output=True,
    )
    return str(dst)


def atom_xyz(resname, resseq, name, pdb=PDB_1M17):
    for l in pathlib.Path(pdb).read_text().splitlines():
        if (
            l.startswith("ATOM")
            and l[21] == "A"
            and l[17:20] == resname
            and int(l[22:26]) == resseq
            and l[12:16].strip() == name
        ):
            return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    raise KeyError((resname, resseq, name))


def load_poses(directory=REF):
    """45 docked poses (5 seeds x 9, exhaustiveness 32) + the crystal pose last (index 45)."""
    directory = pathlib.Path(directory)
    return np.load(directory / "poses_xyz.npy"), json.load(open(directory / "poses_meta.json"))
