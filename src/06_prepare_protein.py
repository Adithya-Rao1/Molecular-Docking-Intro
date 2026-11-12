"""Step 6. 
Protein for rescoring: chain A of 1M17, no waters/ligand, missing heavy atoms added, hydrogens added at pH 7.4, no loop building (the 293/294 chain gap is untouched). 

OPTIONAL: The published rescoring used the file shipped as data/inputs/prot_fixed.pdb, which is the default input of steps 7-8.
PDBFixer's hydrogen placement has slight variance as regenerating moves every hydrogen by up to ~2 A while heavy atoms are identical
Thus, a regenerated file gives slightly different OpenMM energies. To reproduce blog results, pass it explicitly: 
`07_openmm_mmgbsa.py --protein work/protein/prot_fixed.pdb`.
"""
from openmm import app
from pdbfixer import PDBFixer
from redock.common import PDB_1M17, work

raw = [l for l in PDB_1M17.read_text().splitlines() if l.startswith("ATOM") and l[21] == "A"]
work("protein", "prot_in.pdb").write_text("\n".join(raw) + "\nEND\n")
fx = PDBFixer(str(work("protein", "prot_in.pdb")))
fx.findMissingResidues(); fx.missingResidues = {}
fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.4)
app.PDBFile.writeFile(fx.topology, fx.positions, open(work("protein", "prot_fixed.pdb"), "w"))
print("wrote", work("protein", "prot_fixed.pdb"))
