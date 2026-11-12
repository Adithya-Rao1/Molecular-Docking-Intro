"""Step 7. 
OpenMM MM-GBSA rescoring of the 45 docked poses + the crystal pose (index 45).

Amber ff14SB protein, OpenFF 2.1.0 ligand with AM1-BCC charges, OBC2 implicit solvent, no cutoff. 
Each pose is minimized with the protein backbone (N, CA, C) restrained at 10 kcal/mol/A^2, then a single-trajectory protocol is used:
dE = E(complex) - E(receptor) - E(ligand). 
No entropy or MD so the scores are only for ranking and don't represent binding free energies.

  python 07_openmm_mmgbsa.py              # all 46 poses (about an hour on one A10 with the OpenCL platform)
  python 07_openmm_mmgbsa.py --only 0 45  # a quick subset (first Vina pose + the crystal pose)

Reads data/results/poses_{xyz.npy,meta.json} (or work/ if you regenerated them)
"""
import argparse, json, pathlib, sys, time
import numpy as np
import openmm as mm, openmm.app as app, openmm.unit as u
from openff.toolkit import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem
from redock.common import INPUTS, PROTEIN_FIXED, REF, WORK, load_poses, work

ap = argparse.ArgumentParser()
ap.add_argument("--only", type=int, nargs="*", help="pose indices to process (default: all 46)")
ap.add_argument("--poses", default=str(WORK if (WORK / "poses_xyz.npy").exists() else REF), help="directory with poses_xyz.npy/poses_meta.json")
ap.add_argument("--protein", default=str(PROTEIN_FIXED), help="protein PDB with hydrogens (default: the exact file used for the published numbers)")
args = ap.parse_args()
poses, meta = load_poses(args.poses)
prot_pdb = pathlib.Path(args.protein)
if not prot_pdb.exists():
    sys.exit(f"{prot_pdb} not found")
prot = app.PDBFile(str(prot_pdb))
# --- ligand
heavy=Chem.MolFromMolFile(str(INPUTS/"erlotinib.sdf"),removeHs=True)
lig_h=Chem.AddHs(heavy,addCoords=True)
off=Molecule.from_rdkit(lig_h,allow_undefined_stereo=True)
t=time.time(); off.assign_partial_charges("am1bcc"); print("AM1-BCC charges %.0fs"%(time.time()-t),flush=True)
gen=SMIRNOFFTemplateGenerator(molecules=[off],forcefield="openff-2.1.0")
ff=app.ForceField("amber14/protein.ff14SB.xml","implicit/obc2.xml"); ff.registerTemplateGenerator(gen.generator)
nH=lig_h.GetNumAtoms()-29
def build(x_heavy):
    """complex Modeller with ligand heavy atoms at x_heavy (A); ligand H placed by RDKit then relaxed in the minimisation."""
    m=Chem.Mol(lig_h); c=m.GetConformer()
    for i in range(29): c.SetAtomPosition(i,x_heavy[i].tolist())
    # put hydrogens back on the heavy atoms: embed H only by minimising with heavy atoms fixed (MMFF)
    from rdkit.Chem import AllChem
    props=AllChem.MMFFGetMoleculeProperties(m); f=AllChem.MMFFGetMoleculeForceField(m,props)
    for i in range(29): f.AddFixedPoint(i)
    f.Minimize(maxIts=500)
    pos=[mm.Vec3(*(np.array(c.GetAtomPosition(i))/10)) for i in range(m.GetNumAtoms())]
    ltop=off.to_topology().to_openmm()
    md=app.Modeller(prot.topology,prot.positions); md.add(ltop,pos*u.nanometer)
    return md
def system(top):
    return ff.createSystem(top,nonbondedMethod=app.NoCutoff,constraints=None)
def _pick():
    for name,pr in (("OpenCL",{"Precision":"mixed"}),("CPU",{})):
        try:
            p=mm.Platform.getPlatformByName(name); c=mm.Context(mm.System(),mm.VerletIntegrator(0.001),p,pr) if False else None
            sys0=mm.System(); sys0.addParticle(1.0); ctx0=mm.Context(sys0,mm.VerletIntegrator(0.001),p,pr); print("platform",name,flush=True); return p,pr
        except Exception as e: print("platform",name,"failed:",str(e)[:80],flush=True)
    raise SystemExit("no usable platform")
plat,props=_pick()
def energy_of(sysm,positions,top):
    integ=mm.VerletIntegrator(0.001); ctx=mm.Context(sysm,integ,plat,props); ctx.setPositions(positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
KC=4.184
res=[]
for n,(x,mt) in enumerate(zip(poses,meta)):
    if args.only is not None and n not in args.only: continue
    mt=dict(mt,index=n)
    md=build(x); top=md.topology; N=top.getNumAtoms(); nl=lig_h.GetNumAtoms(); lig_idx=list(range(N-nl,N)); prot_idx=list(range(N-nl))
    s_free=system(top)
    # restrained minimisation: protein backbone N,CA,C held (k=10 kcal/mol/A^2); everything else free
    s_res=system(top); rest=mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"); rest.addGlobalParameter("k",10*KC*100); 
    for p in ("x0","y0","z0"): rest.addPerParticleParameter(p)
    pos0=md.positions
    for a in top.atoms():
        if a.index<N-nl and a.name in ("N","CA","C"): rest.addParticle(a.index,[pos0[a.index].x,pos0[a.index].y,pos0[a.index].z])
    s_res.addForce(rest)
    integ=mm.VerletIntegrator(0.001); ctx=mm.Context(s_res,integ,plat,props); ctx.setPositions(pos0)
    mm.LocalEnergyMinimizer.minimize(ctx,tolerance=1.0,maxIterations=5000)
    xyz=ctx.getState(getPositions=True).getPositions(asNumpy=True)
    # single-trajectory MM-GBSA from the minimised complex: E_complex - E_receptor - E_ligand (all with OBC2 implicit solvent)
    Ec=energy_of(s_free,xyz,top)
    def sub(idx):
        mdl=app.Modeller(top,xyz); mdl.delete([a for a in top.atoms() if a.index not in idx]); return mdl
    mr=sub(set(prot_idx)); ml=sub(set(lig_idx))
    Er=energy_of(system(mr.topology),mr.positions,mr.topology); El=energy_of(system(ml.topology),ml.positions,ml.topology)
    # ligand strain-free reference: also relax the ligand alone
    integ2=mm.VerletIntegrator(0.001); sl=system(ml.topology); c2=mm.Context(sl,integ2,plat,props); c2.setPositions(ml.positions)
    mm.LocalEnergyMinimizer.minimize(c2,tolerance=1.0,maxIterations=5000); El_relaxed=c2.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
    dE=Ec-Er-El
    lx=np.array(xyz.value_in_unit(u.angstrom))[N-nl:N-nl+29]; shift=float(np.sqrt(((lx-x)**2).sum(1).mean()))
    res.append(dict(**mt,dE_gbsa=dE,dE_gbsa_strain=Ec-Er-El_relaxed,lig_moved_A=shift))
    print("%2d/%d crystal=%s seed %s rank %s vina %s RMSD %.1f | MM-GBSA dE %.1f (strain-corrected %.1f) | ligand moved %.2f A"%(n+1,len(poses),mt.get("crystal",False),mt["seed"],mt["rank"],("%.2f"%mt["vina"]) if mt["vina"] is not None else "-",mt["rmsd"],dE,Ec-Er-El_relaxed,shift),flush=True)
json.dump(res,open(work("openmm_mmgbsa_results.json"),"w"),indent=1)
