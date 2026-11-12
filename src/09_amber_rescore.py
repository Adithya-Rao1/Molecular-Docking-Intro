"""Step 9. 
Amber rescoring of every pose. tleap -> restrained sander minimization -> MMPBSA.py (GB and PB).

Per pose (directory work/amber/pNN): 
1.) ff14SB protein + GAFF2/AM1-BCC ligand, mbondi2 radii; 1000-step minimization with the protein backbone (N, CA, C) restrained at 10 kcal/mol/A^2 (igb=5)
2.) single-frame MMPBSA.py (GB: igb=5, 0.15 M salt; PB: istrng=150). Single trajectory, no entropy. About 12 minutes per pose on one core.

  python 09_amber_rescore.py                 # all 46 poses, --workers processes in parallel
  python 09_amber_rescore.py 0 45            # a subset (pose indices)

Needs 08_amber_prepare.py first.
"""
import argparse, json, pathlib, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from rdkit import Chem
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from redock.common import INPUTS, REF, WORK, load_poses, work
import importlib
ligand_h = importlib.import_module("08_amber_prepare").ligand_h
A = work("amber", "x").parent
poses, meta = load_poses(WORK if (WORK / "poses_xyz.npy").exists() else REF)
heavy = Chem.MolFromMolFile(str(INPUTS / "erlotinib.sdf"), removeHs=True)
def sh(cmd,cwd):
    r=subprocess.run(cmd,shell=True,cwd=cwd,capture_output=True,text=True)
    if r.returncode: raise RuntimeError(cmd+"\n"+r.stdout[-800:]+r.stderr[-800:])
    return r.stdout
def parse(dat):
    t=pathlib.Path(dat).read_text(); out={}
    for name,key in (("GENERALIZED BORN:","gb"),("POISSON BOLTZMANN:","pb")):
        i=t.find(name); j=t.find("DELTA TOTAL",i); 
        if i>=0 and j>=0: out[key]=float(t[j:].split()[2])
    return out
def run(i):
    d=A/f"p{i:02d}"; d.mkdir(exist_ok=True)
    try:
        if (d/"FINAL.dat").exists(): return i,parse(d/"FINAL.dat"),None
        m=ligand_h(heavy,poses[i]); conf=m.GetConformer(); lines=(A/"lig_ref.mol2").read_text().splitlines(); out=[]; sect=None; k=0
        for l in lines:
            if l.startswith("@<TRIPOS>"): sect=l
            if sect=="@<TRIPOS>ATOM" and not l.startswith("@") and l.strip():
                p=l.split(); x,y,z=conf.GetAtomPosition(k); k+=1
                l="%7s %-8s %9.4f %9.4f %9.4f %-5s %3s %-5s %9s"%(p[0],p[1],x,y,z,p[5],p[6],p[7],p[8])
            out.append(l)
        assert k==m.GetNumAtoms(),(k,m.GetNumAtoms())
        (d/"pose.mol2").write_text("\n".join(out)+"\n")
        (d/"leap.in").write_text("source leaprc.protein.ff14SB\nsource leaprc.gaff2\nset default PBRadii mbondi2\nloadamberparams ../lig.frcmod\nLIG = loadmol2 pose.mol2\nREC = loadpdb ../prot_noH.pdb\nCOM = combine {REC LIG}\nsaveamberparm LIG lig.prmtop lig.rst7\nsaveamberparm REC rec.prmtop rec.rst7\nsaveamberparm COM com.prmtop com.rst7\nquit\n")
        sh("tleap -f leap.in > leap.log 2>&1",d)
        (d/"min.in").write_text("min\n &cntrl\n  imin=1, maxcyc=1000, ncyc=500, ntb=0, igb=5, cut=999., ntr=1, restraint_wt=10.0,\n  restraintmask='(@N,CA,C)', ntpr=1000,\n /\n")
        sh("sander -O -i min.in -o min.out -p com.prmtop -c com.rst7 -r com.min.rst7 -ref com.rst7",d)
        (d/"cpptraj.in").write_text("parm com.prmtop\ntrajin com.min.rst7\ntrajout com.min.nc netcdf\nrun\nquit\n")
        sh("cpptraj -i cpptraj.in > cpptraj.log 2>&1",d)
        (d/"mmpbsa.in").write_text("&general\n startframe=1, endframe=1, interval=1, verbose=1,\n/\n&gb\n igb=5, saltcon=0.150,\n/\n&pb\n istrng=150, inp=2, radiopt=0, fillratio=4.0,\n/\n")
        sh("MMPBSA.py -O -i mmpbsa.in -o FINAL.dat -sp com.prmtop -cp com.prmtop -rp rec.prmtop -lp lig.prmtop -y com.min.nc > mmpbsa.log 2>&1",d)
        return i,parse(d/"FINAL.dat"),None
    except Exception as e: return i,None,str(e)[:1200]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("idx", type=int, nargs="*"); ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    idx = a.idx or list(range(len(poses)))
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r, err in ex.map(run, idx):
            print(i, meta[i].get("seed"), meta[i].get("rank"), "vina", meta[i].get("vina"), "rmsd", meta[i]["rmsd"], "|", r if r else "ERROR " + str(err), flush=True)
            res.append(dict(index=i, **meta[i], amber=r, error=err))
    json.dump(res, open(A / ("results_all.json" if len(idx) == len(poses) else "results_i%d.json" % idx[0]), "w"), indent=1)
