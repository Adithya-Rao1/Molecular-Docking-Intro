"""Step 1. 
Redock erlotinib into 1M17 chain A with AutoDock Vina, using the dockprep (Metis) input files.

Exhaustiveness 8 and 32, seeds 1-5, 9 poses each. 
For each run we record the top score, the RMSD of the top-ranked pose to the crystal pose, and the best RMSD among the nine poses.

The default here is one thread per run with the ten (exhaustiveness, seed) runs executed in parallel processes (--jobs):

  python 01_dock.py                       # --cpu 1 --jobs 10. Takes about 7 minutes on 10 cores
  python 01_dock.py --cpu 30 --jobs 1     # The original setting. Fastest, but performance varies run to run from original blog post since multithreading isn't deterministic.

Output: work/vina_out/pose_ex{E}_s{S}.pdbqt and work/dock_results.json (same schema as data/results/dock_results.json).
"""
import argparse, json, subprocess, sys, time
from redock.common import INPUTS, WORK, RmsdScorer, metis_ligand, perm_for, read_box, work

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", type=int, default=1, help="Vina threads per run")
ap.add_argument("--jobs", type=int, default=10, help="runs executed in parallel (processes)")
ap.add_argument("--ex", type=int, nargs="+", default=[8, 32])
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
a = ap.parse_args()


def one(ex, seed, cpu):
    from vina import Vina
    center, size = read_box()
    lig = metis_ligand()
    perm = perm_for((INPUTS / "erlotinib.pdbqt").read_text(), lig)
    v = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
    v.set_receptor(str(INPUTS / "receptor.pdbqt"))
    v.set_ligand_from_file(str(INPUTS / "erlotinib.pdbqt"))
    v.compute_vina_maps(center=center, box_size=size)
    t = time.time()
    v.dock(exhaustiveness=ex, n_poses=9)
    dt = time.time() - t
    en = v.energies(n_poses=9)[:, 0].tolist()
    p = work("vina_out", f"pose_ex{ex}_s{seed}.pdbqt")
    v.write_poses(str(p), n_poses=9, overwrite=True)
    r = RmsdScorer().of_poses(p.read_text(), perm, lig)
    return dict(ex=ex, seed=seed, top_score=en[0], top_rmsd=r[0], best_rmsd=min(r), scores=en, rmsds=r, secs=round(dt, 1))


if a.single:
    assert len(a.ex) == 1 and len(a.seeds) == 1
    row = one(a.ex[0], a.seeds[0], a.cpu)
    work("dock_parts", f"ex{row['ex']}_s{row['seed']}.json").write_text(json.dumps(row))
    sys.exit(0)

combos = [(e, s) for e in a.ex for s in a.seeds]
if a.jobs <= 1:
    rows = [one(e, s, a.cpu) for e, s in combos]
else:
    procs, rows = [], []
    pending = list(combos)
    running = []
    while pending or running:
        while pending and len(running) < a.jobs:
            e, s = pending.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, "--single", "--ex", str(e), "--seeds", str(s), "--cpu", str(a.cpu)]))
        for p in list(running):
            if p.poll() is not None:
                if p.returncode:
                    sys.exit("a docking subprocess failed")
                running.remove(p)
        time.sleep(1)
    rows = [json.loads(work("dock_parts", f"ex{e}_s{s}.json").read_text()) for e, s in combos]
for r in rows:
    print(f"ex {r['ex']} seed {r['seed']}: top {r['top_score']:.2f} kcal/mol, RMSD(top) {r['top_rmsd']:.2f}, best-of-9 {r['best_rmsd']:.2f} ({r['secs']:.0f}s)", flush=True)
work("dock_results.json").write_text(json.dumps(rows, indent=1))
