#!/usr/bin/env python3
"""
Check the numbers quoted in the blog post, and (optionally) compare a fresh re-run against the shipped results.

  python verify.py              # the shipped reference results reproduce every number in the post (seconds, numpy only)
  python verify.py --compare    # also compare whatever you regenerated in work/ with the reference results
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
REF, WORK = ROOT / "data" / "results", ROOT / "work"
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def ranks(meta, scores):
    """Rank (1 = best score) of the best native-like (RMSD <= 2.0 A) pose among each seed's nine poses."""
    n = 45
    rm = np.array([m["rmsd"] for m in meta[:n]])
    seed = np.array([m["seed"] for m in meta[:n]])
    y = np.asarray(scores[:n])
    out = []
    for s in range(1, 6):
        idx = np.where(seed == s)[0]
        order = idx[np.argsort(y[idx])]
        out.append(next(p + 1 for p, i in enumerate(order) if rm[i] <= 2.0))
    return out


def load(d):
    meta = json.load(open(d / "poses_meta.json"))
    gb = json.load(open(d / "openmm_mmgbsa_results.json"))
    am_path = d / "amber" / "amber_results.json"
    am = json.load(open(am_path if am_path.exists() else d / "amber" / "results_all.json"))
    return meta, gb, am


def claims(d, label):
    print(f"\n== claims in the post vs {label} ==")
    rows = json.load(open(d / "dock_results.json")) if (d / "dock_results.json").exists() else None
    if rows:
        top = [r["top_rmsd"] for r in rows]
        best = [r["best_rmsd"] for r in rows]
        check("top-ranked pose 7.6-8.7 A from the crystal pose in 10/10 runs", len(rows) == 10 and 7.55 <= min(top) and max(top) <= 8.75, f"{min(top):.1f}-{max(top):.1f}")
        check("best of the nine poses 0.9-2.1 A in every run", 0.85 <= min(best) and max(best) <= 2.15, f"{min(best):.1f}-{max(best):.1f}")
    meta, gb, am = load(d)
    check("45 poses + crystal pose", len(meta) == 46 and meta[45].get("crystal") is True)
    exp = {"Vina": [7, 5, 7, 4, 3], "OpenMM MM-GBSA": [1, 2, 2, 1, 1], "Amber MM-GBSA": [3, 3, 3, 1, 1], "Amber MM-PBSA": [1, 2, 2, 1, 1]}
    got = {
        "Vina": ranks(meta, [m["vina"] for m in meta]),
        "OpenMM MM-GBSA": ranks(meta, [g["dE_gbsa"] for g in gb]),
        "Amber MM-GBSA": ranks(meta, [a["amber"]["gb"] for a in am]),
        "Amber MM-PBSA": ranks(meta, [a["amber"]["pb"] for a in am]),
    }
    for k in exp:
        check(f"rank table, {k}", got[k] == exp[k], f"{got[k]} vs {exp[k]}")
    check("crystal pose: OpenMM dE ~ -49.4", abs(gb[45]["dE_gbsa"] + 49.4) < 0.1, f"{gb[45]['dE_gbsa']:.1f}")
    check("crystal pose: Amber GB ~ -40.6, PB ~ -1.4", abs(am[45]["amber"]["gb"] + 40.6) < 0.1 and abs(am[45]["amber"]["pb"] + 1.4) < 0.1, f"{am[45]['amber']['gb']:.1f}, {am[45]['amber']['pb']:.1f}")
    check("Vina top pose, seed 1: Amber GB -35.5, PB +9.4", abs(am[0]["amber"]["gb"] + 35.5) < 0.1 and abs(am[0]["amber"]["pb"] - 9.4) < 0.1)
    seed = np.array([m["seed"] for m in meta[:45]])
    pb_top = []
    for s in range(1, 6):
        idx = np.where(seed == s)[0]
        pb_top.append(meta[int(idx[np.argmin([am[i]["amber"]["pb"] for i in idx])])]["rmsd"])
    check(
        "MM-PBSA rank-1 pose is native-like in 3 of 5 seeds; the others are 4.7-4.9 A",
        sum(r <= 2.0 for r in pb_top) == 3 and all(4.6 <= r <= 5.0 for r in pb_top if r > 2.0),
        str(np.round(pb_top, 1).tolist()),
    )


def compare():
    print("\n== regenerated results (work/) vs reference ==")
    if (WORK / "dock_results.json").exists():
        a = json.load(open(WORK / "dock_results.json"))
        b = json.load(open(REF / "dock_results.json"))
        bm = {(r["ex"], r["seed"]): r for r in b}
        same = [np.allclose(r["scores"], bm[(r["ex"], r["seed"])]["scores"], atol=0.01) for r in a if (r["ex"], r["seed"]) in bm]
        check("Vina scores identical to reference (single-thread run; see README)", all(same), f"{sum(same)}/{len(same)} runs")
        check("top-pose RMSD within 0.1 A of reference", all(abs(r["top_rmsd"] - bm[(r["ex"], r["seed"])]["top_rmsd"]) < 0.1 for r in a if (r["ex"], r["seed"]) in bm))
    else:
        print("skip  work/dock_results.json not found")
    if (WORK / "standard_prep_comparison.json").exists():
        c = json.load(open(WORK / "standard_prep_comparison.json"))
        for name, rows in c.items():
            top = np.array([r[1] for r in rows])
            best = np.array([r[2] for r in rows])
            check(
                f"standard prep, {name}: top pose ~8 A off in most seeds, near-native pose found",
                np.median(top) >= 7.5 and np.median(best) <= 2.0 and (top < 2).sum() <= 1,
                f"top-RMSD median {np.median(top):.1f}, best-of-9 median {np.median(best):.1f}, seeds with top < 2 A: {(top < 2).sum()}/5",
            )
    else:
        print("skip  work/standard_prep_comparison.json not found")
    if (WORK / "openmm_mmgbsa_results.json").exists():
        a = json.load(open(WORK / "openmm_mmgbsa_results.json"))
        b = json.load(open(REF / "openmm_mmgbsa_results.json"))
        for r in a:
            i = r["index"]
            check(f"OpenMM dE, pose {i} within 1.0 kcal/mol of reference", abs(r["dE_gbsa"] - b[i]["dE_gbsa"]) < 1.0, f"{r['dE_gbsa']:.2f} vs {b[i]['dE_gbsa']:.2f}")
    else:
        print("skip  work/openmm_mmgbsa_results.json not found")
    parts = sorted((WORK / "amber").glob("results_*.json")) if (WORK / "amber").exists() else []
    if parts:
        b = json.load(open(REF / "amber" / "amber_results.json"))
        for p in parts:
            for r in json.load(open(p)):
                i = r["index"]
                good = bool(r["amber"]) and abs(r["amber"]["gb"] - b[i]["amber"]["gb"]) < 1.0 and abs(r["amber"]["pb"] - b[i]["amber"]["pb"]) < 1.0
                check(f"Amber GB/PB, pose {i} within 1.0 kcal/mol of reference", good, f"{r['amber']} vs {b[i]['amber']}")
    else:
        print("skip  no work/amber/results_*.json found")
    if (WORK / "rank_table.json").exists():
        t = json.load(open(WORK / "rank_table.json"))
        want = {"AutoDock Vina score": [7, 5, 7, 4, 3], "OpenMM MM-GBSA": [1, 2, 2, 1, 1], "Amber MM-GBSA": [3, 3, 3, 1, 1], "Amber MM-PBSA": [1, 2, 2, 1, 1]}
        check("rank table from regenerated data matches the post", t == want)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    claims(REF, "shipped reference results")
    if a.compare:
        compare()
    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)
