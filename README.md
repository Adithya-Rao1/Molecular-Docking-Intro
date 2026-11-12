# Redocking erlotinib into EGFR (PDB 1M17)

## Three ways to use

| Level | Scripts | Time | Needs |
|---|---|---|---|
| 1. Verify numbers | `python verify.py` and `make figures` | seconds | Python with numpy, matplotlib |
| 2. Redock | `make dock` and `make -j3 diagnose` | `dock` about 4 minutes on 10 cores. `diagnose` runs three single-thread jobs in parallel, taking about 75 minutes | docking environment (Vina, Meeko, Open Babel, RDKit) |
| 3. Rescore | `make openmm amber` | hours (see timings) | rescoring environment (OpenMM, AmberTools), a GPU helps for OpenMM |
| Figures 1–3 | `make pymol` | minutes | pymol environment |

Every script writes to `work/`, so to compare your run with the blog post:
`python verify.py --compare`.

## Layout

```
data/1M17.pdb                 Stamos et al. 2002 (RCSB, 2.6 Å). Only chain A is usef; ligand AQ4 (A:999) is the reference pose.
data/inputs/                  Docking inputs: receptor.pdbqt, erlotinib.pdbqt/.sdf, vina.conf (box), co-crystal ligand
data/results/                 Reference results: dock_results.json, poses_xyz.npy + poses_meta.json (46 poses),
                              openmm_mmgbsa_results.json, amber/amber_results.json, standard_prep_comparison.json
logs/                         Printed output of steps 3-5 from one single-thread run (post's values), the OpenMM run, and logs/archival_session/ (older multithreaded output)
figures/                      The five figures used in the post
src/redock/common.py          Shared helpers (paths, PDBQT parsing, RMSD, receptor preparation)
src/01_dock.py ... 11_*.py    The pipeline, one script per step (below)
environment/                  Environments for each stage
verify.py, Makefile, SHA256SUMS
```

## The pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_dock.py` | Vina 1.2.7, exhaustiveness 8 and 32, seeds 1–5, nine poses, RMSD to the crystal pose |
| 2 | `02_collect_poses.py` | The 45 (exhaustiveness 32) poses + crystal pose (index 45) as one array |
| 3 | `03_diagnose_top_pose.py` | Hinge geometry and score terms of the wrong poses, 35 Å box, exhaustiveness 128, Vinardo |
| 4 | `04_crystal_score_and_waters.py` | Crystal pose scored in place (−6.12 kcal/mol; −6.95 kcal/mol after minimization), with and without waters |
| 5 | `05_compare_standard_prep.py` | dockprep vs Open Babel/Meeko preparation |
| 6 | `06_prepare_protein.py` | PDBFixer with chain A, pH 7.4 hydrogens, no loop building |
| 7 | `07_openmm_mmgbsa.py` | OpenMM MM-GBSA (ff14SB, OpenFF 2.1.0, OBC2) of all 46 poses |
| 8–9 | `08_amber_prepare.py`, `09_amber_rescore.py` | AmberTools using tleap, restrained minimization, MMPBSA.py (GB and PB) |
| 10 | `10_analyze_rescoring.py` | Score-vs-RMSD figure and the rank table |
| 11 | `11_make_pose_files.py`, `pymol/render_complex.py`, `pymol/compose.py` | PyMOL renders |

## Setting up the environments

```bash
# docking (steps 1-5): Python 3.12
python3.12 -m venv .venv-dock && .venv-dock/bin/pip install -r environment/docking.requirements.txt
# rescoring (steps 6-9, 11a): OpenMM 8.6.1, AmberTools 26, OpenFF, PDBFixer
micromamba create -f environment/rescoring.environment.yml
# PyMOL (step 11b)
micromamba create -f environment/pymol.environment.yml
# analysis and figure composition (steps 10, 11c)
pip install -r environment/analysis.requirements.txt
```

Use Makefile: 
`make dock PY_DOCK=.venv-dock/bin/python`. `make` uses `micromamba run -n rescoring` and `-n pymol` by default.

## Reproducing experiments details

- **System.** PDB 1M17 chain A with waters and the ligand removed where ligand AQ4 is erlotinib. Box center is (21.697, 0.303, 52.092) Å, with size (23.709, 12.684, 15.099) Å. This includes the co-crystal ligand plus 4 Å padding.
- **Docking.** AutoDock Vina 1.2.7, nine poses, seeds 1–5, one thread per run. RMSD is heavy-atom, symmetry-corrected (spyrmsd) against the crystal pose. According to standard convention, "native-like" means ≤ 2 Å.
- **Inputs.** `data/inputs/` holds the receptor and ligand PDBQT files that were used. The post's preparation step used the Metis docking-prep tool.
- **OpenMM MM-GBSA.** ff14SB protein (no waters), OpenFF 2.1.0 ligand with AM1-BCC charges, OBC2, and no cutoff. The protein backbone (N, CA, C) restrained at 10 kcal/mol/Å² is energy minimized with a single-trajectory protocol ΔE = E(complex) − E(receptor) − E(ligand). No entropy or MD in this experiment. OpenMM ran on the OpenCL platform.
- **Amber.** ff14SB + GAFF2 (AM1-BCC via antechamber), mbondi2 radii, 1000-step minimization with the same backbone restraint, `MMPBSA.py` single frame, 0.15 M salt, no entropy. Amber needs hydrogens stripped and re-added by tleap, and a `TER` at the chain gap between HID293 and LEU294 (done by `08_amber_prepare.py`).
- **Energy meaning.** The energy values (−33 to −60 kcal/mol, GB) are not binding free energies as they don't include any entropy terms or further molecular dynamics. Thus, use them to rank poses only.

## Reproducibility notes

- **Vina threads.** Every script defaults to one thread per run for reproducibility in Vina, and step 1 runs the ten (exhaustiveness, seed) jobs in parallel processes (`--jobs`).
- **Diagnostic numbers (steps 3–5).** `logs/diagnose_top_pose.log`, `crystal_score_and_waters.log` and `standard_prep_comparison.log` are the outputs of a single-thread (`--cpu 1`) run on an A10 GPU instance. We did not repeat steps 3–5 to confirm they are identical (step 1 was repeated and is) due to computation restraints. Thus, poses, ranks, and the intermolecular terms can differ in a rerun. Therefore, compare only the conclusions (top pose about 8 Å off, near-native pose among rank 9). `logs/archival_session/` holds older output from a multithreaded run.
- **Protein input.** PDBFixer hydrogen placement is not deterministic between runs, so the protein used for the published rescoring is shipped as `data/inputs/prot_fixed.pdb` and used by default. `06_prepare_protein.py` regenerates one for comparison so expect small energy differences.
- **Rescoring.** OpenMM and Amber energies are not fully reproducible across hardware or library builds. `verify.py --compare` allows 1.0 kcal/mol against the shipped values and reports what it found.
- **Data.** The poses in `data/results/` are the exhaustiveness 32 poses from step 1 mapped to the ligand's SDF atom order. The reference results were produced on Ubuntu 22.04, Intel Xeon Platinum 8358, Python 3.12 (docking) and 3.11 (rescoring). See `environment/` for more information.