# Reproduce the erlotinib / EGFR redocking study. See README.md.
# Three Python environments (environment/): docking (pip), rescoring (conda: OpenMM + AmberTools), pymol (conda).
PY_DOCK     ?= python
PY_RESCORE  ?= micromamba run -n rescoring python
PY_PYMOL    ?= micromamba run -n pymol pymol
PY_ANALYSIS ?= python
RESCORE     ?= micromamba run -n rescoring
JOBS        ?= 10
# Vina threads per run; 1 is deterministic (see README)
CPU         ?= 1
export PYTHONPATH := $(CURDIR)/src

.PHONY: logs diag3 diag4 diag5 verify figures dock diagnose openmm amber protein pymol all clean checksums

# Level 1 (seconds, numpy + matplotlib only): check the shipped results against the post, and redraw the two data figures.
verify:
	python verify.py
figures:
	$(PY_ANALYSIS) src/10_analyze_rescoring.py

# Level 2 (minutes on a 30-core machine): redock with Vina, rebuild the pose array, run the diagnostic experiments.
dock:
	$(PY_DOCK) src/01_dock.py --cpu $(CPU) --jobs $(JOBS)
	$(PY_DOCK) src/02_collect_poses.py
# Steps 3-5 are independent single-thread jobs: `make -j3 diagnose` runs them side by side.
diagnose: diag3 diag4 diag5
logs:
	mkdir -p work/logs
diag3: logs
	$(PY_DOCK) src/03_diagnose_top_pose.py --cpu $(CPU) > work/logs/diagnose_top_pose.log
diag4: logs
	$(PY_DOCK) src/04_crystal_score_and_waters.py --cpu $(CPU) > work/logs/crystal_score_and_waters.log
diag5: logs
	$(PY_DOCK) src/05_compare_standard_prep.py --cpu $(CPU) > work/logs/standard_prep_comparison.log

# Level 3 (hours): physics-based rescoring of all 46 poses. OpenMM needs a GPU for speed (OpenCL or CUDA); Amber is CPU.
openmm:
	$(PY_RESCORE) src/07_openmm_mmgbsa.py
amber:
	$(PY_RESCORE) src/08_amber_prepare.py
	$(PY_RESCORE) src/09_amber_rescore.py --workers 14

# Optional: regenerate the protein with PDBFixer (hydrogen positions differ from the published file; see step 6).
protein:
	$(PY_RESCORE) src/06_prepare_protein.py

# Figures 1-3 (PyMOL renders composed with matplotlib).
pymol:
	$(PY_RESCORE) src/11_make_pose_files.py
	cd work/pymol && $(PY_PYMOL) -cq ../../src/pymol/render_complex.py
	cd work/pymol && $(PY_ANALYSIS) ../../src/pymol/compose.py

all: dock diagnose openmm amber figures pymol verify
	python verify.py --compare
clean:
	rm -rf work
checksums:
	find data environment figures -type f | sort | xargs shasum -a 256 > SHA256SUMS
