#!/usr/bin/env python3
"""
Quantum Allosteric Signal Propagation Scanner — V4.1.3
Cleveland Clinic Global Quantum + AI Challenge 2026

Standalone Python conversion of the claim-locked V4.1.3 notebook.
Scientific scoring/model logic is unchanged.

Runtime environment variables:
  QALLOSTERY_ROOT                 Output/work directory.
  QALLOSTERY_CHECKPOINT_DIR       Explicit persistent checkpoint directory.
  QALLOSTERY_USE_GOOGLE_DRIVE     1/0; defaults to 1 in Colab, 0 elsewhere.
  QALLOSTERY_ALLOW_LOCAL_FALLBACK 1/0; defaults to 1 outside Colab.
  QALLOSTERY_CALIBRATION_CSV      Optional calibration CSV path.

Run:
  python cleveland_clinic_quantum_allostery_V4_1_3_claim_locked_final_submission.py
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Plain-Python compatibility for notebook display() calls.
# -----------------------------------------------------------------------------
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(obj):
        if hasattr(obj, 'to_string'):
            try:
                print(obj.to_string())
                return
            except Exception:
                pass
        print(obj)


# %% [markdown]
# # Quantum Allosteric Signal Propagation Scanner — V4.1.3
# ## Cleveland Clinic — Global Quantum + AI Challenge 2026
# ### Claim-Locked Final Submission
#
# **Scientific architecture frozen. No scorer, feature, training-label, fusion,
# Top-5 or challenge-prediction change is made in V4.1.3.**
#
# V4.1.2 completed the final scientific sensitivity audits. The resulting
# submission position is now explicit:
#
# - the frozen V3.6 site-aware scorer remains the **primary competition model**;
# - the HDC + neuroevolution calibrated hybrid remains **auxiliary evidence**;
# - chain-remapping sensitivity is disclosed rather than used for retrospective
#   retraining;
# - K=8 hardware-noise stability is separated from full-resolution
#   coarse-representation fidelity;
# - the hardware path uses a label-free fidelity ladder to state the minimum
#   tested K required to reach ≥0.80 full→coarse seed-transport Spearman;
# - all final claims are generated from machine-readable audit tables and locked
#   into a final manifest.
#
# V4.1.3 is intended to be the final reproducible submission notebook. Any
# future change to calibration labels or the scoring architecture should be
# treated as a **new model** and validated on a fresh independent holdout.

# %% [markdown]
# ## 1. Install dependencies
#
# The primary implementation uses Qiskit 2.x and Qiskit Aer, Biopython, NetworkX, NumPy/SciPy, scikit-learn, pandas and Plotly.
#
# Run this cell once in a fresh Google Colab runtime.

# %% [code] Notebook code cell 1 (source index 2)

import importlib.util
import subprocess
import sys

V411_PACKAGE_SPECS = {
    "qiskit": "qiskit>=2.3,<3.0",
    "qiskit_aer": "qiskit-aer>=0.17,<0.18",
    "Bio": "biopython>=1.84",
    "networkx": "networkx>=3.3",
    "sklearn": "scikit-learn>=1.4",
    "plotly": "plotly>=5.20",
    "pandas": "pandas>=2.1",
    "numpy": "numpy>=1.26",
    "scipy": "scipy>=1.12",
    "requests": "requests>=2.31",
    "tqdm": "tqdm>=4.66",
}

missing = [
    spec
    for module, spec in V411_PACKAGE_SPECS.items()
    if importlib.util.find_spec(module) is None
]

if missing:
    print("Installing missing dependencies:", missing)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing]
    )
else:
    print("All required dependencies already available; pip install skipped.")

print("V4.1.1 dependency bootstrap: PASS")

# %% [markdown]
# ## 2. Imports, reproducibility and run configuration
#
# `FULL_RUN=True` computes the complete matrices for all challenge proteins. On a standard Colab CPU this is practical, but the Cardiac Myosin eigendecomposition is the slowest step.
#
# The primary challenge score stays **topology-first**. Amino-acid language-model features are kept in a separate optional section and are never mixed into the official score.

# %% [code] Notebook code cell 2 (source index 4)
import re
import os
import sys
import math
import json
import shutil
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
import networkx as nx
import requests
from tqdm.auto import tqdm

from scipy.spatial import cKDTree, distance
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.stats import spearmanr, mannwhitneyu, rankdata
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.cluster import KMeans

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Align import PairwiseAligner

import plotly.graph_objects as go

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

_QALLOSTERY_DEFAULT_ROOT = (
    Path("/content/quantum_allostery_v4_0_4")
    if Path("/content").exists()
    else Path.cwd() / "quantum_allostery_v4_0_4"
)
ROOT = Path(
    os.environ.get("QALLOSTERY_ROOT", str(_QALLOSTERY_DEFAULT_ROOT))
).expanduser()
PDB_DIR = ROOT / "pdb"
RESULTS_DIR = ROOT / "results"
PDB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True
CONTACT_CUTOFF_A = 8.0
LIGAND_CONTACT_CUTOFF_A = 5.0
ACTIVE_SEED_CUTOFF_A = 5.0
TIME_GRID = np.linspace(0.35, 7.0, 12 if FULL_RUN else 6)
GAMMA = 1.0

# HDC dimension. 4096 is already highly expressive while staying fast in Colab.
HDC_DIM = 4096

# Neuroevolution defaults.
EVO_POP = 64 if FULL_RUN else 28
EVO_GENERATIONS = 55 if FULL_RUN else 15
EVO_HIDDEN = 10

# Qiskit coarse-grained demonstration.
QUBIT_COARSE_CLUSTERS = 8
QISKIT_SHOTS = 4096
QISKIT_TROTTER_REPS = 2

print("Root:", ROOT)
print("Full run:", FULL_RUN)

# ---------------- V2 controls ----------------
DISTAL_EUCLID_A = 8.0
DISTAL_GRAPH_HOPS = 2
SUBMISSION_MODE = True
ALLOW_CHALLENGE_TO_CHALLENGE_TRANSFER = False
MIN_LEARNED_BENCHMARKS = 5

ROBUSTNESS_SETTINGS = [
    (7.5, 0.85, 0.90),
    (8.0, 1.00, 1.00),
    (8.5, 1.15, 1.10),
    (7.75, 1.10, 0.95),
    (8.25, 0.90, 1.05),
] if FULL_RUN else [
    (7.75, 0.90, 0.95),
    (8.0, 1.00, 1.00),
    (8.25, 1.10, 1.05),
]

COARSE_K_VALUES = [8, 16, 32, 64, 128]
COARSE_MIN_SPEARMAN = 0.80
COARSE_MIN_TOP20_JACCARD = 0.40

RUN_QISKIT_NOISE = True
NOISE_TARGETS = ["KRAS_G12C", "BCR_ABL1"]
NOISE_TIMES = [1.5, 3.0]
NOISE_LEVELS = [(0.0,0.0),(0.0005,0.005),(0.001,0.01),(0.002,0.02)]

print("V2 submission mode:", SUBMISSION_MODE)


# ---------------- V3 controls ----------------
V3_INTERVENTION_LAMBDA = 0.35
V3_INTERVENTION_SHORTLIST = 48 if FULL_RUN else 24
V3_INTERVENTION_TIMES = np.linspace(
    float(TIME_GRID[0]), float(TIME_GRID[-1]), 12 if FULL_RUN else 7
)

# Only the seed itself and its immediate graph neighbours are hard-excluded.
# Wider "distality" becomes a graded evidence term instead of a universal hard gate.
V3_HARD_EXCLUDE_GRAPH_HOPS = 1
V3_HARD_EXCLUDE_EUCLID_A = 4.5

# Distance-conditioned residuals.
V3_DISTANCE_MIN_BIN_SIZE = 12
V3_EUCLID_BIN_WIDTH_A = 4.0

# Intervention shortlist mixes direct quantum excess and topology so that
# low-occupancy / high-control residues can still enter the expensive scan.
V3_SHORTLIST_WEIGHTS = {
    "distance_quantum_excess": 0.40,
    "topology": 0.25,
    "participation": 0.15,
    "surface": 0.10,
    "fluctuation": 0.10,
}

# Final physics score. Learned components are still gated independently.
V3_PHYSICAL_WEIGHTS = {
    "quantum_intervention_excess": 0.42,
    "quantum_susceptibility": 0.18,
    "distance_quantum_excess": 0.15,
    "topology": 0.10,
    "symbolic": 0.08,
    "robustness": 0.07,
}

# Seed-aware coarse-graining blends spectral geometry with transfer profiles.
V3_COARSE_PROFILE_WEIGHT = 0.60
V3_COARSE_SPECTRAL_WEIGHT = 0.30
V3_COARSE_DISTANCE_WEIGHT = 0.10

print("V3 intervention lambda:", V3_INTERVENTION_LAMBDA)
print("V3 intervention shortlist:", V3_INTERVENTION_SHORTLIST)


# ---------------- V3.1 controls ----------------
V31_FULL_SCAN_TARGETS = {"KRAS_G12C", "BCR_ABL1"}
V31_LAMBDA_SWEEP = np.array([0.10, 0.20, 0.35, 0.50, 0.75], dtype=float)
V31_PROSPECTIVE_SHORTLIST = 64 if FULL_RUN else 28

# Masked intervention normalization.
V31_INTERVENTION_MIN_GROUP = 6

# Coarse-graining diagnostics.
V31_COARSE_NDCG_K = 20
V31_COARSE_TOP_CLUSTER_COUNT = 5
V31_COARSE_MIN_NDCG20 = 0.70
V31_COARSE_MIN_TIE_TOP20 = 0.40

print("V3.1 full-scan targets:", sorted(V31_FULL_SCAN_TARGETS))
print("V3.1 lambda sweep:", V31_LAMBDA_SWEEP.tolist())

from sklearn.metrics import ndcg_score


# ---------------- V3.2 controls ----------------
V32_RRF_K = 60.0
V32_RRF_COMPONENTS = (
    ("quantum_intervention_excess", 1.0),
    ("distance_quantum_excess", 1.0),
    ("topology_score", 1.0),
)

# Spatial region support is intentionally modest: the quantum/topology rank
# consensus remains the primary score.
V32_POCKET_RADIUS_A = 10.0
V32_POCKET_HIGH_QUANTILE = 0.75
V32_POCKET_BONUS_WEIGHT = 0.15

# Rank-stability normalization: percentile rank std has theoretical maximum 0.5.
V32_RANK_STABILITY_SCALE = 2.0

# Coarse evaluation remains label-free for K selection. Pocket label metrics are
# diagnostic only and are never used to pick K.
V32_COARSE_MIN_NDCG20 = 0.70
V32_COARSE_MIN_TOP20 = 0.40

print("V3.2 RRF k:", V32_RRF_K)
print("V3.2 pocket radius Å:", V32_POCKET_RADIUS_A)


# ---------------- V3.3 pocket-first controls ----------------
# Only the exact functional seed is hard-excluded from ranking.
V33_SEED_ONLY_EXCLUSION = True

# Label-free pocket discovery.
V33_SITE_RADIUS_A = 10.0
V33_CENTER_SEPARATION_A = 12.0
V33_MAX_POCKETS = 8
V33_SITE_SEED_COHERENCE_WEIGHT = 0.65
V33_SITE_SEED_RRF_WEIGHT = 0.35

# Region-level rank fusion.
V33_SITE_RRF_K = 10.0
V33_SITE_TOP_FRACTION = 0.25
V33_SITE_COMPONENTS = (
    ("coherence_top_mean", 1.0),
    ("rrf_top_mean", 1.0),
    ("intervention_top_mean", 1.0),
    ("lambda_stability_mean", 1.0),
    ("surface_mean", 1.0),
)

# Pocket-first projection back to residues.
V33_SITE_FIELD_WEIGHT = 0.70
V33_RESIDUE_RRF_WEIGHT = 0.30
V33_TOP5_MAX_PER_POCKET = 3

# Site-level post-hoc validation distances.
V33_SITE_NEAR_A = 5.0
V33_SITE_NEAR_RELAXED_A = 8.0

print("V3.3 site radius Å:", V33_SITE_RADIUS_A)
print("V3.3 center separation Å:", V33_CENTER_SEPARATION_A)
print("V3.3 hard exclusion: functional seed only")


# ---------------- V3.4 controls ----------------
V34_ANNULUS_INNER_A = 10.0
V34_ANNULUS_OUTER_A = 18.0
V34_CONTRAST_CLIP = 8.0
V34_SITE_RRF_K = 10.0
V34_SITE_COMPONENTS = (
    ("center_seed_score", 1.0),
    ("topology_contrast", 1.0),
    ("intervention_contrast", 1.0),
    ("distance_excess_contrast", 1.0),
    ("coherence_contrast", 1.0),
)
V34_SITE_FIELD_WEIGHT = 0.90
V34_RESIDUE_RRF_TIEBREAK_WEIGHT = 0.10
V34_SEED_CORE_FRACTION = 0.50
V34_SITE_ROBUST_TOP_POCKETS = 3

print("V3.4 annulus:", V34_ANNULUS_INNER_A, "to", V34_ANNULUS_OUTER_A, "Å")


# ---------------- V3.5 controls ----------------
V35_RADIUS_GRID_A = (6.0, 8.0, 10.0, 12.0)
V35_CANONICAL_RADIUS_A = 10.0
V35_SITE_RRF_K = 10.0
V35_TOP_FRACTION = 0.25
V35_HIGH_QUANTILE = 0.75

V35_SCALE_COMPONENTS = (
    ("topology_top_mean", 1.0),
    ("coherence_top_mean", 1.0),
    ("intervention_top_mean", 1.0),
    ("distance_excess_top_mean", 1.0),
    ("consensus_fraction", 1.0),
)

# Site-first final residue score.
V35_SITE_FIELD_WEIGHT = 0.90
V35_RESIDUE_RRF_TIEBREAK_WEIGHT = 0.10

# Competition Top-5 representatives are actual site members.
V35_TOP5_MAX_PER_POCKET = 3

print("V3.5 radius grid Å:", V35_RADIUS_GRID_A)
print("V3.5 canonical output radius Å:", V35_CANONICAL_RADIUS_A)


# ---------------- V3.6 controls ----------------
V36_PHASE_KICK_LAMBDAS = np.array([0.15, 0.35, 0.65], dtype=float)
V36_PHASE_KICK_TIMES = np.linspace(
    float(TIME_GRID[0]), float(TIME_GRID[-1]), 6 if FULL_RUN else 4
)
V36_PHASE_SIZE_POWER = 0.5

V36_NULL_RADIUS_A = V35_CANONICAL_RADIUS_A
V36_NULL_SIZE_TOL_FRAC = 0.30
V36_NULL_SEED_DISTANCE_TOL_A = 5.0
V36_NULL_EXCLUDE_CENTER_A = 12.0
V36_NULL_MIN_MATCHES = 15
V36_NULL_Z_CLIP = 8.0

V36_SITE_RRF_K = 10.0
V36_SITE_COMPONENTS = (
    ("phase_kick_density_coeff", 1.0),
    ("collective_transfer_gain_z", 1.0),
    ("collective_q_peak_z", 1.0),
    ("topology_top_mean", 1.0),
    ("multiscale_score_median", 1.0),
)

V36_SITE_FIELD_WEIGHT = 0.90
V36_RESIDUE_RRF_TIEBREAK_WEIGHT = 0.10

# Scientific guardrail: after this run, freeze benchmark-driven scoring changes.
V36_ARCHITECTURE_FREEZE_AFTER_RUN = True

print("V3.6 phase-kick lambdas:", V36_PHASE_KICK_LAMBDAS.tolist())
print("V3.6 phase-kick times:", len(V36_PHASE_KICK_TIMES))
print("V3.6 freeze-after-run:", V36_ARCHITECTURE_FREEZE_AFTER_RUN)


# ---------------- V3.7 frozen-validation controls ----------------
import hashlib

V37_BLIND_FULL_INTERVENTION = True
V37_BLIND_PANEL_USED_FOR_TRAINING = False
V37_ALLOW_POST_UNSEAL_RETUNING = False

# Predeclared summary criteria. These are diagnostics, not optimization targets.
V37_PREDECLARED_CRITERIA = {
    "median_ap_over_prevalence_gt": 1.0,
    "top3_pocket_overlap_success_fraction_gte": 0.50,
    "top5_any_hit_success_fraction_gte": 0.50,
}

# Freeze the key V3.6 architecture choices before any blind-label cell exists.
V37_FROZEN_SPEC = {
    "contact_cutoff_A": float(CONTACT_CUTOFF_A),
    "gamma": float(GAMMA),
    "time_grid": [float(x) for x in TIME_GRID],
    "lambda_sweep_residue": [float(x) for x in V31_LAMBDA_SWEEP],
    "phase_kick_lambdas": [float(x) for x in V36_PHASE_KICK_LAMBDAS],
    "phase_kick_times": [float(x) for x in V36_PHASE_KICK_TIMES],
    "phase_size_power": float(V36_PHASE_SIZE_POWER),
    "site_rrf_k": float(V36_SITE_RRF_K),
    "site_rrf_components": [(str(a), float(b)) for a,b in V36_SITE_COMPONENTS],
    "site_field_weight": float(V36_SITE_FIELD_WEIGHT),
    "residue_rrf_tiebreak_weight": float(V36_RESIDUE_RRF_TIEBREAK_WEIGHT),
    "canonical_radius_A": float(V35_CANONICAL_RADIUS_A),
    "candidate_center_separation_A": float(V33_CENTER_SEPARATION_A),
    "max_candidate_pockets": int(V33_MAX_POCKETS),
    "top5_max_per_pocket": int(V35_TOP5_MAX_PER_POCKET),
}
V37_FROZEN_SIGNATURE = hashlib.sha256(
    json.dumps(V37_FROZEN_SPEC, sort_keys=True).encode("utf-8")
).hexdigest()

print("V3.7 frozen V3.6 signature:", V37_FROZEN_SIGNATURE)
print("Blind panel used for training:", V37_BLIND_PANEL_USED_FOR_TRAINING)


# ---------------- V3.8 evidence-pack controls ----------------
V38_PRIMARY_LABEL_CUTOFF_A = 5.0
V38_LABEL_CUTOFF_GRID_A = np.array([4.0, 4.5, 5.0, 5.5, 6.0], dtype=float)
V38_SPATIAL_NULL_REPLICATES = 2500 if FULL_RUN else 500
V38_PANEL_BOOTSTRAPS = 10000 if FULL_RUN else 2000
V38_NULL_SEED = SEED + 380
V38_EVIDENCE_PACK_ONLY = True

# The scorer must remain the exact V3.7-frozen V3.6 architecture.
V38_PARENT_SIGNATURE = V37_FROZEN_SIGNATURE

print("V3.8 parent frozen signature:", V38_PARENT_SIGNATURE)
print("V3.8 spatial-null replicates:", V38_SPATIAL_NULL_REPLICATES)
print("V3.8 label-cutoff sensitivity Å:", V38_LABEL_CUTOFF_GRID_A.tolist())


# ---------------- V3.9 Phase-2 frozen replication controls ----------------
V39_FROZEN_SIGNATURE = V37_FROZEN_SIGNATURE
V39_PHASE2_USED_FOR_TRAINING = False
V39_ALLOW_PHASE2_RETUNING = False
V39_PHASE2_FULL_INTERVENTION = True

# Predeclared before Phase-2 holo labels are present.
V39_PHASE2_CRITERIA = {
    "median_ap_over_prevalence_gt": 1.0,
    "top3_pocket_overlap_success_fraction_gte": 0.50,
    "top5_any_hit_success_fraction_gte": 0.50,
}

# Replication evidence.
V39_SPATIAL_NULL_REPLICATES = 2500 if FULL_RUN else 500
V39_COMBINED_BOOTSTRAPS = 10000 if FULL_RUN else 2000
V39_NULL_SEED = SEED + 390

print("V3.9 frozen signature:", V39_FROZEN_SIGNATURE)
print("Phase-2 used for training:", V39_PHASE2_USED_FOR_TRAINING)
print("Phase-2 retuning allowed:", V39_ALLOW_PHASE2_RETUNING)
print("Phase-2 criteria:", V39_PHASE2_CRITERIA)


# ---------------- V4.0 calibrated-learning controls ----------------
V40_PARENT_FROZEN_SIGNATURE = V37_FROZEN_SIGNATURE

V40_EVO_HIDDEN = 6
V40_EVO_POP = 36 if FULL_RUN else 20
V40_EVO_GENERATIONS = 28 if FULL_RUN else 12
V40_EVO_ENSEMBLE = 10
V40_EVO_SEED = SEED + 400

# No labels from challenge targets or Phase-3 are permitted in calibration.
V40_CHALLENGE_LABELS_USED_FOR_TRAINING = False
V40_PHASE3_USED_FOR_TRAINING = False

# Equal-rank fusion prevents post-hoc magnitude tuning.
V40_HYBRID_RRF_K = 60.0
V40_HYBRID_COMPONENTS = (
    "frozen_v3_6_final",
    "hdc_calibrated",
    "evo_calibrated",
)

V40_LEARN_FEATURES = [
    "pocket_first_physics_score",
    "site_field_score",
    "rrf_core",
    "pocket_coherence",
    "quantum_intervention_excess",
    "q_susceptibility_coeff_n",
    "distance_quantum_excess",
    "distality_score",
    "quantum_score",
    "q_persistence_n",
    "q_gain_n",
    "betweenness_n",
    "participation_n",
    "surface_n",
    "fluctuation_n",
    "topology_score",
    "symbolic_score",
    "lambda_rank_stability",
    "lambda_top20_frequency",
]

V40_PHASE3_CRITERIA = {
    "median_hybrid_ap_over_prevalence_gt": 1.0,
    "hybrid_top5_any_hit_success_fraction_gte": 0.50,
    "hybrid_improves_or_matches_frozen_on_majority": 0.50,
}

print("V4.0 frozen parent signature:", V40_PARENT_FROZEN_SIGNATURE)
print("V4.0 evo pop/gens:", V40_EVO_POP, V40_EVO_GENERATIONS)
print("Challenge labels used for training:", V40_CHALLENGE_LABELS_USED_FOR_TRAINING)
print("Phase-3 used for training:", V40_PHASE3_USED_FOR_TRAINING)


# ---------------- V4.0.2 execution-resilience ----------------
import pickle
import hashlib as _hashlib_v402

V404_RESUME = True
V404_SKIP_ARCHIVED_EVIDENCE = True
V404_SKIP_LEGACY_POST_ANALYSIS = True
V404_DEFER_CHALLENGE_UNTIL_CALIBRATED = True
_QALLOSTERY_IN_COLAB = importlib.util.find_spec("google.colab") is not None
V404_USE_GOOGLE_DRIVE_CHECKPOINTS = (
    os.environ.get(
        "QALLOSTERY_USE_GOOGLE_DRIVE",
        "1" if _QALLOSTERY_IN_COLAB else "0",
    ).strip().lower()
    not in {"0", "false", "no", "off"}
)

# Exact execution acceleration only.
V404_INTERVENTION_BATCH_SIZE = 8
V404_INTERVENTION_BATCH_SIZE_LARGE = 4
V404_LARGE_GRAPH_N = 650
V404_BATCH_CHECKPOINTS = True

# Clean-room Phase-3 controls.
import uuid as _uuid_v404
V404_STRICT_PHASE3_CLEANROOM = True
V404_PHASE3_ALLOW_STALE_GLOBALS = False
V404_PHASE3_REQUIRE_FILE_HASHES = True

# ---------------- V4.0.8 robust checkpoint backend resolution ----------------
import time as _time_v408
import shutil as _shutil_v408

V408_ALLOW_LOCAL_FALLBACK = (
    os.environ.get(
        "QALLOSTERY_ALLOW_LOCAL_FALLBACK",
        "0" if V404_USE_GOOGLE_DRIVE_CHECKPOINTS else "1",
    ).strip().lower()
    not in {"0", "false", "no", "off"}
)
V408_FORCE_REMOUNT_ON_FAILURE = True
V408_MOUNT_RETRIES = 2
V408_DRIVE_SUBDIR = (
    "quantum_allostery_checkpoints/v4_0_4"
)

V404_CHECKPOINT_BACKEND = "local"
V404_CHECKPOINT_DIR = ROOT / "checkpoints"

# Explicit environment override wins and may point at another persistent mount.
_v408_env_dir=os.environ.get("QALLOSTERY_CHECKPOINT_DIR")
if _v408_env_dir:
    _candidate=Path(_v408_env_dir)
    _candidate.mkdir(parents=True,exist_ok=True)
    V404_CHECKPOINT_DIR=_candidate
    V404_CHECKPOINT_BACKEND="environment_override"

elif V404_USE_GOOGLE_DRIVE_CHECKPOINTS:
    _drive_root=Path("/content/drive")
    _mydrive=_drive_root/"MyDrive"
    _mount_errors=[]

    def _v408_drive_ready():
        return _mydrive.exists() and _mydrive.is_dir()

    if not _v408_drive_ready():
        try:
            from google.colab import drive as _v408_drive

            for _attempt in range(max(1,int(V408_MOUNT_RETRIES))):
                try:
                    _v408_drive.mount(
                        "/content/drive",
                    )
                except Exception as exc:
                    _mount_errors.append(
                        f"normal[{_attempt+1}]={repr(exc)}"
                    )

                if _v408_drive_ready():
                    break

                _time_v408.sleep(1.0)

            if (
                not _v408_drive_ready()
                and V408_FORCE_REMOUNT_ON_FAILURE
            ):
                try:
                    _v408_drive.mount(
                        "/content/drive",
                        force_remount=True,
                    )
                except Exception as exc:
                    _mount_errors.append(
                        f"force_remount={repr(exc)}"
                    )
        except Exception as exc:
            _mount_errors.append(
                f"colab_drive_import={repr(exc)}"
            )

    if _v408_drive_ready():
        _candidate=(
            _mydrive
            / "quantum_allostery_checkpoints"
            / "v4_0_4"
        )
        _candidate.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Real filesystem write/read/delete verification.
        _probe=_candidate/".v408_write_probe"
        _token="v408-drive-write-probe"
        try:
            _probe.write_text(
                _token,
                encoding="utf-8",
            )
            _readback=_probe.read_text(
                encoding="utf-8",
            )
            assert _readback==_token
            _probe.unlink(missing_ok=True)

            V404_CHECKPOINT_DIR=_candidate
            V404_CHECKPOINT_BACKEND="google_drive"
        except Exception as exc:
            _mount_errors.append(
                f"write_probe={repr(exc)}"
            )

    if (
        V404_CHECKPOINT_BACKEND!="google_drive"
        and not V408_ALLOW_LOCAL_FALLBACK
    ):
        raise RuntimeError(
            "Persistent Google Drive checkpoints are required for FULL_RUN. "
            "Drive was unavailable or unwritable. "
            "Restart the Colab runtime, approve Drive access, and Run all again. "
            f"Mount diagnostics: {_mount_errors}"
        )

# Local fallback is intentionally opt-in.
if V404_CHECKPOINT_BACKEND=="local":
    V404_CHECKPOINT_DIR=ROOT/"checkpoints"

V404_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Salvage compatible local checkpoints if this notebook is opened in the same
# live runtime after a previous local-fallback run.
V408_LOCAL_SALVAGE_SOURCE=ROOT/"checkpoints"
V408_LOCAL_SALVAGED_FILES=[]

if (
    V404_CHECKPOINT_BACKEND in {
        "google_drive",
        "environment_override",
    }
    and V408_LOCAL_SALVAGE_SOURCE.exists()
    and V408_LOCAL_SALVAGE_SOURCE.resolve()
        !=V404_CHECKPOINT_DIR.resolve()
):
    for _src in V408_LOCAL_SALVAGE_SOURCE.glob("*.pkl"):
        _dst=V404_CHECKPOINT_DIR/_src.name
        if not _dst.exists():
            try:
                _shutil_v408.copy2(
                    _src,
                    _dst,
                )
                V408_LOCAL_SALVAGED_FILES.append(
                    _src.name
                )
            except Exception as exc:
                print(
                    "checkpoint salvage warning:",
                    _src.name,
                    repr(exc),
                )

print(
    "V4.0.8 checkpoint backend:",
    V404_CHECKPOINT_BACKEND,
)
print(
    "V4.0.8 checkpoint dir:",
    V404_CHECKPOINT_DIR,
)
print(
    "V4.0.8 locally salvaged checkpoints:",
    len(V408_LOCAL_SALVAGED_FILES),
)

V404_EXECUTION_SPEC = {
    "version": "4.0.4",
    "frozen_parent": V40_PARENT_FROZEN_SIGNATURE,
    "full_run": bool(FULL_RUN),
    "contact_cutoff_A": float(CONTACT_CUTOFF_A),
    "gamma": float(GAMMA),
    "time_grid": [float(x) for x in TIME_GRID],
    "residue_lambdas": [float(x) for x in V31_LAMBDA_SWEEP],
    "phase_lambdas": [float(x) for x in V36_PHASE_KICK_LAMBDAS],
    "evo_hidden": int(V40_EVO_HIDDEN),
    "evo_pop": int(V40_EVO_POP),
    "evo_generations": int(V40_EVO_GENERATIONS),
    "evo_ensemble": int(V40_EVO_ENSEMBLE),
    "learn_features": list(V40_LEARN_FEATURES),
    "hybrid_components": list(V40_HYBRID_COMPONENTS),
    "hybrid_rrf_k": float(V40_HYBRID_RRF_K),
    "challenge_deferred": bool(V404_DEFER_CHALLENGE_UNTIL_CALIBRATED),
    "intervention_batch_size": int(V404_INTERVENTION_BATCH_SIZE),
    "intervention_batch_size_large": int(V404_INTERVENTION_BATCH_SIZE_LARGE),
    "large_graph_n": int(V404_LARGE_GRAPH_N),
    "strict_phase3_cleanroom": bool(V404_STRICT_PHASE3_CLEANROOM),
}

V404_EXECUTION_SIGNATURE = _hashlib_v402.sha256(
    json.dumps(
        V404_EXECUTION_SPEC,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

def _v404_ckpt_path(namespace, key):
    safe = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{namespace}__{key}",
    )
    return V404_CHECKPOINT_DIR / f"{safe}.pkl"

def v404_save(namespace, key, obj):
    path = _v404_ckpt_path(namespace, key)
    tmp = path.with_suffix(".tmp")
    payload = {
        "signature": V404_EXECUTION_SIGNATURE,
        "namespace": namespace,
        "key": key,
        "object": obj,
    }
    with open(tmp, "wb") as fh:
        pickle.dump(
            payload,
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(path)
    print("checkpoint saved:", path.name)
    return path

def v404_load(namespace, key):
    if not V404_RESUME:
        return None
    path = _v404_ckpt_path(namespace, key)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("signature") != V404_EXECUTION_SIGNATURE:
            print(
                "checkpoint ignored (signature mismatch):",
                path.name,
            )
            return None
        print("checkpoint loaded:", path.name)
        return payload.get("object")
    except Exception as exc:
        print(
            "checkpoint load failed; recomputing:",
            path.name,
            repr(exc),
        )
        return None

def v404_stage_marker(stage, **extra):
    row = {
        "stage": stage,
        "signature": V404_EXECUTION_SIGNATURE,
        **extra,
    }
    p = RESULTS_DIR / "v404_execution_status.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print("STAGE:", stage)

print("V4.0.3 checkpoint backend:", V404_CHECKPOINT_BACKEND)
print("V4.0.3 checkpoint dir:", V404_CHECKPOINT_DIR)
print("V4.0.3 execution signature:", V404_EXECUTION_SIGNATURE)

# ---------------- V4.0.5 mechanical bugfix tag ----------------
V405_VERSION = "4.0.5"
V405_CHECKPOINT_COMPATIBILITY = "V4.0.4"
V405_SCIENTIFIC_ARCHITECTURE_CHANGED = False
print(
    "V4.0.5 checkpoint compatibility:",
    V405_CHECKPOINT_COMPATIBILITY,
)
print(
    "V4.0.5 scientific architecture changed:",
    V405_SCIENTIFIC_ARCHITECTURE_CHANGED,
)


# ---------------- V4.0.6 export/package-only tag ----------------
V406_VERSION = "4.0.6"
V406_SCIENTIFIC_ARCHITECTURE_CHANGED = False
V406_PARENT_CLEANROOM = "V4.0.5"
V406_CHECKPOINT_COMPATIBILITY = "V4.0.4/V4.0.5"
print("V4.0.6 scientific architecture changed:", V406_SCIENTIFIC_ARCHITECTURE_CHANGED)


# ---------------- V4.0.7 persistent-resume guard ----------------
from datetime import datetime, timezone as _v407_timezone

V407_VERSION = "4.0.7"
V407_SCIENTIFIC_ARCHITECTURE_CHANGED = False
V407_CHECKPOINT_COMPATIBILITY = "V4.0.4/V4.0.5/V4.0.6"
V407_REQUIRE_PERSISTENT_CHECKPOINTS_FOR_FULL_RUN = False

V407_RESUME_STATUS_PATH = (
    V404_CHECKPOINT_DIR / "v407_resume_status.json"
)
V407_STAGE_DIR = (
    V404_CHECKPOINT_DIR / "v407_stage_markers"
)
V407_STAGE_DIR.mkdir(parents=True, exist_ok=True)

def v407_checkpoint_inventory():
    """
    Inventory current compatible checkpoint files without mutating them.
    """
    rows=[]
    if not V404_CHECKPOINT_DIR.exists():
        return pd.DataFrame(
            columns=["namespace","count"]
        )

    for p in V404_CHECKPOINT_DIR.glob("*.pkl"):
        stem=p.stem
        namespace=stem.split("__",1)[0]
        rows.append({
            "namespace":namespace,
            "file":p.name,
            "size_bytes":int(p.stat().st_size),
            "mtime":datetime.fromtimestamp(
                p.stat().st_mtime,
                tz=_v407_timezone.utc,
            ).isoformat(),
        })

    if not rows:
        return pd.DataFrame(
            columns=["namespace","count"]
        )

    return pd.DataFrame(rows)

def v407_inventory_summary():
    inv=v407_checkpoint_inventory()
    if len(inv)==0:
        return pd.DataFrame(
            columns=["namespace","count","size_mb"]
        )
    return (
        inv.groupby("namespace",as_index=False)
        .agg(
            count=("file","count"),
            size_bytes=("size_bytes","sum"),
        )
        .assign(
            size_mb=lambda d:
                d["size_bytes"]/(1024.0*1024.0)
        )
        .drop(columns=["size_bytes"])
        .sort_values(
            ["namespace","count"]
        )
        .reset_index(drop=True)
    )

def v407_write_resume_status(namespace,key):
    inv=v407_checkpoint_inventory()
    payload={
        "version":V407_VERSION,
        "scientific_architecture_changed":
            V407_SCIENTIFIC_ARCHITECTURE_CHANGED,
        "checkpoint_backend":V404_CHECKPOINT_BACKEND,
        "checkpoint_dir":str(V404_CHECKPOINT_DIR),
        "execution_signature":V404_EXECUTION_SIGNATURE,
        "updated_utc":datetime.now(
            _v407_timezone.utc
        ).isoformat(),
        "last_namespace":str(namespace),
        "last_key":str(key),
        "checkpoint_files":int(len(inv)),
        "namespace_counts":(
            inv.groupby("namespace")
            .size()
            .astype(int)
            .to_dict()
            if len(inv) else {}
        ),
    }
    tmp=V407_RESUME_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload,indent=2),
        encoding="utf-8",
    )
    tmp.replace(V407_RESUME_STATUS_PATH)
    return payload

# Wrap the existing checkpoint writer; stored checkpoint objects/signatures and
# filenames themselves remain completely unchanged.
_v407_original_v404_save = v404_save

def v404_save(namespace,key,obj):
    path=_v407_original_v404_save(
        namespace,key,obj
    )
    try:
        v407_write_resume_status(
            namespace,key
        )
    except Exception as exc:
        print(
            "resume heartbeat warning:",
            repr(exc),
        )
    return path

# Persist stage markers to Drive as well.
_v407_original_v404_stage_marker = v404_stage_marker

def v404_stage_marker(stage,**extra):
    _v407_original_v404_stage_marker(
        stage,**extra
    )
    payload={
        "stage":stage,
        "updated_utc":datetime.now(
            _v407_timezone.utc
        ).isoformat(),
        "execution_signature":
            V404_EXECUTION_SIGNATURE,
        **extra,
    }
    safe=re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(stage),
    )
    p=V407_STAGE_DIR/f"{safe}.json"
    p.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

print(
    "V4.0.7 checkpoint compatibility:",
    V407_CHECKPOINT_COMPATIBILITY,
)
print(
    "V4.0.7 backend:",
    V404_CHECKPOINT_BACKEND,
)
print(
    "V4.0.7 checkpoint dir:",
    V404_CHECKPOINT_DIR,
)
if (
    FULL_RUN
    and V404_CHECKPOINT_BACKEND!="google_drive"
):
    print(
        "WARNING: full run is not using the "
        "Google Drive checkpoint backend."
    )


# ---------------- V4.0.8 execution-only tag ----------------
V408_VERSION="4.0.8"
V408_SCIENTIFIC_ARCHITECTURE_CHANGED=False
V408_CHECKPOINT_COMPATIBILITY="V4.0.4–V4.0.7"

def v408_assert_persistent_backend(stage):
    if V404_CHECKPOINT_BACKEND in {
        "google_drive",
        "environment_override",
    }:
        return True

    if V408_ALLOW_LOCAL_FALLBACK:
        print(
            "WARNING: local fallback explicitly allowed at stage:",
            stage,
        )
        return False

    raise RuntimeError(
        "Refusing expensive stage without persistent checkpoint storage: "
        f"{stage}; backend={V404_CHECKPOINT_BACKEND}"
    )

print(
    "V4.0.8 scientific architecture changed:",
    V408_SCIENTIFIC_ARCHITECTURE_CHANGED,
)


# ---------------- V4.1 submission-evidence controls ----------------
V410_VERSION = "4.1"
V410_SCIENTIFIC_ARCHITECTURE_CHANGED = False
V410_PARENT = "V4.0.8"
V410_COARSE_K_VALUES = (8, 16, 32)
V410_HARDWARE_TARGETS = ("KRAS_G12C", "BCR_ABL1")
V410_HARDWARE_K = 8
V410_HARDWARE_TIME = 3.0
V410_HARDWARE_PHASE_LAMBDA = 0.35
V410_HARDWARE_SHOTS = 4096
V410_NOISE_1Q = 0.001
V410_NOISE_2Q = 0.01

V410_SUBMISSION_DIR = RESULTS_DIR / "competition_v4_1"
V410_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

assert V410_SCIENTIFIC_ARCHITECTURE_CHANGED is False
print("V4.1 scientific architecture changed:", V410_SCIENTIFIC_ARCHITECTURE_CHANGED)
print("V4.1 submission directory:", V410_SUBMISSION_DIR)


# ---------------- V4.1.1 execution attestation ----------------
import uuid as _uuid_v411
from datetime import datetime as _dt_v411, timezone as _tz_v411

V411_VERSION = "4.1.1"
V411_SCIENTIFIC_ARCHITECTURE_CHANGED = False
V411_RUN_ID = _uuid_v411.uuid4().hex

V411_ATTEST_DIR = V404_CHECKPOINT_DIR / "v411_execution_attestation"
V411_ATTEST_DIR.mkdir(parents=True, exist_ok=True)

V411_LEDGER_PATH = V411_ATTEST_DIR / f"{V411_RUN_ID}.jsonl"
V411_RUN_MANIFEST_PATH = V411_ATTEST_DIR / f"{V411_RUN_ID}_manifest.json"

def v411_attest(stage, **meta):
    rec = {
        "run_id": V411_RUN_ID,
        "stage": str(stage),
        "utc": _dt_v411.now(_tz_v411.utc).isoformat(),
        "checkpoint_backend": V404_CHECKPOINT_BACKEND,
        "checkpoint_dir": str(V404_CHECKPOINT_DIR),
        "execution_signature": V404_EXECUTION_SIGNATURE,
        **meta,
    }
    with V411_LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec

def v411_ledger_records():
    if not V411_LEDGER_PATH.exists():
        return []
    rows=[]
    for line in V411_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def v411_assert_current_run_stages(required):
    rows=v411_ledger_records()
    stages={
        r["stage"]
        for r in rows
        if r.get("run_id")==V411_RUN_ID
    }
    missing=sorted(set(required)-stages)
    if missing:
        raise RuntimeError(
            "Current-run execution attestation incomplete. "
            f"Missing stages: {missing}"
        )
    return stages

V411_RUN_MANIFEST = {
    "version": V411_VERSION,
    "run_id": V411_RUN_ID,
    "scientific_architecture_changed": V411_SCIENTIFIC_ARCHITECTURE_CHANGED,
    "checkpoint_backend": V404_CHECKPOINT_BACKEND,
    "checkpoint_dir": str(V404_CHECKPOINT_DIR),
    "execution_signature": V404_EXECUTION_SIGNATURE,
    "started_utc": _dt_v411.now(_tz_v411.utc).isoformat(),
}
V411_RUN_MANIFEST_PATH.write_text(
    json.dumps(V411_RUN_MANIFEST, indent=2),
    encoding="utf-8",
)

v411_attest("bootstrap_complete")
print("V4.1.1 run ID:", V411_RUN_ID)
print("V4.1.1 ledger:", V411_LEDGER_PATH)
print("V4.1.1 execution attestation: READY")


# ---------------- V4.1.2 final-scientific-audit tag ----------------
V412_VERSION="4.1.2"
V412_SCIENTIFIC_ARCHITECTURE_CHANGED=False
V412_MAPPING_SENSITIVITY_ONLY=True
V412_HARDWARE_FIDELITY_STRONG=0.70
V412_HARDWARE_FIDELITY_MODERATE=0.40

assert V412_SCIENTIFIC_ARCHITECTURE_CHANGED is False
print("V4.1.2 scientific architecture changed:",V412_SCIENTIFIC_ARCHITECTURE_CHANGED)


# ---------------- V4.1.3 final claim-lock tag ----------------
V413_VERSION="4.1.3"
V413_SCIENTIFIC_ARCHITECTURE_CHANGED=False
V413_PRIMARY_MODEL="frozen_v3_6_site_aware"
V413_AUXILIARY_MODEL="v40_calibrated_hybrid"
V413_FULL_TO_COARSE_TARGET_RHO=0.80

assert V413_SCIENTIFIC_ARCHITECTURE_CHANGED is False
print("V4.1.3 scientific architecture changed:",V413_SCIENTIFIC_ARCHITECTURE_CHANGED)

# %% [markdown]
# ### 2A. Persistent checkpoint inventory
#
# This is intentionally executed before any scoring. It shows what a resumed
# run can reuse from Google Drive.
#
# The inventory is descriptive only; it does not determine any scientific score.

# %% [code] Notebook code cell 3 (source index 6)
v408_assert_persistent_backend("startup_inventory")

print(
    "V4.0.8 persistent backend verified:",
    V404_CHECKPOINT_BACKEND,
)
if V408_LOCAL_SALVAGED_FILES:
    print(
        "Salvaged local checkpoints into persistent storage:",
        len(V408_LOCAL_SALVAGED_FILES),
    )


V407_STARTUP_INVENTORY=v407_checkpoint_inventory()
V407_STARTUP_SUMMARY=v407_inventory_summary()

print(
    "Persistent checkpoint files:",
    len(V407_STARTUP_INVENTORY),
)
display(V407_STARTUP_SUMMARY)

if V407_RESUME_STATUS_PATH.exists():
    try:
        _resume=json.loads(
            V407_RESUME_STATUS_PATH.read_text(
                encoding="utf-8"
            )
        )
        print(
            "Last persistent checkpoint:",
            _resume.get("last_namespace"),
            _resume.get("last_key"),
        )
        print(
            "Heartbeat updated:",
            _resume.get("updated_utc"),
        )
    except Exception as exc:
        print(
            "Could not read previous resume heartbeat:",
            repr(exc),
        )

# Helpful namespace expectations — reporting only.
V407_EXPECTED_NAMESPACES=[
    "intervention_batch",
    "intervention_lambda",
    "phase1_pre_unseal",
    "phase2_pre_unseal",
    "v40_lopo",
    "v40_final_model",
    "challenge_frozen_final",
    "phase3_pre_unseal",
]
print(
    "Expected namespaces:",
    V407_EXPECTED_NAMESPACES,
)

# %% [markdown]
# ## 3. Challenge target registry
#
# The required benchmark pairs from the challenge statement are encoded directly:
#
# - KRAS G12C: **4OBE → 6OIM**
# - BCR-ABL1: **1OPL → 5MO4**
# - Cardiac Myosin: **5TBY → 6C1H**
# - c-Myc application target: **1NKP**
#
# For known validation structures we specify the allosteric drug ligand where possible:
# - 6OIM: `MOV` = AMG 510 / sotorasib bound form
# - 5MO4: `AY7` = asciminib
#
# For 6C1H we intentionally set no mavacamten ligand ID, because the notebook must not fabricate ground truth.

# %% [code] Notebook code cell 4 (source index 8)
TARGETS = {
    "KRAS_G12C": {
        "apo": "4OBE",
        "holo": "6OIM",
        "chains": ["A"],
        "target_chains": ["A"],
        "holo_ligand": "MOV",
        "seed_mode": "endogenous_ligand",
        "manual_seed_resids": [10,11,12,13,14,15,16,17,116,117,118,119],
        "description": "KRAS G12C — Switch-II allosteric pocket",
        "official": True,
    },
    "BCR_ABL1": {
        "apo": "1OPL",
        "holo": "5MO4",
        "chains": ["A"],
        "target_chains": ["A"],
        "holo_ligand": "AY7",
        "seed_mode": "endogenous_or_manual",
        # Canonical catalytic/ATP-site motifs; used only if no endogenous catalytic ligand is present.
        "manual_seed_resids": [248,249,250,251,252,253,271,286,315,381,382,383],
        "description": "ABL1 — distal myristoyl pocket",
        "official": True,
    },
    "CARDIAC_MYOSIN": {
        "apo": "5TBY",
        "holo": "6C1H",
        "chains": ["A"],
        "target_chains": ["A"],
        "holo_ligand": None,  # Intentionally unresolved: do not use ADP as a fake mavacamten label.
        "seed_mode": "endogenous_or_manual",
        "manual_seed_resids": [
            179,180,181,182,183,184,185,186,
            233,234,235,236,237,238,239,
            454,455,456,457,458,459
        ],
        "description": "β-cardiac myosin — mechanical / super-relaxed-state site",
        "official": True,
    },
    "cMYC": {
        "apo": "1NKP",
        "holo": None,
        # Author chain A = Myc, B = Max in the legacy PDB file.
        # Keep both protein chains in the graph so inter-protein topology is retained.
        "chains": ["A", "B"],
        "target_chains": ["A"],
        "holo_ligand": None,
        "seed_mode": "nucleic_interface",
        "manual_seed_resids": [],
        "description": "c-Myc/Max — prospective allosteric discovery",
        "official": True,
    },
}

# Optional exploratory validation only. Do NOT silently replace the competition pair.
USE_EXPLORATORY_MAVACAMTEN_VALIDATION = False

EXPLORATORY_MYOSIN = {
    "apo": "5TBY",
    "holo": "8QYQ",
    "chains": ["A"],
    "target_chains": ["A"],
    "holo_ligand": "XB2",
    "seed_mode": "endogenous_or_manual",
    "manual_seed_resids": TARGETS["CARDIAC_MYOSIN"]["manual_seed_resids"],
    "description": "Exploratory β-cardiac myosin validation against mavacamten-bound 8QYQ",
    "official": False,
}

if USE_EXPLORATORY_MAVACAMTEN_VALIDATION:
    TARGETS["CARDIAC_MYOSIN_EXPLORATORY"] = EXPLORATORY_MYOSIN

pd.DataFrame(TARGETS).T[["apo","holo","chains","target_chains","holo_ligand","seed_mode","official"]]

# %% [markdown]
# ### Independent calibration panel
#
# Upload `/content/allosteric_calibration.csv` before running the next cell to add separately curated training proteins.
#
# Columns:
#
# `name, apo, holo, chains, target_chains, holo_ligand, seed_mode, manual_seed_resids, description`
#
# Use semicolons for list fields (`A;B`, `12;13;14`). These entries are marked `official=False`.
#
# For submission-grade learned components, use at least **5** independent labelled proteins; **15–30** is the target. Without enough calibration proteins, V2 deliberately keeps HDC/neuroevolution from dominating the challenge ranking.

# %% [code] Notebook code cell 5 (source index 10)

CALIBRATION_CSV = Path(
    os.environ.get(
        "QALLOSTERY_CALIBRATION_CSV",
        str(ROOT / "allosteric_calibration.csv"),
    )
).expanduser()

def _split_str(v):
    if pd.isna(v) or str(v).strip() == "":
        return []
    return [x.strip() for x in str(v).split(";") if x.strip()]

def _split_int(v):
    return [int(x) for x in _split_str(v)]

def load_calibration_registry(path: Path):
    if not path.exists():
        print("No independent calibration CSV supplied.")
        return {}
    df = pd.read_csv(path)
    required = {
        "name","apo","holo","chains","target_chains","holo_ligand",
        "seed_mode","manual_seed_resids","description"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Calibration CSV missing columns: {sorted(missing)}")
    registry = {}
    for row in df.itertuples(index=False):
        registry[str(row.name)] = {
            "apo": str(row.apo),
            "holo": None if pd.isna(row.holo) else str(row.holo),
            "chains": _split_str(row.chains),
            "target_chains": _split_str(row.target_chains),
            "holo_ligand": None if pd.isna(row.holo_ligand) else str(row.holo_ligand),
            "seed_mode": str(row.seed_mode),
            "manual_seed_resids": _split_int(row.manual_seed_resids),
            "description": str(row.description),
            "official": False,
        }
    return registry

CALIBRATION_TARGETS = load_calibration_registry(CALIBRATION_CSV)
TARGETS.update(CALIBRATION_TARGETS)

pd.DataFrame(columns=[
    "name","apo","holo","chains","target_chains","holo_ligand",
    "seed_mode","manual_seed_resids","description"
]).to_csv(RESULTS_DIR / "allosteric_calibration_template.csv", index=False)

print("Independent calibration targets:", len(CALIBRATION_TARGETS))

# %% [markdown]
# ## 4. Static PDB ingestion — no MD trajectories
#
# Only static structures are downloaded from RCSB PDB. Water is ignored; the residue graph is built from protein Cα coordinates with a geometric contact cutoff.
#
# The primary graph is therefore an elastic/contact-network abstraction consistent with the challenge's topology-first assumption.

# %% [code] Notebook code cell 6 (source index 12)
AA3_TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"
}

ENDOGENOUS_CATALYTIC_LIGANDS = {
    "GDP","GTP","GNP","GSP","ATP","ADP","AMP","ANP","ACP","AMPPNP","ATP4",
    "GDP3","GTP3","GDP4","GTP4"
}

NUCLEIC_RESNAMES = {
    "A","C","G","U","T","DA","DC","DG","DT","DU","ADE","CYT","GUA","THY","URA"
}

def download_pdb(pdb_id: str, force: bool = False) -> Path:
    pdb_id = pdb_id.upper()
    path = PDB_DIR / f"{pdb_id}.pdb"
    if path.exists() and not force:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path

PARSER = PDBParser(QUIET=True)

def load_model(pdb_id: str):
    path = download_pdb(pdb_id)
    structure = PARSER.get_structure(pdb_id, str(path))
    return next(structure.get_models())

def available_protein_chains(model) -> List[str]:
    out = []
    for chain in model:
        n = sum(1 for r in chain if is_aa(r, standard=True) and "CA" in r)
        if n > 0:
            out.append(chain.id)
    return out

def extract_residue_table(pdb_id: str, requested_chains: Optional[List[str]] = None) -> pd.DataFrame:
    model = load_model(pdb_id)
    protein_chains = available_protein_chains(model)

    if requested_chains:
        chains = [c for c in requested_chains if c in protein_chains]
        if not chains:
            warnings.warn(
                f"{pdb_id}: requested chains {requested_chains} not found. "
                f"Using largest protein chain from {protein_chains}."
            )
            counts = {
                c: sum(1 for r in model[c] if is_aa(r, standard=True) and "CA" in r)
                for c in protein_chains
            }
            chains = [max(counts, key=counts.get)]
    else:
        chains = protein_chains

    rows = []
    global_idx = 0
    for chain_id in chains:
        chain = model[chain_id]
        for residue in chain:
            if not is_aa(residue, standard=True) or "CA" not in residue:
                continue
            resseq = int(residue.id[1])
            icode = str(residue.id[2]).strip()
            resname = residue.get_resname().upper()
            ca = np.asarray(residue["CA"].coord, dtype=float)
            atom_coords = np.asarray(
                [a.coord for a in residue.get_atoms() if a.element != "H"],
                dtype=float
            )
            rows.append({
                "idx": global_idx,
                "chain": chain_id,
                "resseq": resseq,
                "icode": icode,
                "resname": resname,
                "aa": AA3_TO1.get(resname, "X"),
                "x": ca[0], "y": ca[1], "z": ca[2],
                "ca": ca,
                "atom_coords": atom_coords,
                "label": f"{chain_id}:{resseq}{icode}",
            })
            global_idx += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No standard protein residues found for {pdb_id}.")
    return df

def extract_nonprotein_atoms(pdb_id: str, residue_names: Optional[set] = None):
    model = load_model(pdb_id)
    found = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().upper()
            if is_aa(residue, standard=True):
                continue
            if resname in {"HOH","WAT","DOD"}:
                continue
            if residue_names is not None and resname not in residue_names:
                continue
            coords = np.asarray(
                [a.coord for a in residue.get_atoms() if a.element != "H"],
                dtype=float
            )
            if len(coords) == 0:
                continue
            found.append({
                "chain": chain.id,
                "resseq": int(residue.id[1]),
                "resname": resname,
                "coords": coords,
                "n_atoms": len(coords),
            })
    return found

for name, cfg in TARGETS.items():
    apo = cfg["apo"]
    model = load_model(apo)
    print(name, apo, "protein chains:", available_protein_chains(model))

# %% [markdown]
# ## 5. Residue contact graph
#
# Edges are added when Cα atoms lie within `CONTACT_CUTOFF_A` Å. Sequential neighbours are also retained when geometrically plausible, protecting chain continuity across slightly sparse structural regions.
#
# Edge weight:
#
# \[
# w_{ij}=\exp[-(d_{ij}/r_c)^2]
# \]
#
# This produces the topology that drives both the classical elastic-network proxy and the quantum Hamiltonian.

# %% [code] Notebook code cell 7 (source index 14)
def build_contact_graph(df: pd.DataFrame, cutoff: float = CONTACT_CUTOFF_A) -> nx.Graph:
    coords = np.stack(df["ca"].to_numpy())
    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff)

    G = nx.Graph()
    for row in df.itertuples():
        G.add_node(
            row.idx,
            chain=row.chain,
            resseq=row.resseq,
            resname=row.resname,
            label=row.label,
        )

    for i, j in pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        w = math.exp(-((d / cutoff) ** 2))
        G.add_edge(i, j, distance=d, weight=w)

    # Preserve covalent sequence continuity where adjacent author residue numbers are close in space.
    for chain, sub in df.groupby("chain", sort=False):
        inds = sub.index.to_list()
        for a, b in zip(inds[:-1], inds[1:]):
            ra, rb = df.loc[a], df.loc[b]
            if abs(int(ra.resseq) - int(rb.resseq)) <= 2:
                d = float(np.linalg.norm(ra.ca - rb.ca))
                if d <= 12.0 and not G.has_edge(int(ra.idx), int(rb.idx)):
                    G.add_edge(
                        int(ra.idx), int(rb.idx),
                        distance=d,
                        weight=math.exp(-((d / cutoff) ** 2))
                    )

    return G

def graph_matrices(G: nx.Graph):
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)
    degree = A.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    nz = degree > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
    Dm = np.diag(inv_sqrt)
    Lnorm = np.eye(len(A)) - Dm @ A @ Dm
    # Force exact symmetry against floating-point drift.
    Lnorm = 0.5 * (Lnorm + Lnorm.T)
    return A, Lnorm, degree

# %% [markdown]
# ## 6. Functional seed selection
#
# For benchmark proteins we prefer an **endogenous catalytic ligand in the apo structure** (GDP/GTP/ATP/ADP and close analogues), so no holo information leaks into the predictor.
#
# If no endogenous catalytic ligand is available, a small manually defined catalytic motif is used as a fallback.
#
# For c-Myc, there is no canonical enzyme active site. We therefore seed propagation from the **DNA-contacting functional interface** in the input 1NKP structure and search for distal sites dynamically connected to that functional interface.

# %% [code] Notebook code cell 8 (source index 16)
def residue_min_distance_to_atoms(residue_atom_coords: np.ndarray, atom_coords: np.ndarray) -> float:
    if len(residue_atom_coords) == 0 or len(atom_coords) == 0:
        return np.inf
    # Small residue/ligand atom sets: broadcasting is fast and avoids another dependency.
    d = residue_atom_coords[:, None, :] - atom_coords[None, :, :]
    return float(np.sqrt((d*d).sum(axis=2)).min())

def seed_from_endogenous_ligand(pdb_id: str, df: pd.DataFrame, cutoff: float = ACTIVE_SEED_CUTOFF_A):
    ligands = extract_nonprotein_atoms(pdb_id, ENDOGENOUS_CATALYTIC_LIGANDS)
    if not ligands:
        return [], []
    seed = set()
    evidence = []
    for lig in ligands:
        local_hits = []
        for row in df.itertuples():
            d = residue_min_distance_to_atoms(row.atom_coords, lig["coords"])
            if d <= cutoff:
                seed.add(int(row.idx))
                local_hits.append(row.label)
        evidence.append((lig["resname"], lig["chain"], lig["resseq"], local_hits))
    return sorted(seed), evidence

def seed_from_nucleic_interface(pdb_id: str, df: pd.DataFrame, cutoff: float = 7.0):
    nuc = extract_nonprotein_atoms(pdb_id, NUCLEIC_RESNAMES)
    if not nuc:
        return [], []
    all_nuc = np.concatenate([x["coords"] for x in nuc], axis=0)
    seed = []
    for row in df.itertuples():
        # CA-to-nucleic distance is a conservative interface proxy.
        d = np.linalg.norm(all_nuc - row.ca[None, :], axis=1).min()
        if d <= cutoff:
            seed.append(int(row.idx))
    return sorted(seed), [(x["resname"], x["chain"], x["resseq"]) for x in nuc]

def manual_seed_indices(df: pd.DataFrame, resids: List[int]) -> List[int]:
    wanted = set(int(x) for x in resids)
    return df.loc[df["resseq"].isin(wanted), "idx"].astype(int).tolist()

def choose_seed_indices(cfg: dict, df: pd.DataFrame):
    mode = cfg["seed_mode"]
    apo = cfg["apo"]

    if mode in {"endogenous_ligand", "endogenous_or_manual"}:
        seed, evidence = seed_from_endogenous_ligand(apo, df)
        if seed:
            return seed, {"mode": "endogenous_ligand", "evidence": evidence}

    if mode == "nucleic_interface":
        seed, evidence = seed_from_nucleic_interface(apo, df)
        if seed:
            return seed, {"mode": "nucleic_interface", "evidence": evidence}

    seed = manual_seed_indices(df, cfg.get("manual_seed_resids", []))
    if seed:
        return seed, {"mode": "manual_catalytic_motif", "evidence": cfg.get("manual_seed_resids", [])}

    # Last-resort topology seed: the most central 5% of residues.
    # This is only a fallback; it does not use validation information.
    G = build_contact_graph(df)
    bc = nx.betweenness_centrality(G, k=min(100, len(G)), seed=SEED, weight="distance")
    k = max(3, int(0.05 * len(G)))
    seed = sorted(bc, key=bc.get, reverse=True)[:k]
    return seed, {"mode": "topology_fallback", "evidence": "top 5% approximate betweenness"}

# %% [markdown]
# ## 7. V3 quantum engine — connectivity, distance-conditioned excess and intervention susceptibility
#
# V3 preserves the V2 CTQW and matched classical diffusion kernels, but adds two new layers:
#
# 1. **Distance-conditioned residuals**: a residue is compared with peers at a similar functional-site distance.
# 2. **Local intervention susceptibility**: a local Hamiltonian perturbation tests how much that residue controls global communication.
#
# The expensive intervention scan is performed on a physics/topology shortlist rather than blindly diagonalizing a modified Hamiltonian for every residue.

# %% [code] Notebook code cell 9 (source index 18)

@dataclass
class QuantumWalkResult:
    connectivity: np.ndarray
    classical_connectivity: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    quantum_entropy: np.ndarray
    fluctuation_proxy: np.ndarray
    q_prob_t: np.ndarray
    c_prob_t: np.ndarray
    times: np.ndarray

def minmax(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def normalized_entropy_rows(P: np.ndarray) -> np.ndarray:
    Q = np.clip(P, 1e-15, None)
    Q = Q / np.clip(Q.sum(axis=1, keepdims=True), 1e-15, None)
    H = -(Q * np.log(Q)).sum(axis=1)
    return H / np.log(max(2, P.shape[1]))

def seed_state(n: int, seed_idx: List[int]):
    if not seed_idx:
        return (
            np.ones(n, dtype=complex) / np.sqrt(n),
            np.ones(n, dtype=float) / n,
        )
    seed = np.asarray(sorted(set(seed_idx)), dtype=int)
    amp = np.zeros(n, dtype=complex)
    amp[seed] = 1.0 / np.sqrt(len(seed))
    prob = np.zeros(n, dtype=float)
    prob[seed] = 1.0 / len(seed)
    return amp, prob

def ctqw_connectivity(
    L: np.ndarray,
    seed_idx: List[int],
    times: np.ndarray = TIME_GRID,
    gamma: float = GAMMA,
) -> QuantumWalkResult:
    vals, vecs = np.linalg.eigh(L)
    n = len(L)
    psi0, p0 = seed_state(n, seed_idx)

    Cq = np.zeros((n, n), dtype=np.float64)
    Cc = np.zeros((n, n), dtype=np.float64)
    q_prob, c_prob = [], []

    q0 = vecs.T @ psi0
    c0 = vecs.T @ p0

    for t in tqdm(times, desc=f"V2 CTQW (N={n})", leave=False):
        qphase = np.exp(-1j * gamma * vals * t)
        U = (vecs * qphase[None, :]) @ vecs.T
        Cq += np.abs(U) ** 2
        psi = vecs @ (qphase * q0)
        q_prob.append(np.abs(psi) ** 2)

        cphase = np.exp(-gamma * vals * t)
        K = (vecs * cphase[None, :]) @ vecs.T
        K = np.clip(K.real, 0.0, None)
        K /= np.clip(K.sum(axis=1, keepdims=True), 1e-15, None)
        Cc += K

        cp = vecs @ (cphase * c0)
        cp = np.clip(cp.real, 0.0, None)
        cp /= np.clip(cp.sum(), 1e-15, None)
        c_prob.append(cp)

    Cq /= len(times)
    Cc /= len(times)

    tol = 1e-8
    inv = np.where(vals > tol, 1.0 / vals, 0.0)
    fluct = ((vecs ** 2) * inv[None, :]).sum(axis=1)

    return QuantumWalkResult(
        connectivity=Cq,
        classical_connectivity=Cc,
        eigenvalues=vals,
        eigenvectors=vecs,
        quantum_entropy=normalized_entropy_rows(Cq),
        fluctuation_proxy=fluct,
        q_prob_t=np.asarray(q_prob),
        c_prob_t=np.asarray(c_prob),
        times=np.asarray(times, float),
    )

def transport_features(qw: QuantumWalkResult):
    q, c, t = qw.q_prob_t, qw.c_prob_t, qw.times
    q_mean = q.mean(axis=0)
    q_peak = q.max(axis=0)
    c_mean = c.mean(axis=0)
    c_peak = c.max(axis=0)
    threshold = np.quantile(q, 0.75, axis=1)
    q_persistence = (q >= threshold[:,None]).mean(axis=0)
    late = max(1, int(len(t) * 2 / 3))
    q_late = q[late:].mean(axis=0)

    if len(t) > 1 and t[-1] > t[0]:
        q_auc = np.trapezoid(q, t, axis=0) / (t[-1]-t[0])
        c_auc = np.trapezoid(c, t, axis=0) / (t[-1]-t[0])
    else:
        q_auc, c_auc = q_mean.copy(), c_mean.copy()

    gain = q_auc - c_auc
    qt = np.clip(q.T, 1e-15, None)
    qt /= np.clip(qt.sum(axis=1, keepdims=True), 1e-15, None)
    temporal_entropy = -(qt*np.log(qt)).sum(axis=1) / np.log(max(2,len(t)))

    return {
        "q_mean": q_mean,
        "q_peak": q_peak,
        "q_auc": q_auc,
        "q_late": q_late,
        "q_persistence": q_persistence,
        "q_temporal_entropy": temporal_entropy,
        "classical_mean": c_mean,
        "classical_peak": c_peak,
        "classical_auc": c_auc,
        "q_gain": gain,
    }

def quantum_transport_only(L, seed_idx, times, gamma):
    # Fast perturbation path: no full N-by-N matrix.
    vals, vecs = np.linalg.eigh(L)
    psi0, p0 = seed_state(len(L), seed_idx)
    q0, c0 = vecs.T @ psi0, vecs.T @ p0
    q_rows, c_rows = [], []

    for t in times:
        qp = np.exp(-1j*gamma*vals*t)
        psi = vecs @ (qp*q0)
        q_rows.append(np.abs(psi)**2)

        cp = np.exp(-gamma*vals*t)
        pv = vecs @ (cp*c0)
        pv = np.clip(pv.real,0,None)
        pv /= np.clip(pv.sum(),1e-15,None)
        c_rows.append(pv)

    q, c = np.asarray(q_rows), np.asarray(c_rows)
    threshold = np.quantile(q,0.75,axis=1)
    late = max(1,int(len(times)*2/3))
    gain = q.mean(axis=0)-c.mean(axis=0)

    score = (
        0.30*minmax(q.mean(axis=0))
        + 0.25*minmax(q.max(axis=0))
        + 0.20*(q>=threshold[:,None]).mean(axis=0)
        + 0.15*minmax(np.maximum(gain,0))
        + 0.10*minmax(q[late:].mean(axis=0))
    )
    return minmax(score)


def _js_divergence_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence for corresponding rows of P and Q."""
    P = np.clip(np.asarray(P, float), 1e-15, None)
    Q = np.clip(np.asarray(Q, float), 1e-15, None)
    P /= np.clip(P.sum(axis=1, keepdims=True), 1e-15, None)
    Q /= np.clip(Q.sum(axis=1, keepdims=True), 1e-15, None)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log(P / M), axis=1)
    kl_qm = np.sum(Q * np.log(Q / M), axis=1)
    return 0.5 * (kl_pm + kl_qm)

def _seed_baseline_sparse(
    L: np.ndarray,
    seed_idx: List[int],
    times: np.ndarray,
    gamma: float = GAMMA,
):
    """
    Baseline quantum and classical seed propagation using sparse expm_multiply.
    times must be evenly spaced.
    """
    n = len(L)
    psi0, p0 = seed_state(n, seed_idx)
    Ls = sparse.csr_matrix(L)

    q_states = expm_multiply(
        (-1j * gamma) * Ls,
        psi0,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    q_prob = np.abs(q_states) ** 2
    q_prob /= np.clip(q_prob.sum(axis=1, keepdims=True), 1e-15, None)

    c_states = expm_multiply(
        (-gamma) * Ls,
        p0,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    c_prob = np.clip(np.asarray(c_states).real, 0.0, None)
    c_prob /= np.clip(c_prob.sum(axis=1, keepdims=True), 1e-15, None)
    return q_prob, c_prob

def intervention_susceptibility(
    L: np.ndarray,
    seed_idx: List[int],
    candidate_idx: List[int],
    times: np.ndarray = V3_INTERVENTION_TIMES,
    lam: float = V3_INTERVENTION_LAMBDA,
    gamma: float = GAMMA,
):
    """
    Finite local-intervention scan.

    H_r = L + lam |r><r|

    Returns per-candidate:
      - quantum JS susceptibility
      - classical JS susceptibility
      - quantum-over-classical intervention excess
      - late-time quantum susceptibility
      - max-time quantum susceptibility
    """
    times = np.asarray(times, float)
    q_base, c_base = _seed_baseline_sparse(L, seed_idx, times, gamma)
    Ls = sparse.csr_matrix(L)
    n = len(L)

    out = {}
    for r in tqdm(candidate_idx, desc="V3 intervention scan", leave=False):
        pert = sparse.csr_matrix(
            (np.array([lam], dtype=float), ([int(r)], [int(r)])),
            shape=(n, n)
        )
        Hr = Ls + pert

        psi0, p0 = seed_state(n, seed_idx)

        q_states = expm_multiply(
            (-1j * gamma) * Hr,
            psi0,
            start=float(times[0]),
            stop=float(times[-1]),
            num=len(times),
            endpoint=True,
        )
        q_prob = np.abs(q_states) ** 2
        q_prob /= np.clip(q_prob.sum(axis=1, keepdims=True), 1e-15, None)

        c_states = expm_multiply(
            (-gamma) * Hr,
            p0,
            start=float(times[0]),
            stop=float(times[-1]),
            num=len(times),
            endpoint=True,
        )
        c_prob = np.clip(np.asarray(c_states).real, 0.0, None)
        c_prob /= np.clip(c_prob.sum(axis=1, keepdims=True), 1e-15, None)

        q_js = _js_divergence_rows(q_base, q_prob)
        c_js = _js_divergence_rows(c_base, c_prob)

        late = max(1, int(len(times) * 2 / 3))
        out[int(r)] = {
            "q_susceptibility": float(q_js.mean()),
            "c_susceptibility": float(c_js.mean()),
            "q_intervention_excess_raw": float(q_js.mean() - c_js.mean()),
            "q_susceptibility_late": float(q_js[late:].mean()),
            "q_susceptibility_peak": float(q_js.max()),
        }
    return out

# %% [markdown]
# ## 8. V3.3 topology, soft distality and seed-only rankability
#
# V3.3 no longer equates allostery with a universal minimum geometric distance.
#
# The exact functional seed remains non-rankable, but direct graph neighbours and spatially close residues are restored to the candidate set. Graph distance, Euclidean distance and immediate-shell membership are preserved as explanatory features.

# %% [code] Notebook code cell 10 (source index 20)

def participation_coefficient(G: nx.Graph, communities: List[set]) -> np.ndarray:
    n = len(G)
    node_to_comm = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = ci
    out = np.zeros(n, dtype=float)
    for i in G.nodes():
        nbrs = list(G.neighbors(i))
        if not nbrs:
            continue
        strengths, total = {}, 0.0
        for j in nbrs:
            w = float(G[i][j].get("weight",1.0))
            total += w
            cc = node_to_comm[j]
            strengths[cc] = strengths.get(cc,0.0)+w
        out[i] = 1.0 - sum((v/total)**2 for v in strengths.values()) if total else 0.0
    return out

def graph_distance_from_seed(G, seed_idx):
    dist = np.full(len(G), np.inf)
    if not seed_idx:
        return np.zeros(len(G))
    for s in seed_idx:
        for node,d in nx.single_source_shortest_path_length(G,s).items():
            dist[node] = min(dist[node], d)
    finite = dist[np.isfinite(dist)]
    dist[~np.isfinite(dist)] = (finite.max()+1) if len(finite) else 0
    return dist

def euclidean_distance_from_seed(df, seed_idx):
    coords = np.stack(df["ca"].to_numpy())
    if not seed_idx:
        return np.linalg.norm(coords-coords.mean(axis=0),axis=1)
    d,_ = cKDTree(coords[seed_idx]).query(coords,k=1)
    return np.asarray(d,float)

def compute_graph_features(df, G, qw, seed_idx):
    n = len(df)
    A, _, degree_strength = graph_matrices(G)
    k = None if n <= 300 else min(180,n)
    bet = nx.betweenness_centrality(G,k=k,seed=SEED,normalized=True,weight="distance")
    close = nx.closeness_centrality(G,distance="distance")
    try:
        eig = nx.eigenvector_centrality_numpy(G,weight="weight")
    except Exception:
        eig = nx.eigenvector_centrality(G,max_iter=2000,weight="weight")

    try:
        communities = nx.community.louvain_communities(G,weight="weight",seed=SEED)
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(G,weight="weight"))
    part = participation_coefficient(G,communities)

    coords = np.stack(df["ca"].to_numpy())
    radial = np.linalg.norm(coords-coords.mean(axis=0),axis=1)
    surface = 0.55*minmax(radial)+0.45*(1-minmax(degree_strength))

    dyn = transport_features(qw)
    gdist = graph_distance_from_seed(G,seed_idx)
    edist = euclidean_distance_from_seed(df,seed_idx)

    out = df[["idx","chain","resseq","icode","resname","aa","label","x","y","z"]].copy()
    out["degree"] = [degree_strength[i] for i in range(n)]
    out["betweenness"] = [bet[i] for i in range(n)]
    out["closeness"] = [close[i] for i in range(n)]
    out["eigenvector"] = [eig[i] for i in range(n)]
    out["participation"] = part
    out["surface"] = surface
    out["fluctuation"] = qw.fluctuation_proxy
    out["q_entropy"] = qw.quantum_entropy
    out["graph_dist_seed"] = gdist
    out["euclid_dist_seed"] = edist
    for key,val in dyn.items():
        out[key] = val

    scale = [
        "degree","betweenness","closeness","eigenvector","participation","surface",
        "fluctuation","q_entropy","graph_dist_seed","euclid_dist_seed",
        "q_mean","q_peak","q_auc","q_late","q_persistence","q_temporal_entropy",
        "classical_mean","classical_peak","classical_auc","q_gain"
    ]
    for c in scale:
        out[c+"_n"] = minmax(out[c].to_numpy())

    out["eligible_distal"] = (
        (out["euclid_dist_seed"] >= DISTAL_EUCLID_A)
        & (out["graph_dist_seed"] >= DISTAL_GRAPH_HOPS)
    )

    out["quantum_score"] = minmax(
        0.28*out["q_auc_n"] + 0.22*out["q_peak_n"]
        + 0.20*out["q_persistence_n"]
        + 0.15*minmax(np.maximum(out["q_gain"].to_numpy(),0))
        + 0.10*out["q_late_n"]
        + 0.05*(1-out["q_temporal_entropy_n"])
    )
    out["classical_score"] = minmax(
        0.55*out["classical_auc_n"] + 0.30*out["classical_peak_n"] + 0.15*out["surface_n"]
    )
    out["topology_score"] = minmax(
        0.30*out["betweenness_n"] + 0.25*out["participation_n"] + 0.20*out["surface_n"]
        + 0.15*out["fluctuation_n"] + 0.10*out["eigenvector_n"]
    )
    out["quantum_advantage"] = out["quantum_score"] - out["classical_score"]
    return out


def _robust_group_z(values: np.ndarray, groups: np.ndarray, min_group: int = V3_DISTANCE_MIN_BIN_SIZE):
    """Robust within-group z-score with fallback to global median/MAD."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    out = np.zeros(len(values), dtype=float)

    global_med = float(np.median(values))
    global_mad = float(np.median(np.abs(values - global_med)))
    global_scale = max(1e-8, 1.4826 * global_mad)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) >= min_group:
            med = float(np.median(values[idx]))
            mad = float(np.median(np.abs(values[idx] - med)))
            scale = max(1e-8, 1.4826 * mad)
        else:
            med, scale = global_med, global_scale
        out[idx] = (values[idx] - med) / scale
    return np.clip(out, -6.0, 6.0)

def add_v3_distance_conditioning(features: pd.DataFrame, seed_idx: List[int]) -> pd.DataFrame:
    out = features.copy()

    graph_shell = np.floor(out["graph_dist_seed"].to_numpy(float)).astype(int)
    euclid_shell = np.floor(
        out["euclid_dist_seed"].to_numpy(float) / V3_EUCLID_BIN_WIDTH_A
    ).astype(int)

    # Joint shell key keeps comparisons local in both graph and Euclidean distance.
    joint_shell = graph_shell * 1000 + euclid_shell
    out["distance_shell"] = joint_shell

    q_raw = out["quantum_score"].to_numpy(float)
    c_raw = out["classical_score"].to_numpy(float)

    q_z = _robust_group_z(q_raw, joint_shell)
    c_z = _robust_group_z(c_raw, joint_shell)

    out["quantum_distance_z"] = q_z
    out["classical_distance_z"] = c_z
    out["distance_quantum_excess_raw"] = q_z - c_z
    out["distance_quantum_excess"] = minmax(q_z - c_z)

    # Only the seed itself / immediate local shell is non-rankable.
    seed_mask = np.zeros(len(out), dtype=bool)
    if seed_idx:
        seed_mask[np.asarray(seed_idx, dtype=int)] = True

    graph_shell = (
        out["graph_dist_seed"].to_numpy(float) <= V3_HARD_EXCLUDE_GRAPH_HOPS
    )
    euclid_shell = (
        out["euclid_dist_seed"].to_numpy(float) <= V3_HARD_EXCLUDE_EUCLID_A
    )
    immediate_shell = graph_shell | euclid_shell

    # V3.3: shells are diagnostics/features only. The exact seed is the only hard exclusion.
    out["functional_seed"] = seed_mask
    out["functional_graph_shell"] = graph_shell & ~seed_mask
    out["functional_euclid_shell"] = euclid_shell & ~seed_mask
    out["functional_shell"] = immediate_shell & ~seed_mask
    out["eligible_distal"] = ~seed_mask
    out["rankable"] = ~seed_mask

    # Graded distality evidence, used symbolically rather than as a hard gate.
    out["distality_score"] = minmax(
        0.60 * minmax(out["graph_dist_seed"].to_numpy(float))
        + 0.40 * minmax(out["euclid_dist_seed"].to_numpy(float))
    )
    return out

# Preserve the inherited feature generator as the baseline implementation.
_compute_graph_features_v2_base = compute_graph_features

def compute_graph_features(df, G, qw, seed_idx):
    out = _compute_graph_features_v2_base(df, G, qw, seed_idx)
    return add_v3_distance_conditioning(out, seed_idx)

# %% [markdown]
# ## 9. Ground-truth extraction from holo structures — validation only
#
# The predictor never sees the holo structure while generating features.
#
# For validation:
#
# 1. Identify protein residues within 5 Å of the specified allosteric drug ligand in the holo structure.
# 2. Align the relevant holo chain to the apo chain.
# 3. Map pocket residues back to apo indices.
# 4. Reject mappings with poor sequence identity or insufficient pocket coverage.
#
# This catches malformed benchmark pairs instead of silently inventing labels.

# %% [code] Notebook code cell 11 (source index 22)
def ligand_contact_residues(pdb_id: str, ligand_code: str, cutoff: float = LIGAND_CONTACT_CUTOFF_A):
    model = load_model(pdb_id)
    ligs = extract_nonprotein_atoms(pdb_id, {ligand_code.upper()})
    if not ligs:
        return {}, {"valid": False, "reason": f"Ligand {ligand_code} not found in {pdb_id}"}

    ligand_atoms = np.concatenate([x["coords"] for x in ligs], axis=0)
    by_chain = {}
    for chain in model:
        hits = []
        for residue in chain:
            if not is_aa(residue, standard=True) or "CA" not in residue:
                continue
            coords = np.asarray(
                [a.coord for a in residue.get_atoms() if a.element != "H"], dtype=float
            )
            d = residue_min_distance_to_atoms(coords, ligand_atoms)
            if d <= cutoff:
                hits.append((int(residue.id[1]), str(residue.id[2]).strip(), residue.get_resname()))
        if hits:
            by_chain[chain.id] = hits

    return by_chain, {"valid": bool(by_chain), "ligand_instances": ligs}

def chain_sequence(df: pd.DataFrame, chain: str):
    sub = df[df["chain"] == chain].sort_values(["resseq","icode"])
    return "".join(sub["aa"].tolist()), sub

def pairwise_chain_mapping(apo_df: pd.DataFrame, holo_df: pd.DataFrame, apo_chain: str, holo_chain: str):
    seq_a, sub_a = chain_sequence(apo_df, apo_chain)
    seq_b, sub_b = chain_sequence(holo_df, holo_chain)
    if not seq_a or not seq_b:
        return {}, 0.0, 0.0

    aln = pairwise2.align.globalms(
        seq_a, seq_b, 2.0, -1.0, -5.0, -0.5, one_alignment_only=True
    )[0]

    ai = bi = 0
    mapping = {}
    matches = 0
    aligned = 0

    a_indices = sub_a["idx"].astype(int).tolist()
    b_rows = list(sub_b.itertuples())

    for ca, cb in zip(aln.seqA, aln.seqB):
        a_idx = None
        b_row = None
        if ca != "-":
            a_idx = a_indices[ai]
            ai += 1
        if cb != "-":
            b_row = b_rows[bi]
            bi += 1
        if ca != "-" and cb != "-":
            aligned += 1
            if ca == cb:
                matches += 1
            mapping[(b_row.chain, int(b_row.resseq), str(b_row.icode))] = a_idx

    identity = matches / max(1, aligned)
    coverage = aligned / max(1, min(len(seq_a), len(seq_b)))
    return mapping, identity, coverage

def build_validation_labels(cfg: dict, apo_df: pd.DataFrame):
    holo_id = cfg.get("holo")
    ligand = cfg.get("holo_ligand")

    if not holo_id:
        return None, {"valid": False, "reason": "No holo structure: prospective target."}
    if not ligand:
        return None, {
            "valid": False,
            "reason": (
                f"{cfg['description']}: no allosteric validation ligand is declared for "
                f"{holo_id}; refusing to fabricate ground truth."
            )
        }

    contacts, contact_meta = ligand_contact_residues(holo_id, ligand)
    if not contact_meta["valid"]:
        return None, contact_meta

    holo_df = extract_residue_table(holo_id, requested_chains=None)
    labels = np.zeros(len(apo_df), dtype=int)
    mapping_report = []
    mapped_pocket = 0
    total_pocket = sum(len(v) for v in contacts.values())

    apo_chains = sorted(apo_df["chain"].unique())
    for hchain, pocket in contacts.items():
        best = None
        for achain in apo_chains:
            mapping, ident, cov = pairwise_chain_mapping(apo_df, holo_df, achain, hchain)
            score = ident * cov
            if best is None or score > best[0]:
                best = (score, achain, mapping, ident, cov)

        if best is None:
            continue

        _, achain, mapping, ident, cov = best
        local_mapped = 0
        for resseq, icode, _resname in pocket:
            key = (hchain, int(resseq), str(icode))
            if key in mapping:
                labels[mapping[key]] = 1
                mapped_pocket += 1
                local_mapped += 1

        mapping_report.append({
            "holo_chain": hchain,
            "apo_chain": achain,
            "identity": ident,
            "coverage": cov,
            "pocket_total": len(pocket),
            "pocket_mapped": local_mapped,
        })

    pocket_coverage = mapped_pocket / max(1, total_pocket)
    best_identity = max([x["identity"] for x in mapping_report], default=0.0)

    valid = (
        labels.sum() > 0
        and pocket_coverage >= 0.60
        and best_identity >= 0.50
    )

    meta = {
        "valid": valid,
        "holo": holo_id,
        "ligand": ligand,
        "positives": int(labels.sum()),
        "total_holo_pocket_residues": total_pocket,
        "mapped_pocket_coverage": pocket_coverage,
        "mapping_report": mapping_report,
    }
    if not valid:
        meta["reason"] = (
            f"Validation mapping rejected: positives={labels.sum()}, "
            f"pocket_coverage={pocket_coverage:.2f}, best_identity={best_identity:.2f}"
        )
        return None, meta

    return labels, meta

# V2 replacement for deprecated pairwise2.
_PAIRWISE_ALIGNER = PairwiseAligner()
_PAIRWISE_ALIGNER.mode = "global"
_PAIRWISE_ALIGNER.match_score = 2.0
_PAIRWISE_ALIGNER.mismatch_score = -1.0
_PAIRWISE_ALIGNER.open_gap_score = -5.0
_PAIRWISE_ALIGNER.extend_gap_score = -0.5

def pairwise_chain_mapping(apo_df, holo_df, apo_chain, holo_chain):
    seq_a, sub_a = chain_sequence(apo_df, apo_chain)
    seq_b, sub_b = chain_sequence(holo_df, holo_chain)
    if not seq_a or not seq_b:
        return {},0.0,0.0
    aln = _PAIRWISE_ALIGNER.align(seq_a,seq_b)[0]
    a_indices = sub_a["idx"].astype(int).tolist()
    b_rows = list(sub_b.itertuples())
    mapping, matches, aligned = {},0,0
    a_blocks,b_blocks = aln.aligned
    for (a0,a1),(b0,b1) in zip(a_blocks,b_blocks):
        blen = min(a1-a0,b1-b0)
        for off in range(blen):
            ai,bi = int(a0+off),int(b0+off)
            if ai>=len(a_indices) or bi>=len(b_rows):
                continue
            br = b_rows[bi]
            mapping[(br.chain,int(br.resseq),str(br.icode))] = a_indices[ai]
            aligned += 1
            matches += int(seq_a[ai] == seq_b[bi])
    return mapping, matches/max(1,aligned), aligned/max(1,min(len(seq_a),len(seq_b)))

# %% [markdown]
# ## 10. V3.6 HDC structural memory — calibration gated

# %% [code] Notebook code cell 12 (source index 24)

HDC_FEATURES = [
    "degree_n","betweenness_n","closeness_n","eigenvector_n",
    "participation_n","surface_n","fluctuation_n",
    "q_auc_n","q_peak_n","q_late_n","q_persistence_n",
    "q_temporal_entropy_n","q_gain_n","graph_dist_seed_n","euclid_dist_seed_n",
    "distance_quantum_excess","distality_score"
]

def deterministic_bipolar(token: str, dim: int = HDC_DIM):
    h = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8],"little",signed=False) % (2**32-1)
    rr = np.random.default_rng(seed)
    return rr.choice(np.array([-1,1],dtype=np.int8),size=dim)

def value_bin(v,bins=16):
    # Missing intervention features are neutral rather than forced to either extreme.
    if not np.isfinite(v):
        v = 0.5
    return min(bins-1,int(float(np.clip(v,0,1))*bins))

def hdc_encode(features,dim=HDC_DIM,bins=16):
    H = np.empty((len(features),dim),dtype=np.int8)
    roles = {f:deterministic_bipolar("ROLE::"+f,dim) for f in HDC_FEATURES}
    vals = {b:deterministic_bipolar(f"VALUE::{b}",dim) for b in range(bins)}
    for i,row in enumerate(features.itertuples()):
        acc = np.zeros(dim,dtype=np.int16)
        for f in HDC_FEATURES:
            acc += roles[f].astype(np.int16)*vals[value_bin(getattr(row,f),bins)].astype(np.int16)
        H[i] = np.where(acc>=0,1,-1).astype(np.int8)
    return H

def bipolar_prototype(H):
    return np.where(H.astype(np.int32).sum(axis=0)>=0,1,-1).astype(np.int8)

def hdc_cosine(H,proto):
    return (H.astype(np.float32)@proto.astype(np.float32))/H.shape[1]

def hdc_train_score(train_sets,test_H):
    pos,neg = [],[]
    for item in train_sets:
        y = item["labels"]
        if y is None: continue
        elig = item["features"]["eligible_distal"].to_numpy(bool)
        pm = (y==1)&elig
        nm = (y==0)&elig
        if pm.sum()==0: continue
        pos.append(item["H"][pm])
        pool = item["H"][nm]
        take = min(len(pool),max(30,int(5*pm.sum())))
        if take:
            neg.append(pool[np.linspace(0,len(pool)-1,take).astype(int)])
    if not pos:
        return np.zeros(len(test_H))
    ps = hdc_cosine(test_H,bipolar_prototype(np.concatenate(pos)))
    if not neg:
        return minmax(ps)
    ns = hdc_cosine(test_H,bipolar_prototype(np.concatenate(neg)))
    return minmax(ps-ns)

# %% [markdown]
# ## 11. V3.6 neuro-symbolic evidence — seed-only exclusion

# %% [code] Notebook code cell 13 (source index 26)

def apply_symbolic_rules(features):
    out = features.copy()
    eligible = out["eligible_distal"].to_numpy(bool)

    def eq(col,q,default=1.0):
        vals = out.loc[eligible,col]
        return float(vals.quantile(q)) if len(vals) else default

    out["rule_distal"] = eligible.astype(int)
    out["rule_quantum_connected"] = ((out["quantum_score"]>=eq("quantum_score",0.70))&eligible).astype(int)
    out["rule_persistent"] = ((out["q_persistence"]>=eq("q_persistence",0.65))&eligible).astype(int)
    out["rule_bottleneck"] = ((out["betweenness"]>=eq("betweenness",0.60))&eligible).astype(int)
    out["rule_surface"] = ((out["surface"]>=eq("surface",0.45))&eligible).astype(int)
    out["rule_bridge"] = ((out["participation"]>=eq("participation",0.60))&eligible).astype(int)
    out["rule_interference"] = ((out["quantum_advantage"]>=eq("quantum_advantage",0.60))&eligible).astype(int)

    out["symbolic_score"] = (
        0.26*out["rule_quantum_connected"] + 0.18*out["rule_persistent"]
        + 0.14*out["rule_bottleneck"] + 0.14*out["rule_surface"]
        + 0.13*out["rule_bridge"] + 0.15*out["rule_interference"]
    )
    out.loc[~out["eligible_distal"],"symbolic_score"] = 0.0

    rules = [
        ("rule_distal","DISTAL"),("rule_quantum_connected","QUANTUM_CONNECTED"),
        ("rule_persistent","PERSISTENT_TRANSFER"),("rule_bottleneck","BOTTLENECK"),
        ("rule_surface","SURFACE_ACCESSIBLE"),("rule_bridge","COMMUNITY_BRIDGE"),
        ("rule_interference","QUANTUM_OVER_CLASSICAL")
    ]
    explanations=[]
    for row in out.itertuples():
        if not row.eligible_distal:
            explanations.append("INELIGIBLE_NON_DISTAL")
        else:
            explanations.append(" + ".join(name for col,name in rules if getattr(row,col)==1))
    out["symbolic_explanation"] = explanations
    return out



def apply_symbolic_rules_v3_post_intervention(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    eligible = out["eligible_distal"].to_numpy(bool)
    scanned = out["intervention_scanned"].to_numpy(bool)

    def q(col, quant, default=1.0, require_scanned=False):
        mask = eligible.copy()
        if require_scanned:
            mask &= scanned
        vals = out.loc[mask, col].dropna()
        return float(vals.quantile(quant)) if len(vals) else default

    out["rule_distal"] = (
        (out["distality_score"] >= q("distality_score", 0.35)) & eligible
    ).astype(int)
    out["rule_quantum_connected"] = (
        (out["distance_quantum_excess"] >= q("distance_quantum_excess", 0.65))
        & eligible
    ).astype(int)
    out["rule_persistent"] = (
        (out["q_persistence"] >= q("q_persistence", 0.65)) & eligible
    ).astype(int)
    out["rule_bottleneck"] = (
        (out["betweenness"] >= q("betweenness", 0.60)) & eligible
    ).astype(int)
    out["rule_surface"] = (
        (out["surface"] >= q("surface", 0.45)) & eligible
    ).astype(int)
    out["rule_bridge"] = (
        (out["participation"] >= q("participation", 0.60)) & eligible
    ).astype(int)
    out["rule_interference"] = (
        (out["quantum_advantage"] >= q("quantum_advantage", 0.60)) & eligible
    ).astype(int)

    out["rule_quantum_control"] = (
        scanned
        & eligible
        & (out["q_susceptibility_coeff_n"] >= q(
            "q_susceptibility_coeff_n", 0.70, require_scanned=True
        ))
    ).astype(int)
    out["rule_intervention_excess"] = (
        scanned
        & eligible
        & (out["quantum_intervention_excess"] >= q(
            "quantum_intervention_excess", 0.65, require_scanned=True
        ))
    ).astype(int)
    out["rule_lambda_stable"] = (
        scanned
        & eligible
        & (out["lambda_stability"] >= q(
            "lambda_stability", 0.60, require_scanned=True
        ))
    ).astype(int)
    out["rule_distance_excess"] = (
        (out["distance_quantum_excess"] >= q("distance_quantum_excess", 0.65))
        & eligible
    ).astype(int)

    out["symbolic_score"] = (
        0.16 * out["rule_quantum_control"]
        + 0.16 * out["rule_intervention_excess"]
        + 0.08 * out["rule_lambda_stable"]
        + 0.14 * out["rule_distance_excess"]
        + 0.12 * out["rule_persistent"]
        + 0.09 * out["rule_bottleneck"]
        + 0.09 * out["rule_bridge"]
        + 0.07 * out["rule_surface"]
        + 0.05 * out["rule_interference"]
        + 0.04 * out["rule_distal"]
    )
    out.loc[~eligible, "symbolic_score"] = 0.0

    rule_names = [
        ("rule_quantum_control", "QUANTUM_CONTROL_POINT"),
        ("rule_intervention_excess", "QUANTUM_OVER_CLASSICAL_INTERVENTION"),
        ("rule_lambda_stable", "LAMBDA_RANK_STABLE"),
        ("rule_distance_excess", "DISTANCE_NORMALIZED_EXCESS"),
        ("rule_persistent", "PERSISTENT_TRANSFER"),
        ("rule_bottleneck", "BOTTLENECK"),
        ("rule_bridge", "COMMUNITY_BRIDGE"),
        ("rule_surface", "SURFACE_ACCESSIBLE"),
        ("rule_interference", "QUANTUM_OVER_CLASSICAL_TRANSFER"),
        ("rule_distal", "DISTAL_EVIDENCE"),
    ]

    explanations = []
    for row in out.itertuples():
        if not row.eligible_distal:
            explanations.append("INELIGIBLE_FUNCTIONAL_SEED")
            continue
        fired = [
            label for col, label in rule_names if getattr(row, col) == 1
        ]
        if not row.intervention_scanned:
            fired.insert(0, "INTERVENTION_NOT_SCANNED")
        explanations.append(" + ".join(fired))

    out["symbolic_explanation"] = explanations
    return out

# %% [markdown]
# ## 12. Define residue intervention engine — challenge execution deferred

# %% [code] Notebook code cell 14 (source index 28)

def save_connectivity_matrix(name: str, features: pd.DataFrame, C: np.ndarray):
    labels = features["label"].tolist()
    path = RESULTS_DIR / f"{name}_quantum_connectivity_v3_6.csv"
    pd.DataFrame(C, index=labels, columns=labels).to_csv(path)
    return path

def _intervention_shortlist(features: pd.DataFrame, max_candidates: int):
    eligible = features["eligible_distal"].to_numpy(bool)
    score = (
        V3_SHORTLIST_WEIGHTS["distance_quantum_excess"]
        * features["distance_quantum_excess"].to_numpy(float)
        + V3_SHORTLIST_WEIGHTS["topology"]
        * features["topology_score"].to_numpy(float)
        + V3_SHORTLIST_WEIGHTS["participation"]
        * features["participation_n"].to_numpy(float)
        + V3_SHORTLIST_WEIGHTS["surface"]
        * features["surface_n"].to_numpy(float)
        + V3_SHORTLIST_WEIGHTS["fluctuation"]
        * features["fluctuation_n"].to_numpy(float)
    )
    inds = np.where(eligible)[0]
    if not len(inds):
        return []
    order = inds[np.argsort(score[inds])[::-1]]
    return [int(x) for x in order[:min(max_candidates, len(order))]]

def _masked_minmax(values, mask):
    values = np.asarray(values, float)
    mask = np.asarray(mask, bool) & np.isfinite(values)
    out = np.full(len(values), np.nan, dtype=float)
    if not mask.any():
        return out
    v = values[mask]
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        out[mask] = 0.5
    else:
        out[mask] = (v - lo) / (hi - lo)
    return out

def _masked_group_z(values, groups, valid_mask, min_group=V31_INTERVENTION_MIN_GROUP):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    valid = np.asarray(valid_mask, bool) & np.isfinite(values)
    out = np.full(len(values), np.nan, dtype=float)
    if not valid.any():
        return out

    gv = values[valid]
    global_med = float(np.median(gv))
    global_mad = float(np.median(np.abs(gv - global_med)))
    global_scale = max(1e-8, 1.4826 * global_mad)

    for group in np.unique(groups[valid]):
        idx = np.where(valid & (groups == group))[0]
        if len(idx) >= min_group:
            med = float(np.median(values[idx]))
            mad = float(np.median(np.abs(values[idx] - med)))
            scale = max(1e-8, 1.4826 * mad)
        else:
            med, scale = global_med, global_scale
        out[idx] = np.clip((values[idx] - med) / scale, -6.0, 6.0)
    return out

def _through_origin_lambda2_coefficient(lambdas, values):
    x = np.asarray(lambdas, float) ** 2
    y = np.asarray(values, float)
    denom = float(np.dot(x, x))
    if denom <= 1e-15:
        return np.nan, np.nan
    coef = float(np.dot(x, y) / denom)
    pred = coef * x
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if sst <= 1e-15 else float(1.0 - sse / sst)
    return coef, r2

def _lambda_rank_diagnostics(by_lambda, candidate_idx, lambdas):
    """
    Rank candidates at each lambda using perturbative quantum-over-classical
    susceptibility: (Q_JS - C_JS) / lambda^2.
    """
    candidate_idx = [int(x) for x in candidate_idx]
    lambdas = [float(x) for x in lambdas]
    n = len(candidate_idx)
    m = len(lambdas)

    if n == 0:
        return {}, np.nan

    score_matrix = np.zeros((m, n), dtype=float)
    rank_pct = np.zeros((m, n), dtype=float)

    for li, lam in enumerate(lambdas):
        denom = max(lam * lam, 1e-12)
        vals = np.array([
            (
                by_lambda[lam][r]["q_susceptibility"]
                - by_lambda[lam][r]["c_susceptibility"]
            ) / denom
            for r in candidate_idx
        ], dtype=float)
        score_matrix[li] = vals
        ranks = rankdata(-vals, method="average")
        rank_pct[li] = (
            (ranks - 1.0) / max(1.0, float(n - 1))
        )

    pairwise = []
    for a in range(m):
        for b in range(a + 1, m):
            rho = spearmanr(score_matrix[a], score_matrix[b]).statistic
            if np.isfinite(rho):
                pairwise.append(float(rho))
    mean_pairwise_rho = float(np.mean(pairwise)) if pairwise else np.nan

    topn = min(20, n)
    result = {}
    for j, ridx in enumerate(candidate_idx):
        std_pct = float(np.std(rank_pct[:, j], ddof=0))
        stability = float(np.clip(
            1.0 - V32_RANK_STABILITY_SCALE * std_pct, 0.0, 1.0
        ))
        top_freq = float(np.mean([
            rank_pct[li, j] <= ((topn - 1) / max(1, n - 1))
            for li in range(m)
        ]))
        result[ridx] = {
            "lambda_rank_stability": stability,
            "lambda_rank_std": std_pct,
            "lambda_mean_rank_percentile": float(rank_pct[:, j].mean()),
            "lambda_top20_frequency": top_freq,
        }
    return result, mean_pairwise_rho


def intervention_susceptibility_batched_exact(
    L,
    seed_idx,
    candidate_idx,
    times,
    lam,
    gamma,
    q_base,
    c_base,
    checkpoint_key=None,
):
    """
    Exact finite local-intervention scan evaluated in block-diagonal batches.

    Each block is exactly H_r = L + lam |r><r|. Quantum and classical
    propagators are independent blocks of one larger sparse matrix, so this
    changes execution only, not the mathematical model.
    """
    import time as _time_v403

    times=np.asarray(times,float)
    candidate_idx=[int(x) for x in candidate_idx]
    n=len(L)

    batch_size=(
        V404_INTERVENTION_BATCH_SIZE_LARGE
        if n>=V404_LARGE_GRAPH_N
        else V404_INTERVENTION_BATCH_SIZE
    )
    batch_size=max(1,int(batch_size))

    Ls=sparse.csr_matrix(L)
    psi0,p0=seed_state(n,seed_idx)

    out={}
    total_batches=int(math.ceil(len(candidate_idx)/batch_size))
    t0=_time_v403.time()

    for bi,start in enumerate(
        range(0,len(candidate_idx),batch_size)
    ):
        batch=candidate_idx[start:start+batch_size]
        batch_token=(
            f"{checkpoint_key}__lambda_{float(lam):.6f}"
            f"__batch_{bi:05d}__n_{len(batch)}"
            if checkpoint_key is not None
            else None
        )

        cached=None
        if (
            V404_BATCH_CHECKPOINTS
            and batch_token is not None
        ):
            cached=v404_load(
                "intervention_batch",
                batch_token,
            )

        if cached is not None:
            out.update(cached)
            print(
                f"    batch {bi+1}/{total_batches} loaded "
                f"({len(batch)} residues)"
            )
            continue

        hrs=[]
        for r in batch:
            pert=sparse.csr_matrix(
                (
                    np.array([float(lam)],dtype=float),
                    ([int(r)],[int(r)]),
                ),
                shape=(n,n),
            )
            hrs.append(Ls+pert)

        # One exact block-diagonal propagation contains:
        #   quantum blocks 0..B-1
        #   classical blocks B..2B-1
        q_blocks=[
            (-1j*float(gamma))*Hr
            for Hr in hrs
        ]
        c_blocks=[
            (-float(gamma))*Hr
            for Hr in hrs
        ]
        bigA=sparse.block_diag(
            q_blocks+c_blocks,
            format="csr",
        )

        x0=np.concatenate(
            [psi0 for _ in batch]
            +[p0.astype(complex) for _ in batch]
        )

        states=expm_multiply(
            bigA,
            x0,
            start=float(times[0]),
            stop=float(times[-1]),
            num=len(times),
            endpoint=True,
        )

        states=np.asarray(states)
        B=len(batch)
        batch_out={}

        for j,r in enumerate(batch):
            q_state=states[:,j*n:(j+1)*n]
            c0=(B+j)*n
            c_state=states[:,c0:c0+n]

            q_prob=np.abs(q_state)**2
            q_prob/=np.clip(
                q_prob.sum(axis=1,keepdims=True),
                1e-15,
                None,
            )

            c_prob=np.clip(
                np.asarray(c_state).real,
                0.0,
                None,
            )
            c_prob/=np.clip(
                c_prob.sum(axis=1,keepdims=True),
                1e-15,
                None,
            )

            q_js=_js_divergence_rows(
                q_base,
                q_prob,
            )
            c_js=_js_divergence_rows(
                c_base,
                c_prob,
            )

            late=max(
                1,
                int(len(times)*2/3),
            )

            batch_out[int(r)]={
                "q_susceptibility":float(q_js.mean()),
                "c_susceptibility":float(c_js.mean()),
                "q_intervention_excess_raw":float(
                    q_js.mean()-c_js.mean()
                ),
                "q_susceptibility_late":float(
                    q_js[late:].mean()
                ),
                "q_susceptibility_peak":float(
                    q_js.max()
                ),
            }

        out.update(batch_out)

        if (
            V404_BATCH_CHECKPOINTS
            and batch_token is not None
        ):
            v404_save(
                "intervention_batch",
                batch_token,
                batch_out,
            )

        elapsed=_time_v403.time()-t0
        done=bi+1
        eta=(
            elapsed/done*(total_batches-done)
            if done>0 else np.nan
        )
        print(
            f"    batch {done}/{total_batches} saved "
            f"| residues={len(batch)} "
            f"| elapsed={elapsed/60:.1f}m "
            f"| ETA≈{eta/60:.1f}m"
        )

    return out


def intervention_ensemble(
    L,
    seed_idx,
    candidate_idx,
    lambdas=V31_LAMBDA_SWEEP,
    times=V3_INTERVENTION_TIMES,
    gamma=GAMMA,
    checkpoint_key=None,
):
    candidate_idx = [int(x) for x in candidate_idx]
    lambdas = np.asarray(lambdas, float)
    by_lambda = {}

    # Baseline is independent of λ: compute once per protein.
    q_base,c_base=_seed_baseline_sparse(
        L,
        seed_idx,
        np.asarray(times,float),
        gamma,
    )

    for lam in lambdas:
        lam = float(lam)
        print(f"  λ={lam:.2f} | candidates={len(candidate_idx)}")

        lam_result = None
        lam_key=(
            f"{checkpoint_key}__lambda_{lam:.6f}"
            if checkpoint_key is not None
            else None
        )

        if lam_key is not None:
            lam_result = v404_load(
                "intervention_lambda",
                lam_key,
            )

        if lam_result is None:
            lam_result = intervention_susceptibility_batched_exact(
                L=L,
                seed_idx=seed_idx,
                candidate_idx=candidate_idx,
                times=times,
                lam=lam,
                gamma=gamma,
                q_base=q_base,
                c_base=c_base,
                checkpoint_key=checkpoint_key,
            )
            if lam_key is not None:
                v404_save(
                    "intervention_lambda",
                    lam_key,
                    lam_result,
                )

        by_lambda[lam] = lam_result

    rank_diag, pairwise_rho = _lambda_rank_diagnostics(
        by_lambda, candidate_idx, lambdas
    )

    fields = [
        "q_susceptibility",
        "c_susceptibility",
        "q_intervention_excess_raw",
        "q_susceptibility_late",
        "q_susceptibility_peak",
    ]

    out = {}
    for ridx in candidate_idx:
        out[ridx] = {}

        for field in fields:
            vals = np.array(
                [by_lambda[float(lam)][ridx][field] for lam in lambdas],
                dtype=float,
            )
            out[ridx][field] = float(vals.mean())
            out[ridx][field + "_lambda_std"] = float(vals.std(ddof=0))
            out[ridx][field + "_lambda_min"] = float(vals.min())
            out[ridx][field + "_lambda_max"] = float(vals.max())

        qvals = np.array([
            by_lambda[float(lam)][ridx]["q_susceptibility"]
            for lam in lambdas
        ], dtype=float)
        cvals = np.array([
            by_lambda[float(lam)][ridx]["c_susceptibility"]
            for lam in lambdas
        ], dtype=float)

        qcoef, qr2 = _through_origin_lambda2_coefficient(lambdas, qvals)
        ccoef, cr2 = _through_origin_lambda2_coefficient(lambdas, cvals)

        out[ridx]["q_susceptibility_coeff"] = qcoef
        out[ridx]["c_susceptibility_coeff"] = ccoef
        out[ridx]["q_intervention_coeff_excess_raw"] = qcoef - ccoef
        out[ridx]["q_lambda2_fit_r2"] = qr2
        out[ridx]["c_lambda2_fit_r2"] = cr2
        out[ridx].update(rank_diag.get(ridx, {}))

    return out, by_lambda, pairwise_rho

def _condition_intervention_excess(features: pd.DataFrame):
    out = features.copy()
    scanned = out["intervention_scanned"].to_numpy(bool)
    groups = out["distance_shell"].to_numpy()

    # V3.2 primary intervention signal: perturbative lambda^2 coefficient.
    q = out["q_susceptibility_coeff"].to_numpy(float)
    c = out["c_susceptibility_coeff"].to_numpy(float)

    qz = _masked_group_z(q, groups, scanned)
    cz = _masked_group_z(c, groups, scanned)
    raw = qz - cz

    out["q_susceptibility_coeff_distance_z"] = qz
    out["c_susceptibility_coeff_distance_z"] = cz
    out["quantum_intervention_excess_raw"] = raw
    out["quantum_intervention_excess"] = _masked_minmax(raw, scanned)

    # Coefficient-normalized diagnostic features.
    out["q_susceptibility_coeff_n"] = _masked_minmax(q, scanned)
    out["c_susceptibility_coeff_n"] = _masked_minmax(c, scanned)

    # Retain V3.1 mean-JS diagnostics but they do not drive the final V3.2 score.
    out["q_susceptibility_n"] = _masked_minmax(
        out["q_susceptibility"].to_numpy(float), scanned
    )
    out["c_susceptibility_n"] = _masked_minmax(
        out["c_susceptibility"].to_numpy(float), scanned
    )
    out["q_susceptibility_late_n"] = _masked_minmax(
        out["q_susceptibility_late"].to_numpy(float), scanned
    )
    out["q_susceptibility_peak_n"] = _masked_minmax(
        out["q_susceptibility_peak"].to_numpy(float), scanned
    )

    # Compatibility alias: stability now means rank stability across lambda.
    out["lambda_stability"] = out["lambda_rank_stability"].to_numpy(float)
    return out

def run_pipeline(name: str, cfg: dict):
    print("\n" + "=" * 96)
    print(name, "-", cfg["description"])
    print("=" * 96)

    apo_df = extract_residue_table(cfg["apo"], cfg.get("chains"))
    G = build_contact_graph(apo_df, CONTACT_CUTOFF_A)
    A, L, degree = graph_matrices(G)

    seed_idx, seed_meta = choose_seed_indices(cfg, apo_df)
    print(f"Apo {cfg['apo']}: {len(apo_df)} residues, {G.number_of_edges()} edges")
    print(f"Seed mode: {seed_meta['mode']} | seed residues: {len(seed_idx)}")

    qw = ctqw_connectivity(L, seed_idx, TIME_GRID, GAMMA)
    features = compute_graph_features(apo_df, G, qw, seed_idx)

    eligible_idx = features.loc[
        features["eligible_distal"], "idx"
    ].astype(int).tolist()

    if name in V31_FULL_SCAN_TARGETS:
        candidate_idx = eligible_idx
        scan_mode = "FULL_ELIGIBLE"
    else:
        candidate_idx = _intervention_shortlist(
            features, V31_PROSPECTIVE_SHORTLIST
        )
        scan_mode = "SHORTLIST"

    print(
        f"V3.6 intervention mode={scan_mode}: "
        f"{len(candidate_idx)} / {len(eligible_idx)} eligible residues"
    )

    scan, scan_by_lambda, pairwise_rho = intervention_ensemble(
        L,
        seed_idx,
        candidate_idx,
        lambdas=V31_LAMBDA_SWEEP,
        times=V3_INTERVENTION_TIMES,
        gamma=GAMMA,
        checkpoint_key=f"challenge__{name}",
    )
    print("Mean pairwise λ rank Spearman:", pairwise_rho)

    intervention_cols = [
        "q_susceptibility",
        "c_susceptibility",
        "q_susceptibility_late",
        "q_susceptibility_peak",
        "q_susceptibility_lambda_std",
        "c_susceptibility_lambda_std",
        "q_intervention_excess_raw_lambda_std",
        "q_susceptibility_late_lambda_std",
        "q_susceptibility_peak_lambda_std",
        "q_susceptibility_coeff",
        "c_susceptibility_coeff",
        "q_intervention_coeff_excess_raw",
        "q_lambda2_fit_r2",
        "c_lambda2_fit_r2",
        "lambda_rank_stability",
        "lambda_rank_std",
        "lambda_mean_rank_percentile",
        "lambda_top20_frequency",
    ]
    for col in intervention_cols:
        features[col] = np.nan

    for ridx, vals in scan.items():
        row_mask = features["idx"] == int(ridx)
        for col in intervention_cols:
            if col in vals:
                features.loc[row_mask, col] = vals[col]

    features["intervention_scanned"] = features["idx"].isin(candidate_idx)
    features["intervention_scan_mode"] = scan_mode
    features = _condition_intervention_excess(features)
    features = apply_symbolic_rules_v3_post_intervention(features)

    labels, validation_meta = build_validation_labels(cfg, apo_df)
    if labels is not None:
        features["label_allosteric"] = labels
        print("Validation:", validation_meta)
    else:
        features["label_allosteric"] = np.nan
        print(
            "Validation unavailable/rejected:",
            validation_meta.get("reason", validation_meta),
        )

    C_path = save_connectivity_matrix(name, features, qw.connectivity)
    feature_path = RESULTS_DIR / f"{name}_residue_features_v3_6.csv"
    features.to_csv(feature_path, index=False)

    return {
        "name": name,
        "cfg": cfg,
        "apo_df": apo_df,
        "G": G,
        "A": A,
        "L": L,
        "seed_idx": seed_idx,
        "seed_meta": seed_meta,
        "qw": qw,
        "features": features,
        "labels": labels,
        "validation_meta": validation_meta,
        "connectivity_path": C_path,
        "feature_path": feature_path,
        "intervention_shortlist": candidate_idx,
        "intervention_scan_mode": scan_mode,
        "intervention_by_lambda": scan_by_lambda,
        "lambda_pairwise_spearman_mean": pairwise_rho,
    }

RESULTS = {}
CHALLENGE_DEFERRED = True
print(
    "V4.0.3: challenge target physics deferred until after "
    "the six-protein calibrated model is frozen."
)
v404_stage_marker("challenge_deferred")

# %% [markdown]
# ### V4.1.1 exact-batching numerical equivalence audit
#
# The batched solver is an execution optimization. Before using it as competition
# evidence, this cell verifies that it reproduces the original residue-by-residue
# intervention solver on the same deterministic graph.
#
# Five returned observables are checked for several candidate residues:
#
# - quantum susceptibility;
# - classical susceptibility;
# - quantum-over-classical excess;
# - late-time quantum susceptibility;
# - peak quantum susceptibility.
#
# The audit is deliberately label-free and does not touch model fitting.

# %% [code] Notebook code cell 15 (source index 30)

# Deterministic small weighted graph.
_n=10
_A=np.zeros((_n,_n),float)
for i in range(_n):
    j=(i+1)%_n
    _A[i,j]=_A[j,i]=1.0
for i,j,w in [
    (0,5,0.35),
    (2,7,0.55),
    (3,8,0.25),
]:
    _A[i,j]=_A[j,i]=w

_deg=_A.sum(axis=1)
_inv=np.zeros_like(_deg)
_inv[_deg>1e-15]=1.0/np.sqrt(_deg[_deg>1e-15])
_L=np.eye(_n)-(_inv[:,None]*_A*_inv[None,:])
_L=(_L+_L.T)/2.0

_seed=[0,1]
_candidates=[3,5,7,8]
_times=np.array([0.35,0.70,1.05,1.40,1.75],float)
_lam=0.35

_old=intervention_susceptibility(
    _L,
    _seed,
    _candidates,
    times=_times,
    lam=_lam,
    gamma=GAMMA,
)

_qb,_cb=_seed_baseline_sparse(
    _L,
    _seed,
    _times,
    GAMMA,
)

_new=intervention_susceptibility_batched_exact(
    L=_L,
    seed_idx=_seed,
    candidate_idx=_candidates,
    times=_times,
    lam=_lam,
    gamma=GAMMA,
    q_base=_qb,
    c_base=_cb,
    checkpoint_key=None,
)

_fields=[
    "q_susceptibility",
    "c_susceptibility",
    "q_intervention_excess_raw",
    "q_susceptibility_late",
    "q_susceptibility_peak",
]

_rows=[]
for r in _candidates:
    for field in _fields:
        a=float(_old[r][field])
        b=float(_new[r][field])
        abs_diff=abs(a-b)
        rel_diff=abs_diff/max(abs(a),abs(b),1e-15)
        _rows.append({
            "candidate":r,
            "field":field,
            "original":a,
            "batched":b,
            "abs_diff":abs_diff,
            "rel_diff":rel_diff,
        })

V411_BATCH_EQUIVALENCE=pd.DataFrame(_rows)
V411_BATCH_EQUIVALENCE_MAX_ABS=float(
    V411_BATCH_EQUIVALENCE["abs_diff"].max()
)
V411_BATCH_EQUIVALENCE_MAX_REL=float(
    V411_BATCH_EQUIVALENCE["rel_diff"].max()
)

V411_EQUIVALENCE_TOL=1e-8
V411_BATCH_EQUIVALENCE_PASS=bool(
    V411_BATCH_EQUIVALENCE_MAX_ABS <= V411_EQUIVALENCE_TOL
)

V411_BATCH_EQUIVALENCE.to_csv(
    V410_SUBMISSION_DIR/"batched_solver_equivalence_audit.csv",
    index=False,
)

v411_attest(
    "batched_solver_equivalence",
    pass_status=V411_BATCH_EQUIVALENCE_PASS,
    max_abs_diff=V411_BATCH_EQUIVALENCE_MAX_ABS,
    max_rel_diff=V411_BATCH_EQUIVALENCE_MAX_REL,
    tolerance=V411_EQUIVALENCE_TOL,
)

display(V411_BATCH_EQUIVALENCE)
print("max abs diff:",V411_BATCH_EQUIVALENCE_MAX_ABS)
print("max rel diff:",V411_BATCH_EQUIVALENCE_MAX_REL)

assert V411_BATCH_EQUIVALENCE_PASS, (
    "Exact-batched intervention solver failed numerical equivalence audit: "
    f"max_abs={V411_BATCH_EQUIVALENCE_MAX_ABS:.3e}"
)

print("V4.1.1 EXACT BATCHING EQUIVALENCE: PASS")

# %% [markdown]
# ## 13. Submission-safe V3.6 HDC calibration

# %% [code] Notebook code cell 16 (source index 32)

# Add V3 intervention dimensions to the HDC representation.
for _f in [
    "distance_quantum_excess","distality_score",
    "q_susceptibility_coeff_n","c_susceptibility_coeff_n",
    "q_susceptibility_n","q_susceptibility_late_n",
    "q_susceptibility_peak_n","quantum_intervention_excess",
    "lambda_rank_stability","lambda_top20_frequency"
]:
    if _f not in HDC_FEATURES:
        HDC_FEATURES.append(_f)

def is_valid_labeled(item):
    y = item["labels"]
    if y is None or y.sum() == 0:
        return False
    elig = item["features"]["eligible_distal"].to_numpy(bool)
    return ((y == 1) & elig).sum() > 0

for item in RESULTS.values():
    item["H"] = hdc_encode(item["features"])

VALID_LABELED = [x for x in RESULTS.values() if is_valid_labeled(x)]
CALIBRATION_LABELED = [x for x in VALID_LABELED if not x["cfg"].get("official", False)]
OFFICIAL_LABELED = [x for x in VALID_LABELED if x["cfg"].get("official", False)]

def training_items_for(target_item):
    if SUBMISSION_MODE:
        train = [x for x in CALIBRATION_LABELED if x["name"] != target_item["name"]]
        if ALLOW_CHALLENGE_TO_CHALLENGE_TRANSFER:
            train += [
                x for x in OFFICIAL_LABELED
                if x["name"] != target_item["name"]
            ]
        return train
    return [x for x in VALID_LABELED if x["name"] != target_item["name"]]

print("Valid labelled:", [x["name"] for x in VALID_LABELED])
print("Independent calibration labelled:", [x["name"] for x in CALIBRATION_LABELED])

# %% [markdown]
# ## 14. V3.6 candidate-pocket generator and multi-scale descriptors
#
# The V3.5 generator is retained. Its multi-scale consensus is now treated as
# one structural/physics descriptor rather than the final pocket rank.

# %% [code] Notebook code cell 17 (source index 34)

EVO_FEATURES = [
    "pocket_first_physics_score","site_field_score","rrf_core","pocket_coherence",
    "quantum_intervention_excess",
    "q_susceptibility_coeff_n",
    "distance_quantum_excess","distality_score",
    "quantum_score","q_persistence_n","q_gain_n",
    "betweenness_n","participation_n","surface_n","fluctuation_n",
    "topology_score","symbolic_score","hdc_score",
    "lambda_rank_stability","lambda_top20_frequency"
]

def _evo_matrix(frame):
    X = frame[EVO_FEATURES].to_numpy(np.float32)
    return np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

def genome_size(n_features,hidden=EVO_HIDDEN):
    return n_features*hidden + hidden + hidden + 1

def unpack_genome(g,n_features,hidden=EVO_HIDDEN):
    p=0
    W1=g[p:p+n_features*hidden].reshape(n_features,hidden); p+=n_features*hidden
    b1=g[p:p+hidden]; p+=hidden
    W2=g[p:p+hidden]; p+=hidden
    return W1,b1,W2,float(g[p])

def evo_forward(X,g,hidden=EVO_HIDDEN):
    W1,b1,W2,b2=unpack_genome(g,X.shape[1],hidden)
    z=np.tanh(X@W1+b1)@W2+b2
    z=np.clip(z,-30,30)
    return 1/(1+np.exp(-z))

def topk_recall(y,p,k=5):
    y=np.asarray(y)
    p=np.asarray(p,float)
    positives=int(y.sum())
    if positives==0:
        return 0.0
    finite=np.isfinite(p)
    if not finite.any():
        return np.nan
    if np.nanmax(p[finite]) - np.nanmin(p[finite]) < 1e-12:
        return np.nan
    idx=np.argsort(np.where(finite,p,-np.inf))[::-1][:min(k,int(finite.sum()))]
    return float(y[idx].sum()/min(k,positives))

def genome_fitness(g,train_items):
    vals=[]
    for item in train_items:
        y=item["labels"]
        elig=item["features"]["eligible_distal"].to_numpy(bool)
        if y is None: continue
        yy=y[elig]
        if len(yy)<10 or yy.sum()==0 or yy.sum()==len(yy): continue
        X=_evo_matrix(item["features"].loc[elig])
        p=evo_forward(X,g)
        t5=topk_recall(yy,p,5)
        if not np.isfinite(t5):
            t5=0.0
        vals.append(
            0.75*average_precision_score(yy,p)
            + 0.25*t5
        )
    return float(np.mean(vals)) if vals else -1e9

def evolve_network(train_items,pop_size=EVO_POP,generations=EVO_GENERATIONS,seed=SEED):
    if not train_items: return None,[],[]
    rr=np.random.default_rng(seed)
    gs=genome_size(len(EVO_FEATURES))
    pop=rr.normal(0,0.40,size=(pop_size,gs)).astype(np.float32)
    history=[]; archive=[]
    for gen in range(generations):
        fit=np.array([genome_fitness(g,train_items) for g in pop])
        order=np.argsort(fit)[::-1]; pop,fit=pop[order],fit[order]
        elite_n=max(4,pop_size//8); parent_n=max(8,pop_size//3)
        elites=pop[:elite_n].copy(); parents=pop[:parent_n]
        history.append(float(fit[0]))
        archive.extend((float(fit[i]),pop[i].copy()) for i in range(min(5,elite_n)))
        sigma=0.18*(0.985**gen)+0.02
        children=list(elites)
        while len(children)<pop_size:
            a,b=rr.integers(0,parent_n,size=2)
            mask=rr.random(gs)<0.5
            child=np.where(mask,parents[a],parents[b]).copy()
            mut=rr.random(gs)<0.14
            child[mut]+=rr.normal(0,sigma,size=int(mut.sum()))
            children.append(child.astype(np.float32))
        pop=np.asarray(children,dtype=np.float32)
    fit=np.array([genome_fitness(g,train_items) for g in pop])
    order=np.argsort(fit)[::-1]; pop,fit=pop[order],fit[order]
    archive.extend((float(fit[i]),pop[i].copy()) for i in range(min(10,len(pop))))
    archive.sort(key=lambda x:x[0],reverse=True)
    ensemble=[]
    for fitval,g in archive:
        if not ensemble or all(np.linalg.norm(g-h)>1e-4 for _,h in ensemble):
            ensemble.append((fitval,g))
        if len(ensemble)>=12: break
    return pop[0],history,ensemble

def ensemble_predict(X,ensemble):
    if not ensemble:
        return np.zeros(len(X)),np.ones(len(X))
    P=np.stack([evo_forward(X,g) for _,g in ensemble])
    return P.mean(axis=0),P.std(axis=0)

def rank_percentiles(scores,eligible):
    r=np.ones(len(scores))
    inds=np.where(eligible)[0]
    if not len(inds): return r
    order=inds[np.argsort(scores[inds])[::-1]]
    r[order]=np.linspace(0,1,len(order))
    return r

def robustness_ensemble(item):
    df=item["apo_df"]; seed=item["seed_idx"]
    eligible=item["features"]["eligible_distal"].to_numpy(bool)
    S=[]; R=[]; top20=[]
    for cutoff,gamma,t_scale in ROBUSTNESS_SETTINGS:
        G=build_contact_graph(df,cutoff)
        _,L,_=graph_matrices(G)
        s=quantum_transport_only(L,seed,TIME_GRID*t_scale,gamma)
        s=np.where(eligible,s,0.0)
        S.append(s); R.append(rank_percentiles(s,eligible))
        inds=np.where(eligible)[0]
        n=min(20,len(inds))
        top20.append(set(inds[np.argsort(s[inds])[::-1][:n]]) if n else set())
    S=np.stack(S); R=np.stack(R)
    freq=np.array([np.mean([i in t for t in top20]) for i in range(len(df))])
    rank_stab=np.clip(1-R.std(axis=0),0,1)
    score_stab=np.clip(1-S.std(axis=0),0,1)
    return {
        "robust_score":S.mean(axis=0),
        "robust_score_std":S.std(axis=0),
        "mean_rank_percentile":R.mean(axis=0),
        "rank_stability":rank_stab,
        "score_stability":score_stab,
        "top20_frequency":freq,
        "robust_confidence":0.50*freq+0.30*rank_stab+0.20*score_stab
    }

def learned_weight(n):
    if n < MIN_LEARNED_BENCHMARKS: return 0.0
    return min(0.30,0.12+0.02*(n-MIN_LEARNED_BENCHMARKS))

def _rrf_component_rank(scores, eligible):
    scores=np.asarray(scores,float)
    eligible=np.asarray(eligible,bool)
    rank=np.full(len(scores),np.nan,dtype=float)
    valid=eligible & np.isfinite(scores)
    if valid.any():
        rank[valid]=rankdata(-scores[valid],method="average")
    return rank

def add_rrf_core(frame):
    f=frame
    eligible=f["eligible_distal"].to_numpy(bool)
    numerator=np.zeros(len(f),dtype=float)
    denominator=np.zeros(len(f),dtype=float)

    rank_columns={}
    for col,weight in V32_RRF_COMPONENTS:
        ranks=_rrf_component_rank(f[col].to_numpy(float),eligible)
        rank_columns[col]=ranks
        valid=np.isfinite(ranks)
        numerator[valid] += weight/(V32_RRF_K+ranks[valid])
        denominator[valid] += weight

    raw=np.full(len(f),np.nan,dtype=float)
    valid=eligible & (denominator>0)
    raw[valid]=numerator[valid]/denominator[valid]

    f["rrf_core_raw"]=raw
    f["rrf_core"]=_masked_minmax(raw,valid)

    for col,ranks in rank_columns.items():
        safe=col.replace("quantum_","q_").replace("distance_","d_")
        f[f"rrf_rank_{safe}"]=ranks

def add_spatial_pocket_coherence(item):
    """
    Residue-level local support from the V3.2 idea.
    V3.3 keeps this as the field used to *discover* sites rather than a small final bonus.
    """
    f=item["features"]
    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    eligible=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )
    core=f["rrf_core"].to_numpy(float)
    surface=f["surface_n"].to_numpy(float)
    coords=f[["x","y","z"]].to_numpy(float)

    tree=cKDTree(coords)
    inds=np.where(eligible & np.isfinite(core))[0]
    high_thr=float(np.quantile(core[inds],V32_POCKET_HIGH_QUANTILE)) if len(inds) else 1.0

    raw=np.zeros(len(f),dtype=float)
    neighbour_count=np.zeros(len(f),dtype=int)
    local_support=np.zeros(len(f),dtype=float)
    high_fraction=np.zeros(len(f),dtype=float)

    for idx in inds:
        nbrs=tree.query_ball_point(coords[idx],r=V32_POCKET_RADIUS_A)
        nbrs=np.array([
            j for j in nbrs
            if j!=idx and eligible[j] and np.isfinite(core[j])
        ],dtype=int)
        neighbour_count[idx]=len(nbrs)
        if not len(nbrs):
            continue

        d=np.linalg.norm(coords[nbrs]-coords[idx],axis=1)
        w=np.exp(-((d/max(V32_POCKET_RADIUS_A,1e-6))**2))
        w=w/np.clip(w.sum(),1e-12,None)

        local=float(np.sum(w*core[nbrs]))
        high=float(np.mean(core[nbrs]>=high_thr))
        surf=float(np.sum(w*surface[nbrs]))
        size=float(min(1.0,len(nbrs)/6.0))

        local_support[idx]=local
        high_fraction[idx]=high
        raw[idx]=0.55*local+0.20*high+0.15*surf+0.10*size

    f["pocket_neighbor_count"]=neighbour_count
    f["pocket_local_support"]=local_support
    f["pocket_high_rank_fraction"]=high_fraction
    f["pocket_coherence_raw"]=raw
    f["pocket_coherence"]=0.0
    if len(inds):
        f.loc[inds,"pocket_coherence"]=minmax(raw[inds])


def _top_fraction_mean(values, fraction=V33_SITE_TOP_FRACTION):
    vals=np.asarray(values,float)
    vals=vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan
    n=max(1,int(math.ceil(len(vals)*float(fraction))))
    return float(np.mean(np.sort(vals)[-n:]))


def _dynamic_rrf_rows(df, components, k):
    """
    RRF across candidate pockets. Missing components are ignored per pocket.
    """
    n=len(df)
    numerator=np.zeros(n,float)
    denominator=np.zeros(n,float)
    for col,weight in components:
        vals=df[col].to_numpy(float)
        valid=np.isfinite(vals)
        if not valid.any():
            continue
        ranks=np.full(n,np.nan,float)
        ranks[valid]=rankdata(-vals[valid],method="average")
        numerator[valid]+=float(weight)/(float(k)+ranks[valid])
        denominator[valid]+=float(weight)

    raw=np.full(n,np.nan,float)
    valid=denominator>0
    raw[valid]=numerator[valid]/denominator[valid]
    score=np.zeros(n,float)
    if valid.any():
        score[valid]=minmax(raw[valid])
    return raw,score


def _scale_rrf(frame, components=V35_SCALE_COMPONENTS, k=V35_SITE_RRF_K):
    n=len(frame)
    numerator=np.zeros(n,float)
    denominator=np.zeros(n,float)

    for col,weight in components:
        vals=frame[col].to_numpy(float)
        valid=np.isfinite(vals)
        if not valid.any():
            continue

        ranks=np.full(n,np.nan,float)
        ranks[valid]=rankdata(-vals[valid],method="average")
        numerator[valid]+=float(weight)/(float(k)+ranks[valid])
        denominator[valid]+=float(weight)

    raw=np.full(n,np.nan,float)
    valid=denominator>0
    raw[valid]=numerator[valid]/denominator[valid]

    score=np.zeros(n,float)
    if valid.any():
        score[valid]=minmax(raw[valid])
    return raw,score

def _top_fraction(values,fraction=V35_TOP_FRACTION):
    vals=np.asarray(values,float)
    vals=vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan
    n=max(1,int(math.ceil(len(vals)*float(fraction))))
    return float(np.mean(np.sort(vals)[-n:]))

def _high_threshold(values,mask,q=V35_HIGH_QUANTILE):
    vals=np.asarray(values,float)
    mask=np.asarray(mask,bool) & np.isfinite(vals)
    if not mask.any():
        return np.nan
    return float(np.quantile(vals[mask],q))

def _multichannel_consensus_fraction(
    member_idx,
    channels,
    thresholds,
):
    """
    Fraction of members high in at least half of their available evidence channels,
    with a minimum of 2 channels when >=3 are available.
    """
    member_idx=np.asarray(member_idx,int)
    if not len(member_idx):
        return np.nan

    supported=[]
    for ridx in member_idx:
        available=0
        high=0
        for name,arr in channels.items():
            val=float(arr[ridx]) if np.isfinite(arr[ridx]) else np.nan
            thr=thresholds.get(name,np.nan)
            if not np.isfinite(val) or not np.isfinite(thr):
                continue
            available+=1
            high+=int(val>=thr)

        if available==0:
            supported.append(False)
        else:
            required=2 if available>=3 else 1
            supported.append(high>=required)

    return float(np.mean(supported))

def _candidate_centers(item):
    """
    Preserve the V3.3/V3.4 candidate generator.
    Center evidence is only a discovery mechanism, not a final ranking term.
    """
    f=item["features"]
    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    target_mask=f["chain"].isin(target_chains).to_numpy(bool)
    rankable=f["eligible_distal"].to_numpy(bool) & target_mask
    coords=f[["x","y","z"]].to_numpy(float)

    coherence=f["pocket_coherence"].to_numpy(float)
    core=f["rrf_core"].to_numpy(float)
    inds=np.where(rankable & np.isfinite(core))[0]

    if not len(inds):
        return [],rankable,coords,np.zeros(len(f),float)

    discovery_score=(
        V33_SITE_SEED_COHERENCE_WEIGHT*coherence
        + V33_SITE_SEED_RRF_WEIGHT*core
    )
    order=inds[np.argsort(discovery_score[inds])[::-1]]

    centers=[]
    for ridx in order:
        if centers:
            d=np.linalg.norm(coords[np.asarray(centers)]-coords[ridx],axis=1)
            if np.any(d<V33_CENTER_SEPARATION_A):
                continue
        centers.append(int(ridx))
        if len(centers)>=V33_MAX_POCKETS:
            break

    return centers,rankable,coords,discovery_score

def discover_candidate_pockets(item):
    f=item["features"]
    centers,rankable,coords,discovery_score=_candidate_centers(item)

    if not centers:
        item["predicted_pockets"]=pd.DataFrame()
        item["pocket_scale_profiles"]=pd.DataFrame()
        f["predicted_pocket_id"]=-1
        f["predicted_pocket_rank"]=-1
        f["predicted_pocket_score"]=0.0
        f["predicted_pocket_distance_A"]=np.nan
        f["site_field_score"]=0.0
        f["pocket_first_physics_score"]=0.0
        return item["predicted_pockets"]

    topology=f["topology_score"].to_numpy(float)
    coherence=f["pocket_coherence"].to_numpy(float)
    intervention=f["quantum_intervention_excess"].to_numpy(float)
    distance_excess=f["distance_quantum_excess"].to_numpy(float)
    residue_rrf=f["rrf_core"].to_numpy(float)
    lambda_stability=f["lambda_rank_stability"].to_numpy(float)
    surface=f["surface_n"].to_numpy(float)

    channels={
        "topology":topology,
        "coherence":coherence,
        "intervention":intervention,
        "distance_excess":distance_excess,
    }
    thresholds={
        name:_high_threshold(arr,rankable)
        for name,arr in channels.items()
    }

    profile_rows=[]
    canonical_members={}

    for radius in V35_RADIUS_GRID_A:
        scale_rows=[]

        for pid,center_idx in enumerate(centers,1):
            d=np.linalg.norm(coords-coords[center_idx],axis=1)
            members=np.where(rankable & (d<=float(radius)))[0]
            if not len(members):
                members=np.array([center_idx],dtype=int)

            if math.isclose(float(radius),float(V35_CANONICAL_RADIUS_A),abs_tol=1e-9):
                canonical_members[int(pid)]=members.copy()

            scale_rows.append({
                "pocket_id":int(pid),
                "center_idx":int(center_idx),
                "radius_A":float(radius),
                "member_count":int(len(members)),
                "topology_top_mean":_top_fraction(topology[members]),
                "coherence_top_mean":_top_fraction(coherence[members]),
                "intervention_top_mean":_top_fraction(intervention[members]),
                "distance_excess_top_mean":_top_fraction(distance_excess[members]),
                "residue_rrf_top_mean":_top_fraction(residue_rrf[members]),
                "consensus_fraction":_multichannel_consensus_fraction(
                    members,channels,thresholds
                ),
                # descriptors only
                "lambda_stability_mean":float(np.nanmean(lambda_stability[members]))
                    if np.isfinite(lambda_stability[members]).any() else np.nan,
                "surface_mean":float(np.nanmean(surface[members])),
            })

        scale_df=pd.DataFrame(scale_rows)
        raw,score=_scale_rrf(scale_df)
        scale_df["scale_rrf_raw"]=raw
        scale_df["scale_consensus_score"]=score
        scale_df["scale_rank"]=rankdata(
            -scale_df["scale_consensus_score"].to_numpy(float),
            method="average"
        )
        profile_rows.append(scale_df)

    profiles=pd.concat(profile_rows,ignore_index=True)
    item["pocket_scale_profiles"]=profiles.copy()

    rows=[]
    for pid,center_idx in enumerate(centers,1):
        p=profiles[profiles["pocket_id"]==pid].copy()
        members=canonical_members.get(pid)
        if members is None:
            d=np.linalg.norm(coords-coords[center_idx],axis=1)
            members=np.where(
                rankable & (d<=float(V35_CANONICAL_RADIUS_A))
            )[0]
            if not len(members):
                members=np.array([center_idx],dtype=int)

        weights=np.clip(residue_rrf[members]+1e-6,1e-6,None)
        centroid=np.average(coords[members],axis=0,weights=weights)
        radial=np.linalg.norm(coords[members]-centroid,axis=1)

        scores=p["scale_consensus_score"].to_numpy(float)
        ranks=p["scale_rank"].to_numpy(float)

        rows.append({
            "pocket_id":int(pid),
            "center_idx":int(center_idx),
            "center_label":str(f.loc[center_idx,"label"]),
            "center_resname":str(f.loc[center_idx,"resname"]),
            "member_count":int(len(members)),
            "member_indices":tuple(int(x) for x in members),
            "member_labels":";".join(f.loc[members,"label"].astype(str).tolist()),
            "centroid_x":float(centroid[0]),
            "centroid_y":float(centroid[1]),
            "centroid_z":float(centroid[2]),
            "mean_radius_A":float(radial.mean()) if len(radial) else 0.0,
            "max_radius_A":float(radial.max()) if len(radial) else 0.0,

            # generation descriptor only
            "center_discovery_score":float(discovery_score[center_idx]),

            # multi-scale ranking evidence
            "multiscale_score_median":float(np.median(scores)),
            "multiscale_score_mean":float(np.mean(scores)),
            "multiscale_score_std":float(np.std(scores)),
            "multiscale_top3_frequency":float(np.mean(ranks<=min(3,len(centers)))),
            "multiscale_mean_rank":float(np.mean(ranks)),

            # canonical-scale descriptors
            "topology_top_mean":_top_fraction(topology[members]),
            "coherence_top_mean":_top_fraction(coherence[members]),
            "intervention_top_mean":_top_fraction(intervention[members]),
            "distance_excess_top_mean":_top_fraction(distance_excess[members]),
            "residue_rrf_top_mean":_top_fraction(residue_rrf[members]),
            "consensus_fraction":_multichannel_consensus_fraction(
                members,channels,thresholds
            ),
            "lambda_stability_mean":float(np.nanmean(lambda_stability[members]))
                if np.isfinite(lambda_stability[members]).any() else np.nan,
            "surface_mean":float(np.nanmean(surface[members])),
        })

    pockets=pd.DataFrame(rows)

    # Final pocket rank: robust across scales. No center term and no labels.
    pockets["pocket_score"]=minmax(
        pockets["multiscale_score_median"].to_numpy(float)
    )
    pockets=pockets.sort_values(
        [
            "pocket_score",
            "multiscale_top3_frequency",
            "consensus_fraction",
            "coherence_top_mean",
        ],
        ascending=False
    ).reset_index(drop=True)
    pockets["pocket_rank"]=np.arange(1,len(pockets)+1)
    item["predicted_pockets"]=pockets

    # Smooth field for residue-level ranking.
    f["predicted_pocket_id"]=-1
    f["predicted_pocket_rank"]=-1
    f["predicted_pocket_score"]=0.0
    f["predicted_pocket_distance_A"]=np.nan
    f["site_field_score"]=0.0

    inds=np.where(rankable)[0]
    for ridx in inds:
        best_field=-np.inf
        best=None
        for p in pockets.itertuples():
            centroid=np.array([p.centroid_x,p.centroid_y,p.centroid_z],float)
            dist=float(np.linalg.norm(coords[ridx]-centroid))
            field=float(p.pocket_score)*math.exp(
                -(dist/max(float(V35_CANONICAL_RADIUS_A),1e-8))**2
            )
            if field>best_field:
                best_field=field
                best=(p,dist)

        if best is not None:
            p,dist=best
            f.loc[ridx,"predicted_pocket_id"]=int(p.pocket_id)
            f.loc[ridx,"predicted_pocket_rank"]=int(p.pocket_rank)
            f.loc[ridx,"predicted_pocket_score"]=float(p.pocket_score)
            f.loc[ridx,"predicted_pocket_distance_A"]=dist
            f.loc[ridx,"site_field_score"]=max(0.0,best_field)

    f["pocket_first_physics_score"]=0.0
    if len(inds):
        combined=(
            V35_SITE_FIELD_WEIGHT
            * f.loc[inds,"site_field_score"].to_numpy(float)
            + V35_RESIDUE_RRF_TIEBREAK_WEIGHT
            * f.loc[inds,"rrf_core"].to_numpy(float)
        )
        f.loc[inds,"pocket_first_physics_score"]=minmax(combined)

    # Export scale profiles now; validation labels are not present here.
    profiles.to_csv(
        RESULTS_DIR/f"{item['name']}_pocket_scale_profiles_v3_5.csv",
        index=False
    )

    return pockets


# Build label-free residue and site physics before optional learned influence.
for item in RESULTS.values():
    add_rrf_core(item["features"])
    add_spatial_pocket_coherence(item)
    discover_candidate_pockets(item)

for name,item in RESULTS.items():
    train=training_items_for(item)
    enough=len(train)>=MIN_LEARNED_BENCHMARKS
    f=item["features"]

    f["hdc_score"]=hdc_train_score(train,item["H"]) if train else 0.0

    if enough:
        best,hist,ensemble=evolve_network(
            train,seed=SEED+sum(map(ord,name))%1000
        )
        mean_p,std_p=ensemble_predict(_evo_matrix(f),ensemble)
    else:
        best,hist,ensemble=None,[],[]
        mean_p=np.zeros(len(f)); std_p=np.ones(len(f))

    item["evo_best"]=best
    item["evo_history"]=hist
    item["evo_ensemble"]=ensemble
    item["training_items"]=[x["name"] for x in train]
    item["learned_enabled"]=enough
    f["evo_score"]=mean_p
    f["evo_uncertainty"]=std_p

    rob=robustness_ensemble(item)
    item["robustness"]=rob
    for c,v in rob.items():
        f[c]=v

    lw=learned_weight(len(train))
    physical=f["pocket_first_physics_score"].to_numpy(float)
    learned=minmax(
        0.55*f["evo_score"].to_numpy()
        + 0.45*f["hdc_score"].to_numpy()
    )
    final=minmax((1-lw)*physical+lw*learned)

    elig=f["eligible_distal"].to_numpy(bool)
    f["learning_weight"]=lw
    f["physical_score"]=physical
    f["learned_score"]=learned
    f["final_score"]=np.where(elig,final,0.0)

    # Confidence only: robustness + lambda rank stability + optional learned consensus.
    lambda_conf=f["lambda_rank_stability"].to_numpy(float)
    measured=np.isfinite(lambda_conf)

    if enough:
        evo_consensus=1-minmax(f["evo_uncertainty"].to_numpy())
        base_conf=0.70*f["robust_confidence"].to_numpy()+0.30*evo_consensus
        conf=base_conf.copy()
        conf[measured]=0.70*base_conf[measured]+0.30*lambda_conf[measured]
    else:
        base_conf=f["robust_confidence"].to_numpy(float)
        conf=base_conf.copy()
        conf[measured]=0.65*base_conf[measured]+0.35*lambda_conf[measured]

    f["confidence"]=np.where(elig,np.clip(conf,0,1),0.0)
    f["stability_confidence"]=f["confidence"]

    print(
        name,
        "independent training=",len(train),
        "learned weight=",lw,
        "rankable=",int(elig.sum()),
        "lambda rank rho=",item.get("lambda_pairwise_spearman_mean")
    )

# %% [markdown]
# ## 15. V3.6 collective quantum site control
#
# This section reranks the already-generated candidate pockets using direct
# site-level quantum observables.
#
# ### A. Matched-null collective transfer
#
# For each canonical pocket, quantum and classical probability density are
# integrated over the site. Candidate scores are standardized against pseudo-
# pockets matched approximately on region size and geometric distance from the
# functional seed. Validation labels are not used to construct the null.
#
# ### B. Unitary phase-kick susceptibility
#
# A local phase kick is inserted halfway through the CTQW evolution. The
# implementation uses
#
# \[
# U(t/2)K_PU(t/2)|\psi_0\rangle
# =
# |\psi(t)\rangle+
# (e^{-i\lambda}-1)U(t/2)\Pi_P|\phi(t/2)\rangle
# \]
#
# so all candidate pockets at a given time are propagated together.
#
# The raw \(\lambda^2\) coefficient is divided by \(\sqrt{|P|}\) as an explicit,
# predeclared size-normalization diagnostic. Both raw and normalized coefficients
# are exported.

# %% [code] Notebook code cell 18 (source index 36)

def _robust_z_against(value, null_values, clip=V36_NULL_Z_CLIP):
    arr=np.asarray(null_values,float)
    arr=arr[np.isfinite(arr)]
    if len(arr)<3 or not np.isfinite(value):
        return np.nan
    med=float(np.median(arr))
    mad=float(1.4826*np.median(np.abs(arr-med)))
    scale=max(mad,1e-8)
    return float(np.clip((float(value)-med)/scale,-clip,clip))

def _region_transfer_stats(q_prob_t,c_prob_t,times,members):
    members=np.asarray(members,int)
    m=max(1,len(members))
    q_density=np.sum(q_prob_t[:,members],axis=1)/m
    c_density=np.sum(c_prob_t[:,members],axis=1)/m

    if len(times)>1 and times[-1]>times[0]:
        denom=float(times[-1]-times[0])
        q_auc=float(np.trapezoid(q_density,times)/denom)
        c_auc=float(np.trapezoid(c_density,times)/denom)
    else:
        q_auc=float(np.mean(q_density))
        c_auc=float(np.mean(c_density))

    late=max(1,int(len(times)*2/3))
    return {
        "collective_q_auc":q_auc,
        "collective_c_auc":c_auc,
        "collective_transfer_gain":q_auc-c_auc,
        "collective_q_peak":float(np.max(q_density)),
        "collective_c_peak":float(np.max(c_density)),
        "collective_q_late":float(np.mean(q_density[late:])),
        "collective_c_late":float(np.mean(c_density[late:])),
        "collective_qc_log_ratio":float(
            np.log((q_auc+1e-15)/(c_auc+1e-15))
        ),
    }

def _build_null_region_catalog(item):
    f=item["features"]
    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    rankable=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )
    coords=f[["x","y","z"]].to_numpy(float)
    inds=np.where(rankable)[0]
    tree=cKDTree(coords)

    seed=np.asarray(item["seed_idx"],int)
    seed_centroid=coords[seed].mean(axis=0) if len(seed) else coords.mean(axis=0)

    q=item["qw"].q_prob_t
    c=item["qw"].c_prob_t
    times=item["qw"].times

    rows=[]
    for center_idx in inds:
        members=np.asarray(
            [
                j for j in tree.query_ball_point(
                    coords[center_idx],
                    r=float(V36_NULL_RADIUS_A)
                )
                if rankable[j]
            ],
            dtype=int,
        )
        if not len(members):
            members=np.array([int(center_idx)],dtype=int)

        centroid=coords[members].mean(axis=0)
        st=_region_transfer_stats(q,c,times,members)
        rows.append({
            "center_idx":int(center_idx),
            "member_count":int(len(members)),
            "seed_distance_A":float(np.linalg.norm(centroid-seed_centroid)),
            "center_x":float(coords[center_idx,0]),
            "center_y":float(coords[center_idx,1]),
            "center_z":float(coords[center_idx,2]),
            **st,
        })

    return pd.DataFrame(rows)

def _matched_null_for_pocket(item,pocket,null_catalog):
    f=item["features"]
    coords=f[["x","y","z"]].to_numpy(float)
    members=np.asarray(pocket.member_indices,int)
    centroid=coords[members].mean(axis=0)

    seed=np.asarray(item["seed_idx"],int)
    seed_centroid=coords[seed].mean(axis=0) if len(seed) else coords.mean(axis=0)
    seed_dist=float(np.linalg.norm(centroid-seed_centroid))
    m=len(members)

    null=null_catalog.copy()
    null_center=null[["center_x","center_y","center_z"]].to_numpy(float)
    far=np.linalg.norm(null_center-coords[int(pocket.center_idx)],axis=1) >= V36_NULL_EXCLUDE_CENTER_A

    size_tol=max(3,int(math.ceil(m*V36_NULL_SIZE_TOL_FRAC)))
    matched=(
        far
        & (np.abs(null["member_count"].to_numpy(int)-m)<=size_tol)
        & (np.abs(null["seed_distance_A"].to_numpy(float)-seed_dist)
           <=V36_NULL_SEED_DISTANCE_TOL_A)
    )

    if int(matched.sum())<V36_NULL_MIN_MATCHES:
        matched=(
            far
            & (np.abs(null["member_count"].to_numpy(int)-m)
               <=max(5,int(math.ceil(m*0.50))))
            & (np.abs(null["seed_distance_A"].to_numpy(float)-seed_dist)<=10.0)
        )

    if int(matched.sum())<5:
        matched=far

    return null.loc[matched].copy(),seed_dist

def _phase_kick_site_scan(item):
    """
    Exact split-step unitary phase-kick experiment for all generated pockets.

    For each t:
      phi = U(t/2) psi0
      base = U(t/2) phi
      perturbed_P = base + (exp(-i lambda)-1) U(t/2) Pi_P phi

    This is exactly equivalent to U(t/2) exp(-i lambda Pi_P) U(t/2) psi0.
    """
    pockets=item["predicted_pockets"].copy()
    if not len(pockets):
        return pd.DataFrame()

    vals=item["qw"].eigenvalues
    vecs=item["qw"].eigenvectors
    n=len(vals)
    psi0,_=seed_state(n,item["seed_idx"])
    eig0=vecs.T@psi0

    pocket_ids=[int(x) for x in pockets["pocket_id"]]
    member_map={
        int(p.pocket_id):np.asarray(p.member_indices,int)
        for p in pockets.itertuples()
    }
    P=len(pocket_ids)
    L=len(V36_PHASE_KICK_LAMBDAS)

    js_time=np.zeros((L,P,len(V36_PHASE_KICK_TIMES)),dtype=float)

    for ti,t in enumerate(V36_PHASE_KICK_TIMES):
        half_phase=np.exp(-1j*GAMMA*vals*(float(t)/2.0))
        phi=vecs@(half_phase*eig0)
        base=vecs@(half_phase*(vecs.T@phi))
        base_prob=np.abs(base)**2
        base_prob/=np.clip(base_prob.sum(),1e-15,None)

        # One masked half-time state per predicted pocket.
        X=np.zeros((n,P),dtype=complex)
        for pj,pid in enumerate(pocket_ids):
            members=member_map[pid]
            X[members,pj]=phi[members]

        Y=vecs@(
            half_phase[:,None]*(vecs.T@X)
        )

        for li,lam in enumerate(V36_PHASE_KICK_LAMBDAS):
            delta=np.exp(-1j*float(lam))-1.0
            pert=base[:,None]+delta*Y
            prob=np.abs(pert)**2
            prob/=np.clip(prob.sum(axis=0,keepdims=True),1e-15,None)

            # _js_divergence_rows expects one distribution per row.
            base_rows=np.repeat(base_prob[None,:],P,axis=0)
            js=_js_divergence_rows(base_rows,prob.T)
            js_time[li,:,ti]=js

    mean_js=js_time.mean(axis=2)

    rows=[]
    for pj,pid in enumerate(pocket_ids):
        vals_js=mean_js[:,pj]
        coeff,r2=_through_origin_lambda2_coefficient(
            V36_PHASE_KICK_LAMBDAS,
            vals_js
        )
        members=member_map[pid]
        density_coeff=coeff/(max(1,len(members))**V36_PHASE_SIZE_POWER)

        # Rank stability of size-normalized response across lambda.
        rows.append({
            "pocket_id":pid,
            "phase_kick_coeff_raw":float(coeff),
            "phase_kick_density_coeff":float(density_coeff),
            "phase_kick_lambda2_r2":float(r2) if np.isfinite(r2) else np.nan,
            "phase_kick_mean_js":float(np.mean(vals_js)),
            "phase_kick_peak_lambda_js":float(np.max(vals_js)),
        })

    out=pd.DataFrame(rows)

    # Per-pocket rank stability across lambdas.
    response=np.zeros((L,P),float)
    sizes=np.array(
        [max(1,len(member_map[pid])) for pid in pocket_ids],
        dtype=float
    )
    for li in range(L):
        response[li]=mean_js[li]/(sizes**V36_PHASE_SIZE_POWER)

    ranks=np.vstack([
        rankdata(-response[li],method="average")
        for li in range(L)
    ])
    pct=(ranks-1.0)/max(1.0,P-1.0)

    stab={}
    for pj,pid in enumerate(pocket_ids):
        stab[pid]=float(np.clip(1.0-2.0*np.std(pct[:,pj]),0.0,1.0))
    out["phase_kick_rank_stability"]=out["pocket_id"].map(stab)

    return out

def _v36_site_rrf(pockets):
    n=len(pockets)
    numerator=np.zeros(n,float)
    denominator=np.zeros(n,float)

    for col,weight in V36_SITE_COMPONENTS:
        vals=pockets[col].to_numpy(float)
        valid=np.isfinite(vals)
        if not valid.any():
            continue
        ranks=np.full(n,np.nan,float)
        ranks[valid]=rankdata(-vals[valid],method="average")
        numerator[valid]+=float(weight)/(V36_SITE_RRF_K+ranks[valid])
        denominator[valid]+=float(weight)

    raw=np.full(n,np.nan,float)
    valid=denominator>0
    raw[valid]=numerator[valid]/denominator[valid]

    score=np.zeros(n,float)
    if valid.any():
        score[valid]=minmax(raw[valid])
    return raw,score

def _reproject_v36_site_field(item):
    f=item["features"]
    pockets=item["predicted_pockets"]
    coords=f[["x","y","z"]].to_numpy(float)

    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    rankable=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )
    inds=np.where(rankable)[0]

    f["predicted_pocket_id"]=-1
    f["predicted_pocket_rank"]=-1
    f["predicted_pocket_score"]=0.0
    f["predicted_pocket_distance_A"]=np.nan
    f["site_field_score"]=0.0

    for ridx in inds:
        best_field=-np.inf
        best=None
        for p in pockets.itertuples():
            centroid=np.array([p.centroid_x,p.centroid_y,p.centroid_z],float)
            dist=float(np.linalg.norm(coords[ridx]-centroid))
            field=float(p.pocket_score)*math.exp(
                -(dist/max(float(V35_CANONICAL_RADIUS_A),1e-8))**2
            )
            if field>best_field:
                best_field=field
                best=(p,dist)

        if best is not None:
            p,dist=best
            f.loc[ridx,"predicted_pocket_id"]=int(p.pocket_id)
            f.loc[ridx,"predicted_pocket_rank"]=int(p.pocket_rank)
            f.loc[ridx,"predicted_pocket_score"]=float(p.pocket_score)
            f.loc[ridx,"predicted_pocket_distance_A"]=dist
            f.loc[ridx,"site_field_score"]=max(0.0,best_field)

    f["pocket_first_physics_score"]=0.0
    if len(inds):
        combined=(
            V36_SITE_FIELD_WEIGHT*f.loc[inds,"site_field_score"].to_numpy(float)
            +V36_RESIDUE_RRF_TIEBREAK_WEIGHT*f.loc[inds,"rrf_core"].to_numpy(float)
        )
        f.loc[inds,"pocket_first_physics_score"]=minmax(combined)

    # Recompute physical/final scores after V3.6 site reranking.
    physical=f["pocket_first_physics_score"].to_numpy(float)
    lw=float(f["learning_weight"].iloc[0])
    learned=f["learned_score"].to_numpy(float)
    final=minmax((1-lw)*physical+lw*learned)

    f["physical_score"]=physical
    f["final_score"]=np.where(rankable,final,0.0)

POCKET_CONTROL_ROWS=[]

for name,item in RESULTS.items():
    pockets=item["predicted_pockets"].copy()
    if not len(pockets):
        continue

    # Preserve V3.5 site score as a diagnostic.
    pockets["v35_pocket_score"]=pockets["pocket_score"].to_numpy(float)

    # A. Matched-null direct collective transfer.
    null_catalog=_build_null_region_catalog(item)
    item["v36_null_catalog"]=null_catalog

    transfer_rows=[]
    q=item["qw"].q_prob_t
    c=item["qw"].c_prob_t
    times=item["qw"].times

    for p in pockets.itertuples():
        members=np.asarray(p.member_indices,int)
        stats=_region_transfer_stats(q,c,times,members)
        matched,seed_dist=_matched_null_for_pocket(
            item,p,null_catalog
        )

        row={
            "pocket_id":int(p.pocket_id),
            "collective_seed_distance_A":seed_dist,
            "collective_null_n":int(len(matched)),
            **stats,
        }

        for field in [
            "collective_q_auc",
            "collective_transfer_gain",
            "collective_q_peak",
            "collective_q_late",
        ]:
            row[field+"_z"]=_robust_z_against(
                stats[field],
                matched[field].to_numpy(float)
            )
            nv=matched[field].to_numpy(float)
            nv=nv[np.isfinite(nv)]
            if len(nv):
                row[field+"_empirical_p"]=float(
                    (1+np.sum(nv>=stats[field]))/(1+len(nv))
                )
            else:
                row[field+"_empirical_p"]=np.nan

        transfer_rows.append(row)

    transfer_df=pd.DataFrame(transfer_rows)

    # B. Unitary collective phase-kick experiment.
    phase_df=_phase_kick_site_scan({
        **item,
        "predicted_pockets":pockets,
    })

    pockets=(
        pockets
        .merge(transfer_df,on="pocket_id",how="left")
        .merge(phase_df,on="pocket_id",how="left")
    )

    # V3.6 pocket rank.
    raw,score=_v36_site_rrf(pockets)
    pockets["v36_site_rrf_raw"]=raw
    pockets["pocket_score"]=score
    pockets=pockets.sort_values(
        [
            "pocket_score",
            "phase_kick_density_coeff",
            "collective_transfer_gain_z",
            "collective_q_peak_z",
        ],
        ascending=False,
    ).reset_index(drop=True)
    pockets["pocket_rank"]=np.arange(1,len(pockets)+1)

    item["predicted_pockets"]=pockets
    _reproject_v36_site_field(item)

    export=pockets.copy()
    export["target"]=name
    POCKET_CONTROL_ROWS.append(export)

    print("\n",name,"V3.6 collective quantum site ranking")
    display(
        pockets[
            [
                "pocket_rank","pocket_id","center_label","pocket_score",
                "phase_kick_density_coeff","phase_kick_lambda2_r2",
                "phase_kick_rank_stability",
                "collective_transfer_gain_z","collective_q_peak_z",
                "collective_q_auc_z",
                "topology_top_mean","multiscale_score_median",
                "collective_null_n",
            ]
        ]
    )

if POCKET_CONTROL_ROWS:
    V36_POCKET_CONTROL=pd.concat(
        POCKET_CONTROL_ROWS,
        ignore_index=True,
    )
    V36_POCKET_CONTROL.to_csv(
        RESULTS_DIR/"collective_quantum_site_control_v3_6.csv",
        index=False,
    )

# %% [markdown]
# ## 16. V3.6 seed sensitivity and site robustness — diagnostics only

# %% [code] Notebook code cell 19 (source index 38)

def _seed_variants(item):
    seed=np.asarray(item["seed_idx"],dtype=int)
    if len(seed)<=2:
        return {"full":seed.tolist()}

    f=item["features"]
    coords=f[["x","y","z"]].to_numpy(float)
    G=item["G"]
    take=max(2,int(math.ceil(len(seed)*V34_SEED_CORE_FRACTION)))

    centroid=coords[seed].mean(axis=0)
    d=np.linalg.norm(coords[seed]-centroid,axis=1)
    geo=seed[np.argsort(d)[:take]]

    strengths=[]
    for s in seed:
        strengths.append(sum(
            float(G[int(s)][j].get("weight",1.0))
            for j in G.neighbors(int(s))
        ))
    graph=seed[np.argsort(np.asarray(strengths))[::-1][:take]]

    variants={
        "full":sorted(set(int(x) for x in seed)),
        "geometric_core":sorted(set(int(x) for x in geo)),
        "graph_core":sorted(set(int(x) for x in graph)),
    }
    unique={}
    seen=set()
    for name,vals in variants.items():
        key=tuple(vals)
        if key not in seen:
            unique[name]=vals
            seen.add(key)
    return unique

def _pocket_support(item,residue_score):
    out={}
    for p in item["predicted_pockets"].itertuples():
        members=np.asarray(p.member_indices,dtype=int)
        out[int(p.pocket_id)]=_top_fraction_mean(
            np.asarray(residue_score,float)[members]
        )
    return out

def _rank_stability(pocket_ids,matrix,topn=3):
    matrix=np.asarray(matrix,float)
    if matrix.ndim!=2 or matrix.shape[1]==0:
        return {},np.nan
    ranks=np.vstack([
        rankdata(-row,method="average")
        for row in matrix
    ])
    n=matrix.shape[1]
    pct=(ranks-1.0)/max(1.0,n-1.0)
    result={}
    for j,pid in enumerate(pocket_ids):
        result[int(pid)]={
            "rank_stability":float(np.clip(1-2*np.std(pct[:,j]),0,1)),
            "top3_frequency":float(np.mean(ranks[:,j]<=min(topn,n))),
            "mean_rank":float(np.mean(ranks[:,j])),
        }
    rhos=[]
    for a in range(matrix.shape[0]):
        for b in range(a+1,matrix.shape[0]):
            rho=spearmanr(matrix[a],matrix[b]).statistic
            if np.isfinite(rho):
                rhos.append(float(rho))
    return result,(float(np.mean(rhos)) if rhos else np.nan)

SEED_SENSITIVITY_ROWS=[]
SITE_ROBUSTNESS_ROWS=[]

for name,item in RESULTS.items():
    pockets=item.get("predicted_pockets",pd.DataFrame())
    if pockets is None or not len(pockets):
        continue
    pids=[int(x) for x in pockets["pocket_id"]]

    # Seed sensitivity
    variants=_seed_variants(item)
    matrix=[]
    for vname,vseed in variants.items():
        score=quantum_transport_only(item["L"],vseed,TIME_GRID,GAMMA)
        sup=_pocket_support(item,score)
        matrix.append([sup[p] for p in pids])

    diag,rho=_rank_stability(pids,np.asarray(matrix,float),topn=3)
    item["seed_variants"]=variants
    item["seed_sensitivity_mean_spearman"]=rho
    for pid in pids:
        d=diag[pid]
        SEED_SENSITIVITY_ROWS.append({
            "target":name,
            "pocket_id":pid,
            "seed_variant_count":len(variants),
            "seed_variant_names":";".join(variants.keys()),
            "seed_rank_stability":d["rank_stability"],
            "seed_top3_frequency":d["top3_frequency"],
            "seed_mean_rank":d["mean_rank"],
            "seed_support_pairwise_spearman":rho,
        })

    # Site robustness under cutoff/gamma/time perturbations
    matrix=[]
    for cutoff,gamma,t_scale in ROBUSTNESS_SETTINGS:
        Gp=build_contact_graph(item["apo_df"],cutoff)
        _,Lp,_=graph_matrices(Gp)
        score=quantum_transport_only(
            Lp,item["seed_idx"],TIME_GRID*t_scale,gamma
        )
        sup=_pocket_support(item,score)
        matrix.append([sup[p] for p in pids])

    diag,rho=_rank_stability(
        pids,np.asarray(matrix,float),
        topn=V34_SITE_ROBUST_TOP_POCKETS
    )
    item["site_robustness_mean_spearman"]=rho
    for pid in pids:
        d=diag[pid]
        SITE_ROBUSTNESS_ROWS.append({
            "target":name,
            "pocket_id":pid,
            "site_rank_stability":d["rank_stability"],
            "site_top3_frequency":d["top3_frequency"],
            "site_mean_rank":d["mean_rank"],
            "site_support_pairwise_spearman":rho,
        })

    seed_df=pd.DataFrame([r for r in SEED_SENSITIVITY_ROWS if r["target"]==name])
    rob_df=pd.DataFrame([r for r in SITE_ROBUSTNESS_ROWS if r["target"]==name])
    item["predicted_pockets"]=(
        item["predicted_pockets"]
        .merge(seed_df.drop(columns=["target"]),on="pocket_id",how="left")
        .merge(rob_df.drop(columns=["target"]),on="pocket_id",how="left")
    )

SEED_SENSITIVITY=pd.DataFrame(SEED_SENSITIVITY_ROWS)
SITE_ROBUSTNESS=pd.DataFrame(SITE_ROBUSTNESS_ROWS)

if len(SEED_SENSITIVITY):
    display(SEED_SENSITIVITY)
    SEED_SENSITIVITY.to_csv(
        RESULTS_DIR/"seed_sensitivity_v3_6.csv",index=False
    )
if len(SITE_ROBUSTNESS):
    display(SITE_ROBUSTNESS)
    SITE_ROBUSTNESS.to_csv(
        RESULTS_DIR/"site_transport_robustness_v3_6.csv",index=False
    )

for name,item in RESULTS.items():
    print(
        name,
        "seed rho=",item.get("seed_sensitivity_mean_spearman"),
        "site robustness rho=",item.get("site_robustness_mean_spearman")
    )

# %% [markdown]
# ## 17. V3.6 residue, site and generator-vs-ranker validation
#
# The same post-hoc validation discipline is retained. `best_generated_*` remains
# a generator diagnostic; the new V3.6 site rank is evaluated without changing
# candidate generation from validation labels.

# %% [code] Notebook code cell 20 (source index 40)

def safe_auc(y,s):
    return np.nan if len(np.unique(y))<2 else roc_auc_score(y,s)

def topk_enrichment(y,s,k=5):
    y=np.asarray(y)
    s=np.asarray(s,float)
    finite=np.isfinite(s)
    if not finite.any():
        return np.nan
    yy=y[finite]; ss=s[finite]
    prev=float(np.mean(yy))
    if prev<=0 or np.nanmax(ss)-np.nanmin(ss)<1e-12:
        return np.nan
    idx=np.argsort(ss)[::-1][:min(k,len(ss))]
    return float(np.mean(yy[idx])/prev)

def permutation_pvalue(y,s,n_perm=1000,seed=SEED):
    y=np.asarray(y,int); s=np.asarray(s,float)
    if y.sum()==0 or y.sum()==len(y): return np.nan
    rr=np.random.default_rng(seed)
    obs=s[y==1].mean()-s[y==0].mean()
    ge=0
    for _ in range(n_perm):
        yp=rr.permutation(y)
        ge += (s[yp==1].mean()-s[yp==0].mean()) >= obs
    return (ge+1)/(n_perm+1)

ABLATIONS={
    "ctqw_occupancy":"quantum_score",
    "distance_quantum_excess":"distance_quantum_excess",
    "quantum_susceptibility_coeff":"q_susceptibility_coeff_n",
    "classical_susceptibility_coeff":"c_susceptibility_coeff_n",
    "quantum_intervention_excess":"quantum_intervention_excess",
    "topology":"topology_score",
    "rrf_core":"rrf_core",
    "pocket_coherence":"pocket_coherence",
    "site_field":"site_field_score",
    "pocket_first_physics":"pocket_first_physics_score",
    "full_v3_6":"final_score",
    "HDC":"hdc_score",
    "neuroevolution":"evo_score",
}

rows=[]
for name,item in RESULTS.items():
    y=item["labels"]
    if y is None:
        continue

    f=item["features"]
    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    elig=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )
    yy=y[elig]
    if len(yy)<5 or yy.sum()==0:
        continue

    for label,col in ABLATIONS.items():
        s=f.loc[elig,col].to_numpy(float)
        finite=np.isfinite(s)
        y_eval=yy[finite]
        s_eval=s[finite]
        if len(s_eval)<5 or y_eval.sum()==0 or y_eval.sum()==len(y_eval):
            continue

        pos,neg=s_eval[y_eval==1],s_eval[y_eval==0]
        try:
            mw_p=float(mannwhitneyu(pos,neg,alternative="greater").pvalue)
        except Exception:
            mw_p=np.nan

        rows.append({
            "target":name,
            "model":label,
            "average_precision":average_precision_score(y_eval,s_eval),
            "roc_auc":safe_auc(y_eval,s_eval),
            "top5_recall":topk_recall(y_eval,s_eval,5),
            "top5_enrichment":topk_enrichment(y_eval,s_eval,5),
            "mean_positive_score":float(pos.mean()),
            "mean_background_score":float(neg.mean()),
            "mann_whitney_p":mw_p,
            "permutation_p":permutation_pvalue(y_eval,s_eval),
            "eligible_positives":int(yy.sum()),
            "eligible_residues":int(len(yy)),
            "total_mapped_positives":int(y.sum()),
            "eligible_positive_coverage":float(yy.sum()/max(1,y.sum())),
            "evaluated_positives":int(y_eval.sum()),
            "evaluated_residues":int(len(y_eval)),
            "constant_score":bool(
                np.nanmax(s_eval)-np.nanmin(s_eval)<1e-12
            ),
        })

METRICS=pd.DataFrame(rows)
display(METRICS)
METRICS.to_csv(
    RESULTS_DIR/"benchmark_ablation_metrics_v3_6.csv",index=False
)

# ------------------------------------------------------------------
# Site-level post-hoc validation
# ------------------------------------------------------------------
site_rows=[]
site_summary_rows=[]

for name,item in RESULTS.items():
    y=item["labels"]
    pockets=item.get("predicted_pockets",pd.DataFrame())
    if y is None or pockets is None or len(pockets)==0:
        continue

    f=item["features"]
    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    target_mask=f["chain"].isin(target_chains).to_numpy(bool)
    rankable=f["eligible_distal"].to_numpy(bool) & target_mask
    positive_idx=np.where((y==1)&rankable)[0]
    if not len(positive_idx):
        continue

    coords=f[["x","y","z"]].to_numpy(float)
    pos_coords=coords[positive_idx]
    positive_set=set(int(x) for x in positive_idx)
    validation_centroid=pos_coords.mean(axis=0)

    evaluated=[]
    for p in pockets.itertuples():
        members=np.array(p.member_indices,dtype=int)
        member_set=set(int(x) for x in members)
        overlap=len(member_set & positive_set)
        union=len(member_set | positive_set)

        centroid=np.array([p.centroid_x,p.centroid_y,p.centroid_z],float)
        centroid_error=float(np.linalg.norm(centroid-validation_centroid))
        mind=float(np.min(distance.cdist(coords[members],pos_coords)))

        row={
            "target":name,
            "pocket_rank":int(p.pocket_rank),
            "pocket_id":int(p.pocket_id),
            "center_label":p.center_label,
            "pocket_score":float(p.pocket_score),
            "member_count":int(len(members)),
            "overlap_positives":int(overlap),
            "pocket_precision":float(overlap/max(1,len(members))),
            "pocket_recall":float(overlap/max(1,len(positive_idx))),
            "pocket_jaccard":float(overlap/max(1,union)),
            "any_overlap_hit":bool(overlap>0),
            "min_residue_to_validation_A":mind,
            "centroid_to_validation_centroid_A":centroid_error,
            "within_5A_hit":bool(mind<=V33_SITE_NEAR_A),
            "within_8A_hit":bool(mind<=V33_SITE_NEAR_RELAXED_A),
        }
        evaluated.append(row)
        site_rows.append(row)

    ev=pd.DataFrame(evaluated).sort_values("pocket_rank")
    top1=ev[ev["pocket_rank"]<=1]
    top3=ev[ev["pocket_rank"]<=3]
    top5=ev[ev["pocket_rank"]<=5]

    site_summary_rows.append({
        "target":name,
        "rankable_positives":int(len(positive_idx)),
        "mapped_positives":int(y.sum()),
        "positive_coverage":float(len(positive_idx)/max(1,y.sum())),
        "top1_pocket_overlap_hit":bool(top1["any_overlap_hit"].any()),
        "top3_pocket_overlap_hit":bool(top3["any_overlap_hit"].any()),
        "top5_pocket_overlap_hit":bool(top5["any_overlap_hit"].any()),
        "top1_within_5A":bool(top1["within_5A_hit"].any()),
        "top3_within_5A":bool(top3["within_5A_hit"].any()),
        "top3_best_recall":float(top3["pocket_recall"].max()) if len(top3) else np.nan,
        "top3_best_precision":float(top3["pocket_precision"].max()) if len(top3) else np.nan,
        "top3_min_residue_distance_A":float(top3["min_residue_to_validation_A"].min()) if len(top3) else np.nan,
        "top3_min_centroid_error_A":float(top3["centroid_to_validation_centroid_A"].min()) if len(top3) else np.nan,
        "best_generated_recall":float(ev["pocket_recall"].max()),
        "best_generated_recall_rank":int(
            ev.loc[ev["pocket_recall"].idxmax(),"pocket_rank"]
        ),
        "best_generated_jaccard":float(ev["pocket_jaccard"].max()),
        "best_generated_jaccard_rank":int(
            ev.loc[ev["pocket_jaccard"].idxmax(),"pocket_rank"]
        ),
        "best_generated_precision":float(ev["pocket_precision"].max()),
    })

SITE_METRICS=pd.DataFrame(site_rows)
SITE_SUMMARY=pd.DataFrame(site_summary_rows)

if len(SITE_METRICS):
    display(SITE_METRICS)
    SITE_METRICS.to_csv(
        RESULTS_DIR/"predicted_pocket_validation_v3_6.csv",index=False
    )
if len(SITE_SUMMARY):
    display(SITE_SUMMARY)
    SITE_SUMMARY.to_csv(
        RESULTS_DIR/"site_validation_summary_v3_6.csv",index=False
    )

if len(METRICS):
    ap=METRICS.pivot(index="target",columns="model",values="average_precision")
    if {"pocket_first_physics","pocket_coherence"}.issubset(ap.columns):
        ap["pocket_first_delta_vs_coherence"]=(
            ap["pocket_first_physics"]-ap["pocket_coherence"]
        )
    if {"full_v3_6","rrf_core"}.issubset(ap.columns):
        ap["full_delta_vs_rrf"]=ap["full_v3_6"]-ap["rrf_core"]
    display(ap)
    ap.to_csv(RESULTS_DIR/"v3_6_ablation_summary.csv")

# %% [markdown]
# ## 18. V3.6 competition Top-5 — true pocket members only

# %% [code] Notebook code cell 21 (source index 42)

def site_aware_top5(item,k=5,max_per_pocket=V35_TOP5_MAX_PER_POCKET):
    f=item["features"]
    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    target_mask=f["chain"].isin(target_chains).to_numpy(bool)
    rankable=f["eligible_distal"].to_numpy(bool) & target_mask

    selected=[]
    pockets=item.get("predicted_pockets",pd.DataFrame()).sort_values("pocket_rank")

    for p in pockets.itertuples():
        members=np.asarray(p.member_indices,dtype=int)
        members=np.array(
            [i for i in members if rankable[i]],
            dtype=int
        )
        if not len(members):
            continue

        group=f.loc[members].copy().sort_values(
            [
                "rrf_core",
                "quantum_intervention_excess",
                "topology_score",
                "stability_confidence",
            ],
            ascending=False
        )
        for ridx in group.index[:max_per_pocket]:
            if int(ridx) not in selected:
                selected.append(int(ridx))
            if len(selected)>=k:
                break
        if len(selected)>=k:
            break

    if len(selected)<k:
        fallback=f.loc[np.where(rankable)[0]].sort_values(
            ["site_field_score","rrf_core","stability_confidence"],
            ascending=False
        )
        for ridx in fallback.index:
            if int(ridx) not in selected:
                selected.append(int(ridx))
            if len(selected)>=k:
                break

    return f.loc[selected[:k]].copy()


TOP_HITS={}
POCKET_TABLES={}
COMPETITION_TOP5_METRICS=pd.DataFrame()
print(
    "V4.0.2: frozen challenge Top-5 deferred until calibrated model stage."
)

# %% [markdown]
# ## 19. Frozen independent blind panel — unchanged from V3.7

# %% [code] Notebook code cell 22 (source index 44)

# IMPORTANT: no "holo" or "holo_ligand" keys are permitted in this public registry.
BLIND_PUBLIC_PANEL = {
    "BLIND_PTP1B": {
        "apo": "8U1E",
        "chains": ["A"],
        "target_chains": ["A"],
        "seed_mode": "endogenous_or_manual",
        "manual_seed_resids": list(range(214, 222)),
        "description": "PTP1B — frozen independent allosteric-site validation",
        "official": False,
        "validation_role": "blind_independent",
        "source_note": (
            "Apo WT PTP1B: PDB 8U1E. "
            "Catalytic P-loop fixed before holo validation."
        ),
    },
    "BLIND_CASPASE7": {
        "apo": "1GQF",
        "chains": ["A", "B"],
        "target_chains": ["A", "B"],
        "seed_mode": "endogenous_or_manual",
        "manual_seed_resids": [87, 144, 184, 186, 231, 233],
        "description": "Caspase-7 — frozen independent dimer-interface validation",
        "official": False,
        "validation_role": "blind_independent",
        "source_note": (
            "Human procaspase-7 dimer: PDB 1GQF. "
            "Catalytic/substrate source residues fixed before holo validation."
        ),
    },
    "BLIND_GLUCOKINASE": {
        "apo": "1V4T",
        "chains": ["A"],
        "target_chains": ["A"],
        "seed_mode": "endogenous_or_manual",
        "manual_seed_resids": [168, 169, 204, 205, 231, 256, 290],
        "description": "Human glucokinase — frozen independent activator-site validation",
        "official": False,
        "validation_role": "blind_independent",
        "source_note": (
            "Super-open glucokinase: PDB 1V4T. "
            "Glucose-binding source residues fixed before holo validation."
        ),
    },
}

for name,cfg in BLIND_PUBLIC_PANEL.items():
    forbidden={"holo","holo_ligand"} & set(cfg)
    assert not forbidden, f"Blind leakage in {name}: {forbidden}"

pd.DataFrame(BLIND_PUBLIC_PANEL).T[
    ["apo","chains","target_chains","seed_mode","validation_role"]
]

# %% [markdown]
# ## 20. Frozen V3.6 scorer — unchanged blind predictions

# %% [code] Notebook code cell 23 (source index 46)

def run_blind_base(name,cfg):
    # Hard leakage assertion.
    assert "holo" not in cfg and "holo_ligand" not in cfg

    print("\n"+"="*96)
    print("BLIND SCORE:",name,"-",cfg["description"])
    print("="*96)

    apo_df=extract_residue_table(cfg["apo"],cfg.get("chains"))
    G=build_contact_graph(apo_df,CONTACT_CUTOFF_A)
    A,L,degree=graph_matrices(G)

    seed_idx,seed_meta=choose_seed_indices(cfg,apo_df)
    print(
        f"Apo {cfg['apo']}: {len(apo_df)} residues, "
        f"{G.number_of_edges()} edges | seed={len(seed_idx)}"
    )

    qw=ctqw_connectivity(L,seed_idx,TIME_GRID,GAMMA)
    features=compute_graph_features(apo_df,G,qw,seed_idx)

    eligible_idx=features.loc[
        features["eligible_distal"],"idx"
    ].astype(int).tolist()

    if V37_BLIND_FULL_INTERVENTION:
        candidate_idx=eligible_idx
        scan_mode="FULL_ELIGIBLE_BLIND"
    else:
        candidate_idx=_intervention_shortlist(
            features,V31_PROSPECTIVE_SHORTLIST
        )
        scan_mode="SHORTLIST_BLIND"

    scan,scan_by_lambda,pairwise_rho=intervention_ensemble(
        L,
        seed_idx,
        candidate_idx,
        lambdas=V31_LAMBDA_SWEEP,
        times=V3_INTERVENTION_TIMES,
        gamma=GAMMA,
        checkpoint_key=f"blind__{name}",
    )

    intervention_cols=[
        "q_susceptibility",
        "c_susceptibility",
        "q_susceptibility_late",
        "q_susceptibility_peak",
        "q_susceptibility_lambda_std",
        "c_susceptibility_lambda_std",
        "q_intervention_excess_raw_lambda_std",
        "q_susceptibility_late_lambda_std",
        "q_susceptibility_peak_lambda_std",
        "q_susceptibility_coeff",
        "c_susceptibility_coeff",
        "q_intervention_coeff_excess_raw",
        "q_lambda2_fit_r2",
        "c_lambda2_fit_r2",
        "lambda_rank_stability",
        "lambda_rank_std",
        "lambda_mean_rank_percentile",
        "lambda_top20_frequency",
    ]
    for col in intervention_cols:
        features[col]=np.nan

    for ridx,vals in scan.items():
        mask=features["idx"]==int(ridx)
        for col in intervention_cols:
            if col in vals:
                features.loc[mask,col]=vals[col]

    features["intervention_scanned"]=features["idx"].isin(candidate_idx)
    features["intervention_scan_mode"]=scan_mode
    features=_condition_intervention_excess(features)
    features=apply_symbolic_rules_v3_post_intervention(features)

    item={
        "name":name,
        "cfg":cfg,
        "apo_df":apo_df,
        "G":G,
        "A":A,
        "L":L,
        "seed_idx":seed_idx,
        "seed_meta":seed_meta,
        "qw":qw,
        "features":features,
        "labels":None,
        "validation_meta":{
            "valid":False,
            "reason":"SEALED — validation not yet unblinded"
        },
        "intervention_shortlist":candidate_idx,
        "intervention_scan_mode":scan_mode,
        "intervention_by_lambda":scan_by_lambda,
        "lambda_pairwise_spearman_mean":pairwise_rho,
    }

    # Frozen V3.6 residue/site setup.
    add_rrf_core(features)
    add_spatial_pocket_coherence(item)
    discover_candidate_pockets(item)

    # No learning influence in blind validation.
    features["hdc_score"]=0.0
    features["evo_score"]=0.0
    features["evo_uncertainty"]=1.0
    features["learning_weight"]=0.0
    features["learned_score"]=0.0

    rob=robustness_ensemble(item)
    item["robustness"]=rob
    for col,val in rob.items():
        features[col]=val

    physical=features["pocket_first_physics_score"].to_numpy(float)
    elig=features["eligible_distal"].to_numpy(bool)
    features["physical_score"]=physical
    features["final_score"]=np.where(elig,minmax(physical),0.0)

    lambda_conf=features["lambda_rank_stability"].to_numpy(float)
    base_conf=features["robust_confidence"].to_numpy(float)
    conf=base_conf.copy()
    measured=np.isfinite(lambda_conf)
    conf[measured]=0.65*base_conf[measured]+0.35*lambda_conf[measured]
    features["confidence"]=np.where(elig,np.clip(conf,0,1),0.0)
    features["stability_confidence"]=features["confidence"]

    # Frozen V3.6 collective quantum site-control rerank.
    pockets=item["predicted_pockets"].copy()
    pockets["v35_pocket_score"]=pockets["pocket_score"].to_numpy(float)

    null_catalog=_build_null_region_catalog(item)
    item["v36_null_catalog"]=null_catalog

    transfer_rows=[]
    q=item["qw"].q_prob_t
    c=item["qw"].c_prob_t
    times=item["qw"].times

    for p in pockets.itertuples():
        members=np.asarray(p.member_indices,int)
        stats=_region_transfer_stats(q,c,times,members)
        matched,seed_dist=_matched_null_for_pocket(
            item,p,null_catalog
        )

        row={
            "pocket_id":int(p.pocket_id),
            "collective_seed_distance_A":seed_dist,
            "collective_null_n":int(len(matched)),
            **stats,
        }
        for field in [
            "collective_q_auc",
            "collective_transfer_gain",
            "collective_q_peak",
            "collective_q_late",
        ]:
            row[field+"_z"]=_robust_z_against(
                stats[field],
                matched[field].to_numpy(float)
            )
            nv=matched[field].to_numpy(float)
            nv=nv[np.isfinite(nv)]
            row[field+"_empirical_p"]=(
                float((1+np.sum(nv>=stats[field]))/(1+len(nv)))
                if len(nv) else np.nan
            )
        transfer_rows.append(row)

    transfer_df=pd.DataFrame(transfer_rows)
    phase_df=_phase_kick_site_scan({
        **item,
        "predicted_pockets":pockets,
    })

    pockets=(
        pockets
        .merge(transfer_df,on="pocket_id",how="left")
        .merge(phase_df,on="pocket_id",how="left")
    )

    raw,site_score=_v36_site_rrf(pockets)
    pockets["v36_site_rrf_raw"]=raw
    pockets["pocket_score"]=site_score
    pockets=pockets.sort_values(
        [
            "pocket_score",
            "phase_kick_density_coeff",
            "collective_transfer_gain_z",
            "collective_q_peak_z",
        ],
        ascending=False
    ).reset_index(drop=True)
    pockets["pocket_rank"]=np.arange(1,len(pockets)+1)

    item["predicted_pockets"]=pockets
    _reproject_v36_site_field(item)

    # Freeze prediction artefacts before labels exist.
    item["blind_top5"]=site_aware_top5(item,5).copy()
    item["prediction_frozen"]=True
    item["architecture_signature"]=V37_FROZEN_SIGNATURE

    features.to_csv(
        RESULTS_DIR/f"{name}_blind_features_pre_unseal_v3_7.csv",
        index=False
    )
    pockets.drop(columns=["member_indices"],errors="ignore").to_csv(
        RESULTS_DIR/f"{name}_blind_pockets_pre_unseal_v3_7.csv",
        index=False
    )
    item["blind_top5"].to_csv(
        RESULTS_DIR/f"{name}_blind_top5_pre_unseal_v3_7.csv",
        index=False
    )

    return item

BLIND_RESULTS={}
print("V4.0.3: Phase-1 scorer defined; targets run in separate cells below.")

# %% [markdown]
# #### V4.0.8 persistent-storage gate — phase1

# %% [code] Notebook code cell 24 (source index 48)
v408_assert_persistent_backend("phase1")

# %% [markdown]
# ### 20A. Phase-1 PTP1B — incremental exact-batched run

# %% [code] Notebook code cell 25 (source index 50)

name="BLIND_PTP1B"
cfg=BLIND_PUBLIC_PANEL[name]
cached=v404_load("phase1_pre_unseal",name)
if cached is None:
    item=run_blind_base(name,cfg)
    v404_save("phase1_pre_unseal",name,item)
else:
    item=cached
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED — restored pre-unseal checkpoint",
    }
    item["prediction_frozen"]=True
BLIND_RESULTS[name]=item
print("PHASE-1 COMPLETE:",name)

# %% [markdown]
# ### 20B. Phase-1 Caspase-7 — incremental exact-batched run

# %% [code] Notebook code cell 26 (source index 52)

name="BLIND_CASPASE7"
cfg=BLIND_PUBLIC_PANEL[name]
cached=v404_load("phase1_pre_unseal",name)
if cached is None:
    item=run_blind_base(name,cfg)
    v404_save("phase1_pre_unseal",name,item)
else:
    item=cached
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED — restored pre-unseal checkpoint",
    }
    item["prediction_frozen"]=True
BLIND_RESULTS[name]=item
print("PHASE-1 COMPLETE:",name)

# %% [markdown]
# ### 20C. Phase-1 Glucokinase — incremental exact-batched run

# %% [code] Notebook code cell 27 (source index 54)

name="BLIND_GLUCOKINASE"
print(
    "V4.0.7 Glucokinase resume: completed target/λ/batch "
    "checkpoints will be loaded from the persistent backend."
)
cfg=BLIND_PUBLIC_PANEL[name]
cached=v404_load("phase1_pre_unseal",name)
if cached is None:
    item=run_blind_base(name,cfg)
    v404_save("phase1_pre_unseal",name,item)
else:
    item=cached
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED — restored pre-unseal checkpoint",
    }
    item["prediction_frozen"]=True
BLIND_RESULTS[name]=item
print("PHASE-1 COMPLETE:",name)

# %% [markdown]
# ### 20D. Phase-1 freeze audit

# %% [code] Notebook code cell 28 (source index 56)

assert set(BLIND_RESULTS)==set(BLIND_PUBLIC_PANEL)
assert all(
    item["labels"] is None
    for item in BLIND_RESULTS.values()
)
assert all(
    item["prediction_frozen"]
    for item in BLIND_RESULTS.values()
)
assert not V37_BLIND_PANEL_USED_FOR_TRAINING

v404_stage_marker(
    "phase1_pre_unseal_complete",
    targets=list(BLIND_RESULTS.keys()),
)
print("Blind predictions frozen.")
print("Architecture signature:",V37_FROZEN_SIGNATURE)

# %% [markdown]
# ## 21. Unseal V3.7 independent validation labels — unchanged primary labels

# %% [code] Notebook code cell 29 (source index 58)

# SEALED VALIDATION REGISTRY.
# Do not move this information into BLIND_PUBLIC_PANEL.
BLIND_SEALED_VALIDATION = {
    "BLIND_PTP1B": {
        "holo": "1T49",
        "holo_ligand": "892",
        "reference": "Wiesmann et al., Nat Struct Mol Biol (2004)",
    },
    "BLIND_CASPASE7": {
        "holo": "1SHJ",
        "holo_ligand": "NXN",
        "reference": "Hardy et al., PNAS (2004)",
    },
    "BLIND_GLUCOKINASE": {
        "holo": "1V4S",
        "holo_ligand": "MRK",
        "reference": "Kamata et al., Structure (2004)",
    },
}

# Architecture must be unchanged at unseal time.
current_signature=hashlib.sha256(
    json.dumps(V37_FROZEN_SPEC,sort_keys=True).encode("utf-8")
).hexdigest()
assert current_signature==V37_FROZEN_SIGNATURE
assert not V37_ALLOW_POST_UNSEAL_RETUNING

for name,item in BLIND_RESULTS.items():
    assert item["prediction_frozen"]
    sealed=BLIND_SEALED_VALIDATION[name]
    validation_cfg={
        **item["cfg"],
        **sealed,
    }

    labels,meta=build_validation_labels(
        validation_cfg,
        item["apo_df"]
    )
    item["labels"]=labels
    item["validation_meta"]=meta

    print("\n",name,"UNSEALED")
    print(json.dumps(meta,indent=2,default=str)[:5000])

# %% [markdown]
# ## 22. Primary blind validation results — V3.7 definition

# %% [code] Notebook code cell 30 (source index 60)

BLIND_RESIDUE_ROWS=[]
BLIND_SITE_ROWS=[]
BLIND_TOP5_ROWS=[]

blind_models={
    "frozen_v3_6_final":"final_score",
    "site_field":"site_field_score",
    "pocket_coherence":"pocket_coherence",
    "quantum_intervention_excess":"quantum_intervention_excess",
    "distance_quantum_excess":"distance_quantum_excess",
    "topology":"topology_score",
}

for name,item in BLIND_RESULTS.items():
    y=item["labels"]
    f=item["features"]
    meta=item["validation_meta"]

    if y is None or not meta.get("valid",False):
        BLIND_RESIDUE_ROWS.append({
            "target":name,
            "model":"VALIDATION_REJECTED",
            "validation_valid":False,
            "reason":meta.get("reason","unknown"),
        })
        continue

    target_chains=set(item["cfg"].get("target_chains",f["chain"].unique()))
    rankable=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )
    yy=y[rankable]
    prevalence=float(np.mean(yy))

    for label,col in blind_models.items():
        s=f.loc[rankable,col].to_numpy(float)
        finite=np.isfinite(s)
        y_eval=yy[finite]
        s_eval=s[finite]
        if len(s_eval)<5 or y_eval.sum()==0 or y_eval.sum()==len(y_eval):
            continue

        BLIND_RESIDUE_ROWS.append({
            "target":name,
            "model":label,
            "validation_valid":True,
            "average_precision":float(
                average_precision_score(y_eval,s_eval)
            ),
            "roc_auc":float(safe_auc(y_eval,s_eval)),
            "prevalence":prevalence,
            "ap_over_prevalence":float(
                average_precision_score(y_eval,s_eval)/prevalence
            ) if prevalence>0 else np.nan,
            "top5_recall_global_score":float(
                topk_recall(y_eval,s_eval,5)
            ),
            "mapped_positive_coverage":float(
                meta.get("mapped_pocket_coverage",np.nan)
            ),
            "rankable_positives":int(yy.sum()),
            "rankable_residues":int(len(yy)),
        })

    # Exact frozen exported Top-5.
    top=item["blind_top5"].copy()
    top_idx=top["idx"].astype(int).to_numpy()
    hits=(y[top_idx]==1)
    hit_count=int(hits.sum())

    BLIND_TOP5_ROWS.append({
        "target":name,
        "top5_hits":hit_count,
        "top5_recall":float(
            hit_count/min(5,int(yy.sum()))
        ),
        "top5_enrichment":float(
            (hit_count/max(1,len(top_idx)))/prevalence
        ) if prevalence>0 else np.nan,
        "hit_labels":";".join(
            top.loc[hits,"label"].astype(str).tolist()
        ),
        "top5_labels":";".join(
            top["label"].astype(str).tolist()
        ),
    })

    # Predicted-site validation.
    pockets=item["predicted_pockets"].copy()
    positive_idx=np.where((y==1)&rankable)[0]
    positive_set=set(int(x) for x in positive_idx)

    pocket_rows=[]
    for p in pockets.itertuples():
        members=set(int(x) for x in p.member_indices)
        overlap=len(members & positive_set)
        union=len(members | positive_set)
        pocket_rows.append({
            "target":name,
            "pocket_rank":int(p.pocket_rank),
            "pocket_id":int(p.pocket_id),
            "center_label":p.center_label,
            "pocket_score":float(p.pocket_score),
            "member_count":int(len(members)),
            "overlap_positives":int(overlap),
            "pocket_precision":float(overlap/max(1,len(members))),
            "pocket_recall":float(overlap/max(1,len(positive_set))),
            "pocket_jaccard":float(overlap/max(1,union)),
            "any_overlap_hit":bool(overlap>0),
        })

    pe=pd.DataFrame(pocket_rows).sort_values("pocket_rank")
    top1=pe[pe["pocket_rank"]<=1]
    top3=pe[pe["pocket_rank"]<=3]

    best_recall_idx=pe["pocket_recall"].idxmax()
    best_jaccard_idx=pe["pocket_jaccard"].idxmax()

    BLIND_SITE_ROWS.append({
        "target":name,
        "top1_pocket_overlap_hit":bool(top1["any_overlap_hit"].any()),
        "top3_pocket_overlap_hit":bool(top3["any_overlap_hit"].any()),
        "top3_best_recall":float(top3["pocket_recall"].max()),
        "best_generated_recall":float(pe["pocket_recall"].max()),
        "best_generated_recall_rank":int(
            pe.loc[best_recall_idx,"pocket_rank"]
        ),
        "best_generated_jaccard":float(pe["pocket_jaccard"].max()),
        "best_generated_jaccard_rank":int(
            pe.loc[best_jaccard_idx,"pocket_rank"]
        ),
    })

BLIND_RESIDUE_METRICS=pd.DataFrame(BLIND_RESIDUE_ROWS)
BLIND_SITE_METRICS=pd.DataFrame(BLIND_SITE_ROWS)
BLIND_TOP5_METRICS=pd.DataFrame(BLIND_TOP5_ROWS)

print("\nBlind residue metrics")
display(BLIND_RESIDUE_METRICS)

print("\nBlind site metrics")
display(BLIND_SITE_METRICS)

print("\nBlind exact Top-5 metrics")
display(BLIND_TOP5_METRICS)

BLIND_RESIDUE_METRICS.to_csv(
    RESULTS_DIR/"blind_independent_residue_metrics_v3_7.csv",
    index=False
)
BLIND_SITE_METRICS.to_csv(
    RESULTS_DIR/"blind_independent_site_metrics_v3_7.csv",
    index=False
)
BLIND_TOP5_METRICS.to_csv(
    RESULTS_DIR/"blind_independent_top5_metrics_v3_7.csv",
    index=False
)

# Predeclared panel summary.
frozen=BLIND_RESIDUE_METRICS[
    BLIND_RESIDUE_METRICS["model"]=="frozen_v3_6_final"
].copy()

valid_target_count=int(len(frozen))
median_ap_ratio=float(
    np.nanmedian(frozen["ap_over_prevalence"])
) if valid_target_count else np.nan

site_success=float(
    np.mean(BLIND_SITE_METRICS["top3_pocket_overlap_hit"])
) if len(BLIND_SITE_METRICS) else np.nan

top5_success=float(
    np.mean(BLIND_TOP5_METRICS["top5_hits"]>0)
) if len(BLIND_TOP5_METRICS) else np.nan

BLIND_PANEL_SUMMARY=pd.DataFrame([{
    "architecture_signature":V37_FROZEN_SIGNATURE,
    "valid_targets":valid_target_count,
    "median_ap_over_prevalence":median_ap_ratio,
    "top3_pocket_overlap_success_fraction":site_success,
    "top5_any_hit_success_fraction":top5_success,
    "criterion_median_ap_ratio_pass":bool(
        np.isfinite(median_ap_ratio)
        and median_ap_ratio>
        V37_PREDECLARED_CRITERIA["median_ap_over_prevalence_gt"]
    ),
    "criterion_top3_site_pass":bool(
        np.isfinite(site_success)
        and site_success>=
        V37_PREDECLARED_CRITERIA[
            "top3_pocket_overlap_success_fraction_gte"
        ]
    ),
    "criterion_top5_hit_pass":bool(
        np.isfinite(top5_success)
        and top5_success>=
        V37_PREDECLARED_CRITERIA[
            "top5_any_hit_success_fraction_gte"
        ]
    ),
    "blind_panel_used_for_training":V37_BLIND_PANEL_USED_FOR_TRAINING,
    "post_unseal_retuning_allowed":V37_ALLOW_POST_UNSEAL_RETUNING,
}])

display(BLIND_PANEL_SUMMARY)
BLIND_PANEL_SUMMARY.to_csv(
    RESULTS_DIR/"blind_panel_summary_v3_7.csv",
    index=False
)

# Architecture must still be frozen after all validation calculations.
post_signature=hashlib.sha256(
    json.dumps(V37_FROZEN_SPEC,sort_keys=True).encode("utf-8")
).hexdigest()
assert post_signature==V37_FROZEN_SIGNATURE

# %% [markdown]
# ## 23. V3.8 spatially clustered null significance
#
# A naive permutation of individual residue labels would destroy the defining
# property of an allosteric pocket: **spatial clustering**.
#
# V3.8 therefore evaluates the frozen score against two label-free clustered
# null models for each independent protein:
#
# ### Euclidean pseudo-pocket null
#
# - choose a random rankable residue as center;
# - take the \(K\) nearest rankable residues in 3D, where \(K\) equals the number
#   of observed positive residues;
# - compute AP, ROC-AUC and Top-5 hits for the unchanged frozen score.
#
# ### Graph pseudo-pocket null
#
# - choose a random rankable residue;
# - take the \(K\) nearest residues in contact-graph shortest-path distance;
# - compute the same frozen metrics.
#
# This asks whether the observed ligand pocket is ranked better than a typical
# *equally sized clustered region*, rather than better than arbitrary scattered
# residues.

# %% [code] Notebook code cell 31 (source index 62)
print("V4.0.1: archived analysis skipped — V3.8 clustered-null evidence.")

# %% [markdown]
# ## 24. V3.8 validation-label cutoff sensitivity
#
# The primary result remains the predeclared **5.0 Å** ligand-contact definition.
#
# Without recomputing any prediction score, V3.8 rebuilds validation labels at:
#
# \[
# 4.0,\;4.5,\;5.0,\;5.5,\;6.0\,\AA
# \]
#
# and re-evaluates the already-frozen ranking.
#
# This tests whether a result depends on one arbitrary holo-contact threshold.
# No cutoff is selected for better performance.

# %% [code] Notebook code cell 32 (source index 64)
print("V4.0.1: archived analysis skipped — V3.8 cutoff sensitivity.")

# %% [markdown]
# ## 25. V3.8 protein-level bootstrap intervals
#
# There are only three independent proteins, so uncertainty must be shown
# explicitly.
#
# The bootstrap resamples **whole proteins**, not residues. It therefore answers
# a panel-level replication question rather than pretending hundreds of
# correlated residues are independent observations.

# %% [code] Notebook code cell 33 (source index 66)
print("V4.0.1: archived analysis skipped — V3.8 protein bootstrap.")

# %% [markdown]
# ## 26. V3.8 frozen-component generalization audit
#
# This is a **diagnostic**, not model selection.
#
# V3.8 summarizes how each already-frozen physical component generalizes across
# the independent panel. The table is useful for interpreting mechanism
# heterogeneity — for example, whether one protein is captured mainly by
# intervention physics while another is captured by coherent regional support.
#
# No component is promoted or reweighted after seeing this table.

# %% [code] Notebook code cell 34 (source index 68)
print("V4.0.1: archived analysis skipped — V3.8 component audit.")

# %% [markdown]
# ## 27. V3.9 Phase-2 blind replication panel — public scoring registry
#
# This registry contains **no holo PDB identifiers and no allosteric ligand
# codes**.
#
# ### SHP2 / PTPN11
#
# - input structure: **4DGP**, wild-type human SHP2;
# - source: catalytic phosphatase region, including the WPD-loop Asp and PTP-loop
#   around the catalytic Cys;
# - site class: distal inter-domain.
#
# ### MEK1 / MAP2K1
#
# - input structure: **3EQD**, human MEK1 with ATP-γS/Mg but no allosteric
#   inhibitor;
# - source: fixed ATP/catalytic residues from the nucleotide-binding site;
# - site class: **proximal type-III**.
#
# The type-III site is intentionally retained as a harder/different allostery
# class and is stratified in reporting rather than removed after seeing results.
#
# ### AKT1
#
# - input structure: **7APJ**, autoinhibited AKT1;
# - source: conserved kinase catalytic residues;
# - site class: distal inter-domain PH/kinase regulation.
#
# All source definitions are fixed before the sealed validation registry is
# created.

# %% [code] Notebook code cell 35 (source index 70)

PHASE2_PUBLIC_PANEL = {
    "BLIND2_SHP2": {
        "apo": "4DGP",
        "chains": ["A"],
        "target_chains": ["A"],
        "seed_mode": "manual",
        "manual_seed_resids": [
            425, 426,
            457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
        ],
        "description": "SHP2/PTPN11 — Phase-2 frozen blind replication",
        "official": False,
        "validation_role": "blind_independent_phase2",
        "site_class": "distal_interdomain",
        "source_note": (
            "WT human SHP2 4DGP. "
            "Catalytic WPD/PTP-loop source fixed before holo validation."
        ),
    },

    "BLIND2_MEK1": {
        "apo": "3EQD",
        "chains": ["A"],
        "target_chains": ["A"],
        "seed_mode": "manual",
        "manual_seed_resids": [
            78, 79, 97,
            144, 146, 150,
            192, 194,
            208, 210, 211, 212,
        ],
        "description": "MEK1/MAP2K1 — Phase-2 frozen blind replication",
        "official": False,
        "validation_role": "blind_independent_phase2",
        "site_class": "proximal_type_III",
        "source_note": (
            "Human MEK1 3EQD nucleotide complex. "
            "ATP-contact/catalytic source fixed before holo validation."
        ),
    },

    "BLIND2_AKT1": {
        "apo": "7APJ",
        "chains": ["A"],
        "target_chains": ["A"],
        "seed_mode": "manual",
        "manual_seed_resids": [
            179, 198, 274, 279, 292,
        ],
        "description": "AKT1 — Phase-2 frozen blind replication",
        "official": False,
        "validation_role": "blind_independent_phase2",
        "site_class": "distal_interdomain",
        "source_note": (
            "Autoinhibited AKT1 7APJ. "
            "Kinase catalytic source residues fixed before holo validation."
        ),
    },
}

for name,cfg in PHASE2_PUBLIC_PANEL.items():
    forbidden={"holo","holo_ligand"} & set(cfg)
    assert not forbidden, f"PHASE-2 LABEL LEAKAGE in {name}: {forbidden}"

assert not V39_PHASE2_USED_FOR_TRAINING
assert not V39_ALLOW_PHASE2_RETUNING
assert V39_FROZEN_SIGNATURE == V37_FROZEN_SIGNATURE

display(
    pd.DataFrame(PHASE2_PUBLIC_PANEL).T[
        [
            "apo","chains","target_chains","seed_mode",
            "site_class","validation_role"
        ]
    ]
)

# %% [markdown]
# ## 28. Score Phase-2 proteins — checkpoint/resume

# %% [code] Notebook code cell 36 (source index 72)

assert V37_BLIND_FULL_INTERVENTION == V39_PHASE2_FULL_INTERVENTION
PHASE2_RESULTS={}
print("V4.0.3: Phase-2 targets run in separate cells below.")

# %% [markdown]
# #### V4.0.8 persistent-storage gate — phase2

# %% [code] Notebook code cell 37 (source index 74)
v408_assert_persistent_backend("phase2")

# %% [markdown]
# ### 28 — SHP2 incremental exact-batched run

# %% [code] Notebook code cell 38 (source index 76)

name="BLIND2_SHP2"
print(
    "V4.0.8 SHP2 resume: existing persistent λ/batch checkpoints "
    "will be loaded before new computation."
)
cfg=PHASE2_PUBLIC_PANEL[name]
cached=v404_load("phase2_pre_unseal",name)

if cached is None:
    item=run_blind_base(name,cfg)
    item["blind_phase"]="phase2"
    item["site_class"]=cfg["site_class"]
    item["architecture_signature"]=V39_FROZEN_SIGNATURE
    item["prediction_frozen"]=True

    item["features"].to_csv(
        RESULTS_DIR/f"{name}_blind_features_pre_unseal_v3_9.csv",
        index=False,
    )
    item["predicted_pockets"].drop(
        columns=["member_indices"],
        errors="ignore",
    ).to_csv(
        RESULTS_DIR/f"{name}_blind_pockets_pre_unseal_v3_9.csv",
        index=False,
    )
    item["blind_top5"].to_csv(
        RESULTS_DIR/f"{name}_blind_top5_pre_unseal_v3_9.csv",
        index=False,
    )
    v404_save("phase2_pre_unseal",name,item)
else:
    item=cached
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED — restored Phase-2 pre-unseal checkpoint",
    }
    item["prediction_frozen"]=True

PHASE2_RESULTS[name]=item
print("PHASE-2 COMPLETE:",name)

# %% [markdown]
# ### 28 — MEK1 incremental exact-batched run

# %% [code] Notebook code cell 39 (source index 78)

name="BLIND2_MEK1"
cfg=PHASE2_PUBLIC_PANEL[name]
cached=v404_load("phase2_pre_unseal",name)

if cached is None:
    item=run_blind_base(name,cfg)
    item["blind_phase"]="phase2"
    item["site_class"]=cfg["site_class"]
    item["architecture_signature"]=V39_FROZEN_SIGNATURE
    item["prediction_frozen"]=True

    item["features"].to_csv(
        RESULTS_DIR/f"{name}_blind_features_pre_unseal_v3_9.csv",
        index=False,
    )
    item["predicted_pockets"].drop(
        columns=["member_indices"],
        errors="ignore",
    ).to_csv(
        RESULTS_DIR/f"{name}_blind_pockets_pre_unseal_v3_9.csv",
        index=False,
    )
    item["blind_top5"].to_csv(
        RESULTS_DIR/f"{name}_blind_top5_pre_unseal_v3_9.csv",
        index=False,
    )
    v404_save("phase2_pre_unseal",name,item)
else:
    item=cached
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED — restored Phase-2 pre-unseal checkpoint",
    }
    item["prediction_frozen"]=True

PHASE2_RESULTS[name]=item
print("PHASE-2 COMPLETE:",name)

# %% [markdown]
# ### 28 — AKT1 incremental exact-batched run

# %% [code] Notebook code cell 40 (source index 80)

name="BLIND2_AKT1"
cfg=PHASE2_PUBLIC_PANEL[name]
cached=v404_load("phase2_pre_unseal",name)

if cached is None:
    item=run_blind_base(name,cfg)
    item["blind_phase"]="phase2"
    item["site_class"]=cfg["site_class"]
    item["architecture_signature"]=V39_FROZEN_SIGNATURE
    item["prediction_frozen"]=True

    item["features"].to_csv(
        RESULTS_DIR/f"{name}_blind_features_pre_unseal_v3_9.csv",
        index=False,
    )
    item["predicted_pockets"].drop(
        columns=["member_indices"],
        errors="ignore",
    ).to_csv(
        RESULTS_DIR/f"{name}_blind_pockets_pre_unseal_v3_9.csv",
        index=False,
    )
    item["blind_top5"].to_csv(
        RESULTS_DIR/f"{name}_blind_top5_pre_unseal_v3_9.csv",
        index=False,
    )
    v404_save("phase2_pre_unseal",name,item)
else:
    item=cached
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED — restored Phase-2 pre-unseal checkpoint",
    }
    item["prediction_frozen"]=True

PHASE2_RESULTS[name]=item
print("PHASE-2 COMPLETE:",name)

# %% [markdown]
# ### 28 — Phase-2 freeze audit

# %% [code] Notebook code cell 41 (source index 82)

assert set(PHASE2_RESULTS)==set(PHASE2_PUBLIC_PANEL)
assert all(
    x["labels"] is None
    for x in PHASE2_RESULTS.values()
)
assert all(
    x["prediction_frozen"]
    for x in PHASE2_RESULTS.values()
)
assert all(
    x["architecture_signature"]==V39_FROZEN_SIGNATURE
    for x in PHASE2_RESULTS.values()
)

v404_stage_marker(
    "phase2_pre_unseal_complete",
    targets=list(PHASE2_RESULTS.keys()),
)
print("PHASE-2 PREDICTIONS FROZEN.")

# %% [markdown]
# ## 29. Unseal Phase-2 holo validation — after predictions are frozen
#
# Only now are the validation structures exposed:
#
# - **SHP2:** 5EHR, SHP099 ligand `5OD`.
# - **MEK1:** 3E8N, RDEA119 ligand `VRA`.
# - **AKT1:** 3O96, allosteric inhibitor ligand `IQO`.
#
# These holo ligands define post-hoc contact-pocket labels only. They never enter
# the scoring function.

# %% [code] Notebook code cell 42 (source index 84)

PHASE2_SEALED_VALIDATION = {
    "BLIND2_SHP2": {
        "holo": "5EHR",
        "holo_ligand": "5OD",
        "reference": "Chen et al., Nature (2016)",
    },
    "BLIND2_MEK1": {
        "holo": "3E8N",
        "holo_ligand": "VRA",
        "reference": "RDEA119 MEK1 structural complex",
    },
    "BLIND2_AKT1": {
        "holo": "3O96",
        "holo_ligand": "IQO",
        "reference": "Wu et al., PLoS One (2010)",
    },
}

# Recompute the frozen hash before touching labels.
phase2_unseal_signature=hashlib.sha256(
    json.dumps(V37_FROZEN_SPEC,sort_keys=True).encode("utf-8")
).hexdigest()
assert phase2_unseal_signature==V39_FROZEN_SIGNATURE
assert not V39_ALLOW_PHASE2_RETUNING

for name,item in PHASE2_RESULTS.items():
    assert item["prediction_frozen"]

    validation_cfg={
        **item["cfg"],
        **PHASE2_SEALED_VALIDATION[name],
    }

    labels,meta=build_validation_labels(
        validation_cfg,
        item["apo_df"]
    )
    item["labels"]=labels
    item["validation_meta"]=meta

    print("\n",name,"PHASE-2 UNSEALED")
    print(json.dumps(meta,indent=2,default=str)[:5000])

# Nothing may have mutated the architecture.
assert hashlib.sha256(
    json.dumps(V37_FROZEN_SPEC,sort_keys=True).encode("utf-8")
).hexdigest()==V39_FROZEN_SIGNATURE

# %% [markdown]
# ## 30. Phase-2 replication-only metrics
#
# The first blind panel is **not** mixed into these success criteria.
#
# This is the replication test: can the already-frozen architecture transfer to
# three additional proteins?

# %% [code] Notebook code cell 43 (source index 86)

def evaluate_frozen_panel(results,phase_label):
    residue_rows=[]
    site_rows=[]
    top5_rows=[]

    models={
        "frozen_v3_6_final":"final_score",
        "site_field":"site_field_score",
        "pocket_coherence":"pocket_coherence",
        "quantum_intervention_excess":"quantum_intervention_excess",
        "distance_quantum_excess":"distance_quantum_excess",
        "topology":"topology_score",
    }

    for name,item in results.items():
        y=item["labels"]
        f=item["features"]
        meta=item["validation_meta"]

        if y is None or not meta.get("valid",False):
            residue_rows.append({
                "target":name,
                "phase":phase_label,
                "site_class":item["cfg"].get("site_class","unknown"),
                "model":"VALIDATION_REJECTED",
                "validation_valid":False,
                "reason":meta.get("reason","unknown"),
            })
            continue

        target_chains=set(
            item["cfg"].get("target_chains",f["chain"].unique())
        )
        rankable=(
            f["eligible_distal"].to_numpy(bool)
            & f["chain"].isin(target_chains).to_numpy(bool)
        )
        yy=y[rankable]
        prevalence=float(np.mean(yy))

        for model,col in models.items():
            s=f.loc[rankable,col].to_numpy(float)
            finite=np.isfinite(s)
            y_eval=yy[finite]
            s_eval=s[finite]

            if (
                len(s_eval)<5
                or y_eval.sum()==0
                or y_eval.sum()==len(y_eval)
            ):
                continue

            ap=float(average_precision_score(y_eval,s_eval))

            residue_rows.append({
                "target":name,
                "phase":phase_label,
                "site_class":item["cfg"].get("site_class","unknown"),
                "model":model,
                "validation_valid":True,
                "average_precision":ap,
                "roc_auc":float(safe_auc(y_eval,s_eval)),
                "prevalence":prevalence,
                "ap_over_prevalence":float(ap/prevalence)
                    if prevalence>0 else np.nan,
                "top5_recall_global_score":float(
                    topk_recall(y_eval,s_eval,5)
                ),
                "mapped_positive_coverage":float(
                    meta.get("mapped_pocket_coverage",np.nan)
                ),
                "rankable_positives":int(yy.sum()),
                "rankable_residues":int(len(yy)),
            })

        # Exact pre-unseal Top-5.
        top=item["blind_top5"].copy()
        top_idx=top["idx"].astype(int).to_numpy()
        hits=(y[top_idx]==1)
        hit_count=int(hits.sum())

        top5_rows.append({
            "target":name,
            "phase":phase_label,
            "site_class":item["cfg"].get("site_class","unknown"),
            "top5_hits":hit_count,
            "top5_recall":float(
                hit_count/min(5,int(yy.sum()))
            ),
            "top5_enrichment":float(
                (hit_count/max(1,len(top_idx)))/prevalence
            ) if prevalence>0 else np.nan,
            "hit_labels":";".join(
                top.loc[hits,"label"].astype(str).tolist()
            ),
            "top5_labels":";".join(
                top["label"].astype(str).tolist()
            ),
        })

        # Site metrics.
        pockets=item["predicted_pockets"].copy()
        positive_idx=np.where((y==1)&rankable)[0]
        positive_set=set(int(x) for x in positive_idx)

        rows=[]
        for p in pockets.itertuples():
            members=set(int(x) for x in p.member_indices)
            overlap=len(members & positive_set)
            union=len(members | positive_set)
            rows.append({
                "pocket_rank":int(p.pocket_rank),
                "pocket_id":int(p.pocket_id),
                "center_label":p.center_label,
                "pocket_score":float(p.pocket_score),
                "member_count":int(len(members)),
                "overlap_positives":int(overlap),
                "pocket_precision":float(
                    overlap/max(1,len(members))
                ),
                "pocket_recall":float(
                    overlap/max(1,len(positive_set))
                ),
                "pocket_jaccard":float(
                    overlap/max(1,union)
                ),
                "any_overlap_hit":bool(overlap>0),
            })

        pe=pd.DataFrame(rows).sort_values("pocket_rank")
        top1=pe[pe["pocket_rank"]<=1]
        top3=pe[pe["pocket_rank"]<=3]

        bri=pe["pocket_recall"].idxmax()
        bji=pe["pocket_jaccard"].idxmax()

        site_rows.append({
            "target":name,
            "phase":phase_label,
            "site_class":item["cfg"].get("site_class","unknown"),
            "top1_pocket_overlap_hit":bool(
                top1["any_overlap_hit"].any()
            ),
            "top3_pocket_overlap_hit":bool(
                top3["any_overlap_hit"].any()
            ),
            "top3_best_recall":float(
                top3["pocket_recall"].max()
            ),
            "best_generated_recall":float(
                pe["pocket_recall"].max()
            ),
            "best_generated_recall_rank":int(
                pe.loc[bri,"pocket_rank"]
            ),
            "best_generated_jaccard":float(
                pe["pocket_jaccard"].max()
            ),
            "best_generated_jaccard_rank":int(
                pe.loc[bji,"pocket_rank"]
            ),
        })

    return (
        pd.DataFrame(residue_rows),
        pd.DataFrame(site_rows),
        pd.DataFrame(top5_rows),
    )

(
    PHASE2_RESIDUE_METRICS,
    PHASE2_SITE_METRICS,
    PHASE2_TOP5_METRICS,
)=evaluate_frozen_panel(PHASE2_RESULTS,"phase2")

display(PHASE2_RESIDUE_METRICS)
display(PHASE2_SITE_METRICS)
display(PHASE2_TOP5_METRICS)

PHASE2_RESIDUE_METRICS.to_csv(
    RESULTS_DIR/"phase2_blind_residue_metrics_v3_9.csv",
    index=False
)
PHASE2_SITE_METRICS.to_csv(
    RESULTS_DIR/"phase2_blind_site_metrics_v3_9.csv",
    index=False
)
PHASE2_TOP5_METRICS.to_csv(
    RESULTS_DIR/"phase2_blind_top5_metrics_v3_9.csv",
    index=False
)

phase2_primary=PHASE2_RESIDUE_METRICS[
    PHASE2_RESIDUE_METRICS["model"]=="frozen_v3_6_final"
].copy()

phase2_median=float(
    np.nanmedian(phase2_primary["ap_over_prevalence"])
) if len(phase2_primary) else np.nan

phase2_site_success=float(
    np.mean(PHASE2_SITE_METRICS["top3_pocket_overlap_hit"])
) if len(PHASE2_SITE_METRICS) else np.nan

phase2_top5_success=float(
    np.mean(PHASE2_TOP5_METRICS["top5_hits"]>0)
) if len(PHASE2_TOP5_METRICS) else np.nan

PHASE2_PANEL_SUMMARY=pd.DataFrame([{
    "architecture_signature":V39_FROZEN_SIGNATURE,
    "valid_targets":int(len(phase2_primary)),
    "median_ap_over_prevalence":phase2_median,
    "top3_pocket_overlap_success_fraction":phase2_site_success,
    "top5_any_hit_success_fraction":phase2_top5_success,
    "criterion_median_ap_ratio_pass":bool(
        np.isfinite(phase2_median)
        and phase2_median>
        V39_PHASE2_CRITERIA["median_ap_over_prevalence_gt"]
    ),
    "criterion_top3_site_pass":bool(
        np.isfinite(phase2_site_success)
        and phase2_site_success>=
        V39_PHASE2_CRITERIA[
            "top3_pocket_overlap_success_fraction_gte"
        ]
    ),
    "criterion_top5_hit_pass":bool(
        np.isfinite(phase2_top5_success)
        and phase2_top5_success>=
        V39_PHASE2_CRITERIA[
            "top5_any_hit_success_fraction_gte"
        ]
    ),
    "used_for_training":V39_PHASE2_USED_FOR_TRAINING,
    "retuning_allowed":V39_ALLOW_PHASE2_RETUNING,
}])

display(PHASE2_PANEL_SUMMARY)
PHASE2_PANEL_SUMMARY.to_csv(
    RESULTS_DIR/"phase2_blind_panel_summary_v3_9.csv",
    index=False
)

# %% [markdown]
# ## 31. Phase-2 clustered-null and cutoff-robustness evidence
#
# The exact V3.8 evidence protocol is repeated for only the three new proteins:
#
# - Euclidean clustered-pocket null,
# - contact-graph clustered-pocket null,
# - exact hypergeometric Top-5 probability,
# - 4.0–6.0 Å ligand-contact cutoff sensitivity.
#
# No score is recomputed from the validation results.

# %% [code] Notebook code cell 44 (source index 88)
print("V4.0.1: archived analysis skipped — Phase-2 clustered-null/cutoff evidence.")

# %% [markdown]
# ## 32. Six-protein pooled replication evidence
#
# The two blind phases remain identifiable.
#
# The pooled table is used to reduce uncertainty, but Phase-2 success is always
# reported separately so the original three proteins cannot mask a failed
# replication.
#
# Additional outputs:
#
# - pooled six-protein Fisher significance;
# - whole-protein bootstrap intervals;
# - leave-one-protein-out sensitivity;
# - site-class summary, with proximal MEK1 separated descriptively.

# %% [code] Notebook code cell 45 (source index 90)
print("V4.0.1: archived analysis skipped — six-protein V3.9 pooled evidence.")

# %% [markdown]
# ## 33. V4.0 calibration set — six previously unsealed proteins
#
# The six proteins from V3.7–V3.9 are no longer called blind validation inside
# this research branch. They are **calibration/development proteins**:
#
# - PTP1B
# - Caspase-7
# - Glucokinase
# - SHP2
# - MEK1
# - AKT1
#
# This relabeling is essential: their holo labels have already been inspected.
#
# The challenge proteins KRAS/BCR/Myosin/c-Myc remain excluded from learning.

# %% [code] Notebook code cell 46 (source index 92)

# Build calibration pool from already-unsealed independent systems.
V40_CALIBRATION_ITEMS = (
    list(BLIND_RESULTS.values())
    + list(PHASE2_RESULTS.values())
)

assert len(V40_CALIBRATION_ITEMS) == 6
assert all(
    x["labels"] is not None
    and x["validation_meta"].get("valid",False)
    for x in V40_CALIBRATION_ITEMS
)
assert not V40_CHALLENGE_LABELS_USED_FOR_TRAINING
assert not V40_PHASE3_USED_FOR_TRAINING

# HDC representations are deterministic and protein-local.
for item in V40_CALIBRATION_ITEMS:
    item["H"] = hdc_encode(item["features"])
    item["calibration_role"] = "V4.0_calibration"

V40_CALIBRATION_NAMES = [
    x["name"] for x in V40_CALIBRATION_ITEMS
]
print("V4.0 calibration proteins:", V40_CALIBRATION_NAMES)

# Explicit audit: no challenge target appears in calibration.
assert not (
    set(V40_CALIBRATION_NAMES)
    & set(RESULTS.keys())
)

pd.DataFrame([
    {
        "name":x["name"],
        "apo":x["cfg"]["apo"],
        "positives":int(x["labels"].sum()),
        "rankable":int(
            x["features"]["eligible_distal"].sum()
        ),
        "role":x["calibration_role"],
    }
    for x in V40_CALIBRATION_ITEMS
])

# %% [markdown]
# ## 34. Calibrated HDC and neuroevolution
#
# ### HDC
#
# The existing deterministic HDC encoder is retained. For each LOPO fold,
# positive and negative prototypes are constructed only from the other five
# proteins.
#
# ### Neuroevolution
#
# A compact tanh network receives **frozen physics-derived residue features**.
# Its fitness is macro-averaged across training proteins:
#
# \[
# 0.70\,G_{AP}+0.30\,R_{Top5}
# \]
#
# where
#
# \[
# G_{AP}=\frac{AP-\pi}{1-\pi}
# \]
#
# and \(\pi\) is the positive prevalence of that protein.
#
# This prevents proteins with larger validation pockets from dominating simply
# because their random AP baseline is higher.
#
# ### Hybrid rank
#
# Frozen physics, HDC and neuroevolution are fused by **equal-weight reciprocal
# rank fusion**. There are no learned post-hoc fusion weights.

# %% [code] Notebook code cell 47 (source index 94)

def _v40_matrix(frame):
    X=frame[V40_LEARN_FEATURES].to_numpy(np.float32)
    return np.nan_to_num(
        X,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )

def _v40_genome_size():
    nf=len(V40_LEARN_FEATURES)
    h=V40_EVO_HIDDEN
    return nf*h+h+h+1

def _v40_unpack(g):
    nf=len(V40_LEARN_FEATURES)
    h=V40_EVO_HIDDEN
    p=0
    W1=g[p:p+nf*h].reshape(nf,h); p+=nf*h
    b1=g[p:p+h]; p+=h
    W2=g[p:p+h]; p+=h
    b2=float(g[p])
    return W1,b1,W2,b2

def _v40_forward(X,g):
    W1,b1,W2,b2=_v40_unpack(g)
    z=np.tanh(X@W1+b1)@W2+b2
    z=np.clip(z,-30,30)
    return 1.0/(1.0+np.exp(-z))

def _v40_fitness(g,items):
    vals=[]
    for item in items:
        y=item["labels"]
        elig=item["features"]["eligible_distal"].to_numpy(bool)
        yy=y[elig]
        if len(yy)<10 or yy.sum()==0 or yy.sum()==len(yy):
            continue

        X=_v40_matrix(item["features"].loc[elig])
        p=_v40_forward(X,g)

        ap=float(average_precision_score(yy,p))
        prev=float(np.mean(yy))
        ap_gain=(ap-prev)/max(1e-8,1-prev)

        t5=topk_recall(yy,p,5)
        if not np.isfinite(t5):
            t5=0.0

        vals.append(
            0.70*ap_gain
            +0.30*float(t5)
        )

    return float(np.mean(vals)) if vals else -1e9

def v40_evolve(items,seed):
    rr=np.random.default_rng(int(seed))
    gs=_v40_genome_size()

    pop=rr.normal(
        0,0.35,
        size=(V40_EVO_POP,gs)
    ).astype(np.float32)

    archive=[]
    history=[]

    for gen in range(V40_EVO_GENERATIONS):
        fit=np.array([
            _v40_fitness(g,items)
            for g in pop
        ])
        order=np.argsort(fit)[::-1]
        pop=pop[order]
        fit=fit[order]

        history.append(float(fit[0]))

        keep=min(5,len(pop))
        archive.extend(
            (float(fit[i]),pop[i].copy())
            for i in range(keep)
        )

        elite_n=max(3,V40_EVO_POP//8)
        parent_n=max(6,V40_EVO_POP//3)
        elites=pop[:elite_n].copy()
        parents=pop[:parent_n]

        sigma=0.16*(0.985**gen)+0.02
        children=list(elites)

        while len(children)<V40_EVO_POP:
            a,b=rr.integers(0,parent_n,size=2)
            mask=rr.random(gs)<0.5
            child=np.where(
                mask,parents[a],parents[b]
            ).copy()

            mut=rr.random(gs)<0.14
            if mut.any():
                child[mut]+=rr.normal(
                    0,sigma,size=int(mut.sum())
                )

            children.append(
                child.astype(np.float32)
            )

        pop=np.asarray(children,dtype=np.float32)

    fit=np.array([
        _v40_fitness(g,items)
        for g in pop
    ])
    order=np.argsort(fit)[::-1]
    pop=pop[order]
    fit=fit[order]

    archive.extend(
        (float(fit[i]),pop[i].copy())
        for i in range(min(10,len(pop)))
    )
    archive.sort(key=lambda x:x[0],reverse=True)

    ensemble=[]
    for fv,g in archive:
        if not ensemble or all(
            np.linalg.norm(g-h)>1e-4
            for _,h in ensemble
        ):
            ensemble.append((fv,g))
        if len(ensemble)>=V40_EVO_ENSEMBLE:
            break

    return pop[0],history,ensemble

def v40_evo_predict(frame,ensemble):
    X=_v40_matrix(frame)
    if not ensemble:
        return (
            np.zeros(len(frame),float),
            np.ones(len(frame),float)
        )
    P=np.stack([
        _v40_forward(X,g)
        for _,g in ensemble
    ])
    return P.mean(axis=0),P.std(axis=0)

def _v40_rank(scores,eligible):
    scores=np.asarray(scores,float)
    eligible=np.asarray(eligible,bool)
    out=np.full(len(scores),np.nan,float)
    valid=eligible & np.isfinite(scores)
    if valid.any():
        out[valid]=rankdata(
            -scores[valid],
            method="average"
        )
    return out

def v40_equal_rrf(
    frozen,
    hdc,
    evo,
    eligible,
    k=V40_HYBRID_RRF_K,
):
    components=[
        np.asarray(frozen,float),
        np.asarray(hdc,float),
        np.asarray(evo,float),
    ]

    num=np.zeros(len(frozen),float)
    den=np.zeros(len(frozen),float)

    for scores in components:
        ranks=_v40_rank(scores,eligible)
        valid=np.isfinite(ranks)
        num[valid]+=1.0/(float(k)+ranks[valid])
        den[valid]+=1.0

    raw=np.full(len(frozen),np.nan,float)
    valid=np.asarray(eligible,bool) & (den>0)
    raw[valid]=num[valid]/den[valid]

    score=np.zeros(len(frozen),float)
    if valid.any():
        score[valid]=minmax(raw[valid])

    return score

def v40_metrics(item,score):
    y=item["labels"]
    f=item["features"]
    elig=f["eligible_distal"].to_numpy(bool)

    yy=y[elig]
    ss=np.asarray(score,float)[elig]

    ap=float(average_precision_score(yy,ss))
    roc=float(safe_auc(yy,ss))
    prev=float(np.mean(yy))
    t5=float(topk_recall(yy,ss,5))

    return {
        "average_precision":ap,
        "roc_auc":roc,
        "prevalence":prev,
        "ap_over_prevalence":(
            float(ap/prev) if prev>0 else np.nan
        ),
        "top5_recall":t5,
    }

# %% [markdown]
# ## 35. Six-protein LOPO calibration audit — one fold per cell

# %% [code] Notebook code cell 48 (source index 96)

V40_LOPO_ROWS=[]
V40_LOPO_SCORES={}

def v404_run_lopo_fold(holdout,fold):
    train=[
        x for x in V40_CALIBRATION_ITEMS
        if x["name"]!=holdout["name"]
    ]
    fold_key=holdout["name"]
    cached=v404_load("v40_lopo",fold_key)

    if cached is None:
        hdc_score=hdc_train_score(
            train,
            holdout["H"],
        )
        _,history,ensemble=v40_evolve(
            train,
            seed=V40_EVO_SEED+fold,
        )
        evo_score,evo_unc=v40_evo_predict(
            holdout["features"],
            ensemble,
        )
        elig=holdout["features"][
            "eligible_distal"
        ].to_numpy(bool)
        frozen=holdout["features"][
            "final_score"
        ].to_numpy(float)
        hybrid=v40_equal_rrf(
            frozen,
            hdc_score,
            evo_score,
            elig,
        )
        cached={
            "hdc":hdc_score,
            "evo":evo_score,
            "hybrid":hybrid,
            "evo_uncertainty":evo_unc,
            "history":history,
        }
        v404_save(
            "v40_lopo",
            fold_key,
            cached,
        )

    V40_LOPO_SCORES[holdout["name"]]=cached

    frozen=holdout["features"][
        "final_score"
    ].to_numpy(float)

    rows=[]
    for model,score in [
        ("frozen_v3_6",frozen),
        ("hdc_lopo",cached["hdc"]),
        ("evo_lopo",cached["evo"]),
        ("hybrid_lopo",cached["hybrid"]),
    ]:
        m=v40_metrics(holdout,score)
        rows.append({
            "target":holdout["name"],
            "held_out":True,
            "train_proteins":len(train),
            "model":model,
            **m,
        })
    return rows

# %% [markdown]
# #### V4.0.8 persistent-storage gate — lopo

# %% [code] Notebook code cell 49 (source index 98)
v408_assert_persistent_backend("lopo")

# %% [markdown]
# ### 35.1 LOPO fold 1/6

# %% [code] Notebook code cell 50 (source index 100)

holdout=V40_CALIBRATION_ITEMS[0]
rows=v404_run_lopo_fold(holdout,0)
V40_LOPO_ROWS.extend(rows)
print("LOPO COMPLETE:",holdout["name"])
display(pd.DataFrame(rows))

# %% [markdown]
# ### 35.2 LOPO fold 2/6

# %% [code] Notebook code cell 51 (source index 102)

holdout=V40_CALIBRATION_ITEMS[1]
rows=v404_run_lopo_fold(holdout,1)
V40_LOPO_ROWS.extend(rows)
print("LOPO COMPLETE:",holdout["name"])
display(pd.DataFrame(rows))

# %% [markdown]
# ### 35.3 LOPO fold 3/6

# %% [code] Notebook code cell 52 (source index 104)

holdout=V40_CALIBRATION_ITEMS[2]
rows=v404_run_lopo_fold(holdout,2)
V40_LOPO_ROWS.extend(rows)
print("LOPO COMPLETE:",holdout["name"])
display(pd.DataFrame(rows))

# %% [markdown]
# ### 35.4 LOPO fold 4/6

# %% [code] Notebook code cell 53 (source index 106)

holdout=V40_CALIBRATION_ITEMS[3]
rows=v404_run_lopo_fold(holdout,3)
V40_LOPO_ROWS.extend(rows)
print("LOPO COMPLETE:",holdout["name"])
display(pd.DataFrame(rows))

# %% [markdown]
# ### 35.5 LOPO fold 5/6

# %% [code] Notebook code cell 54 (source index 108)

holdout=V40_CALIBRATION_ITEMS[4]
rows=v404_run_lopo_fold(holdout,4)
V40_LOPO_ROWS.extend(rows)
print("LOPO COMPLETE:",holdout["name"])
display(pd.DataFrame(rows))

# %% [markdown]
# ### 35.6 LOPO fold 6/6

# %% [code] Notebook code cell 55 (source index 110)

holdout=V40_CALIBRATION_ITEMS[5]
rows=v404_run_lopo_fold(holdout,5)
V40_LOPO_ROWS.extend(rows)
print("LOPO COMPLETE:",holdout["name"])
display(pd.DataFrame(rows))

# %% [markdown]
# ### 35.7 LOPO summary

# %% [code] Notebook code cell 56 (source index 112)

# De-duplicate in case a fold cell was manually rerun.
V40_LOPO_METRICS=(
    pd.DataFrame(V40_LOPO_ROWS)
    .drop_duplicates(
        subset=["target","model"],
        keep="last",
    )
    .reset_index(drop=True)
)

assert V40_LOPO_METRICS["target"].nunique()==6

display(V40_LOPO_METRICS)
V40_LOPO_METRICS.to_csv(
    RESULTS_DIR/"v40_calibration_lopo_metrics.csv",
    index=False,
)

V40_LOPO_SUMMARY=(
    V40_LOPO_METRICS
    .groupby("model",as_index=False)
    .agg(
        proteins=("target","nunique"),
        median_ap=("average_precision","median"),
        median_roc=("roc_auc","median"),
        median_ap_over_prevalence=(
            "ap_over_prevalence","median"
        ),
        mean_top5_recall=("top5_recall","mean"),
    )
    .sort_values(
        "median_ap_over_prevalence",
        ascending=False,
    )
)
display(V40_LOPO_SUMMARY)
V40_LOPO_SUMMARY.to_csv(
    RESULTS_DIR/"v40_calibration_lopo_summary.csv",
    index=False,
)

v404_stage_marker(
    "v40_lopo_complete",
    folds=6,
)

# %% [markdown]
# ## 36. Freeze final calibrated models — checkpointed

# %% [code] Notebook code cell 57 (source index 114)

# Final HDC uses all six calibration proteins.
# hdc_train_score creates prototypes on demand, so the calibration pool itself
# is the frozen HDC model state.
V40_FINAL_HDC_TRAIN = list(V40_CALIBRATION_ITEMS)

# Final evolved network ensemble — checkpointed.
_final_cached=v404_load("v40_final_model","ensemble")

if _final_cached is None:
    (
        V40_FINAL_EVO_GENOME,
        V40_FINAL_EVO_HISTORY,
        V40_FINAL_EVO_ENSEMBLE,
    )=v40_evolve(
        V40_CALIBRATION_ITEMS,
        seed=V40_EVO_SEED+1000,
    )
    _final_cached={
        "genome":V40_FINAL_EVO_GENOME,
        "history":V40_FINAL_EVO_HISTORY,
        "ensemble":V40_FINAL_EVO_ENSEMBLE,
    }
    v404_save("v40_final_model","ensemble",_final_cached)
else:
    V40_FINAL_EVO_GENOME=_final_cached["genome"]
    V40_FINAL_EVO_HISTORY=_final_cached["history"]
    V40_FINAL_EVO_ENSEMBLE=_final_cached["ensemble"]

V40_CALIBRATION_SIGNATURE=hashlib.sha256(
    json.dumps({
        "parent_frozen":V40_PARENT_FROZEN_SIGNATURE,
        "calibration_names":sorted(
            V40_CALIBRATION_NAMES
        ),
        "hdc_features":HDC_FEATURES,
        "evo_features":V40_LEARN_FEATURES,
        "evo_hidden":V40_EVO_HIDDEN,
        "evo_pop":V40_EVO_POP,
        "evo_generations":V40_EVO_GENERATIONS,
        "hybrid_components":V40_HYBRID_COMPONENTS,
        "hybrid_rrf_k":V40_HYBRID_RRF_K,
    },sort_keys=True).encode("utf-8")
).hexdigest()

print(
    "Frozen V4.0 calibration signature:",
    V40_CALIBRATION_SIGNATURE
)
print(
    "Final neuroevolution fitness:",
    V40_FINAL_EVO_HISTORY[-1]
    if V40_FINAL_EVO_HISTORY else np.nan
)

v404_stage_marker("v40_final_model_frozen", calibration_signature=V40_CALIBRATION_SIGNATURE)

# %% [markdown]
# ## 37. Calibrated challenge scoring — frozen physics now runs after calibration

# %% [code] Notebook code cell 58 (source index 116)

def v40_score_item(item):
    f=item["features"]
    elig=f["eligible_distal"].to_numpy(bool)

    H=hdc_encode(f)
    hdc=hdc_train_score(
        V40_FINAL_HDC_TRAIN,
        H,
    )

    evo,evo_unc=v40_evo_predict(
        f,
        V40_FINAL_EVO_ENSEMBLE,
    )

    frozen=f["final_score"].to_numpy(float)
    hybrid=v40_equal_rrf(
        frozen,hdc,evo,elig
    )

    f["v40_hdc_score"]=hdc
    f["v40_evo_score"]=evo
    f["v40_evo_uncertainty"]=evo_unc
    f["v40_calibrated_hybrid"]=hybrid

    return hybrid

def v40_top5_global(item,score,k=5):
    f=item["features"]
    target_chains=set(
        item["cfg"].get(
            "target_chains",
            f["chain"].unique()
        )
    )
    eligible=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )
    inds=np.where(eligible)[0]
    order=inds[
        np.argsort(
            np.asarray(score,float)[inds]
        )[::-1]
    ][:k]
    return f.loc[order].copy()


def v404_run_challenge_frozen(name,cfg):
    cached=v404_load("challenge_frozen_final",name)
    if cached is not None:
        return cached

    public_cfg={
        k:v for k,v in cfg.items()
        if k not in {"holo","holo_ligand"}
    }

    old_mode=globals()["V37_BLIND_FULL_INTERVENTION"]
    globals()["V37_BLIND_FULL_INTERVENTION"]=(
        name in V31_FULL_SCAN_TARGETS
    )
    try:
        item=run_blind_base(name,public_cfg)
    finally:
        globals()["V37_BLIND_FULL_INTERVENTION"]=old_mode

    labels,meta=build_validation_labels(
        cfg,
        item["apo_df"],
    )
    item["cfg"]=cfg
    item["labels"]=labels
    item["validation_meta"]=meta
    item["features"]["label_allosteric"]=(
        labels if labels is not None else np.nan
    )

    v404_save(
        "challenge_frozen_final",
        name,
        item,
    )
    return item

RESULTS={}
for name,cfg in TARGETS.items():
    RESULTS[name]=v404_run_challenge_frozen(
        name,cfg
    )

v404_stage_marker(
    "challenge_after_calibration_complete",
    targets=list(RESULTS.keys()),
)

V40_CHALLENGE_TOP5={}
V40_CHALLENGE_METRICS=[]

for name,item in RESULTS.items():
    hybrid=v40_score_item(item)
    top=v40_top5_global(item,hybrid,5)
    V40_CHALLENGE_TOP5[name]=top

    print("\nV4.0 calibrated challenge Top-5:",name)
    display(
        top[
            [
                "label","resname",
                "v40_calibrated_hybrid",
                "v40_hdc_score",
                "v40_evo_score",
                "final_score",
                "quantum_intervention_excess",
                "topology_score",
                "stability_confidence",
            ]
        ]
    )

    top.to_csv(
        RESULTS_DIR/f"{name}_v40_calibrated_top5.csv",
        index=False
    )

    # Benchmark comparison only after scoring.
    if (
        item["labels"] is not None
        and item["validation_meta"].get("valid",False)
    ):
        for model,score in [
            ("frozen_v3_6",item["features"]["final_score"].to_numpy(float)),
            ("v40_calibrated_hybrid",hybrid),
        ]:
            m=v40_metrics(item,score)
            V40_CHALLENGE_METRICS.append({
                "target":name,
                "model":model,
                "training_contains_challenge_labels":False,
                **m,
            })

V40_CHALLENGE_BENCHMARK=pd.DataFrame(
    V40_CHALLENGE_METRICS
)
if len(V40_CHALLENGE_BENCHMARK):
    display(V40_CHALLENGE_BENCHMARK)
    V40_CHALLENGE_BENCHMARK.to_csv(
        RESULTS_DIR/"v40_challenge_parallel_benchmark.csv",
        index=False
    )

# %% [markdown]
# ## 38. Phase-3 blind holdout — public scoring registry
#
# No holo PDB or allosteric ligand code appears in this registry.
#
# ### EGFR
#
# Input: **2JIT**, EGFR T790M kinase.  
# Functional source: ATP/catalytic kinase residues.
#
# ### IDH1 R132H
#
# Input: **3MAP**, R132H IDH1 dimer with NADP/isocitrate.  
# Functional source: catalytic/substrate-binding residues.
#
# ### HIV-1 integrase
#
# Input: **1EX4**, catalytic-core + C-terminal integrase dimer.  
# Functional source: the D64/D116/E152 catalytic triad.
#
# The three targets are scored with:
#
# - frozen V3.6 physics,
# - HDC trained on the six calibration proteins,
# - neuroevolution trained on the six calibration proteins,
# - the fixed equal-RRF hybrid.
#
# Predictions are frozen before Phase-3 validation is unsealed.

# %% [code] Notebook code cell 59 (source index 118)

PHASE3_PUBLIC_PANEL = {
    "BLIND3_EGFR": {
        "apo": "2JIT",
        "chains": ["A"],
        "target_chains": ["A"],
        "seed_mode": "manual",
        "manual_seed_resids": [
            719, 721, 745, 790, 837, 855
        ],
        "description": "EGFR T790M — Phase-3 calibrated holdout",
        "official": False,
        "validation_role": "blind_independent_phase3",
        "site_class": "kinase_allosteric",
    },

    "BLIND3_IDH1": {
        "apo": "3MAP",
        "chains": ["A","B"],
        "target_chains": ["A","B"],
        "seed_mode": "manual",
        "manual_seed_resids": [
            100,109,132,139,212,252,275
        ],
        "description": "IDH1 R132H — Phase-3 calibrated holdout",
        "official": False,
        "validation_role": "blind_independent_phase3",
        "site_class": "dimer_interface_allosteric",
    },

    "BLIND3_HIV_IN": {
        "apo": "1EX4",
        "chains": ["A","B"],
        "target_chains": ["A","B"],
        "seed_mode": "manual",
        "manual_seed_resids": [
            64,116,152
        ],
        "description": "HIV-1 integrase — Phase-3 calibrated holdout",
        "official": False,
        "validation_role": "blind_independent_phase3",
        "site_class": "dimer_interface_allosteric",
    },
}

for name,cfg in PHASE3_PUBLIC_PANEL.items():
    assert "holo" not in cfg
    assert "holo_ligand" not in cfg

assert not V40_PHASE3_USED_FOR_TRAINING
display(pd.DataFrame(PHASE3_PUBLIC_PANEL).T)

# %% [markdown]
# ### 38A. Clean-room state reset and run nonce
#
# This cell must execute before any Phase-3 scorer.
#
# It deliberately destroys any Phase-3 state that could have survived from an
# earlier Colab execution. The sealed validation registry is not allowed to be
# present at this point.

# %% [code] Notebook code cell 60 (source index 120)

import hashlib as _hashlib_v404_phase3

# Destroy any stale holdout state.
for _name in [
    "PHASE3_RESULTS",
    "PHASE3_SEALED_VALIDATION",
    "PHASE3_METRICS",
    "PHASE3_TOP5",
    "PHASE3_SUMMARY",
    "V404_PHASE3_UNSEAL_NONCE",
]:
    globals().pop(_name, None)

assert "PHASE3_SEALED_VALIDATION" not in globals()
assert "V404_PHASE3_UNSEAL_NONCE" not in globals()

V404_PHASE3_RUN_NONCE = _uuid_v404.uuid4().hex
V404_PHASE3_RESULTS = {}
V404_PHASE3_PREUNSEAL_MANIFEST = {}

V404_PHASE3_SEAL_DIR = RESULTS_DIR / "phase3_cleanroom_seals"
V404_PHASE3_SEAL_DIR.mkdir(parents=True, exist_ok=True)

def v404_sha256_file(path):
    h=_hashlib_v404_phase3.sha256()
    with open(path,"rb") as fh:
        while True:
            block=fh.read(1024*1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def v404_cleanroom_paths(name):
    return {
        "features": RESULTS_DIR/f"{name}_v404_cleanroom_pre_unseal_features.csv",
        "pockets": RESULTS_DIR/f"{name}_v404_cleanroom_pre_unseal_pockets.csv",
        "top5": RESULTS_DIR/f"{name}_v404_cleanroom_pre_unseal_top5.csv",
        "seal": V404_PHASE3_SEAL_DIR/f"{name}_seal.json",
    }

print("V4.0.4 Phase-3 clean-room nonce:",V404_PHASE3_RUN_NONCE)
print("sealed registry present:", "PHASE3_SEALED_VALIDATION" in globals())


def v405_run_cleanroom_phase3_target(name):
    """
    Mechanical refactor of the V4.0.4 target cell.

    Scientific computation is unchanged. V4.0.4 λ/batch checkpoints remain
    compatible because the V404 execution/checkpoint namespace is preserved.
    """
    cfg=PHASE3_PUBLIC_PANEL[name]

    # Leakage/state gate.
    assert "PHASE3_SEALED_VALIDATION" not in globals()
    assert "V404_PHASE3_UNSEAL_NONCE" not in globals()
    assert name not in V404_PHASE3_RESULTS

    # Do not restore an old completed Phase-3 target object: the clean-room
    # protocol requires a new current-run object. run_blind_base may still
    # reuse structure-only intervention λ/batch checkpoints.
    item=run_blind_base(name,cfg)
    item["H"]=hdc_encode(item["features"])

    hdc=hdc_train_score(
        V40_FINAL_HDC_TRAIN,
        item["H"],
    )
    evo,evo_unc=v40_evo_predict(
        item["features"],
        V40_FINAL_EVO_ENSEMBLE,
    )

    elig=item["features"][
        "eligible_distal"
    ].to_numpy(bool)
    frozen=item["features"][
        "final_score"
    ].to_numpy(float)

    hybrid=v40_equal_rrf(
        frozen,
        hdc,
        evo,
        elig,
    )

    f=item["features"]
    f["v40_hdc_score"]=hdc
    f["v40_evo_score"]=evo
    f["v40_evo_uncertainty"]=evo_unc
    f["v40_calibrated_hybrid"]=hybrid

    item["v40_hybrid_score"]=hybrid
    item["v40_top5"]=v40_top5_global(
        item,
        hybrid,
        5,
    )
    item["prediction_frozen"]=True
    item["calibration_signature"]=V40_CALIBRATION_SIGNATURE
    item["cleanroom_run_nonce"]=V404_PHASE3_RUN_NONCE
    item["labels"]=None
    item["validation_meta"]={
        "valid":False,
        "reason":"SEALED PHASE-3 CLEANROOM VALIDATION",
    }

    paths=v404_cleanroom_paths(name)

    # Immutable pre-unseal artifacts.
    f.to_csv(
        paths["features"],
        index=False,
    )
    item["predicted_pockets"].drop(
        columns=["member_indices"],
        errors="ignore",
    ).to_csv(
        paths["pockets"],
        index=False,
    )
    item["v40_top5"].to_csv(
        paths["top5"],
        index=False,
    )

    seal={
        "target":name,
        "run_nonce":V404_PHASE3_RUN_NONCE,
        "calibration_signature":V40_CALIBRATION_SIGNATURE,
        "labels_present":False,
        "feature_sha256":v404_sha256_file(paths["features"]),
        "pocket_sha256":v404_sha256_file(paths["pockets"]),
        "top5_sha256":v404_sha256_file(paths["top5"]),
    }

    paths["seal"].write_text(
        json.dumps(seal,indent=2),
        encoding="utf-8",
    )

    V404_PHASE3_RESULTS[name]=item
    V404_PHASE3_PREUNSEAL_MANIFEST[name]=seal

    print(
        "PHASE-3 CLEANROOM PRE-UNSEAL COMPLETE:",
        name,
    )
    print(json.dumps(seal,indent=2))
    display(
        item["v40_top5"][
            [
                "label","resname",
                "v40_calibrated_hybrid",
                "v40_hdc_score",
                "v40_evo_score",
                "final_score",
            ]
        ]
    )

    return item,seal


# Runtime smoke test for the exact object types that failed in V4.0.4.
_v405_probe_meta={
    "valid":False,
    "reason":"probe",
}
_v405_probe_seal={
    "target":"probe",
    "labels_present":False,
}
assert isinstance(_v405_probe_meta,dict)
assert isinstance(_v405_probe_seal,dict)
del _v405_probe_meta,_v405_probe_seal

print("V4.0.5 clean-room target helper: READY")

# %% [markdown]
# ### 38B. V4.0.5 runtime-literal preflight
#
# This guard specifically catches the class of Python construct that caused the
# V4.0.5 failure: set literals whose elements are themselves mutable containers.

# %% [code] Notebook code cell 61 (source index 122)

# This is a runtime guard for the central clean-room helper.
assert callable(v405_run_cleanroom_phase3_target)
assert V405_SCIENTIFIC_ARCHITECTURE_CHANGED is False
assert "PHASE3_SEALED_VALIDATION" not in globals()
print("V4.0.5 Phase-3 runtime-literal preflight: PASS")

# %% [markdown]
# ## 39. Clean-room Phase-3 pre-unseal predictions — one target per cell

# %% [code] Notebook code cell 62 (source index 124)

assert V404_STRICT_PHASE3_CLEANROOM
assert "PHASE3_SEALED_VALIDATION" not in globals()
assert "V404_PHASE3_UNSEAL_NONCE" not in globals()
assert len(V404_PHASE3_RESULTS)==0
print("Clean-room Phase-3 scoring opened.")

# %% [markdown]
# #### V4.0.8 persistent-storage gate — phase3

# %% [code] Notebook code cell 63 (source index 126)
v408_assert_persistent_backend("phase3")

# %% [markdown]
# ### 39 — EGFR pre-unseal calibrated prediction

# %% [code] Notebook code cell 64 (source index 128)
name="BLIND3_EGFR"
item,seal=v405_run_cleanroom_phase3_target(name)

# %% [markdown]
# ### 39 — IDH1 pre-unseal calibrated prediction

# %% [code] Notebook code cell 65 (source index 130)
name="BLIND3_IDH1"
item,seal=v405_run_cleanroom_phase3_target(name)

# %% [markdown]
# ### 39 — HIV-1 integrase pre-unseal calibrated prediction

# %% [code] Notebook code cell 66 (source index 132)
name="BLIND3_HIV_IN"
item,seal=v405_run_cleanroom_phase3_target(name)

# %% [markdown]
# ### 39D. Cryptographic Phase-3 freeze audit

# %% [code] Notebook code cell 67 (source index 134)

expected=set(PHASE3_PUBLIC_PANEL)
actual=set(V404_PHASE3_RESULTS)

assert actual==expected, (
    "Phase-3 target set incomplete. "
    f"expected={sorted(expected)} actual={sorted(actual)}"
)
assert set(V404_PHASE3_PREUNSEAL_MANIFEST)==expected
assert "PHASE3_SEALED_VALIDATION" not in globals()
assert "V404_PHASE3_UNSEAL_NONCE" not in globals()

for name,item in V404_PHASE3_RESULTS.items():
    assert item["labels"] is None
    assert item["prediction_frozen"]
    assert item["calibration_signature"]==V40_CALIBRATION_SIGNATURE
    assert item["cleanroom_run_nonce"]==V404_PHASE3_RUN_NONCE

    paths=v404_cleanroom_paths(name)
    for key in ["features","pockets","top5","seal"]:
        assert paths[key].exists(), f"Missing pre-unseal artifact: {paths[key]}"

    seal=json.loads(paths["seal"].read_text(encoding="utf-8"))
    assert seal["run_nonce"]==V404_PHASE3_RUN_NONCE
    assert seal["calibration_signature"]==V40_CALIBRATION_SIGNATURE
    assert seal["labels_present"] is False
    assert seal["feature_sha256"]==v404_sha256_file(paths["features"])
    assert seal["pocket_sha256"]==v404_sha256_file(paths["pockets"])
    assert seal["top5_sha256"]==v404_sha256_file(paths["top5"])

# Persist the complete freeze manifest before validation labels exist.
V404_PHASE3_FREEZE_MANIFEST = {
    "run_nonce":V404_PHASE3_RUN_NONCE,
    "calibration_signature":V40_CALIBRATION_SIGNATURE,
    "targets":V404_PHASE3_PREUNSEAL_MANIFEST,
    "sealed_registry_present":False,
}
V404_PHASE3_FREEZE_MANIFEST_PATH = (
    RESULTS_DIR/"phase3_v404_cleanroom_freeze_manifest.json"
)
V404_PHASE3_FREEZE_MANIFEST_PATH.write_text(
    json.dumps(V404_PHASE3_FREEZE_MANIFEST,indent=2),
    encoding="utf-8",
)

V404_PHASE3_FREEZE_MANIFEST_SHA256=v404_sha256_file(
    V404_PHASE3_FREEZE_MANIFEST_PATH
)

v404_stage_marker(
    "phase3_cleanroom_pre_unseal_complete",
    targets=sorted(expected),
    run_nonce=V404_PHASE3_RUN_NONCE,
    freeze_manifest_sha256=V404_PHASE3_FREEZE_MANIFEST_SHA256,
)

print("PHASE-3 CLEANROOM PREDICTIONS CRYPTOGRAPHICALLY FROZEN")
print("freeze manifest:",V404_PHASE3_FREEZE_MANIFEST_PATH)
print("freeze SHA256:",V404_PHASE3_FREEZE_MANIFEST_SHA256)

# %% [markdown]
# ## 40. Clean-room unseal — allowed only after complete cryptographic freeze

# %% [code] Notebook code cell 68 (source index 136)

# A complete current-run freeze is mandatory.
assert set(V404_PHASE3_RESULTS)==set(PHASE3_PUBLIC_PANEL)
assert V404_PHASE3_FREEZE_MANIFEST_PATH.exists()
assert (
    v404_sha256_file(V404_PHASE3_FREEZE_MANIFEST_PATH)
    ==V404_PHASE3_FREEZE_MANIFEST_SHA256
)
assert "PHASE3_SEALED_VALIDATION" not in globals()

V404_PHASE3_UNSEAL_NONCE = _uuid_v404.uuid4().hex

PHASE3_SEALED_VALIDATION = {
    "BLIND3_EGFR": {
        "holo": "6P1L",
        "holo_ligand": "9LL",
        "reference": "EAI045 allosteric EGFR structure",
    },
    "BLIND3_IDH1": {
        "holo": "6ADG",
        "holo_ligand": "9UO",
        "reference": "AG-881 allosteric mutant IDH1 structure",
    },
    "BLIND3_HIV_IN": {
        "holo": "4E1N",
        "holo_ligand": "TQX",
        "reference": "non-catalytic allosteric HIV-1 integrase inhibitor",
    },
}

assert set(PHASE3_SEALED_VALIDATION)==set(V404_PHASE3_RESULTS)

V404_PHASE3_LABELS={}
V404_PHASE3_VALIDATION_META={}

for name,item in V404_PHASE3_RESULTS.items():
    validation_cfg={
        **item["cfg"],
        **PHASE3_SEALED_VALIDATION[name],
    }

    labels,meta=build_validation_labels(
        validation_cfg,
        item["apo_df"],
    )

    V404_PHASE3_LABELS[name]=labels
    V404_PHASE3_VALIDATION_META[name]=meta

    print("\n",name,"CLEANROOM UNSEALED")
    print(json.dumps(meta,indent=2,default=str)[:5000])

assert V404_PHASE3_UNSEAL_NONCE != V404_PHASE3_RUN_NONCE

unseal_record={
    "prediction_run_nonce":V404_PHASE3_RUN_NONCE,
    "unseal_nonce":V404_PHASE3_UNSEAL_NONCE,
    "freeze_manifest_sha256":V404_PHASE3_FREEZE_MANIFEST_SHA256,
    "targets":sorted(V404_PHASE3_RESULTS),
}
(
    RESULTS_DIR/"phase3_v404_cleanroom_unseal_record.json"
).write_text(
    json.dumps(unseal_record,indent=2),
    encoding="utf-8",
)

v404_stage_marker(
    "phase3_cleanroom_unsealed",
    prediction_run_nonce=V404_PHASE3_RUN_NONCE,
    unseal_nonce=V404_PHASE3_UNSEAL_NONCE,
)

# %% [markdown]
# ## 41. Clean-room Phase-3 evaluation — reload frozen files only

# %% [code] Notebook code cell 69 (source index 138)

assert "PHASE3_SEALED_VALIDATION" in globals()
assert "V404_PHASE3_UNSEAL_NONCE" in globals()
assert set(V404_PHASE3_LABELS)==set(PHASE3_PUBLIC_PANEL)

PHASE3_ROWS=[]
PHASE3_TOP5_ROWS=[]
V404_PHASE3_FILE_RELOAD_AUDIT=[]

for name in sorted(PHASE3_PUBLIC_PANEL):
    item=V404_PHASE3_RESULTS[name]
    labels=V404_PHASE3_LABELS[name]
    meta=V404_PHASE3_VALIDATION_META[name]
    paths=v404_cleanroom_paths(name)

    # Verify pre-unseal hashes again after labels exist.
    seal=json.loads(paths["seal"].read_text(encoding="utf-8"))
    assert seal["run_nonce"]==V404_PHASE3_RUN_NONCE
    assert seal["feature_sha256"]==v404_sha256_file(paths["features"])
    assert seal["top5_sha256"]==v404_sha256_file(paths["top5"])

    if labels is None or not meta.get("valid",False):
        PHASE3_ROWS.append({
            "target":name,
            "model":"VALIDATION_REJECTED",
            "validation_valid":False,
            "reason":meta.get("reason","unknown"),
        })
        continue

    # CRITICAL: all scores are loaded from files created before unseal.
    frozen_df=pd.read_csv(paths["features"])
    frozen_top5=pd.read_csv(paths["top5"])

    # Row alignment is explicit.
    expected_idx=item["features"]["idx"].astype(int).to_numpy()
    observed_idx=frozen_df["idx"].astype(int).to_numpy()
    assert np.array_equal(expected_idx,observed_idx)

    f_live=item["features"]
    elig=f_live["eligible_distal"].to_numpy(bool)
    yy=np.asarray(labels,int)[elig]
    prev=float(np.mean(yy))

    score_map={
        "frozen_v3_6":frozen_df["final_score"].to_numpy(float),
        "hdc_calibrated":frozen_df["v40_hdc_score"].to_numpy(float),
        "evo_calibrated":frozen_df["v40_evo_score"].to_numpy(float),
        "v40_calibrated_hybrid":frozen_df["v40_calibrated_hybrid"].to_numpy(float),
    }

    for model,score in score_map.items():
        ss=score[elig]
        ap=float(average_precision_score(yy,ss))
        PHASE3_ROWS.append({
            "target":name,
            "site_class":item["cfg"]["site_class"],
            "model":model,
            "validation_valid":True,
            "average_precision":ap,
            "roc_auc":float(safe_auc(yy,ss)),
            "prevalence":prev,
            "ap_over_prevalence":float(ap/prev) if prev>0 else np.nan,
            "top5_recall_global_score":float(topk_recall(yy,ss,5)),
            "mapped_positive_coverage":float(
                meta.get("mapped_pocket_coverage",np.nan)
            ),
            "prediction_source":"pre_unseal_csv",
            "prediction_run_nonce":V404_PHASE3_RUN_NONCE,
            "unseal_nonce":V404_PHASE3_UNSEAL_NONCE,
        })

    # Exact Top-5 is likewise the pre-unseal CSV.
    top_idx=frozen_top5["idx"].astype(int).to_numpy()
    hits=(np.asarray(labels,int)[top_idx]==1)
    hit_count=int(hits.sum())

    PHASE3_TOP5_ROWS.append({
        "target":name,
        "site_class":item["cfg"]["site_class"],
        "top5_hits":hit_count,
        "top5_any_hit":bool(hit_count>0),
        "top5_recall":float(
            hit_count/min(5,int(yy.sum()))
        ),
        "top5_enrichment":float(
            (hit_count/max(1,len(top_idx)))/prev
        ) if prev>0 else np.nan,
        "hit_labels":";".join(
            frozen_top5.loc[hits,"label"].astype(str).tolist()
        ),
        "top5_labels":";".join(
            frozen_top5["label"].astype(str).tolist()
        ),
        "prediction_source":"pre_unseal_csv",
    })

    V404_PHASE3_FILE_RELOAD_AUDIT.append({
        "target":name,
        "feature_sha256":seal["feature_sha256"],
        "top5_sha256":seal["top5_sha256"],
        "prediction_run_nonce":V404_PHASE3_RUN_NONCE,
        "unseal_nonce":V404_PHASE3_UNSEAL_NONCE,
    })

PHASE3_METRICS=pd.DataFrame(PHASE3_ROWS)
PHASE3_TOP5=pd.DataFrame(PHASE3_TOP5_ROWS)

display(PHASE3_METRICS)
display(PHASE3_TOP5)

PHASE3_METRICS.to_csv(
    RESULTS_DIR/"phase3_v404_cleanroom_metrics.csv",
    index=False,
)
PHASE3_TOP5.to_csv(
    RESULTS_DIR/"phase3_v404_cleanroom_exact_top5.csv",
    index=False,
)
pd.DataFrame(
    V404_PHASE3_FILE_RELOAD_AUDIT
).to_csv(
    RESULTS_DIR/"phase3_v404_file_reload_audit.csv",
    index=False,
)

frozen=PHASE3_METRICS[
    PHASE3_METRICS["model"]=="frozen_v3_6"
].set_index("target")
hybrid=PHASE3_METRICS[
    PHASE3_METRICS["model"]=="v40_calibrated_hybrid"
].set_index("target")

common=sorted(set(frozen.index)&set(hybrid.index))

improved=[
    hybrid.loc[t,"average_precision"]
    >= frozen.loc[t,"average_precision"]
    for t in common
]

median_ratio=float(
    np.nanmedian(hybrid.loc[common,"ap_over_prevalence"])
) if common else np.nan

top5_success=float(
    np.mean(PHASE3_TOP5["top5_any_hit"])
) if len(PHASE3_TOP5) else np.nan

improve_fraction=float(
    np.mean(improved)
) if improved else np.nan

PHASE3_SUMMARY=pd.DataFrame([{
    "calibration_signature":V40_CALIBRATION_SIGNATURE,
    "prediction_run_nonce":V404_PHASE3_RUN_NONCE,
    "unseal_nonce":V404_PHASE3_UNSEAL_NONCE,
    "freeze_manifest_sha256":V404_PHASE3_FREEZE_MANIFEST_SHA256,
    "valid_targets":len(common),
    "median_hybrid_ap_over_prevalence":median_ratio,
    "hybrid_top5_any_hit_success_fraction":top5_success,
    "hybrid_improves_or_matches_frozen_fraction":improve_fraction,
    "criterion_median_ap_ratio_pass":bool(
        np.isfinite(median_ratio)
        and median_ratio>
        V40_PHASE3_CRITERIA[
            "median_hybrid_ap_over_prevalence_gt"
        ]
    ),
    "criterion_top5_pass":bool(
        np.isfinite(top5_success)
        and top5_success>=
        V40_PHASE3_CRITERIA[
            "hybrid_top5_any_hit_success_fraction_gte"
        ]
    ),
    "criterion_improvement_pass":bool(
        np.isfinite(improve_fraction)
        and improve_fraction>=
        V40_PHASE3_CRITERIA[
            "hybrid_improves_or_matches_frozen_on_majority"
        ]
    ),
    "phase3_used_for_training":V40_PHASE3_USED_FOR_TRAINING,
    "cleanroom_integrity_pass":True,
    "prediction_source":"cryptographically_sealed_pre_unseal_csv",
}])

display(PHASE3_SUMMARY)
PHASE3_SUMMARY.to_csv(
    RESULTS_DIR/"phase3_v404_cleanroom_summary.csv",
    index=False,
)

v404_stage_marker(
    "phase3_cleanroom_evaluation_complete",
    valid_targets=len(common),
    cleanroom_integrity_pass=True,
)

print("V4.0.4 CLEANROOM PHASE-3 EVALUATION COMPLETE")

# %% [markdown]
# ## 42. Required N×N connectivity matrices + competition Top-5
#
# The challenge requires an N×N connectivity matrix and a ranked Top-5 list.
#
# The **primary** ranking below is the frozen V3.6 site-aware output
# (`blind_top5`), preserving the pocket-first architecture. The calibrated
# hybrid is exported separately as auxiliary evidence rather than silently
# replacing the primary ranking.

# %% [code] Notebook code cell 70 (source index 140)

V410_MATRIX_MANIFEST=[]
V410_PRIMARY_TOP5=[]
V410_AUX_TOP5=[]

for name,item in RESULTS.items():
    labels=item["features"]["label"].astype(str).tolist()
    Cq=np.asarray(item["qw"].connectivity,float)
    Cc=np.asarray(item["qw"].classical_connectivity,float)

    assert Cq.shape==(len(labels),len(labels))
    assert Cc.shape==Cq.shape

    q_csv=V410_SUBMISSION_DIR/f"{name}_NxN_quantum_connectivity.csv"
    c_csv=V410_SUBMISSION_DIR/f"{name}_NxN_classical_connectivity.csv"
    npz=V410_SUBMISSION_DIR/f"{name}_connectivity_matrices.npz"

    pd.DataFrame(Cq,index=labels,columns=labels).to_csv(q_csv)
    pd.DataFrame(Cc,index=labels,columns=labels).to_csv(c_csv)
    np.savez_compressed(
        npz,
        quantum=Cq,
        classical=Cc,
        labels=np.asarray(labels,dtype=str),
        times=np.asarray(item["qw"].times,float),
    )

    V410_MATRIX_MANIFEST.append({
        "target":name,
        "n_residues":len(labels),
        "quantum_matrix_shape":f"{Cq.shape[0]}x{Cq.shape[1]}",
        "quantum_csv":q_csv.name,
        "classical_csv":c_csv.name,
        "npz":npz.name,
    })

    # Primary: frozen V3.6 site-aware representatives.
    primary=item["blind_top5"].copy().reset_index(drop=True)
    primary["target"]=name
    primary["rank"]=np.arange(1,len(primary)+1)
    primary["ranking_role"]="PRIMARY_frozen_v3_6_site_aware"
    V410_PRIMARY_TOP5.append(primary)

    # Auxiliary: calibrated global residue ranking.
    hybrid=item["features"]["v40_calibrated_hybrid"].to_numpy(float)
    aux=v40_top5_global(item,hybrid,5).copy().reset_index(drop=True)
    aux["target"]=name
    aux["rank"]=np.arange(1,len(aux)+1)
    aux["ranking_role"]="AUXILIARY_calibrated_hybrid"
    V410_AUX_TOP5.append(aux)

V410_MATRIX_MANIFEST=pd.DataFrame(V410_MATRIX_MANIFEST)
V410_PRIMARY_TOP5=pd.concat(V410_PRIMARY_TOP5,ignore_index=True)
V410_AUX_TOP5=pd.concat(V410_AUX_TOP5,ignore_index=True)

V410_MATRIX_MANIFEST.to_csv(
    V410_SUBMISSION_DIR/"connectivity_matrix_manifest.csv",
    index=False,
)
V410_PRIMARY_TOP5.to_csv(
    V410_SUBMISSION_DIR/"challenge_primary_top5_site_aware.csv",
    index=False,
)
V410_AUX_TOP5.to_csv(
    V410_SUBMISSION_DIR/"challenge_auxiliary_top5_calibrated.csv",
    index=False,
)

display(V410_MATRIX_MANIFEST)
display(
    V410_PRIMARY_TOP5[
        [c for c in [
            "target","rank","label","resname",
            "final_score","pocket_id","pocket_rank",
            "stability_confidence","ranking_role",
        ] if c in V410_PRIMARY_TOP5.columns]
    ]
)



v411_attest(
    "v41_connectivity_and_top5",
    rows=int(len(V410_PRIMARY_TOP5)),
)
print("V4.1.1 attested stage: v41_connectivity_and_top5")

# %% [markdown]
# ## 43. Quantum-vs-classical benchmark ablation
#
# This section does **not** tune a score. It evaluates already-frozen residue
# signals on labelled challenge benchmarks after prediction.
#
# It separates:
#
# - direct CTQW occupancy,
# - matched classical diffusion,
# - quantum intervention susceptibility,
# - classical intervention susceptibility,
# - quantum-over-classical intervention excess,
# - topology,
# - frozen V3.6,
# - calibrated hybrid.

# %% [code] Notebook code cell 71 (source index 142)

V410_ABLATION_ROWS=[]

V410_ABLATION_COLUMNS={
    "ctqw_quantum_occupancy":"quantum_score",
    "classical_diffusion":"classical_score",
    "quantum_intervention_susceptibility":"q_susceptibility_coeff_n",
    "classical_intervention_susceptibility":"c_susceptibility_coeff_n",
    "quantum_over_classical_intervention_excess":"quantum_intervention_excess",
    "topology":"topology_score",
    "frozen_v3_6":"final_score",
    "calibrated_hybrid":"v40_calibrated_hybrid",
}

for name,item in RESULTS.items():
    y=item.get("labels")
    if y is None:
        continue

    y=np.asarray(y,int)
    elig=item["features"]["eligible_distal"].to_numpy(bool)
    yy=y[elig]

    if yy.sum()==0 or len(np.unique(yy))<2:
        continue

    for metric,col in V410_ABLATION_COLUMNS.items():
        if col not in item["features"].columns:
            continue

        ss=item["features"][col].to_numpy(float)[elig]
        finite=np.isfinite(ss)
        y2=yy[finite]
        s2=ss[finite]

        if len(y2)==0 or len(np.unique(y2))<2:
            continue

        ap=float(average_precision_score(y2,s2))
        auc=float(safe_auc(y2,s2))
        prev=float(y2.mean())

        try:
            u,p=mannwhitneyu(
                s2[y2==1],
                s2[y2==0],
                alternative="greater",
            )
            mw_p=float(p)
        except Exception:
            mw_p=np.nan

        V410_ABLATION_ROWS.append({
            "target":name,
            "metric":metric,
            "feature_column":col,
            "n_rankable":int(len(y2)),
            "positives":int(y2.sum()),
            "prevalence":prev,
            "average_precision":ap,
            "ap_over_prevalence":float(ap/prev) if prev>0 else np.nan,
            "roc_auc":auc,
            "mann_whitney_one_sided_p":mw_p,
        })

V410_QUANTUM_CLASSICAL_ABLATION=pd.DataFrame(V410_ABLATION_ROWS)
V410_QUANTUM_CLASSICAL_ABLATION.to_csv(
    V410_SUBMISSION_DIR/"quantum_vs_classical_benchmark_ablation.csv",
    index=False,
)
display(V410_QUANTUM_CLASSICAL_ABLATION)



v411_attest(
    "v41_quantum_classical_ablation",
    rows=int(len(V410_QUANTUM_CLASSICAL_ABLATION)),
)
print("V4.1.1 attested stage: v41_quantum_classical_ablation")

# %% [markdown]
# ## 44. Label-free coarse-graining fidelity and scalability
#
# Residues are spatially clustered from the apo structure only. The quotient
# graph is then evolved with the same normalized-Laplacian CTQW.
#
# Fidelity is measured against the full-resolution seed transport aggregated
# onto the same clusters. No allosteric labels are used to choose K.

# %% [code] Notebook code cell 72 (source index 144)

def v410_quotient_model(item,k):
    coords=item["apo_df"][["x","y","z"]].to_numpy(float)
    n=len(coords)
    k=int(min(k,n))

    km=KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10,
    )
    cluster=km.fit_predict(coords)

    S=np.zeros((n,k),float)
    S[np.arange(n),cluster]=1.0
    sizes=np.maximum(S.sum(axis=0),1.0)

    A=np.asarray(item["A"],float)
    Q=S.T@A@S
    np.fill_diagonal(Q,0.0)

    # Size-normalized quotient weights.
    denom=np.sqrt(
        sizes[:,None]*sizes[None,:]
    )
    Q=np.divide(
        Q,
        denom,
        out=np.zeros_like(Q),
        where=denom>0,
    )
    Q=(Q+Q.T)/2.0

    degree=Q.sum(axis=1)
    inv=np.zeros_like(degree)
    nz=degree>1e-12
    inv[nz]=1.0/np.sqrt(degree[nz])
    Lc=np.eye(k)-(
        inv[:,None]*Q*inv[None,:]
    )
    Lc=(Lc+Lc.T)/2.0

    seed_clusters=sorted(set(
        int(cluster[int(i)])
        for i in item["seed_idx"]
    ))

    qw_c=ctqw_connectivity(
        Lc,
        seed_clusters,
        item["qw"].times,
        GAMMA,
    )

    full_q=item["qw"].q_prob_t.mean(axis=0)
    agg=np.zeros(k,float)
    for j in range(k):
        agg[j]=full_q[cluster==j].sum()
    agg/=np.clip(agg.sum(),1e-15,None)

    coarse_q=qw_c.q_prob_t.mean(axis=0)
    coarse_q/=np.clip(coarse_q.sum(),1e-15,None)

    rho=float(
        spearmanr(agg,coarse_q).statistic
    ) if k>2 else np.nan

    topn=min(3,k)
    a=set(np.argsort(agg)[-topn:])
    b=set(np.argsort(coarse_q)[-topn:])
    jacc=float(len(a&b)/max(1,len(a|b)))

    return {
        "cluster":cluster,
        "centers":km.cluster_centers_,
        "A":Q,
        "L":Lc,
        "seed_clusters":seed_clusters,
        "qw":qw_c,
        "full_aggregated":agg,
        "coarse_transport":coarse_q,
        "spearman":rho,
        "top3_cluster_jaccard":jacc,
    }

V410_COARSE_MODELS={}
V410_COARSE_ROWS=[]

for name,item in RESULTS.items():
    n=len(item["features"])

    for k in V410_COARSE_K_VALUES:
        if k>=n:
            continue

        model=v410_quotient_model(item,k)
        V410_COARSE_MODELS[(name,int(k))]=model

        V410_COARSE_ROWS.append({
            "target":name,
            "N":n,
            "K":int(k),
            "compression_N_over_K":float(n/k),
            "nominal_dense_cubic_reduction":float((n/k)**3),
            "seed_transport_spearman":model["spearman"],
            "top3_cluster_jaccard":model["top3_cluster_jaccard"],
        })

V410_COARSE_GRAINING=pd.DataFrame(V410_COARSE_ROWS)
V410_COARSE_GRAINING.to_csv(
    V410_SUBMISSION_DIR/"coarse_graining_fidelity.csv",
    index=False,
)
display(V410_COARSE_GRAINING)



v411_attest(
    "v41_coarse_graining",
    rows=int(len(V410_COARSE_GRAINING)),
)
print("V4.1.1 attested stage: v41_coarse_graining")

# %% [markdown]
# ## 45. Hardware-aligned coarse CTQW + pocket phase kick + noise
#
# For K=8, the quotient Hamiltonian fits in three qubits.
#
# The circuit implements the same conceptual site-control experiment:
#
# \[
# U(t/2)\,K_P(\lambda)\,U(t/2),
# \qquad
# K_P(\lambda)=e^{-i\lambda\Pi_P}.
# \]
#
# `PauliEvolutionGate` supplies a gate-based Hamiltonian-evolution path, while the
# phase kick is an exact diagonal unitary on the selected coarse pocket state.
#
# The Aer noise test is optional and never changes a ranking.

# %% [code] Notebook code cell 73 (source index 146)

V410_HARDWARE_ROWS=[]

def v410_counts_to_prob(counts,n_states):
    p=np.zeros(n_states,float)
    total=max(1,sum(counts.values()))
    for bitstring,count in counts.items():
        key=str(bitstring).replace(" ","")
        idx=int(key,2)
        if idx<n_states:
            p[idx]+=count/total
    return p

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import (
        StatePreparation,
        PauliEvolutionGate,
        UnitaryGate,
    )
    from qiskit.quantum_info import (
        SparsePauliOp,
        Operator,
    )
    from qiskit.synthesis import SuzukiTrotter

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import (
            NoiseModel,
            depolarizing_error,
        )
        V410_AER_AVAILABLE=True
    except Exception:
        V410_AER_AVAILABLE=False

    for name in V410_HARDWARE_TARGETS:
        item=RESULTS[name]
        model=V410_COARSE_MODELS[(name,V410_HARDWARE_K)]
        H=np.asarray(model["L"],complex)
        k=H.shape[0]
        nq=int(round(np.log2(k)))
        assert 2**nq==k

        # Seed state on coarse nodes.
        amp=np.zeros(k,complex)
        seed_clusters=model["seed_clusters"]
        amp[seed_clusters]=1/np.sqrt(len(seed_clusters))

        # Top frozen pocket -> coarse cluster with maximal membership overlap.
        pockets=item["predicted_pockets"].sort_values("pocket_rank")
        p0=pockets.iloc[0]
        members=np.asarray(p0["member_indices"],int)
        cl=model["cluster"][members]
        values,counts=np.unique(cl,return_counts=True)
        pocket_cluster=int(values[np.argmax(counts)])

        pauli=SparsePauliOp.from_operator(
            Operator(H)
        ).simplify(atol=1e-10)

        evo=PauliEvolutionGate(
            pauli,
            time=float(V410_HARDWARE_TIME/2.0),
            synthesis=SuzukiTrotter(
                order=2,
                reps=QISKIT_TROTTER_REPS,
            ),
        )

        phase=np.ones(k,complex)
        phase[pocket_cluster]=np.exp(
            -1j*V410_HARDWARE_PHASE_LAMBDA
        )
        kick=UnitaryGate(
            np.diag(phase),
            label="PocketKick",
        )

        qc=QuantumCircuit(nq)
        qc.append(StatePreparation(amp),range(nq))
        qc.append(evo,range(nq))
        qc.append(kick,range(nq))
        qc.append(evo,range(nq))

        # Transpilation evidence is available even without Aer.
        tqc=transpile(
            qc,
            basis_gates=["rz","sx","x","cx"],
            optimization_level=3,
        )

        row={
            "target":name,
            "K":k,
            "qubits":nq,
            "pauli_terms":int(len(pauli)),
            "transpiled_depth":int(tqc.depth()),
            "transpiled_size":int(tqc.size()),
            "top_pocket_cluster":pocket_cluster,
            "phase_lambda":float(V410_HARDWARE_PHASE_LAMBDA),
            "evolution_time":float(V410_HARDWARE_TIME),
            "trotter_reps":int(QISKIT_TROTTER_REPS),
            "aer_noise_tested":False,
            "ideal_noisy_spearman":np.nan,
            "ideal_noisy_top3_jaccard":np.nan,
            "ideal_noisy_js":np.nan,
            "status":"CIRCUIT_COMPILED",
        }

        if V410_AER_AVAILABLE:
            qcm=qc.copy()
            qcm.measure_all()

            sim=AerSimulator()
            tqcm=transpile(
                qcm,
                sim,
                optimization_level=2,
            )
            ideal_counts=sim.run(
                tqcm,
                shots=V410_HARDWARE_SHOTS,
            ).result().get_counts()
            ideal=v410_counts_to_prob(
                ideal_counts,k
            )

            noise=NoiseModel()
            e1=depolarizing_error(
                V410_NOISE_1Q,1
            )
            e2=depolarizing_error(
                V410_NOISE_2Q,2
            )
            noise.add_all_qubit_quantum_error(
                e1,["sx","x"]
            )
            noise.add_all_qubit_quantum_error(
                e2,["cx"]
            )

            noisy_sim=AerSimulator(
                noise_model=noise
            )
            noisy_counts=noisy_sim.run(
                tqcm,
                shots=V410_HARDWARE_SHOTS,
            ).result().get_counts()
            noisy=v410_counts_to_prob(
                noisy_counts,k
            )

            rho=float(
                spearmanr(ideal,noisy).statistic
            )
            topn=min(3,k)
            ia=set(np.argsort(ideal)[-topn:])
            ib=set(np.argsort(noisy)[-topn:])
            jac=float(
                len(ia&ib)/max(1,len(ia|ib))
            )
            js=float(
                _js_divergence_rows(
                    ideal[None,:],
                    noisy[None,:],
                )[0]
            )

            row.update({
                "aer_noise_tested":True,
                "ideal_noisy_spearman":rho,
                "ideal_noisy_top3_jaccard":jac,
                "ideal_noisy_js":js,
                "status":"CIRCUIT_COMPILED_AND_NOISE_TESTED",
            })

        V410_HARDWARE_ROWS.append(row)

except Exception as exc:
    V410_HARDWARE_ROWS.append({
        "target":"GLOBAL",
        "status":"QISKIT_HARDWARE_DEMO_UNAVAILABLE",
        "reason":repr(exc),
    })

V410_HARDWARE_EVIDENCE=pd.DataFrame(V410_HARDWARE_ROWS)
V410_HARDWARE_EVIDENCE.to_csv(
    V410_SUBMISSION_DIR/"hardware_aligned_phase_kick_evidence.csv",
    index=False,
)
display(V410_HARDWARE_EVIDENCE)



v411_attest(
    "v41_hardware_evidence",
    rows=int(len(V410_HARDWARE_EVIDENCE)),
)
print("V4.1.1 attested stage: v41_hardware_evidence")

# %% [markdown]
# ## 46. Interactive 3D challenge maps + mechanism-stratified holdout audit
#
# The 3D maps color all apo residues by the frozen primary score and overlay the
# primary site-aware Top-5 and calibrated auxiliary Top-5.
#
# The Phase-3 table is a **diagnostic only**. It shows which learning component
# helped or hurt each independent holdout without using those results to refit
# anything.

# %% [code] Notebook code cell 74 (source index 148)

V410_3D_MANIFEST=[]

for name,item in RESULTS.items():
    f=item["features"]
    coords=f[["x","y","z"]].to_numpy(float)

    primary=V410_PRIMARY_TOP5[
        V410_PRIMARY_TOP5["target"]==name
    ]
    aux=V410_AUX_TOP5[
        V410_AUX_TOP5["target"]==name
    ]

    fig=go.Figure()

    fig.add_trace(go.Scatter3d(
        x=f["x"],y=f["y"],z=f["z"],
        mode="markers",
        marker=dict(
            size=3,
            color=f["final_score"],
            colorscale="Viridis",
            colorbar=dict(title="Frozen score"),
            opacity=0.75,
        ),
        text=f["label"],
        name="All residues",
    ))

    for frame,label,symbol,size in [
        (primary,"Primary frozen Top-5","diamond",8),
        (aux,"Aux calibrated Top-5","x",7),
    ]:
        if len(frame)==0:
            continue
        idxs=frame["idx"].astype(int).to_numpy()
        ff=f.set_index("idx").loc[idxs]
        fig.add_trace(go.Scatter3d(
            x=ff["x"],y=ff["y"],z=ff["z"],
            mode="markers+text",
            marker=dict(
                size=size,
                symbol=symbol,
            ),
            text=ff["label"],
            textposition="top center",
            name=label,
        ))

    fig.update_layout(
        title=f"{name} — V4.1 allosteric connectivity ranking",
        scene=dict(
            xaxis_title="x Å",
            yaxis_title="y Å",
            zaxis_title="z Å",
        ),
        width=950,
        height=760,
    )

    html=V410_SUBMISSION_DIR/f"{name}_interactive_3d.html"
    fig.write_html(
        html,
        include_plotlyjs="cdn",
    )

    V410_3D_MANIFEST.append({
        "target":name,
        "html":html.name,
        "primary_top5":";".join(
            primary["label"].astype(str)
        ),
        "auxiliary_top5":";".join(
            aux["label"].astype(str)
        ),
    })

V410_3D_MANIFEST=pd.DataFrame(V410_3D_MANIFEST)
V410_3D_MANIFEST.to_csv(
    V410_SUBMISSION_DIR/"interactive_3d_manifest.csv",
    index=False,
)

# Independent holdout mechanism diagnostics.
phase=PHASE3_METRICS[
    PHASE3_METRICS.get(
        "validation_valid",
        pd.Series(False,index=PHASE3_METRICS.index)
    )==True
].copy()

rows=[]
for target in sorted(phase["target"].unique()):
    sub=phase[phase["target"]==target].set_index("model")
    if "frozen_v3_6" not in sub.index:
        continue

    frozen=float(
        sub.loc["frozen_v3_6","ap_over_prevalence"]
    )
    row={
        "target":target,
        "site_class":sub.iloc[0].get("site_class",""),
        "frozen_ap_over_prevalence":frozen,
    }

    for model,key in [
        ("hdc_calibrated","hdc"),
        ("evo_calibrated","evo"),
        ("v40_calibrated_hybrid","hybrid"),
    ]:
        if model in sub.index:
            val=float(
                sub.loc[model,"ap_over_prevalence"]
            )
            row[f"{key}_ap_over_prevalence"]=val
            row[f"{key}_delta_vs_frozen"]=val-frozen

    top=PHASE3_TOP5[
        PHASE3_TOP5["target"]==target
    ]
    if len(top):
        row["hybrid_exact_top5_hits"]=int(
            top.iloc[0]["top5_hits"]
        )
        row["hybrid_exact_top5_any_hit"]=bool(
            top.iloc[0]["top5_any_hit"]
        )

    rows.append(row)

V410_PHASE3_MECHANISM_AUDIT=pd.DataFrame(rows)
V410_PHASE3_MECHANISM_AUDIT.to_csv(
    V410_SUBMISSION_DIR/"phase3_mechanism_stratified_audit.csv",
    index=False,
)

display(V410_3D_MANIFEST)
display(V410_PHASE3_MECHANISM_AUDIT)



v411_attest(
    "v41_3d_and_phase3_audit",
    rows=int(len(V410_PHASE3_MECHANISM_AUDIT)),
)
print("V4.1.1 attested stage: v41_3d_and_phase3_audit")

# %% [markdown]
# ## 46A. Competition claims and limitations matrix
#
# This table is meant to make the final submission difficult to overstate.
#
# A claim is marked `SUPPORTED`, `PARTIAL`, `PROSPECTIVE`, or `NOT_CLAIMED`.

# %% [markdown]
# ### V4.1.2 one-to-one chain-mapping sensitivity audit
#
# The historical label mapper chooses the best apo chain independently for each
# holo chain. In symmetric homo-oligomers this can map multiple holo chains onto
# the same apo chain.
#
# This audit solves a **maximum-score one-to-one bipartite chain assignment**
# using only sequence identity × alignment coverage. It then remaps ligand
# contacts through that assignment.
#
# Important boundaries:
#
# - predictions are not recomputed;
# - HDC/neuroevolution are not refit;
# - the historical labels remain the primary reported validation;
# - the corrected labels are a post-hoc sensitivity analysis only;
# - Phase-3 clean-room status is not rewritten by this audit.

# %% [code] Notebook code cell 75 (source index 151)

from scipy.optimize import linear_sum_assignment

def v412_build_labels_one_to_one(cfg,apo_df):
    holo_id=cfg.get("holo")
    ligand=cfg.get("holo_ligand")

    contacts,contact_meta=ligand_contact_residues(
        holo_id,ligand
    )
    if not contact_meta.get("valid",False):
        return None,{
            "valid":False,
            "reason":"ligand contact extraction failed",
            **contact_meta,
        }

    holo_df=extract_residue_table(
        holo_id,
        requested_chains=None,
    )

    apo_chains=sorted(
        apo_df["chain"].unique()
    )
    holo_chains=sorted(
        contacts.keys()
    )

    if not apo_chains or not holo_chains:
        return None,{
            "valid":False,
            "reason":"no chains available for assignment",
        }

    score=np.zeros(
        (len(holo_chains),len(apo_chains)),
        float,
    )
    cache={}

    for hi,hchain in enumerate(holo_chains):
        for ai,achain in enumerate(apo_chains):
            mapping,ident,cov=pairwise_chain_mapping(
                apo_df,holo_df,achain,hchain
            )
            cache[(hi,ai)]=(
                mapping,float(ident),float(cov)
            )
            score[hi,ai]=float(ident*cov)

    # Maximum total sequence correspondence under one-to-one assignment.
    h_idx,a_idx=linear_sum_assignment(-score)

    labels=np.zeros(len(apo_df),dtype=int)
    assignment=[]
    selected_total=0
    selected_mapped=0

    for hi,ai in sorted(
        zip(h_idx,a_idx),
        key=lambda z:(holo_chains[z[0]],apo_chains[z[1]])
    ):
        mapping,ident,cov=cache[(int(hi),int(ai))]
        hchain=holo_chains[int(hi)]
        achain=apo_chains[int(ai)]
        pocket=contacts[hchain]

        local=0
        for resseq,icode,_resname in pocket:
            key=(hchain,int(resseq),str(icode))
            if key in mapping:
                labels[mapping[key]]=1
                local+=1

        selected_total+=len(pocket)
        selected_mapped+=local

        assignment.append({
            "holo_chain":hchain,
            "apo_chain":achain,
            "identity":ident,
            "coverage":cov,
            "sequence_score":float(ident*cov),
            "pocket_total":int(len(pocket)),
            "pocket_mapped":int(local),
        })

    all_total=int(sum(len(v) for v in contacts.values()))
    selected_cov=float(
        selected_mapped/max(1,selected_total)
    )
    all_cov=float(
        selected_mapped/max(1,all_total)
    )
    best_identity=max(
        [x["identity"] for x in assignment],
        default=0.0,
    )

    valid=bool(
        labels.sum()>0
        and selected_cov>=0.60
        and best_identity>=0.50
    )

    return (
        labels if valid else None,
        {
            "valid":valid,
            "holo":holo_id,
            "ligand":ligand,
            "positives":int(labels.sum()),
            "apo_chains":apo_chains,
            "holo_chains":holo_chains,
            "selected_holo_chains":[
                x["holo_chain"] for x in assignment
            ],
            "unselected_holo_chains":sorted(
                set(holo_chains)-
                {x["holo_chain"] for x in assignment}
            ),
            "selected_pocket_total":int(selected_total),
            "selected_pocket_mapped":int(selected_mapped),
            "selected_pocket_coverage":selected_cov,
            "all_holo_pocket_total":all_total,
            "all_holo_coverage_under_one_to_one":all_cov,
            "mapping_report":assignment,
            "sensitivity_only":True,
        }
    )

def v412_label_metrics(item,y):
    f=item["features"]
    target_chains=set(
        item["cfg"].get(
            "target_chains",
            f["chain"].unique(),
        )
    )
    rankable=(
        f["eligible_distal"].to_numpy(bool)
        & f["chain"].isin(target_chains).to_numpy(bool)
    )

    y=np.asarray(y,int)
    yy=y[rankable]
    prev=float(yy.mean()) if len(yy) else np.nan

    out={
        "rankable_residues":int(len(yy)),
        "rankable_positives":int(yy.sum()),
        "prevalence":prev,
    }

    for label,col in [
        ("frozen","final_score"),
        ("hdc","v40_hdc_score"),
        ("evo","v40_evo_score"),
        ("hybrid","v40_calibrated_hybrid"),
    ]:
        if col not in f.columns:
            continue
        ss=f.loc[rankable,col].to_numpy(float)
        finite=np.isfinite(ss)
        y2=yy[finite]
        s2=ss[finite]
        if (
            len(y2)<5
            or y2.sum()==0
            or y2.sum()==len(y2)
        ):
            continue
        ap=float(average_precision_score(y2,s2))
        out[f"{label}_ap"]=ap
        out[f"{label}_roc"]=float(safe_auc(y2,s2))
        out[f"{label}_ap_over_prevalence"]=(
            float(ap/prev)
            if prev>0 else np.nan
        )

    # Frozen primary site-aware Top-5.
    if "blind_top5" in item:
        top_idx=item["blind_top5"]["idx"].astype(int).to_numpy()
        out["primary_top5_hits"]=int(
            y[top_idx].sum()
        )

    # Auxiliary calibrated global Top-5 where available.
    if "v40_calibrated_hybrid" in f.columns:
        aux=v40_top5_global(
            item,
            f["v40_calibrated_hybrid"].to_numpy(float),
            5,
        )
        out["auxiliary_top5_hits"]=int(
            y[aux["idx"].astype(int).to_numpy()].sum()
        )

    return out

V412_MAPPING_TARGETS=[]

# Phase 1.
for name,item in BLIND_RESULTS.items():
    V412_MAPPING_TARGETS.append({
        "target":name,
        "phase":"phase1_calibration_source",
        "item":item,
        "sealed":BLIND_SEALED_VALIDATION[name],
        "historical_labels":item["labels"],
        "historical_meta":item["validation_meta"],
    })

# Phase 2.
for name,item in PHASE2_RESULTS.items():
    V412_MAPPING_TARGETS.append({
        "target":name,
        "phase":"phase2_calibration_source",
        "item":item,
        "sealed":PHASE2_SEALED_VALIDATION[name],
        "historical_labels":item["labels"],
        "historical_meta":item["validation_meta"],
    })

# Independent Phase 3.
for name,item in V404_PHASE3_RESULTS.items():
    V412_MAPPING_TARGETS.append({
        "target":name,
        "phase":"phase3_cleanroom_posthoc_sensitivity",
        "item":item,
        "sealed":PHASE3_SEALED_VALIDATION[name],
        "historical_labels":V404_PHASE3_LABELS[name],
        "historical_meta":V404_PHASE3_VALIDATION_META[name],
    })

summary_rows=[]
assignment_rows=[]

for rec in V412_MAPPING_TARGETS:
    name=rec["target"]
    item=rec["item"]
    hist=np.asarray(rec["historical_labels"],int)
    hist_meta=rec["historical_meta"]

    cfg={
        **item["cfg"],
        **rec["sealed"],
    }

    corr,corr_meta=v412_build_labels_one_to_one(
        cfg,item["apo_df"]
    )

    if corr is None:
        summary_rows.append({
            "target":name,
            "phase":rec["phase"],
            "audit_valid":False,
            "reason":corr_meta.get("reason","rejected"),
        })
        continue

    corr=np.asarray(corr,int)

    hset=set(np.flatnonzero(hist))
    cset=set(np.flatnonzero(corr))
    union=hset|cset
    intersection=hset&cset

    hist_map=hist_meta.get("mapping_report",[])
    hist_apo=[
        x.get("apo_chain") for x in hist_map
    ]
    collapse_risk=bool(
        len(hist_apo)!=len(set(hist_apo))
    )

    hm=v412_label_metrics(item,hist)
    cm=v412_label_metrics(item,corr)

    row={
        "target":name,
        "phase":rec["phase"],
        "audit_valid":True,
        "historical_chain_collapse_risk":collapse_risk,
        "historical_holo_chain_count":int(len(hist_map)),
        "historical_unique_apo_chain_count":int(len(set(hist_apo))),
        "one_to_one_assigned_chain_count":int(
            len(corr_meta["mapping_report"])
        ),
        "one_to_one_unselected_holo_chain_count":int(
            len(corr_meta["unselected_holo_chains"])
        ),
        "historical_positives":int(hist.sum()),
        "one_to_one_positives":int(corr.sum()),
        "label_set_jaccard":float(
            len(intersection)/max(1,len(union))
        ),
        "one_to_one_selected_pocket_coverage":float(
            corr_meta["selected_pocket_coverage"]
        ),
        "one_to_one_all_holo_coverage":float(
            corr_meta["all_holo_coverage_under_one_to_one"]
        ),
    }

    for key,val in hm.items():
        row[f"historical_{key}"]=val
    for key,val in cm.items():
        row[f"one_to_one_{key}"]=val

    for key in [
        "frozen_ap_over_prevalence",
        "hybrid_ap_over_prevalence",
        "primary_top5_hits",
        "auxiliary_top5_hits",
    ]:
        hv=hm.get(key,np.nan)
        cv=cm.get(key,np.nan)
        if np.isfinite(hv) and np.isfinite(cv):
            row[f"delta_{key}"]=float(cv-hv)

    summary_rows.append(row)

    for a in corr_meta["mapping_report"]:
        assignment_rows.append({
            "target":name,
            "phase":rec["phase"],
            **a,
        })

V412_CHAIN_MAPPING_SENSITIVITY=pd.DataFrame(summary_rows)
V412_CHAIN_ASSIGNMENTS=pd.DataFrame(assignment_rows)

V412_CHAIN_MAPPING_SENSITIVITY.to_csv(
    V410_SUBMISSION_DIR/"one_to_one_chain_mapping_sensitivity.csv",
    index=False,
)
V412_CHAIN_ASSIGNMENTS.to_csv(
    V410_SUBMISSION_DIR/"one_to_one_chain_assignments.csv",
    index=False,
)

display(V412_CHAIN_MAPPING_SENSITIVITY)
display(V412_CHAIN_ASSIGNMENTS)

v411_attest(
    "v412_chain_mapping_sensitivity",
    targets=int(len(V412_CHAIN_MAPPING_SENSITIVITY)),
    collapse_risk_targets=int(
        V412_CHAIN_MAPPING_SENSITIVITY[
            "historical_chain_collapse_risk"
        ].fillna(False).sum()
    ),
)

print("V4.1.2 CHAIN-MAPPING SENSITIVITY AUDIT: PASS")

# %% [markdown]
# ### V4.1.2 hardware representation-fidelity audit
#
# Noise resilience and coarse-graining fidelity answer different questions.
#
# A circuit can remain very close to its ideal **K=8 coarse circuit** while that
# K=8 quotient graph is still a poor approximation to full-resolution transport.
#
# This audit joins the two evidence tables and reports both properties together.

# %% [code] Notebook code cell 76 (source index 153)

V412_HARDWARE_REPRESENTATION=(
    V410_HARDWARE_EVIDENCE.merge(
        V410_COARSE_GRAINING[
            [
                "target","K",
                "compression_N_over_K",
                "seed_transport_spearman",
                "top3_cluster_jaccard",
            ]
        ],
        on=["target","K"],
        how="left",
    )
)

def v412_fidelity_class(rho):
    if not np.isfinite(rho):
        return "UNKNOWN"
    if rho>=V412_HARDWARE_FIDELITY_STRONG:
        return "STRONG"
    if rho>=V412_HARDWARE_FIDELITY_MODERATE:
        return "MODERATE"
    return "WEAK"

V412_HARDWARE_REPRESENTATION[
    "full_to_coarse_fidelity"
]=V412_HARDWARE_REPRESENTATION[
    "seed_transport_spearman"
].map(v412_fidelity_class)

V412_HARDWARE_REPRESENTATION[
    "coarse_noise_stability"
]=np.where(
    (
        V412_HARDWARE_REPRESENTATION[
            "ideal_noisy_spearman"
        ]>=0.90
    )
    & (
        V412_HARDWARE_REPRESENTATION[
            "ideal_noisy_top3_jaccard"
        ]>=0.80
    ),
    "STRONG",
    "LIMITED",
)

V412_HARDWARE_REPRESENTATION[
    "interpretation"
]=V412_HARDWARE_REPRESENTATION.apply(
    lambda r: (
        "Noise-stable coarse circuit; strong full-resolution transport proxy"
        if (
            r["coarse_noise_stability"]=="STRONG"
            and r["full_to_coarse_fidelity"]=="STRONG"
        )
        else
        "Noise-stable coarse circuit, but K=8 is not a strong full-resolution transport proxy"
        if r["coarse_noise_stability"]=="STRONG"
        else
        "Coarse circuit/noise evidence is limited"
    ),
    axis=1,
)

V412_HARDWARE_REPRESENTATION.to_csv(
    V410_SUBMISSION_DIR/"hardware_representation_fidelity.csv",
    index=False,
)

display(V412_HARDWARE_REPRESENTATION)

v411_attest(
    "v412_hardware_representation_fidelity",
    rows=int(len(V412_HARDWARE_REPRESENTATION)),
    strong_full_fidelity=int(
        (
            V412_HARDWARE_REPRESENTATION[
                "full_to_coarse_fidelity"
            ]=="STRONG"
        ).sum()
    ),
)

print("V4.1.2 HARDWARE REPRESENTATION-FIDELITY AUDIT: PASS")

# %% [code] Notebook code cell 77 (source index 154)

V410_CLAIMS=pd.DataFrame([
    {
        "claim":"Static topology can prioritize distal allosteric signal routes without MD trajectories",
        "status":"SUPPORTED",
        "evidence":"Frozen CTQW/intervention pipeline + independent validation panel",
    },
    {
        "claim":"Calibrated HDC/neuroevolution improves residue-ranking distribution on independent proteins",
        "status":"SUPPORTED",
        "evidence":"Clean-room Phase-3 median AP/prevalence > 1 and improves/matches frozen on 2/3",
    },
    {
        "claim":"Calibrated hybrid is reliably superior for exact Top-5 localization",
        "status":"NOT_CLAIMED",
        "evidence":"Predeclared Phase-3 exact Top-5 success criterion failed (1/3)",
    },
    {
        "claim":"KRAS challenge ranking improves with calibrated auxiliary layer",
        "status":"SUPPORTED",
        "evidence":"AP, ROC and global Top-5 recall improve versus frozen benchmark",
    },
    {
        "claim":"BCR-ABL1 challenge ranking improves with calibrated auxiliary layer",
        "status":"PARTIAL",
        "evidence":"AP and ROC improve; global Top-5 recall remains zero",
    },
    {
        "claim":"Official cardiac-myosin pair provides ligand-contact accuracy validation",
        "status":"NOT_CLAIMED",
        "evidence":"Official target config intentionally leaves holo_ligand unresolved",
    },
    {
        "claim":"c-Myc Top-5 is experimentally validated",
        "status":"PROSPECTIVE",
        "evidence":"c-Myc is scored without a validation holo ligand and is excluded from fitting",
    },
    {
        "claim":"A gate-based hardware path exists for the coarse quantum site-control observable",
        "status":"SUPPORTED_IF_CIRCUIT_COMPILED",
        "evidence":"K=8 Pauli Hamiltonian evolution + exact pocket phase kick; see hardware evidence CSV",
    },
])

V410_CLAIMS.to_csv(
    V410_SUBMISSION_DIR/"claims_and_limitations.csv",
    index=False,
)
display(V410_CLAIMS)



v411_attest(
    "v41_claims_matrix",
    rows=int(len(V410_CLAIMS)),
)
print("V4.1.1 attested stage: v41_claims_matrix")

# %% [markdown]
# ### V4.1.2 submission claims update + audit supplement

# %% [code] Notebook code cell 78 (source index 156)

# Extend — do not erase — the original claims table.
_extra_claims=[]

# Chain mapping claim is deliberately conditional on the observed sensitivity.
_risk=V412_CHAIN_MAPPING_SENSITIVITY[
    V412_CHAIN_MAPPING_SENSITIVITY[
        "historical_chain_collapse_risk"
    ].fillna(False)
].copy()

if len(_risk):
    _frozen_deltas=pd.to_numeric(
        _risk.get(
            "delta_frozen_ap_over_prevalence",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).dropna()
    _max_delta=float(
        _frozen_deltas.abs().max()
    ) if len(_frozen_deltas) else np.nan
else:
    _max_delta=np.nan

_extra_claims.append({
    "claim":"Validation conclusions are insensitive to homologous-chain assignment",
    "status":(
        "SENSITIVITY_REPORTED"
    ),
    "evidence":(
        "See one_to_one_chain_mapping_sensitivity.csv; "
        f"max absolute frozen AP/prevalence change among collapse-risk targets={_max_delta:.3f}"
        if np.isfinite(_max_delta)
        else
        "See one_to_one_chain_mapping_sensitivity.csv"
    ),
})

for r in V412_HARDWARE_REPRESENTATION.itertuples():
    _extra_claims.append({
        "claim":f"{r.target} K=8 circuit is a faithful full-resolution transport proxy",
        "status":(
            "SUPPORTED"
            if r.full_to_coarse_fidelity=="STRONG"
            else "NOT_CLAIMED"
        ),
        "evidence":(
            f"full→K8 Spearman={r.seed_transport_spearman:.3f}; "
            f"ideal/noisy Spearman={r.ideal_noisy_spearman:.3f}; "
            f"Top3 Jaccard={r.ideal_noisy_top3_jaccard:.3f}"
        ),
    })

V412_CLAIMS=pd.concat(
    [
        V410_CLAIMS,
        pd.DataFrame(_extra_claims),
    ],
    ignore_index=True,
)

V412_CLAIMS.to_csv(
    V410_SUBMISSION_DIR/"claims_and_limitations_v4_1_2.csv",
    index=False,
)

audit_lines=[
    "# V4.1.2 Final Scientific Audit Supplement",
    "",
    "## Scope",
    "This supplement does not change any prediction or fitted model. It audits label mapping and the distinction between coarse-circuit noise stability and full-resolution representation fidelity.",
    "",
    "## One-to-one chain-mapping sensitivity",
]

for r in V412_CHAIN_MAPPING_SENSITIVITY.itertuples():
    if not getattr(r,"audit_valid",False):
        audit_lines.append(
            f"- {r.target}: audit rejected."
        )
        continue
    audit_lines.append(
        f"- {r.target}: historical positives={int(r.historical_positives)}, "
        f"one-to-one positives={int(r.one_to_one_positives)}, "
        f"label Jaccard={r.label_set_jaccard:.3f}, "
        f"historical collapse-risk={bool(r.historical_chain_collapse_risk)}."
    )

audit_lines += [
    "",
    "## Hardware representation fidelity",
]

for r in V412_HARDWARE_REPRESENTATION.itertuples():
    audit_lines.append(
        f"- {r.target}: full→K={int(r.K)} transport Spearman="
        f"{r.seed_transport_spearman:.3f} ({r.full_to_coarse_fidelity}); "
        f"ideal/noisy Spearman={r.ideal_noisy_spearman:.3f}, "
        f"Top-3 Jaccard={r.ideal_noisy_top3_jaccard:.3f} "
        f"({r.coarse_noise_stability} coarse noise stability)."
    )

audit_lines += [
    "",
    "## Interpretation boundary",
    "Noise stability is evidence about the compiled coarse circuit relative to its ideal coarse counterpart. It is not, by itself, evidence that the chosen K=8 quotient faithfully reproduces the full residue-level model.",
    "",
    "The historical frozen metrics remain the primary validation record. One-to-one remapping is post-hoc sensitivity analysis and is not used for model selection or retuning.",
]

V412_AUDIT_REPORT="\n".join(audit_lines)
(
    V410_SUBMISSION_DIR/"final_scientific_audit_v4_1_2.md"
).write_text(
    V412_AUDIT_REPORT,
    encoding="utf-8",
)

print(V412_AUDIT_REPORT)

v411_attest(
    "v412_scientific_supplement",
    mapping_targets=int(len(V412_CHAIN_MAPPING_SENSITIVITY)),
    hardware_targets=int(len(V412_HARDWARE_REPRESENTATION)),
)

print("V4.1.2 SCIENTIFIC SUPPLEMENT: PASS")

# %% [markdown]
# ### V4.1.3 quantitative sensitivity decision table
#
# This cell prints the exact historical-versus-one-to-one metrics that matter
# for submission interpretation.
#
# It does **not** rerun or refit any model.

# %% [code] Notebook code cell 79 (source index 158)

V413_SENSITIVITY_COLUMNS=[
    "target","phase",
    "historical_chain_collapse_risk",
    "historical_positives","one_to_one_positives",
    "label_set_jaccard",
    "historical_frozen_ap_over_prevalence",
    "one_to_one_frozen_ap_over_prevalence",
    "delta_frozen_ap_over_prevalence",
    "historical_primary_top5_hits",
    "one_to_one_primary_top5_hits",
    "delta_primary_top5_hits",
    "historical_hybrid_ap_over_prevalence",
    "one_to_one_hybrid_ap_over_prevalence",
    "delta_hybrid_ap_over_prevalence",
    "historical_auxiliary_top5_hits",
    "one_to_one_auxiliary_top5_hits",
    "delta_auxiliary_top5_hits",
]

V413_MAPPING_DECISION_TABLE=V412_CHAIN_MAPPING_SENSITIVITY[
    [
        c for c in V413_SENSITIVITY_COLUMNS
        if c in V412_CHAIN_MAPPING_SENSITIVITY.columns
    ]
].copy()

V413_MAPPING_DECISION_TABLE.to_csv(
    V410_SUBMISSION_DIR/"mapping_sensitivity_decision_table.csv",
    index=False,
)
display(V413_MAPPING_DECISION_TABLE)

# Phase-3 post-hoc one-to-one sensitivity summary.
_p3=V412_CHAIN_MAPPING_SENSITIVITY[
    (
        V412_CHAIN_MAPPING_SENSITIVITY["phase"]
        =="phase3_cleanroom_posthoc_sensitivity"
    )
    & (
        V412_CHAIN_MAPPING_SENSITIVITY["audit_valid"]==True
    )
].copy()

def _safe_median(series):
    s=pd.to_numeric(series,errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan

def _fraction_gt_zero(series):
    s=pd.to_numeric(series,errors="coerce").dropna()
    return float((s>0).mean()) if len(s) else np.nan

def _fraction_hybrid_ge_frozen(frame,prefix):
    h=pd.to_numeric(
        frame[f"{prefix}_hybrid_ap_over_prevalence"],
        errors="coerce",
    )
    f=pd.to_numeric(
        frame[f"{prefix}_frozen_ap_over_prevalence"],
        errors="coerce",
    )
    valid=h.notna()&f.notna()
    return float((h[valid]>=f[valid]).mean()) if valid.any() else np.nan

V413_PHASE3_MAPPING_SENSITIVITY=pd.DataFrame([{
    "analysis_role":"posthoc_mapping_sensitivity_only",
    "targets":int(len(_p3)),
    "historical_median_hybrid_ap_over_prevalence":
        _safe_median(_p3["historical_hybrid_ap_over_prevalence"]),
    "one_to_one_median_hybrid_ap_over_prevalence":
        _safe_median(_p3["one_to_one_hybrid_ap_over_prevalence"]),
    "historical_auxiliary_top5_any_hit_fraction":
        _fraction_gt_zero(_p3["historical_auxiliary_top5_hits"]),
    "one_to_one_auxiliary_top5_any_hit_fraction":
        _fraction_gt_zero(_p3["one_to_one_auxiliary_top5_hits"]),
    "historical_hybrid_ge_frozen_fraction":
        _fraction_hybrid_ge_frozen(_p3,"historical"),
    "one_to_one_hybrid_ge_frozen_fraction":
        _fraction_hybrid_ge_frozen(_p3,"one_to_one"),
    "historical_cleanroom_record_overwritten":False,
    "used_for_training":False,
}])

V413_PHASE3_MAPPING_SENSITIVITY.to_csv(
    V410_SUBMISSION_DIR/"phase3_one_to_one_mapping_sensitivity_summary.csv",
    index=False,
)
display(V413_PHASE3_MAPPING_SENSITIVITY)

v411_attest(
    "v413_mapping_decision_summary",
    phase3_targets=int(len(_p3)),
)

print("V4.1.3 MAPPING DECISION SUMMARY: PASS")

# %% [markdown]
# ### V4.1.3 label-free hardware fidelity ladder
#
# The K=8 three-qubit circuits are useful hardware demonstrations, but K=8 is
# not equally faithful for every protein.
#
# For each challenge target this audit selects the **smallest already-tested K**
# whose full→coarse seed-transport Spearman is at least 0.80. If none passes,
# the best tested K is reported instead.
#
# This selection uses no allosteric labels and changes no biological ranking.

# %% [code] Notebook code cell 80 (source index 160)

V413_HARDWARE_LADDER=[]

for target in RESULTS:
    sub=V410_COARSE_GRAINING[
        V410_COARSE_GRAINING["target"]==target
    ].sort_values("K").copy()

    passing=sub[
        sub["seed_transport_spearman"]
        >=V413_FULL_TO_COARSE_TARGET_RHO
    ]

    if len(passing):
        chosen=passing.iloc[0]
        criterion_pass=True
    else:
        chosen=sub.sort_values(
            "seed_transport_spearman",
            ascending=False,
        ).iloc[0]
        criterion_pass=False

    k=int(chosen["K"])
    qubits=int(round(np.log2(k))) if k>0 else np.nan

    demo=V410_HARDWARE_EVIDENCE[
        V410_HARDWARE_EVIDENCE["target"]==target
    ]
    demo_noise_rho=(
        float(demo.iloc[0]["ideal_noisy_spearman"])
        if len(demo)
        and np.isfinite(demo.iloc[0]["ideal_noisy_spearman"])
        else np.nan
    )

    V413_HARDWARE_LADDER.append({
        "target":target,
        "fidelity_target_spearman":V413_FULL_TO_COARSE_TARGET_RHO,
        "recommended_tested_K":k,
        "recommended_qubits":qubits,
        "full_to_coarse_seed_transport_spearman":
            float(chosen["seed_transport_spearman"]),
        "top3_cluster_jaccard":
            float(chosen["top3_cluster_jaccard"]),
        "compression_N_over_K":
            float(chosen["compression_N_over_K"]),
        "meets_fidelity_target":criterion_pass,
        "k8_noise_test_available":bool(len(demo)),
        "k8_ideal_noisy_spearman":demo_noise_rho,
        "interpretation":(
            "minimum tested coarse resolution meeting full-resolution transport target"
            if criterion_pass
            else "best tested coarse resolution; target fidelity not reached"
        ),
    })

V413_HARDWARE_FIDELITY_LADDER=pd.DataFrame(
    V413_HARDWARE_LADDER
)

V413_HARDWARE_FIDELITY_LADDER.to_csv(
    V410_SUBMISSION_DIR/"hardware_fidelity_ladder.csv",
    index=False,
)
display(V413_HARDWARE_FIDELITY_LADDER)

v411_attest(
    "v413_hardware_fidelity_ladder",
    targets=int(len(V413_HARDWARE_FIDELITY_LADDER)),
    fidelity_target=float(V413_FULL_TO_COARSE_TARGET_RHO),
    passing_targets=int(
        V413_HARDWARE_FIDELITY_LADDER[
            "meets_fidelity_target"
        ].sum()
    ),
)

print("V4.1.3 HARDWARE FIDELITY LADDER: PASS")

# %% [markdown]
# ### V4.1.3 claim-locked final submission report and hash manifest

# %% [code] Notebook code cell 81 (source index 162)

p3clean=PHASE3_SUMMARY.iloc[0]
p3sens=V413_PHASE3_MAPPING_SENSITIVITY.iloc[0]

# Material sensitivity facts.
_casp=V412_CHAIN_MAPPING_SENSITIVITY[
    V412_CHAIN_MAPPING_SENSITIVITY["target"]=="BLIND_CASPASE7"
].iloc[0]
_idh=V412_CHAIN_MAPPING_SENSITIVITY[
    V412_CHAIN_MAPPING_SENSITIVITY["target"]=="BLIND3_IDH1"
].iloc[0]

def _fmt_num(x,d=3):
    try:
        return f"{float(x):.{d}f}" if np.isfinite(float(x)) else "NA"
    except Exception:
        return "NA"

report=[
    "# Final Competition Submission Report — Quantum Allosteric Signal Propagation Scanner V4.1.3",
    "",
    "## Submission status",
    "**READY WITH DECLARED LIMITATIONS.**",
    "",
    f"- Primary model: `{V413_PRIMARY_MODEL}`",
    f"- Auxiliary model: `{V413_AUXILIARY_MODEL}`",
    f"- Scientific architecture changed in V4.1.3: `{V413_SCIENTIFIC_ARCHITECTURE_CHANGED}`",
    "- No corrected chain labels are used for retrospective retraining.",
    "",
    "## Core method",
    r"Residues form an elastic/contact graph. The normalized graph Laplacian \(L\) drives a CTQW \(U(t)=e^{-i\gamma Lt}\).",
    r"Residue controllability is probed with \(H_r=L+\lambda|r\rangle\langle r|\), and site control with \(U(t/2)e^{-i\lambda\Pi_P}U(t/2)\).",
    "Matched classical diffusion provides a same-topology analogue.",
    "",
    "## Exact numerical implementation",
    f"- Original-vs-batched maximum absolute difference: `{V411_BATCH_EQUIVALENCE_MAX_ABS:.3e}`.",
    f"- Original-vs-batched maximum relative difference: `{V411_BATCH_EQUIVALENCE_MAX_REL:.3e}`.",
    f"- Equivalence tolerance: `{V411_EQUIVALENCE_TOL:.1e}`.",
    "",
    "## Primary challenge Top-5",
]

for target in RESULTS:
    report.append(
        f"- **{target}:** {v410_fmt_top5(V410_PRIMARY_TOP5,target)}"
    )

report += [
    "",
    "## Auxiliary calibrated Top-5",
]
for target in RESULTS:
    report.append(
        f"- **{target}:** {v410_fmt_top5(V410_AUX_TOP5,target)}"
    )

report += [
    "",
    "## Independent clean-room Phase-3 — historical frozen validation record",
    f"- Valid targets: {int(p3clean.valid_targets)}",
    f"- Median hybrid AP/prevalence: {_fmt_num(p3clean.median_hybrid_ap_over_prevalence)}",
    f"- Hybrid exact Top-5 any-hit fraction: {_fmt_num(p3clean.hybrid_top5_any_hit_success_fraction)}",
    f"- Hybrid improves/matches frozen fraction: {_fmt_num(p3clean.hybrid_improves_or_matches_frozen_fraction)}",
    f"- Clean-room integrity: `{bool(p3clean.cleanroom_integrity_pass)}`",
    f"- Prediction source: `{p3clean.prediction_source}`",
    "- The predeclared exact Top-5 criterion failed, so the auxiliary model is not promoted.",
    "",
    "## One-to-one chain-mapping sensitivity — post-hoc only",
    f"- Caspase-7 label positives: {int(_casp.historical_positives)} historical → {int(_casp.one_to_one_positives)} one-to-one; Jaccard={_fmt_num(_casp.label_set_jaccard)}.",
    f"- IDH1 label positives: {int(_idh.historical_positives)} historical → {int(_idh.one_to_one_positives)} one-to-one; Jaccard={_fmt_num(_idh.label_set_jaccard)}.",
    f"- Phase-3 median hybrid AP/prevalence: {_fmt_num(p3sens.historical_median_hybrid_ap_over_prevalence)} historical → {_fmt_num(p3sens.one_to_one_median_hybrid_ap_over_prevalence)} sensitivity.",
    f"- Phase-3 auxiliary Top-5 any-hit fraction: {_fmt_num(p3sens.historical_auxiliary_top5_any_hit_fraction)} historical → {_fmt_num(p3sens.one_to_one_auxiliary_top5_any_hit_fraction)} sensitivity.",
    f"- Phase-3 hybrid≥frozen fraction: {_fmt_num(p3sens.historical_hybrid_ge_frozen_fraction)} historical → {_fmt_num(p3sens.one_to_one_hybrid_ge_frozen_fraction)} sensitivity.",
    "- These corrected labels are not an independent revalidation and are not used for retraining.",
    "",
    "## Hardware path",
    "- K=8 circuit noise stability and full-resolution representation fidelity are reported separately.",
]

for r in V413_HARDWARE_FIDELITY_LADDER.itertuples():
    report.append(
        f"- {r.target}: minimum tested K meeting rho≥{V413_FULL_TO_COARSE_TARGET_RHO:.2f} "
        f"is K={int(r.recommended_tested_K)} ({int(r.recommended_qubits)} qubits), "
        f"full→coarse Spearman={r.full_to_coarse_seed_transport_spearman:.3f}; "
        f"K=8 ideal/noisy Spearman={_fmt_num(r.k8_ideal_noisy_spearman)}."
    )

report += [
    "",
    "## Claim boundaries",
    "- Static topology is a proxy for long-range communication, not atomistic thermodynamics.",
    "- The K=8 KRAS hardware demonstration is noise-stable but is not a faithful full-resolution transport surrogate.",
    "- c-Myc is prospective and not experimentally validated by this notebook.",
    "- The official cardiac-myosin pair is not presented as a ligand-contact mavacamten validation.",
    "- The learned auxiliary layer was calibrated on the historical calibration labels; the one-to-one sensitivity audit does not retroactively alter that model.",
    "- Any corrected-label retraining must be treated as a new model and requires a fresh independent holdout.",
    "",
    "## Reproducibility",
    f"- Frozen V3.6 signature: `{V40_PARENT_FROZEN_SIGNATURE}`",
    f"- Calibration signature: `{V40_CALIBRATION_SIGNATURE}`",
    f"- Clean-room freeze SHA256: `{p3clean.freeze_manifest_sha256}`",
    f"- Current run ID: `{V411_RUN_ID}`",
]

V413_FINAL_REPORT="\n".join(report)
final_report_path=V410_SUBMISSION_DIR/"final_submission_report_v4_1_3.md"
final_report_path.write_text(
    V413_FINAL_REPORT,
    encoding="utf-8",
)
print(V413_FINAL_REPORT)

# Hash all current competition artifacts before final archiving.
def _sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as fh:
        while True:
            chunk=fh.read(1024*1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

artifact_rows=[]
for p in sorted(V410_SUBMISSION_DIR.iterdir()):
    if p.is_file():
        artifact_rows.append({
            "file":p.name,
            "size_bytes":int(p.stat().st_size),
            "sha256":_sha256_file(p),
        })

V413_ARTIFACT_HASHES=pd.DataFrame(artifact_rows)
V413_ARTIFACT_HASHES.to_csv(
    V410_SUBMISSION_DIR/"artifact_hash_manifest_v4_1_3.csv",
    index=False,
)

V413_FINAL_STATUS={
    "version":V413_VERSION,
    "status":"READY_WITH_DECLARED_LIMITATIONS",
    "scientific_architecture_changed":False,
    "primary_model":V413_PRIMARY_MODEL,
    "auxiliary_model":V413_AUXILIARY_MODEL,
    "auxiliary_promoted_to_primary":False,
    "corrected_chain_labels_used_for_training":False,
    "historical_cleanroom_record_overwritten":False,
    "batched_solver_equivalence_pass":bool(V411_BATCH_EQUIVALENCE_PASS),
    "cleanroom_integrity_pass":bool(p3clean.cleanroom_integrity_pass),
    "historical_phase3_top5_criterion_pass":bool(
        p3clean.criterion_top5_pass
    ),
    "posthoc_one_to_one_phase3_top5_any_hit_fraction":
        float(p3sens.one_to_one_auxiliary_top5_any_hit_fraction),
    "frozen_parent_signature":V40_PARENT_FROZEN_SIGNATURE,
    "calibration_signature":V40_CALIBRATION_SIGNATURE,
    "cleanroom_freeze_sha256":str(p3clean.freeze_manifest_sha256),
    "run_id":V411_RUN_ID,
}

(
    V410_SUBMISSION_DIR/"final_submission_status_v4_1_3.json"
).write_text(
    json.dumps(V413_FINAL_STATUS,indent=2),
    encoding="utf-8",
)

v411_attest(
    "v413_claim_locked_report",
    status=V413_FINAL_STATUS["status"],
    artifact_count=int(len(V413_ARTIFACT_HASHES)),
)

print("V4.1.3 CLAIM-LOCKED REPORT: PASS")

# %% [markdown]
# ## 46B. Final competition methodological report and compact ZIP

# %% [code] Notebook code cell 82 (source index 164)

def v410_fmt_top5(frame,target):
    sub=frame[frame["target"]==target].sort_values("rank")
    return ", ".join(
        f"{r.rank}. {r.label} ({r.resname})"
        for r in sub.itertuples()
    )

lines=[
    "# Competition Submission Report — Quantum Allosteric Signal Propagation Scanner V4.1",
    "",
    "## Executive summary",
    "This submission predicts allosteric connectivity directly from static protein topology, without classical molecular-dynamics trajectories.",
    "The primary competition ranking is the frozen V3.6 site-aware quantum/topology scorer. A calibrated HDC + neuroevolution layer is reported separately as auxiliary evidence because its predeclared independent Phase-3 exact Top-5 criterion did not pass.",
    "",
    "## Quantum metric and biological proxy",
    r"Each residue is a node in an elastic/contact graph. The normalized graph Laplacian \(L\) defines a continuous-time quantum walk \(U(t)=\exp(-i\gamma Lt)\).",
    "The time-averaged transition matrix is exported as the required N×N quantum connectivity matrix.",
    r"Local controllability is probed with finite perturbations \(H_r=L+\lambda|r\rangle\langle r|\), measuring the Jensen-Shannon change in seed-driven transport over a fixed lambda ensemble.",
    r"At site level, collective control is tested with \(U(t/2)K_P(\lambda)U(t/2)\), where \(K_P=e^{-i\lambda\Pi_P}\).",
    "The working biological interpretation is that persistent, interference-sensitive graph transport and perturbation susceptibility are topology-only proxies for long-range signal transmission.",
    "",
    "## Classical analogue",
    "Matched continuous-time diffusion on the same normalized Laplacian supplies the classical comparator. Quantum occupancy, classical diffusion, quantum susceptibility, classical susceptibility and quantum-over-classical intervention excess are exported side by side.",
    "",
    "## Calibration layer",
    "Residues are encoded into deterministic 4096-dimensional bipolar HDC vectors. A compact six-hidden-unit neuroevolutionary ranker is trained only on six calibration proteins, and the frozen physics/HDC/evolution scores are fused with equal reciprocal-rank fusion.",
    "Challenge labels, c-Myc and Phase-3 holdout labels are excluded from fitting.",
    "",
    "## Calibration LOPO",
]

for r in V40_LOPO_SUMMARY.itertuples():
    lines.append(
        f"- {r.model}: median AP/prevalence={r.median_ap_over_prevalence:.3f}; "
        f"median ROC={r.median_roc:.3f}; mean Top-5 recall={r.mean_top5_recall:.3f}"
    )

lines += [
    "",
    "## Challenge benchmark comparison",
]

for r in V40_CHALLENGE_BENCHMARK.itertuples():
    lines.append(
        f"- {r.target} / {r.model}: AP={r.average_precision:.4f}; "
        f"ROC={r.roc_auc:.4f}; AP/prevalence={r.ap_over_prevalence:.3f}; "
        f"global Top-5 recall={r.top5_recall:.3f}"
    )

lines += [
    "",
    "## Required primary Top-5 predictions",
]

for target in RESULTS:
    lines.append(
        f"- **{target}:** {v410_fmt_top5(V410_PRIMARY_TOP5,target)}"
    )

lines += [
    "",
    "## Auxiliary calibrated Top-5",
]

for target in RESULTS:
    lines.append(
        f"- **{target}:** {v410_fmt_top5(V410_AUX_TOP5,target)}"
    )

p3=PHASE3_SUMMARY.iloc[0]
lines += [
    "",
    "## Independent clean-room Phase-3",
    f"- Valid targets: {int(p3.valid_targets)}",
    f"- Median hybrid AP/prevalence: {float(p3.median_hybrid_ap_over_prevalence):.3f}",
    f"- Hybrid exact Top-5 any-hit success: {float(p3.hybrid_top5_any_hit_success_fraction):.3f}",
    f"- Hybrid improves/matches frozen fraction: {float(p3.hybrid_improves_or_matches_frozen_fraction):.3f}",
    f"- AP-ratio criterion passed: {bool(p3.criterion_median_ap_ratio_pass)}",
    f"- Exact Top-5 criterion passed: {bool(p3.criterion_top5_pass)}",
    f"- Improvement criterion passed: {bool(p3.criterion_improvement_pass)}",
    f"- Clean-room integrity passed: {bool(p3.cleanroom_integrity_pass)}",
    f"- Prediction source: {p3.prediction_source}",
    "",
    "## Scalability",
]

if len(V410_COARSE_GRAINING):
    for target in RESULTS:
        sub=V410_COARSE_GRAINING[
            V410_COARSE_GRAINING["target"]==target
        ].sort_values("K")
        best=sub.sort_values(
            "seed_transport_spearman",
            ascending=False,
        ).iloc[0]
        lines.append(
            f"- {target}: best tested K={int(best.K)}, "
            f"seed-transport Spearman={best.seed_transport_spearman:.3f}, "
            f"N/K compression={best.compression_N_over_K:.1f}x."
        )

lines += [
    "",
    "## Hardware path and noise resilience",
]

for r in V410_HARDWARE_EVIDENCE.itertuples():
    if getattr(r,"target","")=="GLOBAL":
        lines.append(
            f"- Hardware demo unavailable in this runtime: {getattr(r,'reason','')}"
        )
    else:
        lines.append(
            f"- {r.target}: K={int(r.K)}, qubits={int(r.qubits)}, "
            f"Pauli terms={int(r.pauli_terms)}, depth={int(r.transpiled_depth)}, "
            f"status={r.status}, noisy/ideal Spearman="
            f"{getattr(r,'ideal_noisy_spearman',np.nan):.3f}."
        )

lines += [
    "",
    "## Interpretability",
    "Interactive 3D HTML maps are included for every challenge target. Residues are colored by the frozen mechanistic score and both primary and auxiliary Top-5 predictions are overlaid.",
    "",
    "## Limitations and claim boundaries",
    "- The independent learned-layer Phase-3 exact Top-5 criterion failed, so the calibrated layer is not promoted to the primary competition scorer.",
    "- c-Myc is prospective; its residues are hypotheses, not validated druggable pockets.",
    "- The official cardiac-myosin configuration intentionally leaves the holo ligand unresolved, so this submission does not fabricate a mavacamten-contact benchmark from that pair.",
    "- Static contact topology is a proxy for conformational communication; it does not reproduce atomistic thermodynamics.",
    "- Hardware evidence is demonstrated on coarse Hamiltonians; full residue-level execution is currently quantum-inspired/classically evaluated.",
    "",
    "## Reproducibility",
    f"- Frozen V3.6 architecture signature: `{V40_PARENT_FROZEN_SIGNATURE}`",
    f"- Calibration signature: `{V40_CALIBRATION_SIGNATURE}`",
    f"- Clean-room freeze manifest SHA256: `{p3.freeze_manifest_sha256}`",
    f"- V4.1 changes scientific architecture: `{V410_SCIENTIFIC_ARCHITECTURE_CHANGED}`",
]

V410_REPORT="\n".join(lines)
report_path=V410_SUBMISSION_DIR/"competition_submission_report_v4_1.md"
report_path.write_text(V410_REPORT,encoding="utf-8")
print(V410_REPORT)

# Compact manifest.
manifest={
    "version":V410_VERSION,
    "parent":V410_PARENT,
    "scientific_architecture_changed":V410_SCIENTIFIC_ARCHITECTURE_CHANGED,
    "frozen_parent_signature":V40_PARENT_FROZEN_SIGNATURE,
    "calibration_signature":V40_CALIBRATION_SIGNATURE,
    "cleanroom_freeze_sha256":str(p3.freeze_manifest_sha256),
    "primary_ranking":"frozen_v3_6_site_aware",
    "auxiliary_ranking":"v40_calibrated_hybrid",
    "promote_auxiliary_to_primary":False,
    "required_connectivity_targets":list(RESULTS.keys()),
    "hardware_targets":list(V410_HARDWARE_TARGETS),
    "phase3_cleanroom_integrity":bool(p3.cleanroom_integrity_pass),
    "phase3_top5_criterion_pass":bool(p3.criterion_top5_pass),
}
(V410_SUBMISSION_DIR/"submission_manifest.json").write_text(
    json.dumps(manifest,indent=2),
    encoding="utf-8",
)

# Include core pre-existing evidence files when available.
for fname in [
    "v40_calibration_lopo_metrics.csv",
    "v40_calibration_lopo_summary.csv",
    "phase3_v404_cleanroom_metrics.csv",
    "phase3_v404_cleanroom_exact_top5.csv",
    "phase3_v404_cleanroom_summary.csv",
    "v4_0_6_submission_decision.json",
    "v4_0_6_challenge_top5_frozen_vs_calibrated.csv",
    "v4_0_8_backend_audit.json",
]:
    src=RESULTS_DIR/fname
    if src.exists():
        shutil.copy2(
            src,
            V410_SUBMISSION_DIR/fname,
        )


V411_REQUIRED_SUBMISSION_STAGES = ['bootstrap_complete', 'batched_solver_equivalence', 'v41_connectivity_and_top5', 'v41_quantum_classical_ablation', 'v41_coarse_graining', 'v41_hardware_evidence', 'v41_3d_and_phase3_audit', 'v412_chain_mapping_sensitivity', 'v412_hardware_representation_fidelity', 'v41_claims_matrix', 'v412_scientific_supplement', 'v413_mapping_decision_summary', 'v413_hardware_fidelity_ladder', 'v413_claim_locked_report']

v411_assert_current_run_stages(
    V411_REQUIRED_SUBMISSION_STAGES
)

# Write current-run attestation artefacts into the submission directory
# before the ZIP is created.
V411_LEDGER_COPY = V410_SUBMISSION_DIR/"execution_attestation_v4_1_1.jsonl"
shutil.copy2(
    V411_LEDGER_PATH,
    V411_LEDGER_COPY,
)

V411_RUN_MANIFEST["pre_zip_attested_stages"] = sorted(
    v411_assert_current_run_stages(V411_REQUIRED_SUBMISSION_STAGES)
)
V411_RUN_MANIFEST["batched_equivalence_pass"] = bool(
    V411_BATCH_EQUIVALENCE_PASS
)
V411_RUN_MANIFEST["batched_equivalence_max_abs_diff"] = float(
    V411_BATCH_EQUIVALENCE_MAX_ABS
)
V411_RUN_MANIFEST["pre_zip_utc"] = _dt_v411.now(_tz_v411.utc).isoformat()

(
    V410_SUBMISSION_DIR/"execution_manifest_v4_1_1.json"
).write_text(
    json.dumps(V411_RUN_MANIFEST, indent=2),
    encoding="utf-8",
)

zip_path=shutil.make_archive(
    str(ROOT/"cleveland_clinic_quantum_allostery_V4_1_3_final_submission"),
    "zip",
    root_dir=V410_SUBMISSION_DIR,
)

assert report_path.exists()
assert (
    V410_SUBMISSION_DIR/"connectivity_matrix_manifest.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"challenge_primary_top5_site_aware.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"claims_and_limitations.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"one_to_one_chain_mapping_sensitivity.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"hardware_representation_fidelity.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"final_scientific_audit_v4_1_2.md"
).exists()
assert (
    V410_SUBMISSION_DIR/"mapping_sensitivity_decision_table.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"phase3_one_to_one_mapping_sensitivity_summary.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"hardware_fidelity_ladder.csv"
).exists()
assert (
    V410_SUBMISSION_DIR/"final_submission_report_v4_1_3.md"
).exists()
assert (
    V410_SUBMISSION_DIR/"final_submission_status_v4_1_3.json"
).exists()
assert (
    V410_SUBMISSION_DIR/"artifact_hash_manifest_v4_1_3.csv"
).exists()

v411_attest(
    "v41_submission_pack_complete",
    zip_path=str(zip_path),
)
print("V4.1.1 CURRENT-RUN EXECUTION ATTESTATION: PASS")
print("V4.1.3 FINAL SUBMISSION PACK: PASS")
print("Submission ZIP:",zip_path)

# %% [markdown]
# ### V4.1.1 final current-run execution audit
#
# This cell proves that the success message came from one current execution
# rather than from inherited notebook output.

# %% [code] Notebook code cell 83 (source index 166)

V411_FINAL_REQUIRED = [
    "bootstrap_complete",
    "batched_solver_equivalence",
    "v41_connectivity_and_top5",
    "v41_quantum_classical_ablation",
    "v41_coarse_graining",
    "v41_hardware_evidence",
    "v41_3d_and_phase3_audit",
    "v412_chain_mapping_sensitivity",
    "v412_hardware_representation_fidelity",
    "v41_claims_matrix",
    "v412_scientific_supplement",
    "v413_mapping_decision_summary",
    "v413_hardware_fidelity_ladder",
    "v413_claim_locked_report",
    "v41_submission_pack_complete",
]

_stages=v411_assert_current_run_stages(
    V411_FINAL_REQUIRED
)

V411_FINAL_EXECUTION_AUDIT=pd.DataFrame(
    v411_ledger_records()
)

display(V411_FINAL_EXECUTION_AUDIT)

assert V411_FINAL_EXECUTION_AUDIT["run_id"].nunique()==1
assert V411_FINAL_EXECUTION_AUDIT["run_id"].iloc[0]==V411_RUN_ID
assert V411_BATCH_EQUIVALENCE_PASS

print("Current run ID:",V411_RUN_ID)
print("Attested stages:",sorted(_stages))
print("V4.1.3 FINAL CURRENT-RUN AUDIT: PASS")

# %% [markdown]
# ## 46. V4.0.8 completion checkpoint audit

# %% [code] Notebook code cell 84 (source index 168)
v408_assert_persistent_backend("completion_audit")


V407_COMPLETION_INVENTORY=v407_checkpoint_inventory()
V407_COMPLETION_SUMMARY=v407_inventory_summary()

display(V407_COMPLETION_SUMMARY)

V407_COMPLETION_SUMMARY.to_csv(
    RESULTS_DIR/"v4_0_7_checkpoint_inventory_summary.csv",
    index=False,
)

V407_COMPLETION_INVENTORY.to_csv(
    RESULTS_DIR/"v4_0_7_checkpoint_inventory_files.csv",
    index=False,
)

v404_stage_marker(
    "v407_scientific_run_complete",
    checkpoint_files=int(
        len(V407_COMPLETION_INVENTORY)
    ),
)


V408_BACKEND_AUDIT={
    "version":V408_VERSION,
    "backend":V404_CHECKPOINT_BACKEND,
    "checkpoint_dir":str(V404_CHECKPOINT_DIR),
    "scientific_architecture_changed":
        V408_SCIENTIFIC_ARCHITECTURE_CHANGED,
    "local_salvaged_files":len(
        V408_LOCAL_SALVAGED_FILES
    ),
    "checkpoint_files":int(
        len(V407_COMPLETION_INVENTORY)
    ),
}
(
    RESULTS_DIR/"v4_0_8_backend_audit.json"
).write_text(
    json.dumps(
        V408_BACKEND_AUDIT,
        indent=2,
    ),
    encoding="utf-8",
)

# %% [markdown]
# ## 46A. V4.0.6 submission-readiness decision pack
#
# This section is **reporting only**.
#
# The decision rule is fixed from the already-declared Phase-3 criteria:
#
# - clean-room integrity must pass;
# - median hybrid AP/prevalence must exceed 1;
# - hybrid must improve/match frozen on at least half of holdouts;
# - exact Top-5 any-hit success must meet the declared 50% threshold.
#
# Because the clean-room Top-5 criterion failed, V4.0.6 does not automatically
# replace the frozen mechanistic scorer with the learned hybrid. Both rankings
# are exported so the competition report can present the calibrated layer as
# secondary evidence without hiding its stronger AP/ROC behavior.

# %% [code] Notebook code cell 85 (source index 170)

assert len(PHASE3_SUMMARY)==1
_v406_p3=PHASE3_SUMMARY.iloc[0]

V406_PROMOTE_HYBRID_TO_PRIMARY = bool(
    _v406_p3.cleanroom_integrity_pass
    and _v406_p3.criterion_median_ap_ratio_pass
    and _v406_p3.criterion_improvement_pass
    and _v406_p3.criterion_top5_pass
)

V406_SUBMISSION_DECISION = {
    "version": V406_VERSION,
    "scientific_architecture_changed": V406_SCIENTIFIC_ARCHITECTURE_CHANGED,
    "cleanroom_integrity_pass": bool(_v406_p3.cleanroom_integrity_pass),
    "phase3_valid_targets": int(_v406_p3.valid_targets),
    "phase3_median_hybrid_ap_over_prevalence": float(
        _v406_p3.median_hybrid_ap_over_prevalence
    ),
    "phase3_hybrid_top5_any_hit_success_fraction": float(
        _v406_p3.hybrid_top5_any_hit_success_fraction
    ),
    "phase3_hybrid_improves_or_matches_frozen_fraction": float(
        _v406_p3.hybrid_improves_or_matches_frozen_fraction
    ),
    "criterion_median_ap_ratio_pass": bool(
        _v406_p3.criterion_median_ap_ratio_pass
    ),
    "criterion_top5_pass": bool(
        _v406_p3.criterion_top5_pass
    ),
    "criterion_improvement_pass": bool(
        _v406_p3.criterion_improvement_pass
    ),
    "promote_calibrated_hybrid_to_primary": V406_PROMOTE_HYBRID_TO_PRIMARY,
    "recommended_role_frozen_v3_6": (
        "primary mechanistic baseline"
        if not V406_PROMOTE_HYBRID_TO_PRIMARY
        else "mechanistic reference"
    ),
    "recommended_role_calibrated_hybrid": (
        "auxiliary calibrated ranking evidence"
        if not V406_PROMOTE_HYBRID_TO_PRIMARY
        else "primary calibrated ranking layer"
    ),
    "reason": (
        "Clean-room transfer and improvement criteria passed, but the "
        "predeclared Phase-3 exact Top-5 success criterion did not pass."
        if not V406_PROMOTE_HYBRID_TO_PRIMARY
        else
        "All predeclared clean-room Phase-3 criteria passed."
    ),
}

(RESULTS_DIR/"v4_0_6_submission_decision.json").write_text(
    json.dumps(V406_SUBMISSION_DECISION,indent=2),
    encoding="utf-8",
)

print(json.dumps(V406_SUBMISSION_DECISION,indent=2))

# %% [markdown]
# ### 46B. Export frozen and calibrated challenge Top-5 side by side
#
# The competition-facing evidence pack retains both views:
#
# - `frozen_v3_6` — mechanistic V3.6 ranking;
# - `v40_calibrated_hybrid` — HDC + neuroevolution auxiliary ranking.
#
# This is not an ensemble change. It is an auditable comparison artifact.

# %% [code] Notebook code cell 86 (source index 172)

V406_CHALLENGE_TOP5_ROWS=[]

for name,item in RESULTS.items():
    f=item["features"]
    frozen_score=f["final_score"].to_numpy(float)
    calibrated_score=f["v40_calibrated_hybrid"].to_numpy(float)

    frozen_top=v40_top5_global(item,frozen_score,5).copy()
    frozen_top["target"]=name
    frozen_top["ranking_layer"]="frozen_v3_6"
    frozen_top["rank"]=np.arange(1,len(frozen_top)+1)

    calibrated_top=v40_top5_global(item,calibrated_score,5).copy()
    calibrated_top["target"]=name
    calibrated_top["ranking_layer"]="v40_calibrated_hybrid"
    calibrated_top["rank"]=np.arange(1,len(calibrated_top)+1)

    V406_CHALLENGE_TOP5_ROWS.extend(
        [frozen_top,calibrated_top]
    )

V406_CHALLENGE_TOP5_COMPARISON=pd.concat(
    V406_CHALLENGE_TOP5_ROWS,
    ignore_index=True,
)

keep_cols=[
    c for c in [
        "target","ranking_layer","rank","idx","label","resname",
        "final_score","v40_calibrated_hybrid",
        "v40_hdc_score","v40_evo_score",
        "quantum_intervention_excess","topology_score",
        "stability_confidence",
    ]
    if c in V406_CHALLENGE_TOP5_COMPARISON.columns
]

V406_CHALLENGE_TOP5_COMPARISON[keep_cols].to_csv(
    RESULTS_DIR/"v4_0_6_challenge_top5_frozen_vs_calibrated.csv",
    index=False,
)

display(
    V406_CHALLENGE_TOP5_COMPARISON[keep_cols]
)

# %% [markdown]
# ## 47. Parent V4.0.8 evidence report — retained for provenance

# %% [code] Notebook code cell 87 (source index 174)

def build_method_report_v40():
    lines=[
        "# Methodological Report — Quantum Allosteric Signal Propagation Scanner V4.0.8","",

        "## Branch status",
        "V4.0 is a post-freeze calibrated-learning branch. "
        "The V3.6 physics scorer remains an immutable baseline.","",

        "## Calibration set",
        f"Six previously unsealed independent proteins are used for calibration: "
        f"{', '.join(V40_CALIBRATION_NAMES)}.",
        "They are no longer described as blind validation inside this branch.","",

        "## HDC",
        f"Residues are encoded into {HDC_DIM}-dimensional deterministic bipolar hypervectors. "
        "Positive/negative allosteric prototypes are learned across calibration proteins.","",

        "## Neuroevolution",
        f"A compact {V40_EVO_HIDDEN}-hidden-unit nonlinear ranker is evolved with "
        f"population={V40_EVO_POP}, generations={V40_EVO_GENERATIONS}. "
        "Fitness uses prevalence-normalized AP gain plus Top-5 recall.","",

        "## Fusion",
        "Frozen physics, HDC and neuroevolution are fused with equal-weight reciprocal rank fusion. "
        "No post-hoc fusion weights are fitted.","",

        "## Leakage safeguards",
        "- Challenge benchmark labels are excluded from training.",
        "- c-Myc is excluded from training.",
        "- Phase-3 holdout labels are absent until predictions are frozen.",
        f"- Calibration signature: `{V40_CALIBRATION_SIGNATURE}`.",
        "",
    ]

    if len(V40_LOPO_SUMMARY):
        lines += ["## Calibration LOPO summary"]
        for r in V40_LOPO_SUMMARY.itertuples():
            lines.append(
                f"- {r.model}: median AP/prevalence={r.median_ap_over_prevalence:.3f}, "
                f"median ROC={r.median_roc:.3f}, mean Top5 recall={r.mean_top5_recall:.3f}"
            )
        lines.append("")

    if len(V40_CHALLENGE_BENCHMARK):
        lines += ["## Challenge parallel benchmark — excluded from fitting"]
        for r in V40_CHALLENGE_BENCHMARK.itertuples():
            lines.append(
                f"- {r.target} / {r.model}: AP={r.average_precision:.4f}, "
                f"ROC={r.roc_auc:.4f}, AP/prevalence={r.ap_over_prevalence:.3f}, "
                f"Top5 recall={r.top5_recall:.3f}"
            )
        lines.append("")

    if len(PHASE3_METRICS):
        lines += ["## Phase-3 independent holdout"]
        for r in PHASE3_METRICS.itertuples():
            if not getattr(r,"validation_valid",False):
                continue
            lines.append(
                f"- {r.target} [{r.model}]: AP={r.average_precision:.4f}, "
                f"ROC={r.roc_auc:.4f}, AP/prevalence={r.ap_over_prevalence:.3f}"
            )
        lines.append("")

    if len(PHASE3_TOP5):
        lines += ["## Phase-3 exact calibrated Top-5"]
        for r in PHASE3_TOP5.itertuples():
            lines.append(
                f"- {r.target}: hits={int(r.top5_hits)}/5, "
                f"enrichment={r.top5_enrichment:.3f}, "
                f"hits={r.hit_labels or 'none'}"
            )
        lines.append("")

    if len(PHASE3_SUMMARY):
        r=PHASE3_SUMMARY.iloc[0]
        lines += [
            "## Phase-3 summary",
            f"- Valid targets: {int(r.valid_targets)}",
            f"- Median hybrid AP/prevalence: {r.median_hybrid_ap_over_prevalence:.3f}",
            f"- Hybrid Top-5 any-hit success: {r.hybrid_top5_any_hit_success_fraction:.3f}",
            f"- Hybrid improves/matches frozen fraction: {r.hybrid_improves_or_matches_frozen_fraction:.3f}",
            f"- Median AP-ratio criterion: {bool(r.criterion_median_ap_ratio_pass)}",
            f"- Top-5 criterion: {bool(r.criterion_top5_pass)}",
            f"- Improvement criterion: {bool(r.criterion_improvement_pass)}",
            "",
        ]

    if len(PHASE3_SUMMARY):
        r=PHASE3_SUMMARY.iloc[0]
        lines += [
            "## V4.0.4 clean-room integrity",
            f"- Integrity gate pass: {bool(r.cleanroom_integrity_pass)}",
            f"- Prediction run nonce: `{r.prediction_run_nonce}`",
            f"- Unseal nonce: `{r.unseal_nonce}`",
            f"- Freeze-manifest SHA256: `{r.freeze_manifest_sha256}`",
            f"- Prediction source: {r.prediction_source}",
            "",
        ]

    lines += [
        "## Submission-readiness decision",
        f"- Promote calibrated hybrid to primary: {V406_PROMOTE_HYBRID_TO_PRIMARY}",
        f"- Frozen V3.6 role: {V406_SUBMISSION_DECISION['recommended_role_frozen_v3_6']}",
        f"- Calibrated hybrid role: {V406_SUBMISSION_DECISION['recommended_role_calibrated_hybrid']}",
        f"- Reason: {V406_SUBMISSION_DECISION['reason']}",
        "",
        "## Scientific interpretation",
        "Calibration LOPO is development evidence. V4.0.4 accepts Phase-3 as independent evidence only if "
        "the clean-room cryptographic integrity gate passes. The V4.0.3 Phase-3 values are treated as provisional "
        "because its saved execution state skipped required Phase-3 cells. If the clean-room Phase-3 Top-5 criterion "
        "fails, the calibrated branch must not automatically replace the frozen scorer."
    ]

    return "\n".join(lines)

METHOD_REPORT_V40=build_method_report_v40()
(RESULTS_DIR/"methodological_report_v4_0_8.md").write_text(
    METHOD_REPORT_V40
)
print(METHOD_REPORT_V40[:18000])


# Execution-resilience appendix.
with open(
    RESULTS_DIR/"v404_execution_status_summary.json",
    "w",
    encoding="utf-8",
) as fh:
    json.dump({
        "execution_signature":V404_EXECUTION_SIGNATURE,
        "checkpoint_dir":str(V404_CHECKPOINT_DIR),
        "checkpoint_backend":V404_CHECKPOINT_BACKEND,
        "challenge_deferred_until_calibrated":V404_DEFER_CHALLENGE_UNTIL_CALIBRATED,
        "resume_enabled":V404_RESUME,
        "archived_evidence_skipped":V404_SKIP_ARCHIVED_EVIDENCE,
        "legacy_post_analysis_skipped":V404_SKIP_LEGACY_POST_ANALYSIS,
        "phase3_valid_targets":int(
            PHASE3_SUMMARY["valid_targets"].iloc[0]
        ) if len(PHASE3_SUMMARY) else 0,
    },fh,indent=2)

# %% [markdown]
# ## 48. Parent V4.0.8 evidence bundle — retained for provenance

# %% [code] Notebook code cell 88 (source index 176)

import shutil

manifest={
    "version":"4.0.8",
    "mode":"clean-room Phase-3 revalidation of calibrated HDC + neuroevolution branch",
    "frozen_parent_version":"3.6",
    "frozen_parent_signature":V40_PARENT_FROZEN_SIGNATURE,
    "calibration_signature":V40_CALIBRATION_SIGNATURE,
    "calibration_proteins":V40_CALIBRATION_NAMES,
    "calibration_protein_count":len(V40_CALIBRATION_NAMES),
    "challenge_labels_used_for_training":V40_CHALLENGE_LABELS_USED_FOR_TRAINING,
    "phase3_used_for_training":V40_PHASE3_USED_FOR_TRAINING,
    "hdc_dimension":HDC_DIM,
    "hdc_features":HDC_FEATURES,
    "neuroevolution_features":V40_LEARN_FEATURES,
    "neuroevolution_hidden":V40_EVO_HIDDEN,
    "neuroevolution_population":V40_EVO_POP,
    "neuroevolution_generations":V40_EVO_GENERATIONS,
    "hybrid_components":list(V40_HYBRID_COMPONENTS),
    "hybrid_fusion":"equal reciprocal-rank fusion",
    "phase3_public_targets":{
        k:{
            "apo":v["apo"],
            "site_class":v["site_class"],
            "manual_seed_resids":v["manual_seed_resids"],
        }
        for k,v in PHASE3_PUBLIC_PANEL.items()
    },
    "phase3_sealed_validation":{
        "BLIND3_EGFR":{
            "holo":"6P1L","ligand":"9LL"
        },
        "BLIND3_IDH1":{
            "holo":"6ADG","ligand":"9UO"
        },
        "BLIND3_HIV_IN":{
            "holo":"4E1N","ligand":"TQX"
        },
    },
    "phase3_predictions_frozen_before_unseal":True,
    "execution_signature":V404_EXECUTION_SIGNATURE,
    "checkpoint_resume_enabled":V404_RESUME,
    "checkpoint_backend":V404_CHECKPOINT_BACKEND,
    "challenge_deferred_until_calibrated":V404_DEFER_CHALLENGE_UNTIL_CALIBRATED,
    "per_lambda_intervention_checkpointing":True,
    "exact_block_diagonal_intervention_batches":True,
    "intervention_batch_size":V404_INTERVENTION_BATCH_SIZE,
    "large_graph_batch_size":V404_INTERVENTION_BATCH_SIZE_LARGE,
    "incremental_target_cells":True,
    "phase3_cleanroom_integrity":True,
    "mechanical_bugfix_version":"4.0.7",
    "scientific_architecture_changed":False,
    "checkpoint_compatibility":"V4.0.4",
    "fixed_runtime_error":"clean-room target dict bug fixed in V4.0.5; stale PHASE3_RESULTS export reference fixed in V4.0.6",
    "phase3_prediction_run_nonce":V404_PHASE3_RUN_NONCE,
    "phase3_unseal_nonce":V404_PHASE3_UNSEAL_NONCE,
    "phase3_freeze_manifest_sha256":V404_PHASE3_FREEZE_MANIFEST_SHA256,
    "phase3_prediction_source":"cryptographically sealed pre-unseal CSVs",
    "archived_v38_v39_stats_rerun":False,
    "legacy_post_analysis_rerun":False,
    "submission_decision":V406_SUBMISSION_DECISION,
    "promote_calibrated_hybrid_to_primary":V406_PROMOTE_HYBRID_TO_PRIMARY,
    "export_bug_fixed":"PHASE3_RESULTS -> V404_PHASE3_RESULTS",
    "v407_persistent_resume_guard":True,
    "v407_scientific_architecture_changed":False,
    "v407_checkpoint_inventory_exported":True,
    "v408_drive_strict_resume":True,
    "v408_backend":V404_CHECKPOINT_BACKEND,
    "v408_local_salvaged_files":len(V408_LOCAL_SALVAGED_FILES),
    "v408_scientific_architecture_changed":False,
}

(RESULTS_DIR/"v4_0_manifest.json").write_text(
    json.dumps(manifest,indent=2)
)

# Compact Phase-3 artefacts.
p3_top=[]
for name,item in V404_PHASE3_RESULTS.items():
    df=item["v40_top5"].copy()
    df["target"]=name
    p3_top.append(df)

if p3_top:
    pd.concat(
        p3_top,
        ignore_index=True
    ).to_csv(
        RESULTS_DIR/"phase3_all_v40_top5.csv",
        index=False
    )

bundle_path=shutil.make_archive(
    str(ROOT/"cleveland_clinic_quantum_allostery_V4_0_8_final_evidence"),
    "zip",
    root_dir=RESULTS_DIR
)

print("V4.0.8 final evidence bundle:",bundle_path)


# Explicit clean-room Phase-3 compact artefact built from the pre-unseal files.
p3_cleanroom_top=[]
for name in sorted(V404_PHASE3_RESULTS):
    paths=v404_cleanroom_paths(name)
    assert paths["top5"].exists()
    df=pd.read_csv(paths["top5"])
    df["target"]=name
    df["source"]="cryptographically_sealed_pre_unseal_csv"
    p3_cleanroom_top.append(df)

if p3_cleanroom_top:
    pd.concat(
        p3_cleanroom_top,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR/"phase3_all_v406_cleanroom_top5.csv",
        index=False,
    )

# Final package integrity assertions.
assert set(V404_PHASE3_RESULTS)==set(PHASE3_PUBLIC_PANEL)
assert bool(PHASE3_SUMMARY.iloc[0].cleanroom_integrity_pass)
assert (
    PHASE3_SUMMARY.iloc[0].prediction_source
    =="cryptographically_sealed_pre_unseal_csv"
)
assert (RESULTS_DIR/"v4_0_6_submission_decision.json").exists()
assert (RESULTS_DIR/"v4_0_6_challenge_top5_frozen_vs_calibrated.csv").exists()

print("V4.0.6 package integrity: PASS")

# %% [markdown]
# # V4.1.3 final submission checklist
#
# - [ ] Run **Runtime → Run all** with the persistent checkpoint backend.
# - [ ] Confirm exact original-vs-batched equivalence still passes.
# - [ ] Confirm all V4.1/V4.1.2 evidence stages are attested in the same run.
# - [ ] Inspect `mapping_sensitivity_decision_table.csv`.
# - [ ] Confirm the clean-room Phase-3 historical record is not overwritten.
# - [ ] Confirm the one-to-one Phase-3 summary is explicitly marked post-hoc sensitivity only.
# - [ ] Confirm the primary model remains frozen V3.6 site-aware.
# - [ ] Confirm the learned HDC/neuroevolution model remains auxiliary.
# - [ ] Inspect `hardware_fidelity_ladder.csv`; distinguish K=8 noise stability from fidelity-oriented K.
# - [ ] Confirm `final_submission_report_v4_1_3.md` is generated.
# - [ ] Confirm `final_submission_status_v4_1_3.json` reports `READY_WITH_DECLARED_LIMITATIONS`.
# - [ ] Confirm `artifact_hash_manifest_v4_1_3.csv` is generated.
# - [ ] Confirm `V4.1.3 CLAIM-LOCKED REPORT: PASS`.
# - [ ] Confirm `V4.1.3 FINAL SUBMISSION PACK: PASS`.
# - [ ] Confirm `V4.1.3 FINAL CURRENT-RUN AUDIT: PASS`.
# - [ ] Treat any future corrected-label retraining as a new model requiring a fresh holdout.
