# Quantum Allosteric Signal Propagation Scanner — V4.1.3

Competition repository package for the Cleveland Clinic **2026 Global Quantum + AI Challenge** problem:

> *Unlocking undruggable targets: quantum simulation of allosteric signal propagation.*

This repository contains the claim-locked V4.1.3 notebook, a standalone Python conversion,
and the accompanying methodological research paper.

## What the approach does

The pipeline starts from static PDB topology and does **not** use classical molecular-dynamics
trajectories as input.

Core components:

- residue-level elastic/contact graph construction
- normalized-Laplacian continuous-time quantum walk (CTQW)
- full N×N time-averaged quantum connectivity matrices
- residue-level Hamiltonian intervention susceptibility
- collective pocket phase-kick analysis
- matched classical diffusion controls
- site-aware frozen V3.6 primary ranking
- HDC + neuroevolution auxiliary calibration
- coarse-graining fidelity analysis
- Qiskit/Aer hardware-aligned phase-kick circuits and noise tests
- interactive 3D visualization
- clean-room / sealed holdout validation and chain-mapping sensitivity audits

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── quantum_allostery_v4_1_3.ipynb
├── src/
│   └── quantum_allostery_v4_1_3.py
├── docs/
│   ├── research_paper.pdf
│   └── research_paper_editable.docx
└── results/
    └── README.md
```

## Required competition outputs

When the full pipeline is executed, it is designed to generate:

1. **N×N connectivity matrices** representing time-averaged quantum connectivity between residues.
2. **Ranked Top-5 allosteric residues** for each target.
3. **Methodological report** explaining the chosen quantum metrics and biological proxy.
4. **Noise-resilience evidence**, **coarse-graining/scalability evidence**, and **interactive 3D maps**.

The primary ranking remains the frozen V3.6 site-aware model. The HDC + neuroevolution
calibrated hybrid is treated as auxiliary evidence because the predeclared independent
Phase-3 exact Top-5 criterion did not pass.

## Challenge targets

The code is configured for the minimum challenge set:

- KRAS G12C
- BCR-ABL1
- Cardiac Myosin
- c-Myc

It also contains independent calibration and holdout panels used for validation.

## Quick start — Google Colab

Open:

```text
notebooks/quantum_allostery_v4_1_3.ipynb
```

and run **Runtime → Run all**.

The notebook supports persistent checkpointing to Google Drive.

## Quick start — Python

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/quantum_allostery_v4_1_3.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\quantum_allostery_v4_1_3.py
```

## Standalone script environment variables

The Python version supports:

```text
QALLOSTERY_ROOT
QALLOSTERY_CHECKPOINT_DIR
QALLOSTERY_USE_GOOGLE_DRIVE
QALLOSTERY_ALLOW_LOCAL_FALLBACK
QALLOSTERY_CALIBRATION_CSV
```

Outside Colab, the script defaults to a local working directory unless overridden.

## Reproducibility and claim boundaries

The repository deliberately distinguishes:

- **primary mechanistic evidence** — frozen V3.6 site-aware ranking
- **auxiliary learned evidence** — HDC + neuroevolution calibration
- **independent holdout evidence** — clean-room sealed Phase-3 evaluation
- **post-hoc sensitivity analysis** — one-to-one homologous-chain remapping

Important limitations are preserved rather than hidden:

- exact Top-5 generalization of the learned auxiliary layer is not claimed
- K=8 circuit noise stability is not automatically treated as full-resolution fidelity
- c-Myc remains prospective
- the cardiac-myosin challenge structure mismatch is not converted into a fabricated ligand-contact validation
- static contact topology is a mechanistic proxy, not atomistic thermodynamics

## Research paper

See:

```text
docs/research_paper.pdf
```

for the full methodological description, equations, validation design, results, hardware path,
and limitations.

## Results directory

The repository does not ship stale checkpoint files or copied notebook outputs as authoritative
new V4.1.3 results. Run the notebook/script to regenerate the final evidence pack in the current
execution environment.

See `results/README.md` for the expected output families.
