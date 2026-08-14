# Expected generated results

The pipeline writes generated evidence to its configured results/submission directory.

Expected competition-facing files include families such as:

- `*_NxN_quantum_connectivity.csv`
- `*_NxN_classical_connectivity.csv`
- `*_connectivity_matrices.npz`
- `challenge_primary_top5_site_aware.csv`
- `challenge_auxiliary_top5_calibrated.csv`
- `quantum_vs_classical_benchmark_ablation.csv`
- `coarse_graining_fidelity.csv`
- `hardware_aligned_phase_kick_evidence.csv`
- `interactive_3d_manifest.csv`
- `*_interactive_3d.html`
- `one_to_one_chain_mapping_sensitivity.csv`
- `hardware_representation_fidelity.csv`
- `hardware_fidelity_ladder.csv`
- `final_submission_report_v4_1_3.md`
- `final_submission_status_v4_1_3.json`
- execution-attestation and artifact-hash manifests

Generated results are intentionally not bundled here as if they were fresh V4.1.3 outputs.
Run the notebook or Python script to regenerate them under the current run attestation.
