# Reviewer guide: package v4.0.0

This guide identifies the exact recorded study and current manuscript revision. The package is publicly available at https://github.com/CharisBerlian/indonesian-slang-normalization-study. Venue anonymity requirements remain to be confirmed. It contains no review-level data or model weights.

## Exact study identity

| Resource | Recorded version/revision |
|---|---|
| PRDECT-ID | DOI `10.17632/574v66hf2v.1` |
| SmSA via IndoNLU | `ce728f6926a36174b9923dfe49d6a6839b6e9bb7` |
| Kamus Alay | `c710fa9f3f8d8727c24f3209f4f6a4b6ecaf6d8b` |
| `indobenchmark/indobert-base-p1` | `c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2` |

Joint data/preprocessing protocol fingerprint:

`56a11ac3a8b5a70634a2b10af9f0f6c867b50cc298685d60001979d3eb86884a`

Complete retrieval URLs and source-file SHA-256 values are in `data/manifest.json`; `configs/resources.json` records revisions. `data/protocol_lock.json` lists all 20 locked file hashes, including omitted split and processed files. Its status string describes the original pre-training audit checkpoint, not the current completion state. The lock does not freeze later model selection. `ARTIFACT_IDENTITY.json` repeats the revisions and protocol fingerprint and binds aggregate-result files by SHA-256. `MANIFEST.json` identifies the exact paper and supplement bytes in this package.

## How the paper maps to the artifacts

| Paper evidence | Included artifacts |
|---|---|
| Audit, retained counts, normalization coverage | `reports/audit_summary.json`, `reports/study_facts.md`, `src/audit_data.py`, `src/normalization.py` |
| Four training/input cells and pipeline effects (Table II) | `results/final_evaluation/paired_results.json`, `results/final_evaluation/fixed_model_diagnostics.json` |
| Direct contrasts (Table III), paired bootstrap definitions | `results/final_evaluation/paired_results.json`, `src/evaluate_models.py` |
| Effect plot (Fig. 2) | `paper/figures/create_normalization_effects.py`, aggregate `paired_results.json` |
| Workflow (Fig. 1) | `paper/figures/create_workflow.py`, supplied vector PDF |
| Seed and class results (Tables S1-S2) | `paired_results.json`, `reports/final_evaluation.md` |
| Corrected/introduced counts and both input directions (Table S3) | `fixed_model_diagnostics.json`, `src/analyze_fixed_models.py` |
| Class supports and configurations (Tables S4-S6) | `reports/study_facts.md`, `results/final_evaluation/study_facts.json`, `configs/protocol.json` |
| Prose and exact table entries | `paper/sections/`, `paper/supplement.tex`, bibliography and build scripts |

Paths above that omit a directory are under `results/final_evaluation/`. Tables are maintained in LaTeX; no single command automatically writes all table sources.

## Check without private inputs

1. Verify archive SHA-256 against its companion file before extraction. On PowerShell use `Get-FileHash -Algorithm SHA256` on the archive.
2. In a fresh extracted folder run `python tools/verify_review_snapshot.py --strict`. It checks every listed file and rejects unexpected files, apart from Git metadata. It checks the four bundled protocol-locked files; the other 16 are intentionally omitted.
3. Make a second disposable working copy for tests and builds. Follow the environment commands in `REPRODUCIBILITY.md`; run the six synthetic test modules using their specified audit or SVM environment.
4. Regenerate the aggregate effect plot and compile both PDFs. This checks presentation reproduction from aggregate evidence, not the original predictions. Compilation may change PDF timestamps/bytes; rebuilt PDFs need not retain archive manifest hashes. Keep the original extraction intact.

The frozen margin remains 0.01 macro-F1; nominal test-group intervals are conditional on fitted models, and seed variability is separate. New analyses prompted by held-out findings must be identified as exploratory. Do not overwrite frozen experiment records.

## What requires additional resources

- **Audit reconstruction:** obtain pinned files with `src/fetch_resources.py`; rebuild in isolation and compare the protocol fingerprint. Resource terms remain as documented in `reports/resource_availability.md`.
- **Fresh training:** retrieve resources and the pinned base model, select on train/validation data, and create a new selection before held-out inference. GPU/cloud execution and authentication are external to this bundle.
- **Exact historical replay:** additionally obtain the original controlled model, selection and per-review prediction artifacts. Aggregate summaries cannot recreate predictions. Historical transformer selection lacked whole-weight hashes before its freeze; later execution hashes and reproduced validation predictions are distinct evidence.

## Package validation record

The preparation report is retained alongside the archive as `artifacts/review_snapshot_v4_verification.md` in the author's workspace. It records the archive hash, extraction checks, fixture results and PDF-build comparison. It is external to the immutable archive so verification does not alter the files being verified. The public repository supplies package access; full historical replay still requires the excluded private artifacts.
