# Reproduction and review snapshot

The local review snapshot contains manuscript sources and PDFs alongside aggregate research code and results; it is not an end-to-end executable bundle or a public release. It excludes source/normalized reviews, the dictionary, per-review split IDs and labels, predictions, token sequences, generated private scripts, credentials and model weights. `reports/resource_availability.md` records source and terms caveats. The project's own code currently has no project-level license grant; availability for external distribution remains a separate author decision. The bundled IEEEtran class retains its existing third-party notice and LPPL terms. No review-access URL is asserted.

This guide distinguishes (a) a fixture-only smoke test and aggregate figure, runnable from the snapshot; (b) reconstruction of the data audit and a new training run, which require separately retrieved resources; and (c) exact replay of the recorded held-out evaluation, which additionally requires the original privately held selection manifest, models and row-level predictions. A fresh training run does not recreate those original checkpoint hashes or necessarily reproduce the same GPU predictions. Do not overwrite the original frozen workspace or infer permission to redistribute the omitted inputs.

## Environments and fixture smoke test

Commands below are PowerShell from a workspace or extracted snapshot root. Use Python 3.12.14 for local stages. Dependency pins are in `requirements-audit-lock.txt`, `requirements-pilot-svm-lock.txt`, and `requirements-figure-lock.txt`; the latter concerns the normalization-effect plot, not the separate workflow generator. Network access is needed to install dependencies. On a different platform compatible wheels may not be available at the recorded versions.

```powershell
python -m venv .venv-svm
& .venv-svm/Scripts/python.exe -m pip install -r requirements-pilot-svm-lock.txt
& .venv-svm/Scripts/python.exe -m unittest discover -s tests -p test_evaluate_models.py -v
& .venv-svm/Scripts/python.exe -m unittest discover -s tests -p test_fixed_model_diagnostics.py -v
& .venv-svm/Scripts/python.exe -m unittest discover -s tests -p test_svm_search.py -v
& .venv-svm/Scripts/python.exe -m unittest discover -s tests -p test_freeze_new_selection.py -v
```

These synthetic fixtures require no raw data, model weights, row-level results or GPU. They test fixed-label metrics, grouped sampling, direction of comparisons, seed-union counting and an explicit held-out-row sentinel. Expected outcome: all tests pass. The separate audit environment also supports four synthetic audit protocol tests:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-audit-lock.txt
& .venv/Scripts/python.exe -m unittest discover -s tests -p test_data_protocol.py -v
```

## Retrieve and reconstruct the data audit in an isolated copy

`data/manifest.json` contains the exact pinned URLs and SHA-256 values; `configs/resources.json` names the revisions. Confirm current access and terms before retrieval. The fetcher downloads only missing raw files and checks any recorded hashes; it does not retrieve pretrained model weights. The audit writes `data/processed/`, `data/splits/`, `reports/audit_summary.json`, reports and `data/protocol_lock.json`. Consequently, perform it in a fresh isolated copy rather than overwriting the archived experiment. The output includes local per-review text, dictionary mappings and IDs that must remain private until their terms are checked.

```powershell
& .venv/Scripts/python.exe src/fetch_resources.py
& .venv/Scripts/python.exe src/audit_data.py
& .venv/Scripts/python.exe src/verify_frozen_protocol.py
```

Expected fingerprint: `56a11ac3a8b5a70634a2b10af9f0f6c867b50cc298685d60001979d3eb86884a` across 20 locked files. The last command verifies the reconstructed lock against its files; also compare the printed protocol ID to the value above, because the audit itself regenerates the lock. Disagreements, unavailable sources, changed URLs, package/platform differences or unapproved policy changes require investigation, not silent replacement of the frozen record. The lock is an audit fingerprint, not a freeze of later model selection.

## New SVM training in an isolated copy

With reconstructed inputs and the SVM environment, `& .venv-svm/Scripts/python.exe src/train_svm_search.py` performs the 24-candidate training/validation grid and writes `results/svm_search/` plus `reports/svm_search.md`. It fits vocabulary and IDF on training examples only; validation determines C. This command trains models and overwrites its target outputs. The recorded local CPU run took 91.20 seconds in total, including 51.16 seconds summed model fit time; this is not a guaranteed runtime on different hardware and excludes normalization-only cost.

This recreates a search, not the original SVM weight-byte freeze. The selected models and hashes in `results/final_selection.json` refer to the original run. Do not replace that selection or use a newly trained model with the old frozen manifest.

## Transformer training: private hosted workflow

After audit reconstruction, the command `& .venv/Scripts/python.exe kaggle/build_full.py` creates `kaggle/private_full.py` and a payload manifest. The generated script embeds train/validation token IDs and labels. It must be handled privately: token IDs can reconstruct text. `src/train_indobert_full.py` is its trainer template, not a standalone entry point. Running it requires a suitable private hosted GPU session, access to the pinned pretrained model and the generated private input; submission, authentication, account permissions and provider resource availability are external to this snapshot. The recorded run used one Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1 and the image digest recorded in the frozen selection; the final script event occurred after 15,479.63 seconds (about 4 h 18 min). The retained original run has a full environment export under private `results/indobert_full/`; a container digest and package versions do not guarantee identical GPU weights. No present command here submits or downloads multi-gigabyte weights.

A fresh run requires a newly constructed and independently verified selection manifest before any held-out prediction. Once the completed private transformer training output has been transferred or mounted, put its `config.json`, `state.json`, `environment.json`, `build_manifest.json`, run records and validation predictions under `results/indobert_full/`. The twelve selected checkpoint directories (each with `config.json` and `model.safetensors`) must be accessible at a private location; use its `checkpoints` directory in the command below. A separate local environment combines the locked audit and SVM dependencies for validation and tokenizer verification:

```powershell
python -m venv .venv-freeze
& .venv-freeze/Scripts/python.exe -m pip install -r requirements-freeze.txt
& .venv-freeze/Scripts/python.exe src/freeze_new_selection.py --checkpoint-root 'C:/path/to/private/indobert_full/checkpoints'
& .venv/Scripts/python.exe kaggle/build_test_eval.py
```

The freeze builder refuses to overwrite an existing selection and refuses to run if its known local held-out output directories already exist. The review snapshot contains historical aggregate files under `results/final_evaluation/`; create a separate clean pre-test working copy that omits that entire directory and `results/indobert_test/`, while retaining the original snapshot unchanged. The guard cannot establish whether held-out predictions were made elsewhere. It recomputes selected train/validation token payload and all candidate validation metrics, enforces the configured validation tie rules and environment, and records SHA-256 for each supplied transformer weight file before writing the new `results/final_selection.json` and companion SHA. It does not itself load those weights to reproduce predictions: the private inference gate below performs that separate check before decoding held-out inputs. Hashing twelve files totalling roughly 6 GB requires local storage and time; mount locations and hosted-job access depend on the user's provider. This step was fixture-tested locally but was not run against a newly trained model set.

`kaggle/build_test_eval.py` accepts `--selection` and `--selection-sha256` and verifies the companion SHA against exact manifest bytes; defaults are the newly created files above. It writes the sensitive `kaggle/private_test_eval.py`. For a new hash-bound selection it emits the strict template `src/evaluate_indobert_test_frozen.py`: attach the corresponding full private training-output directory to the hosted GPU execution and set `INDOBERT_SOURCE_DIR` in that process to its `indobert_full` directory. The generated script checks every frozen weight hash before loading validation models, reproduces all 12 saved validation prediction sets, and only then decodes held-out inputs. It checks each weight again before test prediction. A hosted-job launcher, provider authentication and access are external to this snapshot; no command here submits a job. A fresh run may yield different weights or scores and cannot claim byte-identical recovery of the historical checkpoints.

## Replay the historical frozen inference and analysis, if authorized artifacts are available

These commands require the original complete local workspace: frozen `results/final_selection.json` and companion SHA file; all eight selected SVM joblib files and recorded validation predictions; reconstructed private raw/processed data; the original IndoBERT training configuration and 24 held-out prediction/verification records under `results/indobert_test/`. They are intentionally absent from the review snapshot. Original IndoBERT weights remain in private hosted output, not in this bundle. The original selection contains null pre-freeze transformer weight hashes; the later execution hashes and complete validation-prediction reproduction are weaker, distinct evidence.

```powershell
& .venv-svm/Scripts/python.exe src/infer_svm_final.py --validate-only
& .venv-svm/Scripts/python.exe src/infer_svm_final.py
& .venv-svm/Scripts/python.exe src/run_final_evaluation.py --indobert-dir results/indobert_test
& .venv-svm/Scripts/python.exe src/analyze_fixed_models.py
& .venv-svm/Scripts/python.exe src/generate_study_facts.py
```

The first command verifies the lock, original manifest, eight checkpoints and matching and switched validation predictions; it does not run held-out inference. The second runs SVM held-out inference only after the validation gate. `kaggle/build_test_eval.py` can also construct the historical private transformer inference payload using the exact original selection and tokenized inputs. That precise historical manifest contains null weight hashes and takes the historical evaluator path; its hosted fallback depended on private provider-specific checkpoint access. The distributed historical template by itself is not a one-command substitute for that access. New selections cannot use the historical null-hash exception. A separately reviewed private hosted execution and transfer of all output files is needed before the joint analysis command can run.

The joint command checks hashes, IDs, groups, labels, logits and model metadata; computes shared paired group-bootstrap metrics; and writes `results/final_evaluation/paired_results.json` and `reports/final_evaluation.md`. The next command writes the four-cell supplementary diagnostics, and the last writes aggregate selection/coverage facts. These commands write their targets: run them in a copy if preserving original evidence. The fixed-model intervals added after seeing results are post hoc exploratory; the primary +0.01 rule is unchanged.

## Rebuild the aggregate figure and manuscript

With `results/final_evaluation/paired_results.json` (included in the snapshot) and the pinned figure environment:

```powershell
python -m venv .venv-figure
& .venv-figure/Scripts/python.exe -m pip install -r requirements-figure-lock.txt
& .venv-figure/Scripts/python.exe paper/figures/create_normalization_effects.py
```

This writes `output/pdf/normalization_effects.pdf` and a paper-copy PDF and PNG in `paper/figures/`. The plot contains aggregate results only. The stored report tables are generated by `src/run_final_evaluation.py`, `src/analyze_fixed_models.py` and `src/generate_study_facts.py` when their private prediction inputs are available; there is no separate standalone LaTeX-table generator. Manuscript tables are included in the TeX sources. In a disposable working copy of the snapshot, `./paper/build.ps1` invokes installed `pdflatex` and `bibtex` and writes `output/pdf/introduction_literature_review.pdf`; `./paper/build_supplement.ps1` builds the supplement. TeX, BibTeX, IEEEtran bibliography style and the required TeX packages are external prerequisites and are not pinned in this bundle. Final manuscript and supplement PDF previews are included; rebuilding them requires those tools.

## Snapshot verification and release status

`MANIFEST.json` in the versioned snapshot lists every bundled file, its byte size and SHA-256. The accompanying zip SHA-256 is recorded outside the archive. A versioned snapshot is a local preparation only: archive integrity and fixture tests do not establish release permission, de-identification or venue compliance. Before external review, authors must settle access/identity decisions, source-specific terms and code licensing, update the snapshot to the intended manuscript version, and arrange an authorized review location. No public URL is supplied here.
