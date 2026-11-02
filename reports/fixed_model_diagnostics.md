# Frozen four-cell and fixed-model diagnostics

O/O, O/N, N/O and N/N denote training preprocessing / evaluation input. These analyses reuse the frozen predictions; no models were trained or tuned. Matched-pipeline effect is N/N - O/O. The two fixed-model directions are O/N - O/O and N/O - N/N, respectively. These quantities describe different comparisons and do not identify a causal decomposition of preprocessing, training or separately selected hyperparameters.

Fixed-model intervals were added after inspection of test results and are post hoc, exploratory, descriptive intervals. They use the same 2,000 group-bootstrap draws (seed 42) as the primary evaluation. Each draw averages the three paired IndoBERT seed effects. Intervals are nominal 95% percentile intervals conditional on the fitted models, not simultaneous intervals; the additional comparisons have multiplicity and do not change the primary effects or their prespecified +0.01 decision rule. Intervals containing zero do not establish equivalence.

| Dataset | Model | O/O | O/N | N/O | N/N | Matched N/N - O/O |
|---|---|---:|---:|---:|---:|---:|
| prdec | word | 0.9570 | 0.9583 | 0.9369 | 0.9646 | +0.0076 |
| prdec | character | 0.9595 | 0.9646 | 0.9495 | 0.9608 | +0.0013 |
| prdec | indobert | 0.9844 | 0.9836 | 0.9810 | 0.9810 | -0.0034 |
| smsa | word | 0.7350 | 0.7366 | 0.7358 | 0.7414 | +0.0064 |
| smsa | character | 0.7214 | 0.7244 | 0.7342 | 0.7317 | +0.0103 |
| smsa | indobert | 0.8830 | 0.8836 | 0.9055 | 0.9032 | +0.0202 |

| Dataset | Model | Fixed-model change | Mean difference | Nominal 95% interval | Seed SD |
|---|---|---|---:|---:|---:|
| prdec | word | O/N - O/O | +0.00135 | [-0.00883, +0.01268] | — |
| prdec | word | N/O - N/N | -0.02766 | [-0.04285, -0.01266] | — |
| prdec | character | O/N - O/O | +0.00508 | [-0.00003, +0.01145] | — |
| prdec | character | N/O - N/N | -0.01127 | [-0.02392, +0.00017] | — |
| prdec | indobert | O/N - O/O | -0.00084 | [-0.00339, +0.00169] | 0.0026 |
| prdec | indobert | N/O - N/N | +0.00001 | [-0.00294, +0.00294] | 0.0033 |
| smsa | word | O/N - O/O | +0.00157 | [+0.00000, +0.00497] | — |
| smsa | word | N/O - N/N | -0.00563 | [-0.01681, +0.00171] | — |
| smsa | character | O/N - O/O | +0.00304 | [+0.00000, +0.00762] | — |
| smsa | character | N/O - N/N | +0.00248 | [-0.00453, +0.01275] | — |
| smsa | indobert | O/N - O/O | +0.00062 | [-0.00151, +0.00335] | 0.0025 |
| smsa | indobert | N/O - N/N | +0.00228 | [-0.00007, +0.00537] | 0.0015 |

IndoBERT per-seed matched-pipeline scores are computed on the same reviews; the sample standard deviation across these three paired effects is training-seed variability, not a larger independent test sample.

| Dataset | Seed | O/O | N/N | Paired N/N - O/O |
|---|---:|---:|---:|---:|
| prdec | 13 | 0.9848 | 0.9734 | -0.0114 |
| prdec | 37 | 0.9823 | 0.9886 | +0.0063 |
| prdec | 73 | 0.9861 | 0.9810 | -0.0051 |
| smsa | 13 | 0.8905 | 0.9058 | +0.0153 |
| smsa | 37 | 0.8858 | 0.9008 | +0.0150 |
| smsa | 73 | 0.8726 | 0.9029 | +0.0303 |

Class F1 below compares matched pipelines; IndoBERT F1 values are arithmetic means across the same three seeds. Class supports count unique held-out reviews once. Full four-cell per-class metrics and per-seed details are in the JSON.

| Dataset | Model | Class | Support | O/O F1 | N/N F1 | N/N - O/O |
|---|---|---|---:|---:|---:|---:|
| prdec | word | negative | 412 | 0.9593 | 0.9666 | +0.0073 |
| prdec | word | positive | 381 | 0.9547 | 0.9626 | +0.0079 |
| prdec | character | negative | 412 | 0.9618 | 0.9631 | +0.0012 |
| prdec | character | positive | 381 | 0.9572 | 0.9585 | +0.0013 |
| prdec | indobert | negative | 412 | 0.9852 | 0.9820 | -0.0032 |
| prdec | indobert | positive | 381 | 0.9836 | 0.9801 | -0.0036 |
| smsa | word | negative | 204 | 0.8426 | 0.8337 | -0.0089 |
| smsa | word | neutral | 88 | 0.5263 | 0.5674 | +0.0411 |
| smsa | word | positive | 208 | 0.8363 | 0.8232 | -0.0130 |
| smsa | character | negative | 204 | 0.8145 | 0.8238 | +0.0093 |
| smsa | character | neutral | 88 | 0.5507 | 0.5612 | +0.0104 |
| smsa | character | positive | 208 | 0.7990 | 0.8103 | +0.0113 |
| smsa | indobert | negative | 204 | 0.9372 | 0.9495 | +0.0123 |
| smsa | indobert | neutral | 88 | 0.7769 | 0.8248 | +0.0479 |
| smsa | indobert | positive | 208 | 0.9348 | 0.9352 | +0.0004 |

Transition counts below use the indicated direction under a fixed trained model. “Corrected” means incorrect→correct and “introduced” means correct→incorrect. For IndoBERT, averages are seed-specific counts, not counts of unique reviews. Unique counts are unions across seeds within the indicated category; the same review may appear in both categories across different seeds. For SVMs, the single-seed means and unique counts coincide.

| Dataset | Model | Direction | Corrected: mean / unique | Introduced: mean / unique | Prediction changed: mean / unique |
|---|---|---|---:|---:|---:|
| prdec | word | O/N - O/O | 10.00 / 10 | 9.00 / 9 | 19.00 / 19 |
| prdec | word | N/O - N/N | 7.00 / 7 | 29.00 / 29 | 36.00 / 36 |
| prdec | character | O/N - O/O | 5.00 / 5 | 1.00 / 1 | 6.00 / 6 |
| prdec | character | N/O - N/N | 8.00 / 8 | 17.00 / 17 | 25.00 / 25 |
| prdec | indobert | O/N - O/O | 0.67 / 1 | 1.33 / 3 | 2.00 / 4 |
| prdec | indobert | N/O - N/N | 1.67 / 5 | 1.67 / 4 | 3.33 / 9 |
| smsa | word | O/N - O/O | 1.00 / 1 | 0.00 / 0 | 1.00 / 1 |
| smsa | word | N/O - N/N | 1.00 / 1 | 3.00 / 3 | 4.00 / 4 |
| smsa | character | O/N - O/O | 2.00 / 2 | 0.00 / 0 | 2.00 / 2 |
| smsa | character | N/O - N/N | 1.00 / 1 | 1.00 / 1 | 2.00 / 2 |
| smsa | indobert | O/N - O/O | 0.33 / 1 | 0.33 / 1 | 1.00 / 3 |
| smsa | indobert | N/O - N/N | 1.33 / 4 | 0.00 / 0 | 1.67 / 5 |

The SmSA IndoBERT matched-pipeline difference and its original-trained fixed-model input difference are separate observations. Few fixed-model prediction changes do not show how much of the pipeline difference was caused by training, substitution, or hyperparameter selection.

The original practical margin was recorded as an investigator-selected one-percentage-point reporting threshold; the protocol records no further prior rationale. It is not an externally validated standard. The original criterion remains unchanged; these supplementary analyses add no sensitivity thresholds.

Reproduce with `.venv-svm/Scripts/python.exe src/analyze_fixed_models.py`; inputs and script hashes are stored in `results/final_evaluation/fixed_model_diagnostics.json`. Individual-seed effects, all four per-class matrices and detailed transitions are retained there.
