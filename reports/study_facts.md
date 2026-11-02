# Dataset, selection, and coverage facts

All numbers come from the frozen audit, frozen selections and selected SVM vectorizers. These are aggregate descriptions of the studied data and saved models, not new training or performance-based selection. The dictionaries are operational matches, not gold slang labels or semantic-preservation annotations.

PRDECT-ID contains Indonesian Tokopedia product reviews across 29 categories ([source](https://data.mendeley.com/datasets/574v66hf2v/1)). SmSA contains Indonesian comments and reviews from multiple online platforms ([IndoNLU paper](https://aclanthology.org/2020.aacl-main.85/)); the released SmSA input files were already preprocessed. Label sets, domains and preprocessing differ, so absolute scores should not be used to infer relative dataset difficulty.

| Dataset | Split | Reviews | Groups | Negative | Neutral | Positive |
|---|---|---:|---:|---:|---:|---:|
| prdec | test | 793 | 793 | 412 | — | 381 |
| prdec | train | 3702 | 3698 | 1922 | — | 1780 |
| prdec | valid | 793 | 792 | 412 | — | 381 |
| smsa | test | 500 | 500 | 204 | 88 | 208 |
| smsa | train | 10919 | 10918 | 3407 | 1136 | 6376 |
| smsa | valid | 1260 | 1260 | 394 | 131 | 735 |

Released/retained: PRDECT-ID 5,400/5,288 (112 canonical duplicates excluded); SmSA 12,760/12,679 (81 canonical duplicates excluded). All SmSA removals were from training.

SVM C candidates were 0.1, 1 and 10, selected by validation macro-F1 with smaller C on exact ties. All selected C values are 1 except SmSA word/normalized, where C=10. No validation examples were added to training.

IndoBERT learning rates were tuned at seed 42 by validation macro-F1 with earlier epoch then lower rate on ties; final seeds 13, 37 and 73 were retrained at the selected rate and each selected its own best validation epoch (earlier epoch on ties).

| Dataset | Condition | Tuning LR | Tuning best epoch | Final epochs for seeds 13 / 37 / 73 |
|---|---|---:|---:|---:|
| prdec | original | 2e-05 | 4 | 2 / 3 / 3 |
| prdec | normalized | 2e-05 | 2 | 3 / 2 / 4 |
| smsa | original | 2e-05 | 5 | 3 / 5 / 2 |
| smsa | normalized | 3e-05 | 5 | 2 / 4 / 5 |

Dictionary: 15,006 source rows; 4,202 eligible unique keys; 71 ambiguous keys skipped; 4,131 active mappings. Counted replacements are occurrences, not distinct terms or evidence that replacements are linguistically correct.

| Dataset | Split | Replacement events | Text-changed reviews | Word-token sequence changed | BERT token IDs changed after truncation |
|---|---|---:|---:|---:|---:|
| prdec | test | 1527 | 489 | 489 | 489 |
| prdec | train | 7020 | 2225 | 2225 | 2224 |
| prdec | valid | 1377 | 471 | 471 | 471 |
| smsa | test | 121 | 97 | 97 | 97 |
| smsa | train | 1742 | 1413 | 1413 | 1413 |
| smsa | valid | 239 | 189 | 189 | 189 |

The following held-out TF-IDF vector survival counts compare original and normalized forms through each named fixed, training-fitted vectorizer. Denominators are text-changed reviews. A changed vector means some effect survives vocabulary filtering; it does not identify which replacement was responsible. These counts were calculated descriptively from the saved models after evaluation and were not used to select models.

| Dataset | Fixed vectorizer | Test changed vectors / text-changed reviews |
|---|---|---:|
| prdec | word_original | 488 / 489 |
| prdec | word_normalized | 489 / 489 |
| prdec | character_original | 489 / 489 |
| prdec | character_normalized | 489 / 489 |
| smsa | word_original | 95 / 97 |
| smsa | word_normalized | 95 / 97 |
| smsa | character_original | 97 / 97 |
| smsa | character_normalized | 97 / 97 |

Machine-readable aggregate results: `results/final_evaluation/study_facts.json`. Resource and sharing caveats are in `reports/resource_availability.md`.
