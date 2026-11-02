# Frozen held-out evaluation

The model-selection freeze preceded held-out inference. Primary pipeline effects compare separately trained and selected normalized and original models. Estimates are macro-F1 differences in raw units; 95% percentile intervals use the same 2,000 group-bootstrap draws across models, conditions and seeds within each dataset. These nominal intervals describe test-sample uncertainty conditional on the fitted models; they are not simultaneous intervals. Transformer sample standard deviation across the three seeds is reported separately.

The ±0.01 practical margin is investigator-selected. A lower interval bound above +0.01 supports a practically meaningful gain for the tested setting; an interval containing zero does not establish equivalence.

| Dataset | Model | Original macro-F1 | Normalized macro-F1 | Effect | 95% interval | Effect seed SD |
|---|---|---:|---:|---:|---:|---:|
| prdec | word | 0.9570 | 0.9646 | +0.0076 | [-0.0013, +0.0177] | — |
| prdec | character | 0.9595 | 0.9608 | +0.0013 | [-0.0076, +0.0101] | — |
| prdec | indobert | 0.9844 | 0.9810 | -0.0034 | [-0.0097, +0.0025] | 0.0090 |
| smsa | word | 0.7350 | 0.7414 | +0.0064 | [-0.0135, +0.0297] | — |
| smsa | character | 0.7214 | 0.7317 | +0.0103 | [-0.0046, +0.0267] | — |
| smsa | indobert | 0.8830 | 0.9032 | +0.0202 | [+0.0037, +0.0381] | 0.0088 |

| Dataset | Direct contrast | Estimate | 95% interval |
|---|---|---:|---:|
| prdec | word_minus_character | +0.0063 | [-0.0038, +0.0178] |
| prdec | word_minus_indobert | +0.0110 | [-0.0004, +0.0233] |
| smsa | word_minus_character | -0.0039 | [-0.0303, +0.0226] |
| smsa | word_minus_indobert | -0.0138 | [-0.0384, +0.0105] |

Per-class F1, support and changes are shown per seed in `results/final_evaluation/paired_results.json`, together with confusion matrices and changed/unchanged subgroup scores. The table below averages seed-specific per-class F1 for IndoBERT; counts are repeated observations of the same held-out examples, not independent seed samples.

| Dataset | Model | Class | Support | Original F1 | Normalized F1 | Difference |
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

Fixed-model original-to-normalized input transitions below use the original-trained model. IndoBERT values are mean counts across the three seeds; per-seed results and both trained conditions are retained in the JSON. A correct-to-incorrect transition is an introduced error.

| Dataset | Model | Text-changed / total | Incorrect → correct | Correct → incorrect |
|---|---|---:|---:|---:|
| prdec | word | 489/793 | 10.00 | 9.00 |
| prdec | character | 489/793 | 5.00 | 1.00 |
| prdec | indobert | 489/793 | 0.67 | 1.33 |
| smsa | word | 97/500 | 1.00 | 0.00 |
| smsa | character | 97/500 | 2.00 | 0.00 |
| smsa | indobert | 97/500 | 0.33 | 0.33 |

Changed subsets can be small, especially for SmSA; subgroup metrics are descriptive and do not establish equivalence. These outcomes apply only to the studied datasets, models and dictionary policy. No simultaneous coverage guarantee or powered equivalence claim is made.

Provenance limitation: the frozen selection recorded transformer checkpoint identifiers and metadata but did not contain pre-freeze weight hashes. The provider-version source was subsequently hashed before held-out inference and each checkpoint reproduced its saved validation predictions; this does not establish a pre-freeze cryptographic binding of those weight bytes. The SVM selection did bind checkpoint hashes.

All 20 selected model checkpoints produced both input conditions. Paired predictions and model/inference provenance are local; see `results/final_evaluation/paired_results.json`, the SVM inference metadata and the independently supplied IndoBERT inference artifacts. Dedicated normalization-only runtime was not measured here.
