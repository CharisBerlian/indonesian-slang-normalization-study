# Indonesian slang normalization study: review package v3.0.0

This package accompanies the current six-page manuscript and two-page supplement for **Dictionary-Based Slang Normalization for Indonesian Sentiment Classification: A Controlled Comparison**, by Charis Berlian. Version 3 updates the manuscript, figures and review documentation; it does not replace the frozen experimental protocol or results. Versions 1 and 2 remain historical snapshots.

The GitHub repository remains **private**. Reviewer access has not been arranged, and this named-author package is not anonymous. Preparing or privately hosting the package does not establish venue compliance or grant a project code license.

## Start here

- [Reviewer guide](REVIEWER_GUIDE.md): paper-to-artifact map, exact resource revisions, protocol identifier, and checks without private data.
- [Main paper](output/pdf/indonesian_slang_sentiment_full_study.pdf) and [supplement](output/pdf/indonesian_slang_sentiment_supplement.pdf).
- [Reproduction instructions](REPRODUCIBILITY.md): environments, synthetic tests, reconstruction, new training, and historical replay limits.
- [Machine-readable identity](ARTIFACT_IDENTITY.json) and [file manifest](MANIFEST.json): exact experiment identity and package hashes.
- [Resource availability](reports/resource_availability.md): attribution, exclusions, and unresolved terms.

From a fresh extraction, `python tools/verify_review_snapshot.py --strict` verifies the packaged files, identity links, and the subset of protocol-locked files included here. It does not validate omitted raw or processed data. The companion `.zip.sha256` verifies archive bytes against a separately supplied checksum; it is not an external signature.

## Contents and limits

The package contains current manuscript sources and PDFs, analysis and training source templates, frozen configurations, retrieval URLs and checksums, aggregate reports/results, synthetic tests, figures, and verification tools. The exact inventory is in `MANIFEST.json`; this README and the Git configuration files are included in that inventory.

Raw/normalized reviews, dictionary mappings, row-level IDs, labels, splits, groups and predictions, token payloads, generated private scripts, credentials, and weights are excluded. Test fixtures contain constructed examples. No additional scraping, annotation or training is part of v3 preparation.

Synthetic checks, aggregate-figure regeneration, and manuscript builds need no excluded data. Audit reconstruction and new training require separately retrieved resources. Exact historical replay additionally requires the original private selections, weights and predictions; later transformer weight hashes must not be described as a pre-selection freeze. The original code has no project-level license grant. The bundled IEEEtran class retains its third-party LPPL notice.
