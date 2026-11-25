> Current status (19 November 2026): training and all evaluations completed. Candidate deployment is held due to external Hindi recall regression. See [evaluation report](EVALUATION-2026-11-19.md). Earlier entries below are historical checkpoints.

# Public-corpus migration: work in progress

The production 300K dataset and fitted models have not been replaced.

## Local construction

From `backend`, install `requirements-dataset.txt` into the research environment and run:

```powershell
.venv/Scripts/python.exe -m scripts.build_public_corpus --max-scan 1000000
.venv/Scripts/python.exe -m pytest tests/test_corpus_integrity.py -q
```

The command uses pinned Hugging Face streams. It does not download complete JailbreakDB CSV files, write to MongoDB, or fit models. A bounded hash-priority reservoir samples each scan window, favoring available Hindi, short and keyword records. The summary records the scan window; this is not uniform sampling of all 12 million upstream prompts.

Original prompt bytes are retained in UTF-8. NFKC, zero-width removal, lowercasing and whitespace collapsing apply only to duplicate hashes. Conflicting-label records are excluded. Known external sources (including NotInject and HarmBench) are excluded even when copied into an aggregator. PII-only and system-context-dependent records are rejected. Language is detected using the pinned fastText lid.176.ftz model (176 languages, CC-BY-SA-3.0; https://fasttext.cc/docs/en/language-identification.html), with a minimum confidence of 0.70; Latin script alone does not establish English.

Parent IDs, when provided, and conservative long shared prefix/suffix groups remain in one split. Semantic near-duplicates and missing upstream lineage remain limitations requiring audit before scientific claims. Hash-based group assignment targets 80/10/10, not exactly those partition counts.

Failed candidates have UNVERIFIED in their filename. The integrity JSON must pass before any staging import. Source names and original annotation provenance are retained. Public-source is not synonymous with human-authored; synthetic and unknown origin remain explicit.

## Audited sources

- JailbreakDB: CC-BY-4.0 aggregate card; upstream license and synthetic status must be preserved rather than overwritten. Revision 2c2141bf74ebbe1fd3666a4b38be8856f6e47a92.
- Quickium main/train: 71,117 records before filtering. Preserve source IDs. Exclude PII-only records and SPML entries requiring associated system instructions. Revision 8d67314bd5c08497ed1f20a62009bec58e7c5eb0.
- Quickium train_nc is not used as filler: its WildJailbreak/WildGuard sources are largely synthetic and have separate non-commercial terms.
- Lakera Gandalf: MIT, crowd-sourced with automatic filtering; these are not independently reviewed HMEG annotations.
- deepset: the card and nested dataset_info list different licenses; preserve that ambiguity.
- SPML: source documentation advises against context-free prompt-only training. Excluded until system context can be represented faithfully.
- Octavio multilingual: GPL terms, filter English/Hindi by content rather than assuming all rows match the card languages.
- NotInject: reserve for external benign hard-negative evaluation. It is an upstream generated benchmark, not manually reviewed natural traffic.

The initial Lingua scan was stopped before producing a corpus because its full-language confidence pass was too slow. The reproducible builder now uses fastText; detector metadata and checksum are recorded. The small reserved NotInject set used Lingua during its separate language audit.

## Remaining gates

1. Complete source/license audit and corpus quota/integrity checks.
2. Verify Atlas access and space for both old and staging collections. Never free space by dropping the working dataset first.
3. Import and verify staging; retain rollback until API reads are verified.
4. Train fresh experts using training only; generate five-fold stratified group OOF predictions. Choose fusion/router on validation only.
5. Evaluate untouched test/OOD records and compare candidate/current models on identical records.
6. Promote artifacts only after loading, inference, regression and improvement checks. Existing results retain their original dataset/model attribution.

No new 275K accuracy, trained model, migration success or deployment is claimed by this document.

## Implementation status

`ml/oof.py` checks disjoint groups and exactly one out-of-fold prediction per training record. `ml/ensemble/research_meta.py` provides versioned 12-feature LR/XGBoost fusion; the existing five-feature bundles remain compatible. These modules have regression tests but no new corpus models have been fitted yet.

The Stress Test tab was deployed independently in commit 057fdc3 and verified on Vercel in both themes. It reports the old model diagnostics explicitly. Atlas was rechecked successfully: 300,000 existing records; dbStats reported 237 MiB logical data, 123 MiB storage and 57 MiB indexes. These figures are a preflight snapshot, not a guarantee that staging fits.

## Completed local candidate (18 November 2026)

- 275,000 unique prompts: 137,500 benign and 137,500 security-positive.
- 274,384 English; 616 Hindi; 140,829 short prompts; 38,093 keyword-containing prompts.
- Train 213,953; validation 29,357; test 31,690. Both labels occur in each partition.
- 3,867 duplicate encounters removed; 20 conflicting hashes excluded.
- 226,742 records marked upstream synthetic; 48,258 unknown origin. This is **not** a human-authored-only corpus.
- 53 recorded upstream source names. Sampling scanned 300,000 records per JailbreakDB split and all 71,117 Quickium main/train records.
- 252 separately reserved NotInject examples; no new model metrics yet.
- Structural integrity passes. Semantic near-duplicates and unresolved upstream licenses remain documented limitations; the aggregate CC-BY license must not be presented as individual upstream-license verification.

### Atlas capacity and next action

Read-only preflight: current cluster data/indexes 309,332,220 bytes; candidate BSON 431,644,534 bytes; reserve 67,108,864 bytes. Projected coexistence 808,085,618 bytes exceeds the 536,870,912-byte free limit. No staging writes were attempted on the existing cluster.

The user selected a separate free Atlas project/cluster. Add `MONGODB_MIGRATION_URI` to the ignored `backend/.env`, keeping `MONGODB_URI` unchanged. An optional `MONGODB_MIGRATION_DATABASE` defaults to `hmeg_research`. The new importer rejects the production hostname and any preexisting production dataset in the target.

After the new URI is configured, from backend:

```powershell
.venv/Scripts/python.exe -m scripts.stage_public_corpus data/HMEG_275K_English_Hindi_RealSource.csv
```

This imports `dataset_samples_new` in 1,000-record batches, resumes using the CSV checksum, verifies count, filters, unique hashes and sampled round trips, and leaves production unchanged. `--activate` additionally renames the verified collection inside the **new** cluster; this does not deploy the API or change the old cluster. Model training, API switch, promotion and old-cluster cleanup remain pending. Do not point the public website at the candidate dataset before model/data attribution and API smoke tests are ready.

CSV SHA-256: `e7fb549713b87f01f217109a6fed56e76c22c89713a00d486011d7233e26b7d7`

## New cluster activation completed (18 November 2026)

The configured `MONGODB_MIGRATION_URI` was verified to use a distinct cluster. All 275,000 records were imported and verified, then activated there. An integer-key BSON metadata error was corrected and recovery was tested before completion. The activation checkpoint is now `activated`.

The new-cluster API smoke passed health, statistics, Hindi/keyword/short filters, all 53 sources and reproducibility. The original production collection still contains 300,000 records. The deployed website connection has not been changed.

Fresh candidate training command (from backend):

```powershell
.venv/Scripts/python.exe -m scripts.train_public_corpus data/HMEG_275K_English_Hindi_RealSource.csv
```

It checks the activated corpus checksum, caches frozen semantic embeddings, produces five-fold group-aware OOF scores, trains final experts on training only, compares LR/XGBoost fusion and simple alternatives on validation, and selects routing on validation. Candidate artifacts go under `backend/models/new/<version>/`; no promotion occurs. Latency, baselines, ablations, OOD and old/new comparison remain required before deployment. The inference loader preserves original formatting and candidate routing defaults for these new bundles, while old bundles retain their legacy behavior.

### Active training checkpoint

Candidate run: `5bf4eaa426dbf55293ac68822eabfa94`. Frozen embedding construction is running; the five-fold expert fits and final models follow automatically. The first 25,000 cached embeddings used two CPU threads; the run resumed with six after a numerical-equivalence benchmark. Thread profiles are recorded in the candidate manifest. Training source files are archived under the candidate's `source_snapshot/` directory.

Post-import cluster usage: 431,659,058 bytes of data plus 92,180,480 bytes of indexes, leaving about 12.4 MiB. Keep model binaries and large per-record outputs out of MongoDB. No public database connection or deployed model has been switched.

`evaluate_public_baselines.py` prepares full-training LR, calibrated SVM, Random Forest and XGBoost comparisons, chooses the best baseline using validation, and measures warm local CPU latency. It has not yet run against fitted candidate models.

External preparation saved 1,015 additional records: 108 JBB behavior examples and 907 Aya Hindi examples. 72 exact overlaps with the 275K corpus were removed. Positive examples are explicitly harmful-behavior proxies. Report source-specific results, not a misleading pooled benchmark; Aya is positive-only. NotInject's 252 benign examples remain separate. Qualifire was skipped because access is gated. TrustAIRLab was not used for OOD because the training aggregation includes the potentially overlapping DAN alias.
