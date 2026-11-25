# HMEG candidate evaluation — 19 November 2026

## Decision

**Do not deploy this candidate.** The original improvement gate passes, but an additional source-specific comparison identifies a substantial Hindi recall regression. The deployed model and database remain unchanged.

Candidate: `5bf4eaa426dbf55293ac68822eabfa94`. Current: `4f44b61012ee4344958544a480186c9a`.

## Data and training

275,000 records; train 213,953, validation 29,357, test 31,690. All five experts trained on the complete training partition. Five-fold group-aware OOF predictions train the meta-classifiers. Fusion and routing selected using validation only. XGBoost fusion selected; router 0.10 / 0.90, final threshold 0.50.

226,742 records are upstream synthetic; 48,258 have unknown origin. Hindi representation is only 616 records. Source licenses and semantic/template overlap remain documented limitations. This is not a human-authored-only corpus.

## Paired held-out comparison

31,645 identical test prompts after excluding 45 exact overlaps with all partitions of the old corpus. Actual warm local inference, before serving abstention. No test-driven threshold retuning.

| Metric | Current | Candidate |
|---|---:|---:|
| accuracy | 52.43% | 95.47% |
| precision | 78.42% | 95.55% |
| recall | 22.27% | 96.50% |
| f1 | 34.68% | 96.03% |
| roc_auc | 71.23% | 98.72% |
| fpr | 8.03% | 5.89% |
| fnr | 77.73% | 3.50% |

Average latency: 17.86 → 22.84 ms (local CPU, not hosted).

## External comparisons

| Dataset / metric | Current | Candidate |
|---|---:|---:|
| NotInject, 252 benign prompts, FPR | 29.37% | 25.40% |
| CohereLabs/aya_redteaming: recall | 99.12% | 26.68% |
| JailbreakBench/JBB-Behaviors: fpr | 69.57% | 68.48% |
| JailbreakBench/JBB-Behaviors: recall | 87.50% | 100.00% |

Aya: 907 Hindi positive harmful-behavior proxy prompts. JBB: 108 prompts (92 benign, 16 positive). No exact overlaps with either training corpus remain in these external subsets. Semantic or embedded-goal overlap is not ruled out. Do not pool these source-specific results.

## Completed checks

- All 25 fold/expert fits and five final expert fits completed.
- Four standalone baselines, five flat fusion methods, six ablations, routing validation, and source-specific external evaluation completed.
- 67 regression tests passed; frontend production build passed.
- Candidate health, dataset statistics, Hindi filter, expert catalog, routing and reproducibility API checks passed after Atlas access was restored.
- Inference API passed private prediction, all-five-expert execution, invalid-input and threshold validation checks.
- Local inference peak RSS approximately 356.2 MiB. This is not a guarantee of hosted-container memory usage.

## Next experiment

Collect or review additional Hindi training examples across benign and harmful/instruction-manipulation categories. Diagnose expert scores and routing using training/validation data. Do not move Aya or the existing held-out test rows into training. Since their results have now been inspected, reserve a fresh independent multilingual confirmation set for future promotion. Do not claim the frozen English MiniLM encoder alone establishes multilingual coverage.

## Evidence files

All detailed results remain under `backend/models/new/5bf4eaa426dbf55293ac68822eabfa94/`: model_manifest.json, baseline_results.json, flat_results.json, ablation_results.json, router_threshold_results.json, expert_results.json, test_slices.json, ood_results.json, promotion_comparison.json, old_external_comparison.json, external_old_overlap_audit.json, api_smoke.json, inference_api_smoke.json, deployment_decision.json. No old artifacts were overwritten.
