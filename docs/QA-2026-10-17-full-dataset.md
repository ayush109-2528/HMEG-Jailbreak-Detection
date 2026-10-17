# Full-dataset HMEG verification - 2026-10-17

Model version: 4f44b61012ee4344958544a480186c9a.

All five experts fit 234,398 training rows. Learned fusion fit 33,461 separate validation rows. All 32,141 held-out test rows were used for classification metrics. Both sampling caps were zero. Cross-partition normalized hashes and parent groups were checked for overlap. The local source CSV SHA-256 matched the imported MongoDB dataset. No records were generated or relabeled for this run.

The imported 300,000-row corpus contains 298,985 synthetic and 1,015 nonsynthetic rows. These are corpus-specific exploratory results; generalization to new sources has not been established by this run. The frozen quantized MiniLM encoder uses the first 256 wordpieces; only its Logistic Regression head is trained.

Twenty experiments cover HMEG, five individual experts, five flat fusions, five ablations, and four routing configurations. Classification uses all test rows. End-to-end ensemble latency uses 500 seeded real test prompts, recorded separately. Individual expert latency covers all test prompts. Test expert scores are cached for repeated fusion comparisons; cached execution timers are discarded.

HMEG metrics:

{
  "accuracy": 0.9803988674901216,
  "precision": 0.9642617449664429,
  "recall": 0.9825611215592409,
  "f1": 0.973325429756965,
  "roc_auc": 0.997132342347554,
  "fpr": 0.020838428802034927,
  "fnr": 0.017438878440759104
}

Confusion matrix (rows true, columns predicted; 0 benign, 1 jailbreak): [[20017, 426], [204, 11494]].

All-five inference checks passed on short and 9,750-character inputs. Local Windows peak RSS was 339.01 MiB; this is not a measurement of the hosted Linux container. No additional Python dependencies were needed.

Training recovered Word and Format checkpoints after process interruptions. The recorded training_seconds is elapsed time for the resumed fitting invocation, excluding previously completed Word/Format fitting and interrupted attempts; it must not be interpreted as a fresh full-run benchmark. Meaning embedding extraction now checkpoints every 1,000 rows, keyed by ordered input and encoder identity.

Rollback preserved locally at backend/models/hmeg-five-expert-20k-rollback.joblib. Existing pages and experiment history are retained. Hosting remains inference-only; training runs separately on the PC.

Final validation: 29 backend tests passed; 34 browser tests passed across dark/light themes, accessibility contrast, mobile layouts, dataset navigation, private predictions, dataset-result attribution, and diagram handles. Production build passed. Light-mode architecture and expert-result screenshots were visually inspected. The Experts page reports measured metrics for the active model version with dataset identity and separate split sizes.
