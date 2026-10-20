# Short-input serving safeguard and contrast regression — 2026-10-20

## Scope
The binary model is unchanged (4f44b61012ee4344958544a480186c9a). This is an explicit serving-time abstention policy, not improved model accuracy or a retraining claim. No generated samples were added to training and no held-out rows were fitted.

An isolated Unicode alphabetic token (including combining marks), optionally wrapped in sentence punctuation, yields `prediction=insufficient_context`, `label=null`, `final_probability=null`, and a request for context/manual review. This includes recognizable words and random letters of any length. It does NOT silently allow tokens such as DAN or IGNORE. Digit-containing encodings, symbols, and multiword instructions retain the binary model decision. This narrow scope is not an adversarial-input detector.

The raw binary prediction, label, probability, and action are retained under `raw_model`; expert execution and scores remain intact. Clients must handle nullable label/probability and must not treat null as benign. The frontend shows a neutral inconclusive verdict and labels raw scores as unreliable for this input. Public result storage preserves the full record; private predictions remain unstored.

## Evidence and limitations
A hand-authored 74-input diagnostic set is not a representative accuracy benchmark. Before: 22/30 ordinary words and 36/36 random strings flagged. After applying the policy to those recorded actual predictions: all 66 isolated inputs abstain; 4/5 complete benign sentences remain benign; 1/5 remains a false positive; 3/3 override controls remain jailbreak. Abstention is not a correct benign prediction and must not be counted as one.

The existing CSV test partition contains 32,141 rows; zero meet this isolated-token condition. Therefore the binary benchmark predictions and measured accuracy are unchanged, and that benchmark does not measure this safeguard's effectiveness. Future evaluation must report abstention coverage separately and use reviewed short benign and attack-like inputs in separate training/validation/test groups.

Format and Pattern scores are high for many short inputs; learned fusion can outweigh low Meaning/Instruction scores. The training partition contains 3,327 positive single-whitespace-token rows versus 576 benign rows, suggesting a coverage/skew issue, not proof of causality. Meaning remains a frozen MiniLM representation plus a trained classifier.

## Visual fixes
Light primary-button hover/focus uses a dark teal background with white text. All chart tooltip labels/items use the active theme's panel/text colors. Inconclusive verdicts have explicit readable colors in both themes.

## Validation
44 backend tests passed. Frontend production build passed. Browser regressions cover both routing paths in both themes, inconclusive rendering, primary hover contrast, and chart tooltip contrast. Deployment verification is recorded separately by the operator.
