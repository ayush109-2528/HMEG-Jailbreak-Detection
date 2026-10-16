# HMEG Research Observatory

Detection of Jailbreak Prompts Targeting Large Language Models Using Machine Learning.

This is the new project. The adjacent `jailbreak-detection-dashboard` directory is retained as an untouched backup. MongoDB contains the supplied 300,000-row corpus. The public interface reads research data and runs HMEG inference; it has no public training, upload, or configuration-write route.

## Local startup (Windows)

From `F:\implementation\HMEG-Jailbreak-Detection`:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8006
```

In a second terminal:

```powershell
cd F:\implementation\HMEG-Jailbreak-Detection\frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5176. API documentation: http://127.0.0.1:8006/docs.

For a new machine, create a Python 3.12 environment (`python -m venv .venv`) in backend, install `requirements.txt`, and create `backend/.env` from `.env.example`. Virtual environments are local machine artifacts, not source code. Keep credentials out of Git and frontend variables.

## Connected architecture

- `frontend/src/main.tsx`: React Router, sidebar, lazy-loaded pages, theme, global navigation search.
- `frontend/src/pages`: the ten public pages; no duplicate analyzer/architecture page.
- `frontend/src/api/client.ts` → `VITE_API_URL/api/...`; locally Vite proxies `/api` to port 8006.
- `frontend/src/hooks/useResource.ts`: abortable reads that keep existing rows visible during updates.
- `frontend/src/components` and `charts`: reusable UI, interactive architecture, measured charts.
- `backend/app/main.py`: FastAPI lifecycle, CORS, public routes, bounded inference, model refresh.
- `backend/app/catalog.py`: expert descriptions and whitelisted source-code walkthrough.
- `backend/database`: compact collection mapping, MongoDB connection, indexes and bounded reads.
- `backend/ml/experts`: Word, Format, optional Meaning, Pattern, Instruction.
- `backend/ml/ensemble`: confidence router, flat fusion, learned fusion, HMEG orchestration.
- `backend/ml/training.py` and `ml/evaluation`: disjoint partitions, evaluations, ablations, OOD, routing studies.
- `backend/scripts`: connection check, resumable CSV importer, operator training CLI, hosted-worker queue.
- `backend/models`: trusted serialized model artifacts. Never load pickles submitted by public visitors.
- `backend/tests`: isolated regression tests; their fixtures are not research records.
- `frontend/tests/e2e`: browser regression, visual screenshots and accessibility checks.

## HMEG flow

Word + Format scores produce a weighted fast score. At or below LOW_THRESHOLD, exit benign; at or above HIGH_THRESHOLD, exit jailbreak. Uncertain prompts execute available deeper experts and learned fusion. Every result reports actual execution/skipped status, scores, timings, route and artifact version. Missing models produce HTTP 503, never simulated outputs.

The optional Meaning Expert uses a **frozen neural sentence encoder** plus Logistic Regression. This is a hybrid representation, not a fine-tuned generative model. Without that encoder, the system trains a clearly labelled four-expert configuration. Install `requirements.txt`, run `python -m scripts.download_semantic` from backend, then train with `--semantic`. The CPU implementation uses the official pinned quantized ONNX MiniLM model, masked mean pooling and L2 normalization, capped at 256 wordpieces. Docker downloads the same assets at build time; requests do not download models.

## Dataset and honest evaluation

Input columns: `sample_id,prompt,label,class_name,attack_category,expert_focus,source_dataset,base_record_id,source_record_id,provenance,is_synthetic,transformation,parent_id,research_split,verification_status,prompt_sha256,normalized_sha256`.

Labels: 0 benign, 1 positive research class. The current file includes harmful-behavior proxies as well as jailbreaks. Of 300,000 rows, 298,985 are marked synthetic and 1,015 non-synthetic. Preserve this distinction when explaining results.

```powershell
python -m scripts.check_database
python -m scripts.import_dataset "PATH_TO_DATASET.csv" --dry-run
python -m scripts.import_dataset "PATH_TO_DATASET.csv"
python -m scripts.train --max-train 20000 --max-eval 1000 --publish --studies
```

Import is resumable for the same fingerprint and refuses a different dataset version in a populated database. It does not create fake rows. The first installed model uses 20,000 training / 1,000 validation / 1,000 test rows after lineage/hash overlap removal and deduplication. These are sampled exploratory results, **not a full-corpus evaluation**. `--max-train 0 --max-eval 0` selects all eligible rows and can require substantial memory/time.

`--held-out-source "exact source name"` supports OOD evaluation; OOD artifacts are not automatically published. `--studies` records available-expert ablations and routing settings. Single-guardrail baseline and OOD sections remain empty until those experiments are recorded; external guardrails have no fabricated results. Repeated test inspection must not be used as an unreported model-selection process.

## Hosted deployment and training

See [deployment instructions](docs/DEPLOYMENT.md). Frontend can deploy to Vercel. API and training worker require a Python/container host with a shared persistent model volume. MongoDB stores data/results/jobs; it does not execute ML training. No hosted worker has been provisioned by this repository.

## Tests

```powershell
# backend
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
# frontend (Microsoft Edge installed for the configured headless runner)
npm run build
npm run test:e2e
```

The browser suite expects the API on 8006 and frontend on 5176. Screenshots and reports go into ignored `frontend/test-output`.

## Public API

GET `/api/health`, `/api/dataset/stats`, `/api/dataset/samples`, `/api/dataset/sources`, `/api/database/stats`, `/api/experts`, `/api/experiments`, `/api/experiments/{id}`, `/api/code`, `/api/routing/config`, `/api/reproducibility`, `/api/live-tests`, `/api/live-tests/{id}`.

POST `/api/predict/hmeg`: `{"prompt":"Explain a rainbow","public_consent":false}`. Consent defaults false: prompt content is not persisted. Explicit true makes the prompt/result publicly readable by ID. Do not include secrets in shared analyses. Rate limiting is per API process; use ingress-level rate limits for a multi-instance deployment.

This academic prototype is not a certified safety boundary. It measures a detector, not successful jailbreaks against a target LLM.

## Latest verification

See [regression and visual QA report](docs/QA-2026-10-16.md) for passed tests, screenshots, moved files, and deployment limitations.
