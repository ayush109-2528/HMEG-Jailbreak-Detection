# Two research tracks — local verification and deployment handoff

Verified locally on 25 November 2026. No deployment or Git push was performed.

## What the selector changes

The Research track list switches dashboard data, dataset samples, saved experiments, expert status, routing defaults, live inference and reproducibility together, per browser/request. It does not mutate a global active model. Existing pages and theme remain intact. Requests without `profile` default to synthetic; allowed values are `synthetic` and `source`. Shared-result links include `track` so they open against the correct database.

| Track | Dataset | Model | Evidence |
|---|---|---|---|
| Synthetic dataset | 300,000 original augmented rows | 4f44b61012ee4344958544a480186c9a | 58 existing experiment records preserved |
| Sourced dataset | 275,000 English rows, balanced labels | 2f45a12a0c56ba7afe6f322c6d743511 | 42 measured experiment records imported |

The sourced corpus includes 226,742 upstream-synthetic records and unknown-origin records. Public sourcing is not proof of human authorship. Dataset Explorer links to the original repositories and records source/license provenance. The model remains explicitly labelled a research candidate: its external false-positive gate did not pass, and below-5% FPR is not claimed. Both tracks are available for demonstration, not certified guardrails.

The synthetic Atlas database was not changed. The separate sourced cluster was verified against all 275,000 normalized hashes. Its former 616 Hindi records are retained in `dataset_hindi_archive`; prior dataset metadata is retained as `bilingual_dataset_archive`. The original bilingual CSV also remains local. The sourced API rejects data whose checksum does not match the model's recorded corpus.

## Local run

From `backend`:

```powershell
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8006
```

From `frontend`:

```powershell
$env:VITE_API_URL=''
node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5176 --strictPort
```

Open http://127.0.0.1:5176/dataset and choose a Research track. Live Detection uses that track's model. Local hosting still requires internet access for Atlas dataset reads; inference artifacts are local. Private prompts are not stored.

## Backend deployment settings

Keep the original `MONGODB_URI` and `MONGODB_DATABASE`. Add `SOURCE_MONGODB_URI` for the second Atlas cluster and `SOURCE_MONGODB_DATABASE=hmeg_research`. Locally, the existing `MONGODB_MIGRATION_URI` is accepted as a fallback; prefer the explicit SOURCE name when hosting. Never put either URI in frontend variables or Git.

The Dockerfile now packages both small fitted bundles:

- `backend/models/hmeg.joblib`: original model, preserved.
- `backend/models/source/hmeg.joblib`: copied from the English candidate; no rejected fusion experiment is activated.

Docker sets `MODEL_DIR=/models`, `SOURCE_MODEL_DIR=/models/source`, and downloads the existing pinned semantic assets. The local source-model default points to its versioned models/new directory. Keep `FRONTEND_URL` set to the exact frontend origin. Keep both Atlas network access lists configured for the backend host.

Model files remain Git-ignored. Before your deployment, include these exact trusted artifacts in the deployment source or your existing artifact-distribution mechanism; do not upload the entire models/new folder, datasets, .env, or caches. Build context is `backend`, Dockerfile is `backend/Dockerfile` relative to the repository root. The application binds port 8000 in Docker.

Vercel frontend remains Vite: root `frontend`, build `npm run build`, output `dist`, `VITE_API_URL` set to the backend origin without `/api`. Changing frontend environment variables requires rebuilding it. MongoDB settings belong only on the backend.

## Verification evidence

- 69 backend regression tests passed, including profile isolation and checksum mismatch rejection.
- Production frontend build passed.
- 44 populated-page checks: 11 pages × 2 tracks × 2 themes; two browser live predictions.
- Both real database health checks and public API panel reads passed.
- Concurrent five-expert predictions used the correct distinct models.
- Dataset filtering, pagination, record drawer, source links, and track switching from shared URLs passed.
- One document load across switching and navigation: no full-page refresh.
- Local concurrent API peak RSS: approximately 452 MiB. Measure the Linux container separately; this is not a free-host memory guarantee.

Evidence is under ignored `test-output/`: dual-track-api.json, dual-track-browser.json, dual-track-interactions.json, dual-track-memory.json and sourced-final screenshots. Database preparation report: backend/data/dual_track_database_report.json.

## Remaining deployment verification

Docker is not installed on this PC, so the Linux container build/runtime has not been tested here. Before exposing the updated site, build the image and verify both `/api/health?profile=synthetic` and `/api/health?profile=source`, both dataset statistics endpoints, CORS for your Vercel origin, and one private prediction per track. Confirm totals 300,000 and 275,000, respectively. Confirm memory under concurrent inference on the chosen host.

No further training is required to demonstrate these two existing models. Operator training commands in older deployment notes apply to the original training workflow; do not use `--publish` as a demonstration/setup step.
