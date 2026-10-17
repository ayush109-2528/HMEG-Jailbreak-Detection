# Deployment: Vercel frontend + Python API/worker + MongoDB Atlas

The source is prepared; a hosting account/service and public API URL still need to be configured. No new paid hosting was purchased or provisioned.

## Frontend on Vercel

Set Root Directory to `HMEG-Jailbreak-Detection/frontend` if the repository contains both project folders, or `frontend` if the repository root is this project. Framework: Vite. Install: `npm ci`. Build: `npm run build`. Output: `dist`.

Set `VITE_API_URL=https://YOUR_API_HOST` (origin only, without `/api`), then redeploy. `frontend/vercel.json` handles SPA deep-link refresh. Do not put MongoDB credentials into Vercel frontend variables.

## API and worker host

Use a Linux Docker host with enough RAM for your chosen training sample size (start with 4 GB for the supplied limits; measure before full-corpus training). Supply backend environment variables securely:

- `MONGODB_URI`: Atlas driver connection string
- `MONGODB_DATABASE=hmeg_research`
- `FRONTEND_URL=https://YOUR_VERCEL_DOMAIN` (comma-separated additional approved origins)
- `MODEL_DIR=/models`
- `LOW_THRESHOLD=0.2`, `HIGH_THRESHOLD=0.8`, `FINAL_THRESHOLD=0.5`

Allow the backend host's egress IP in Atlas Network Access. Configure HTTPS at the host/proxy. Use `/api/health` as a readiness inspection; `status=ok` means the API process is alive, while `database_connected` and `models_loaded` report optional dependency readiness separately.

`docker compose up --build -d` runs the API and one worker on a **single Docker host**, sharing the persistent `model_artifacts` volume. Do not assume two separate hosting services automatically share a filesystem. If your provider cannot share a volume between services, use one VM/container host for this Compose deployment or add authenticated object-storage artifact distribution before splitting services.

Copy the locally validated `backend/models/hmeg.joblib` into the API's `/models/hmeg.joblib` volume, or enqueue a training job and publish it. Model artifacts are ignored by Git intentionally. The Docker image deliberately does not embed local secrets or private data.

## Queue a training run (operator terminal)

```sh
docker compose exec worker python -m scripts.worker enqueue --max-train 20000 --max-eval 1000 --studies --publish
docker compose exec worker python -m scripts.worker status
```

The worker atomically claims queued MongoDB jobs and records completed/failed state. Run **one worker replica per model volume**. `--publish` explicitly permits that run to atomically install its artifact; inference reloads it on the next request. For review-before-promotion, omit `--publish`, inspect the experiment records and artifact manifest, then copy the approved artifact to `hmeg.joblib` atomically.

The worker is not self-training from predictions: it uses labelled imported research rows. Public traffic cannot start training or alter labels. Interrupted jobs remain marked running for operator investigation and are not silently retried; inspect the host logs and enqueue a new job when appropriate. Database credentials are intentionally not printed in errors.

## Deployment verification

1. GET `/api/health`: check database and model flags.
2. Open every frontend route directly and refresh it.
3. Inspect dataset totals and page 2; confirm 300,000 records when this corpus is deployed.
4. Submit a non-shared benign example and an attack-pattern example; inspect actual expert statuses.
5. Verify CORS against the exact frontend domain.
6. Queue a small operator training job, inspect its completion, and confirm artifact version changes only when published.

A MongoDB import alone does not update an existing Vercel deployment. Deploy this new frontend and API configuration explicitly; the old Vercel site remains separate.


## Five-expert deployment on the existing Render free service

The Meaning Expert uses the pinned official all-MiniLM-L6-v2 quantized ONNX encoder (~23 MB), with Logistic Regression. PyTorch and sentence-transformers are not required. The encoder is frozen, not fine-tuned. Training and inference use identical pooling, truncation and quantization. The training manifest records encoder revision and checksums; runtime verifies the deployed assets against them.

1. Train on the PC/worker: from backend run `python -m scripts.download_semantic`, then `python -m scripts.train --semantic --max-train 20000 --max-eval 1000 --studies`. Review the candidate before promoting to `models/hmeg.joblib`.
2. Commit the approved `backend/models/hmeg.joblib` explicitly (it is Git-ignored by default), together with the code and pinned requirements. Do not commit .env or raw dataset files.
3. Render: Dockerfile Path `backend/Dockerfile`, build context `backend`, Root Directory empty, `MODEL_DIR=/models`, `SEMANTIC_MODEL_DIR=/models/semantic`. The Dockerfile sets the semantic path and downloads the pinned assets during build. Keep MongoDB and frontend-origin environment variables unchanged.
4. Use one Uvicorn process (the default Docker CMD). Do not add multiple worker processes on the free service. Model assets live in the image, so cold starts do not need downloads or a paid disk.
5. Redeploy Render and Vercel. Verify `/api/health` reports `meaning_available: true`, then test an uncertain prompt and check the Meaning Expert actually executes. Confident Layer 1 exits correctly skip it.

The free service remains an inference host. Training runs on the PC or a separately configured batch worker. Local Windows memory measurements are not a guarantee of Linux hosting capacity; check Render memory and logs after deployment. No paid service is provisioned by these changes.

Published comparison numbers are in the frontend comparison page only. They are literature context, never inserted into MongoDB as locally measured experiments.


## Full-dataset research run

From backend, use `python -m scripts.train --max-train 0 --max-eval 0 --semantic --studies --latency-samples 500 --dataset-csv "PATH_TO_IMPORTED_CSV"`.

Zero limits mean all eligible rows in the existing partitions; they do not merge the partitions. The local CSV is accepted only if its SHA-256 matches MongoDB's imported-dataset metadata, and it uses the same canonical import mapping. This avoids repeated large Atlas downloads. Parent-group and normalized-text overlap is removed and checked before fitting. Classifiers fit training rows, meta-classifiers fit validation scores, and test labels are used only for evaluation.

For the 300,000-row corpus the audited partitions contain 234,398 training, 33,461 validation and 32,141 test rows. The complete test partition is scored individually by each expert; these exact scores are reused for fusion comparisons. Real end-to-end latency is separately measured on a seeded 500-prompt sample for each ensemble variant. Individual-expert latency covers every test prompt. The UI displays classification and latency counts separately. Cached computation timings are never reported as real inference latency.

`SEMANTIC_THREADS` controls ONNX CPU threads (default 2 for inference). The full local run uses 8 and records that setting in its manifest. Training-machine timings should not be presented as Render timings. Fitted experts and the fitted bundle are checkpointed under `backend/models/training_runs/<version>/` before evaluation, and each completed experiment is also written locally. These files remain Git-ignored.

Promote and deploy only after reviewing the completed full-test report, inference memory checks and regression tests. Preserve the previous hmeg.joblib for rollback. Do not tune thresholds or select repeated candidates using the held-out test results.

Interrupted full-data training can be resumed with the same command plus `--resume-run <run-id>`. The saved input identity must match the CSV checksum, split counts, limits, and semantic setting. Completed experts are reused; frozen Meaning embeddings are saved every 1,000 rows under `backend/models/embedding_cache`, keyed by ordered prompts and the encoder manifest. Keep these local checkpoints private and out of Git. An interrupted extraction can repeat at most its unfinished chunk.
