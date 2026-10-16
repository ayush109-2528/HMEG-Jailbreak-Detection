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
