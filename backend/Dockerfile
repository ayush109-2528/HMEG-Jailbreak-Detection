FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY database ./database
COPY ml ./ml
COPY scripts ./scripts
RUN useradd --create-home --uid 10001 hmeg && mkdir -p /models && chown hmeg /models
# Package only the approved inference artifact for hosts without a persistent disk.
COPY --chown=hmeg:hmeg models/hmeg.joblib /models/hmeg.joblib
ENV SEMANTIC_MODEL_DIR=/models/semantic
RUN python -m scripts.download_semantic && chown -R hmeg:hmeg /models/semantic
USER hmeg
ENV MODEL_DIR=/models PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
