FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY database ./database
COPY ml ./ml
COPY scripts ./scripts
RUN useradd --create-home --uid 10001 hmeg && mkdir -p /models && chown hmeg /models
USER hmeg
ENV MODEL_DIR=/models PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
