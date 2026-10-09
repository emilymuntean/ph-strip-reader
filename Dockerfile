FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Render provides $PORT; defaults to 10000 if not set
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}
