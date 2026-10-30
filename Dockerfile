# Reference Dockerfile. If your existing one already installs requirements.txt and copies the
# whole repo (COPY . .), keep yours - just make sure layout.json and chart.json get copied in.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=10000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
