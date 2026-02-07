# Base image (CPU)
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
EXPOSE 8000

# Env vars for rewards and timeout
ENV TIMEOUT_HOURS=12
ENV LOOP_INTERVAL_SECONDS=60
ENV POSITIVE_REWARD=0.8
ENV NEGATIVE_REWARD=-0.5
ENV TIMEOUT_PENALTY=-2.0
ENV AUTO_SAVE_INTERVAL=50

# Single worker so the timeout thread doesn't duplicate
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]