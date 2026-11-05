# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python application
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
COPY --from=frontend-builder --chown=user /build/dist ./frontend/dist

EXPOSE 7860

ENTRYPOINT ["python", "-u", "start.py"]
