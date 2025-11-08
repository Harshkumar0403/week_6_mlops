# ==========================================================
# Base image
# ==========================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ==========================================================
# Install system dependencies
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# Copy requirements and install
# ==========================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# Copy source code and GCP key
# ==========================================================
COPY app.py .
COPY deploy.py .
COPY github-dvc-key.json .
COPY models ./models/

# ==========================================================
# Expose FastAPI port
# ==========================================================
EXPOSE 8000

# ==========================================================
# Command to run the app
# ==========================================================
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]

