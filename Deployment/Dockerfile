FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES="-1" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PYTHONHASHSEED=0 \
    NUMPY_SEED=0 \
    PORT=8080  

# Install system dependencies and cleanup
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libnuma-dev \
    rng-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Modified command to use PORT environment variable
CMD ["sh", "-c", "rngd -r /dev/urandom & uvicorn main:app --host 0.0.0.0 --port $PORT"]