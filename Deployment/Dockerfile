FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and cleanup
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for TensorFlow
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8082

# Command to run the application with proper error handling
CMD ["python", "-c", "import os; os.environ['PYTHONPATH'] = '/app:' + os.environ.get('PYTHONPATH', ''); import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8082)"]