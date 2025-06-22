FROM python:3.10-slim

WORKDIR /

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REPLICATE_API_TOKEN=""

# Run handler
CMD ["python", "-u", "/handler.py"]
