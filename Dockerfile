# PyTorch 이미지 사용 - torch가 미리 설치되어 있음
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# System dependencies for OpenCV (필요시)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-download LaMa model to avoid runtime delays
RUN python -c "from simple_lama import SimpleLama; SimpleLama()"

# Copy handler
COPY handler.py /handler.py

# Set the entrypoint
CMD ["python", "-u", "/handler.py"]
