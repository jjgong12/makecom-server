FROM python:3.10-slim

WORKDIR /

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# PyTorch 설치 추가
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install simple-lama==0.2.0

# Copy handler
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
