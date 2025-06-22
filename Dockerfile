FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY enhancement/requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler files
COPY enhancement/handler.py /handler.py

# No need to copy entire folders for now
# COPY enhancement/ /enhancement/
# COPY thumbnail/ /thumbnail/

# Set the handler
CMD ["python", "-u", "/handler.py"]
