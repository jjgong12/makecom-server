FROM runpod/base:0.6.2-cuda12.2.0
WORKDIR /


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
