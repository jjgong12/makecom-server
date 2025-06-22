FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --no-deps 제거!!!
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
CMD ["python", "-u", "handler.py"]
