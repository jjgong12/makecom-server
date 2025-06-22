FROM python:3.10-alpine

# 필수 시스템 패키지만 설치
RUN apk add --no-cache \
    libgcc libstdc++ libffi-dev \
    libjpeg-turbo libpng libwebp \
    openblas lapack

# Python 패키지 (no-cache, no-deps)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt

COPY handler.py .
CMD ["python", "-u", "handler.py"]
