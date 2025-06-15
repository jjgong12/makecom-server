# Ultra-Safe Dockerfile for RunPod v15.3.3
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 핸들러 파일 복사
COPY handler.py .

# 메모리 최적화를 위한 환경 변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1

# RunPod 핸들러 시작
CMD ["python", "handler.py"]
