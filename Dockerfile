# Ultra-Safe Dockerfile for RunPod v15.3.3
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 작업 디렉토리를 루트로 설정 (RunPod 표준)
WORKDIR /

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# 핸들러 파일 복사 (루트로)
COPY handler.py /handler.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1

# RunPod 핸들러 시작
CMD ["python", "-u", "/handler.py"]
