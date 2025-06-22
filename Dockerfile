FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /
# 일단 주석 처리
# COPY enhancement/handler.py /handler.py
# COPY enhancement/requirements.txt /requirements.txt
RUN echo "Build test"
CMD ["echo", "test"]
