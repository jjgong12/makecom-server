FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "handler.py"]
