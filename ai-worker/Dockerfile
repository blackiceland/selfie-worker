FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget libgl1 libglib2.0-0 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --upgrade pip

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY pipeline.py /app/

VOLUME ["/root/.cache/huggingface", "/app/data"]

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "pipeline:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "off"]

