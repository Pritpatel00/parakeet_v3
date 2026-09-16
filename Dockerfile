FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    RUNPOD_SERVERLESS=1 \
    PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        ca-certificates && \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        runpod \
        accelerate \
        soundfile \
        librosa \
        requests \
        numpy \
        "transformers @ git+https://github.com/huggingface/transformers.git" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
