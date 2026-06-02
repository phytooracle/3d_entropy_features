FROM python:3.9-slim

WORKDIR /opt
COPY . /opt

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:$PYTHONPATH

ENTRYPOINT ["python3", "/opt/3d_entropy.py"]
