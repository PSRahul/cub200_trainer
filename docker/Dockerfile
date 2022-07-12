FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Python 3
RUN apt-get update && apt-get install -y python3 python3-dev python3-pip  build-essential cmake \
    && pip3 install --no-cache-dir --upgrade pip setuptools

