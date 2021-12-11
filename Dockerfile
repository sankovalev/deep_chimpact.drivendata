FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /workdir

RUN apt-get update && apt-get -y install \
    curl git ffmpeg libsm6 libxext6 zip

# Install aws
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install

# Install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set up environment variables
ENV PYTHONPATH $PYTHONPATH:src

COPY ./ ./
