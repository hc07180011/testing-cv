# NTU-Google-Testing

## Image Quality Assessment (IQA)

* [NIMA](https://github.com/idealo/image-quality-assessment)
* [Explainable](https://github.com/marcotcr/lime)
* Auto correction
  * White-balance
  * Color
* Detection
  * Blur

## Flicker Detection

[![Build Status](https://travis-ci.com/hc07180011/NTU-Google-Testing.svg?branch=main)](https://travis-ci.com/hc07180011/NTU-Google-Testing)

* Embedding
  * [Siamese Network](https://keras.io/examples/vision/siamese_network/)
  * [Facenet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)
* Movement
  * [BRISK](http://margaritachli.com/papers/ICCV2011paper.pdf)
* Transformation
  * [Affine Transformation](https://en.wikipedia.org/wiki/Affine_transformation)

## Install

### Ubuntu

#### Tested on
1. [Docker Ubuntu Official Image 18.04](https://hub.docker.com/_/ubuntu/)
2. [Google Cloud Platform - Ubuntu 16.04, 18.04, 20.04](https://cloud.google.com/)

```bash
# Update and install packages
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
  xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# Install pyenv
curl https://pyenv.run | bash
export PATH="${HOME}/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install python3.8.5
pyenv install 3.8.5

# Clone this repo
git clone https://github.com/hc07180011/NTU-Google-Testing.git
cd NTU-Google-Testing/flicker_detection/flicker_detection

# Download the facenet pre-trained model
wget https://hc07180011.synology.me/data/googlecv/facenet_model.lite.h5 -O preprocessing/embedding/models/facenet_model.lite.h5

# Activate environment and install dependencies
~/.pyenv/versions/3.8.5/bin/python -m venv .env
source .env/bin/activate
python3 -m pip install -r requirements.txt

# Run the program
time python3 main.py -d data/test_data.mp4 
```

### Docker

* Prerequisite: [Docker](https://www.docker.com/)

```bash
# Clone this repo
git clone https://github.com/hc07180011/NTU-Google-Testing.git
cd NTU-Google-Testing/flicker_detection/flicker_detection

# Build the container
docker build -t flicker_detection_runner .

# Run the container with a specific input data (put it under data/ directory)
docker run --rm -e data=data/test_data.mp4 -v $PWD/data:/app/data -it flicker_detection_runner
```

## Slides

* [5/4](https://drive.google.com/file/d/1um59arpNZVOS2UmyMSxypDCklbBbKlBi)
* [5/11](https://drive.google.com/file/d/1gEhwh-pY5t-7Ix1nneKWAur47nagUbfy)
* [5/25](https://drive.google.com/file/d/1wh3mGCUHGBR11b5FHrI4YgFZRLMF2ipw)
* [6/1](https://drive.google.com/file/d/1IGPqMAVWqndF0k2e7aXXP-gOXZTWnGjs)
* [6/15](https://drive.google.com/file/d/1y7P_qGNkOVq9wSbiZdiTu8i5b3kdqOog)
* [6/30](https://drive.google.com/file/d/1oXYgYuQcl1E5JUEygJE_t0urHzT1oX55)
* [7/14](https://drive.google.com/file/d/1B_2AIrGZRO07QqMKo3mYzosTIg-A8zgO)
* [7/21](https://drive.google.com/file/d/1a5uiGk7ElbPZHjLxChIcnS1g-iaGe3VV)
* [7/28](https://drive.google.com/file/d/1BXtmFVxO2bWC3oga_7Vbzf_-jwawpzY-)
* [8/4](https://drive.google.com/file/d/14fz6tNubJawxpn6vnBE-WVBHDnlda544)
* [8/11](https://drive.google.com/file/d/1S4hoHK0-3oV1aeijbZ2ICQPIQILVjLPV)
* [8/18](https://drive.google.com/file/d/1vFAdzpc0CTnlqOq4ucu6RE7N_CGKMSY5)
* [8/25](https://drive.google.com/file/d/1LUJFLgKNUu_0yiEAaOTkhpT7DQNlPIWC)
* [9/1](https://drive.google.com/file/d/1DcU3XVbmaR31BtqAaqMTwI9pR1-u0ykm)
* [9/29](https://drive.google.com/file/d/1BJClB6p_dfQWjI_WQFxyuL7LTKVbR8Oj)
* [10/6](https://drive.google.com/file/d/1kma_4n1uy5_-fOrhFYBzGI5tW8A0xmkq)
* [10/13](https://drive.google.com/file/d/1snQrZhz0LZrvv4-JnOBl5HnHNEVNXWze)
* [10/27](https://drive.google.com/file/d/1BbC2ZIP-f33-nByzd58ykvzi4ET_KiAB)

## For more details

* [Proposal](https://docs.google.com/document/d/1vhABHWuuDh31VZ_OTp5DGJH15cjqedEOAQsllqd5iGc/edit?usp=sharing)
