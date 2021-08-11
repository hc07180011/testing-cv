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
wget https://hc07180011.synology.me/data/googlecv/facenet_model.h5 -O preprocessing/embedding/models/facenet_model.h5

# Activate environment and install dependencies
~/.pyenv/versions/3.8.5/bin/python -m venv .env
source .env/bin/activate
python3 -m pip install -r requirements.txt

# Run the program
time python3 main.py -d tests/test_data.mp4 
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

* [5/4](https://drive.google.com/file/d/1aVSWCC9GXZOBZjz8sjVIfGsHkOgVYl5Z/view?usp=sharing)
* [5/11](https://drive.google.com/file/d/1NK_6WXSEU-nph6-zQVSXyBaWTfwp-Ui4/view?usp=sharing)
* [5/25](https://drive.google.com/file/d/1rZaxUIxic1-Nu3rF4XP9oH2YC4J1_EoV/view?usp=sharing)
* [6/1](https://drive.google.com/file/d/1BOe8rsGuGdZxant4D-6FjkE7ptCs0Cfj/view?usp=sharing)
* [6/15](https://drive.google.com/file/d/12M3fkW3vxlglwOrtW_Tppe3iCP7SUKQ0/view?usp=sharing)
* [6/30](https://drive.google.com/file/d/105ZfZX9DWU7Jwpbme24X7q2YIdhFe0Hu/view?usp=sharing)
* [7/14](https://drive.google.com/file/d/1-5u5J8cRsPFO-DoMjQjONRmkHXlfPxGj/view?usp=sharing)
* [7/21](https://drive.google.com/file/d/1DAHQn88BdUHCFZlU0tV-qhOyGIAvTlbp/view?usp=sharing)
* [7/28](https://drive.google.com/file/d/1zrOD3kwofuVjEaQ5XkXXM7ymbs1fujCL/view?usp=sharing)
* [8/4](https://drive.google.com/file/d/1yfbYejz3P3AzQMykJ-yIWynyCUseJIhA/view?usp=sharing)
* [8/11](https://docs.google.com/presentation/d/1CJW8kz393O-I4MTPlevxvmpfsUYRzzkv/edit?usp=sharing&ouid=101634550160064789006&rtpof=true&sd=true)

## For more details

* [Proposal](https://docs.google.com/document/d/1vhABHWuuDh31VZ_OTp5DGJH15cjqedEOAQsllqd5iGc/edit?usp=sharing)
