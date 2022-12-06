# testing-cv

## Image Quality Assessment (IQA)

* [NIMA](https://github.com/idealo/image-quality-assessment)
* [TRIQ](https://github.com/junyongyou/triq)
* [Explainable](https://github.com/marcotcr/lime)
* Auto correction
  * [Exposure](https://github.com/mahmoudnafifi/Exposure_Correction)
  * White-balance
  * [Denoising](https://github.com/swz30/MPRNet)
* Detection
  * ColorCast

## Flicker Detection

[![Build Status](https://app.travis-ci.com/hc07180011/testing-cv.svg?branch=main)](https://app.travis-ci.com/hc07180011/testing-cv)

* Preprocessing
  * [Optical Flow](https://learnopencv.com/optical-flow-in-opencv/)
  * [Image Normalization](https://www.sciencedirect.com/topics/engineering/image-normalization)
  * [VGG19 feature extractor](https://arxiv.org/abs/1409.1556)
* Modeling
  * [End to End VGG BiDirectional LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
  * [End to End VGG 3DViT](https://arxiv.org/abs/2209.07026)

### Flicker Outlier Detection Results

## Install

### Ubuntu

#### Tested on

1. [Docker Ubuntu Official Image 18.04](https://hub.docker.com/_/ubuntu/)
2. [Google Cloud Platform - Ubuntu 16.04, 18.04, 20.04](https://cloud.google.com/)


#### Anaconda
[conda on Linux](https://docs.anaconda.com/anaconda/install/linux/)
```bash=
# create env from yml file
conda env create -f executable.yml
# activate env
conda activate "enviorment name here"
```

### Docker

* Prerequisite: [Docker](https://www.docker.com/)

```bash=
# Clone this repo
git clone https://github.com/hc07180011/testing-cv.git
cd testing-cv/flicker_detection/flicker_detection

# Build the container
docker build -t flicker_detection_runner .

# Run the container with a specific input data (put it under data/ directory)
docker run --rm -e data=data/test_data.mp4 -v $PWD/data:/app/data -it flicker_detection_runner
```

## Latest Updates

* [2022/11/9 Flicker Report](https://docs.google.com/presentation/d/10Tz_Jhj3amssrfxvayAZUWiRgEN8krsj/edit#slide=id.g14fdbdb000f_0_53)
* [2022-2023 Google CV Research Proposal](https://docs.google.com/document/d/1AgCTqS0zgIFc7saLjTUJ98ghYPj6o6Us-aqnJYDb0qI/edit?usp=sharing)

