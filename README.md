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

* Embedding
  * [vgg16 feature extractor](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)
* Movement
  * [BRISK](http://margaritachli.com/papers/ICCV2011paper.pdf)
* Transformation
  * [Affine Transformation](https://en.wikipedia.org/wiki/Affine_transformation)
  * [Photogenic Data Augmentation](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
* Imbalance Dataset sampling
    * [Synthetic Minority Oversampling Technique](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
    * [NearMiss Undersamling](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.NearMiss.html)
* Detection
  * [Bidirectional LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

## Install

### Ubuntu

#### Tested on

1. [Docker Ubuntu Official Image 18.04](https://hub.docker.com/_/ubuntu/)
2. [Google Cloud Platform - Ubuntu 16.04, 18.04, 20.04](https://cloud.google.com/)

#### Pip3
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

# Install python==3.9.5
pyenv install 3.9.5

# Clone this repo
git clone https://github.com/hc07180011/testing-cv.git
cd testing-cv/flicker_detection/flicker_detection

# Download the facenet pre-trained model
wget https://hc07180011.synology.me/data/googlecv/facenet_model.lite.h5 -O preprocessing/embedding/models/facenet_model.lite.h5

# Activate environment and install dependencies
~/.pyenv/versions/3.9.5/bin/python -m venv .env
source .env/bin/activate
python3 -m pip install -r requirements.txt

# Run the program
time python3 main.py -d data/test_data.mp4 
```

#### Anaconda
[conda on Linux](https://docs.anaconda.com/anaconda/install/linux/)
```bash
# create env from yml file
conda env create -f environment.yml
# activate env
conda activate "enviorment name here"
```



### Docker

* Prerequisite: [Docker](https://www.docker.com/)

```bash
# Clone this repo
git clone https://github.com/hc07180011/testing-cv.git
cd testing-cv/flicker_detection/flicker_detection

# Build the container
docker build -t flicker_detection_runner .

# Run the container with a specific input data (put it under data/ directory)
docker run --rm -e data=data/test_data.mp4 -v $PWD/data:/app/data -it flicker_detection_runner
```

## Testing Scripts Usage
```bash
cd test-cv/flicker-detection/flicker-detection/data
mkdir augmented

"""
make sure you have a folder in data/ called flicker-detection
and source videos are stored there
augmented is the output augmented video destination
"""
python3 data_augment/multiprocess_augmentor.py\
--main-folder-path flicker-detection\
--output-folder-path augmented/\
--max-clips 12

# label augmented videos with the same labels as its video of origin
python3 re_label.py

cd test-cv/flicker-detection/flicker-detection/
mkdir .cache

# extract embeddings from vgg16
python3 extract_embeddings.py


# train & test torch model
python3 torch_training.py --train --test
```
#### Caveats
```python
# Note you may have to configure the src and dst file paths
label_path = "data/new_label.json"
mapping_path = "data/mapping_test.json"  
data_dir = "data/vgg16_emb/"
cache_path = ".cache/train_test"
model_path = "h5_models/model.pth"

# sampler arguments below may need changes depending on Operating Specs
ipca = pk.load(open("ipca.pk1", "rb")) if os.path.exists(
        "ipca.pk1") else IncrementalPCA(n_components=2)
nm = NearMiss(version=3, n_jobs=-1, sampling_strategy='majority')

sm = SMOTE(random_state=42, n_jobs=-1, k_neighbors=5)

# Increase mem_split if not enough CPU RAM
ds_train = Streamer(embedding_list_train, label_path,
                    mapping_path, data_dir, mem_split=20, chunk_size=chunk_size, batch_size=batch_size, sampler=None)  

"""
sampler=[('near_miss', nm), ('smote', sm)])
for doing undersampling of majority class 
then oversampling of minority classes
"""
ds_val = Streamer(embedding_list_val, label_path,
                  mapping_path, data_dir, mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)
ds_test = Streamer(embedding_list_test, label_path,
                   mapping_path, data_dir, mem_split=1, chunk_size=chunk_size, batch_size=batch_size, sampler=None)
```

## Latest Updates

* [2022/7/20 Binary Classification Report](https://docs.google.com/presentation/d/1hXtWVv1v_1Zslkf_Qs5KBnzgBhU3J21w/edit#slide=id.g12f726d91f2_1_0)
* [2022/7/27 Multiclass Classification Report](https://docs.google.com/presentation/d/1g7G1kGudxg15lvsAskZe_fuWtqFvfoNU/edit#slide=id.g13da42ab967_0_136)
* [2022/8/03 Variable Frame Rate Solutions Report](https://docs.google.com/presentation/d/1cGxSHK291eURF7IVG3JD8mdHo0DxfyyN/edit#slide=id.g140496d05b2_0_41)
* [2022-2023 Google CV Research Proposal](https://docs.google.com/document/d/1AgCTqS0zgIFc7saLjTUJ98ghYPj6o6Us-aqnJYDb0qI/edit?usp=sharing)
