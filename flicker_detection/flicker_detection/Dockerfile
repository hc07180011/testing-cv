FROM python:3.8

WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN wget https://hc07180011.synology.me/data/googlecv/facenet_model.h5 -O preprocessing/embedding/models/facenet_model.h5
RUN python3 -m pip install -r requirements.txt 

ARG data
ENV data=$data

CMD ["sh", "-c", "python3 main.py -d $data"]
