FROM python:3.9

WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN wget https://hc07180011.synology.me/data/googlecv/facenet_model.lite.h5 -O preprocessing/embedding/models/facenet_model.lite.h5
RUN python3 -m pip install -r requirements.txt 

ARG data
ENV data=$data

ARG command
ENV command=$command

CMD ["sh", "-c", "python3 main.py -d $data $command"]

