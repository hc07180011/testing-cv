import os
import tqdm
import logging
import cv2
import torch
import torchvision
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from mypyfunc.torch_models import CNN_Transformers
from mypyfunc.streamer import MultiStreamer, VideoDataSet
from mypyfunc.logger import init_logger


def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--eval_dir', type=str, default="data/show-data",
                        help='directory of flicker videos')
    parser.add_argument('--model_dir', type=str, default="cnn_transformers_model",
                    help='directory of saved model paramters')
    return parser.parse_args()

def main()->None:
    init_logger()
    args = command_arg()
    eval_dir,model_dir = args.eval_dir,args.model_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Transformers(
        image_size=360,          # image size
        frames=10,               # number of frames
        image_patch_size=36,     # image patch size
        frame_patch_size=10,      # frame patch size
        num_classes=2,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        cnn=torchvision.models.vgg19(pretrained=True),
        dropout=0.1,
        emb_dropout=0.1,
        pool='cls' 
    )  # 16784 of 19456 gpu mb 0.6094
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(
        model_dir, 'model.pth'))['model_state_dict'])
    model.eval()
    
    objective = torch.nn.Softmax()
    test_files = [os.path.join(eval_dir,f) for f in os.listdir(eval_dir)]
    labels = {
        f.split("/",3)[-1].replace(".mp4",""):i for i,f in enumerate(test_files)
    }
    test_ds = VideoDataSet.split_datasets(
        test_files, labels=labels, class_size=1, max_workers=1, undersample=0)
    stream = MultiStreamer(
        test_ds,
        batch_size=0,
        binary=False
    )
    logging.info("streaming...")
    for inputs,filename in tqdm.tqdm(stream):
        inputs = inputs.permute(
                0, 1, 4, 2, 3).float().to(device)
        output = model(inputs)
        pred = torch.topk(objective(output),
                                      k=1, dim=1).indices.flatten()
        if pred:
            logging.debug(f"INDEX - {filename.item()}")
            logging.debug(list(labels.keys())[list(labels.values()).index(filename.item())])
    logging.info("done...")

def test_fps()->None:
    eval_dir = "data/raw-data"#"data/show-data"
    test_files = [os.path.join(eval_dir,f) for f in os.listdir(eval_dir)]
    for f in test_files[:2]:
        cap = cv2.VideoCapture(f)
        fps = cap.get(cv2.CAP_PROP_FPS)

        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]

        while(cap.isOpened()):
            frame_exists, _ = cap.read()
            if frame_exists:
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(ms/1000)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
            else:
                break

        cap.release()
        # for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
        #     print('Frame %d difference:'%i, abs(ts - cts))        

if __name__ == "__main__":
    # main()
    test_fps()