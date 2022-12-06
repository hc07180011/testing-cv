import os
import re
import tqdm
import logging
import torch
import torchvision
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from mypyfunc.torch_models import CNN_Transformers
from mypyfunc.streamer import MultiStreamer, VideoDataSet
from mypyfunc.logger import init_logger


def run(
    model:torch.nn.Module,
    stream:MultiStreamer,
    device:torch.device,
    objective:torch.nn.Module,
    labels:dict,
    log_dir:str,
)->None:
    logs = {
        'issue':[],
        'occur_frame':[],
        'occur_sec':[],
        'log_message':[]
    }
    logging.info("streaming...")
    for inputs,filename in tqdm.tqdm(stream):
        inputs = inputs.permute(
                0, 1, 4, 2, 3).float().to(device)
        output = model(inputs)
        pred = torch.topk(objective(output),
                                      k=1, dim=1).indices.flatten()
        if pred:
            message = pd.NA
            file = list(labels.keys())[list(labels.values()).index(filename.item())]
            info = file.split("_",4)
            logging.debug(info)
            
            if os.path.exists(os.path.join(log_dir,info[-1]+".txt")):
                log = read_log(os.path.join(log_dir,info[-1]+".txt"))
                logging.debug(log[int(info[1]),:])
                message = " ".join(log.iloc[int(info[2]),:].tolist())
                
            logs['issue'].append(info[-1])
            logs['occur_frame'].append(int(info[0]))
            logs['occur_sec'].append(int(info[2]))
            logs['log_message'].append(message)
    logging.info("done...")
    pd.DataFrame(logs).to_csv("bug_report.csv")
    

def read_log(filename:str)->pd.DataFrame:
    with open(filename, 'r') as f:
        lines = []
        for line in f.readlines():
            if "PROCESS" in line:
                continue
            line = re.sub(' +',' ',line)
            line = list(filter(None,line.split(" ",6)))
            if not bool(re.match('\d{4}-\d{2}-\d{2}',line[0])):
                line = [pd.NA]*6 + [" ".join(line)]
            lines.append(line)
    log = pd.DataFrame(lines,columns=['Date','Time','Process','Function','App','Class','Message'])
    log.fillna(method='ffill',inplace=True)
    log['Time'] = pd.to_datetime(log['Time'],format='%H:%M:%S.%f',errors='ignore')
    log['Total_seconds'] = log['Time'].iloc[-1] - log['Time'].iloc[0]
    return log

def command_arg() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--eval_dir', type=str, default="data/new-meta-data",
                        help='directory of flicker videos')
    parser.add_argument('--log_dir', type=str, default="data/logs",
                        help='directory of logcat logs')
    parser.add_argument('--model_dir', type=str, default="cnn_transformers_model",
                    help='directory of saved model paramters')
    return parser.parse_args()

def main()->None:
    init_logger()
    args = command_arg()
    eval_dir,log_dir,model_dir = args.eval_dir,args.log_dir,args.model_dir
    test_files = [os.path.join(eval_dir,f) for f in os.listdir(eval_dir)]
    labels = {
        f.split("/",3)[-1].replace(".mp4",""):i 
        for i,f in enumerate(test_files)
    }
    # device =torch.device('cpu')
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
    model.load_state_dict(torch.load(os.path.join(
        model_dir, 'model.pth'),map_location=device)['model_state_dict'])
    # model = model.module.to(device)
    model.to(device)
    model.eval()
    objective = torch.nn.Softmax()
    
    test_ds = VideoDataSet.split_datasets(
        test_files, labels=labels, class_size=1, max_workers=1, undersample=0)
    stream = MultiStreamer(
        test_ds,
        batch_size=0,
        binary=False
    )
    run(
        model=model,
        stream=stream,
        device=device,
        objective=objective,
        labels=labels,
        log_dir=log_dir,
    )
    
if __name__ == "__main__":
    """
    https://discuss.pytorch.org/t/create-exe-file/56626/4
    """
    main()
    # test_load_cpu()
    # log = read_log("data/logs/device-2022-12-03-234516.txt")
    # log.to_csv("test_log.csv")