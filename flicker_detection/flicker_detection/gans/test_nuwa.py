import os
import itertools
import logging
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuwa_pytorch import VQGanVAE, NUWA


def vid_to_tensor(path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()

    frames = torch.Tensor().to(device=device)
    count, cur = 10, 0
    while success and cur < count:
        success, image = vidcap.read()
        frame = torch.Tensor(
            np.array(image).reshape(1, 3, 1080, 2340)).to(device=device)
        frames = torch.cat((frames, frame), 0)

        logging.info("frames shape: {}".format(
            frames.shape))
        cur += 1
    # frames.to(device=device)
    return frames


def load_device():
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def init_vae(device):
    return VQGanVAE(
        dim=512,
        # default is 3, but can be changed to any value for the training of the segmentation masks (sketches)
        channels=3,
        image_size=(1080, 2340),           # image size
        num_layers=3,             # number of downsampling layers
        num_conv_blocks=2,        # number of convnext blocks
        vq_codebook_size=8192,    # codebook size
        vq_decay=0.8              # codebook exponential decay
    ).to(device=device)


def init_nuwa(vae, device):
    return NUWA(
        vae=vae,
        dim=512,
        text_num_tokens=20000,                # number of text tokens
        text_enc_depth=12,                    # text encoder depth
        text_enc_heads=8,                     # number of attention heads for encoder
        # max sequence length of text conditioning tokens (keep at 256 as in paper, or shorter, if your text is not that long)
        text_max_seq_len=256,
        max_video_frames=10,                  # number of video frames
        image_size=256,                       # size of each frame of video
        dec_depth=64,                         # video decoder depth
        dec_heads=8,                          # number of attention heads in decoder
        # reversible networks - from reformer, decoupling memory usage from depth
        dec_reversible=True,
        enc_reversible=True,                  # reversible encoders, if you need it
        attn_dropout=0.05,                    # dropout for attention
        ff_dropout=0.05,                      # dropout for feedforward
        # kernel size of the sparse 3dna attention. can be a single value for frame, height, width, or different values (to simulate axial attention, etc)
        sparse_3dna_kernel_size=(5, 3, 3),
        # cycle dilation of 3d conv attention in decoder, for more range
        sparse_3dna_dilation=(1, 2, 4),
        # cheap relative positions for sparse 3dna transformer, by shifting along spatial dimensions by one
        shift_video_tokens=True
    ).to(device=device)


if __name__ == "__main__":
    """
    https://github.com/lucidrains/nuwa-pytorch
    """
    from sys import path
    from pathlib import Path
    from os.path import dirname as dir
    googlecv = Path().parent.absolute().parent
    path = path.append(dir(str(googlecv)+"/"))
    __package__ = "mypyfunc"
    from mypyfunc.logger import init_logger

    init_logger()
    device = load_device()
    vae = init_vae(device)

    videos_root = os.path.join(os.getcwd(), '../data/flicker-detection')

    logging.info("Extracting VAE encoder")
    recon_images = []
    for vid in os.listdir(videos_root):
        path = os.path.join(
            videos_root, videos_root+'/'+vid)
        frames = vid_to_tensor(path)
        print((type(frames)))
        loss = vae(frames, return_loss=True)
        torch.cuda.empty_cache()
        loss.backward()

        discr_loss = vae(frames,
                         return_discr_loss=True)
        discr_loss.backward()

        # generated images
        recon_images.extend(vae(frames))

        break

    torch.cuda.synchronize()
    logging.info("Done extracting VAE encoder")

    logging.info("initializing NUWA")
    nuwa = init_nuwa(vae, device)

    loss = nuwa(video=recon_images, return_loss=True)
    loss.backward()

    video = nuwa.generate(video=recon_images)
