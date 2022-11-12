import os
import logging
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch import nn
import skvideo.io
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from collections import OrderedDict
from typing import Callable
import warnings
warnings.filterwarnings('ignore')


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        cnn: nn.Module,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layer_dim: int,
        bidirectional=False,
    ) -> None:
        super(CNN_LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Output dim classes
        self.output_dim = output_dim
        # NO of hidden layers for lstm
        self.n_directions = 2 if bidirectional else 1
        # Base cnn
        self.extractor = cnn.features
        self.avgpool = cnn.avgpool
        self.fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features=25088, out_features=input_dim)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5, inplace=False)),
        ]))
        # LSTM1  Layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                            batch_first=True, bidirectional=bidirectional)
        # Linear Dense
        self.fc1 = nn.Linear(hidden_dim*self.n_directions, self.output_dim)
        # Linear Dense
        # initialize weights & bias with stdv -> 0.05
        self._initialization()

    def init_hidden(self, x: torch.Tensor) -> torch.FloatTensor:
        h0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"  # cuda
        ).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"
        ).requires_grad_()
        return h0, c0

    def forward(self, x) -> torch.Tensor:
        # Get features (4,10,3,360,360) -> (40,3,360,360)
        batch_size, chunk_size = x.shape[:2]
        out = self.extractor(x.flatten(end_dim=1))
        out = self.avgpool(out).flatten(start_dim=1)
        out = self.fc(out)
        out = out.reshape((batch_size, chunk_size, out.shape[-1]))
        # Shape back to batch x chunk
        out = out.reshape((batch_size, chunk_size, out.shape[-1]))
        # One time step
        out, self.hidden_state = self.lstm(out, self.init_hidden(x))
        # Dense lstm
        out = self.fc1(out)
        return out[:, -1]

    def _initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                for name, param in m.named_parameters():
                    nn.init.normal_(param, std=0.05)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'weight_hh' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)


"""
https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
https://towardsdatascience.com/recreating-keras-functional-api-with-pytorch-cc2974f7143c
"""

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CNN_Transformers(nn.Module):
    def __init__(
        self, *,
        image_size: int,
        image_patch_size: int,
        frames: int,
        frame_patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        cnn: nn.Module,
        pool: str = 'cls',
        dim_head: int = 64,
        dropout: int = 0.,
        emb_dropout: int = 0.
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width) * (frames // frame_patch_size)

        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.extractor = cnn.features
        self.avgpool = cnn.avgpool
        self.fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features=25088, out_features=dim)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5, inplace=False)),
        ]))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self._initialization()

    def forward(self, video):
        batch_size, chunk_size = video.shape[:2]
        x = self.extractor(video.flatten(end_dim=1))
        x = self.avgpool(x).flatten(start_dim=1)
        x = self.fc(x)
        x = x.reshape((batch_size, chunk_size, x.shape[-1]))
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def _initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                for name, param in m.named_parameters():
                    nn.init.normal_(param, std=0.05)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'weight_hh' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)


if __name__ == '__main__':
    """
    torch.Size([4, 1000, 1024])
    """
    flicker_path = '../data/flicker1/'
    # model = CNN_Transformers(
    #     image_size=360,          # image size
    #     frames=10,               # number of frames
    #     image_patch_size=36,     # image patch size
    #     frame_patch_size=1,      # frame patch size
    #     num_classes=2,
    #     dim=256,
    #     depth=6,
    #     heads=8,
    #     mlp_dim=2048,
    #     cnn=torchvision.models.vgg16(pretrained=True),
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    input_dim = 256  # (4,10,61952)
    output_dim = 2
    hidden_dim = 64
    layer_dim = 2
    bidirectional = True
    model = CNN_LSTM(
        cnn=torchvision.models.vgg16(pretrained=True),
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        bidirectional=bidirectional,
    )
    videos = os.listdir(flicker_path)
    test_batch = np.zeros((4, 10, 360, 360, 3))
    for i, video in enumerate(videos[:4]):
        test_batch[i] = skvideo.io.vread(os.path.join(flicker_path, video))
    test_batch = torch.from_numpy(test_batch).permute(0, 1, 4, 2, 3).float()
    out = model(test_batch)
    print(out.shape)
