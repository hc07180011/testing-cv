import os
import logging
import torch
import torchvision
import skvideo.io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torchviz import make_dot
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
        self.n_directions = 2 if bidirectional else 1

        # Base cnn features layer
        self.extractor = cnn.features
        self.avgpool = cnn.avgpool
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                            batch_first=True, bidirectional=bidirectional)
        # Linear Dense
        self.fc1 = nn.Linear(hidden_dim*self.n_directions, hidden_dim//2)
        # Linear Dense
        self.fc2 = nn.Linear(hidden_dim//2, self.output_dim)
        # initialize weights & bias with stdv -> 0.05
        self.initialization()

    def init_hidden(self, x: torch.Tensor) -> torch.FloatTensor:
        h0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"
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
        batch_size, chunk_size = x.shape[:2]
        # Get features (4,10,360,360,3)
        out = self.extractor(x.flatten(end_dim=1))#.flatten(start_dim=1)
        out = self.avgpool(out).flatten(start_dim=1)
        # Shape back to batch x chunk
        out = out.reshape((batch_size, chunk_size, out.shape[-1]))
        # One time step
        out, self.hidden_state = self.lstm(out, self.init_hidden(x))
        # Dense lstm
        out = self.fc1(out)
        # Dense for softmax
        out = self.fc2(out)
        return out[:, -1]

    def initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(param.data, std=0.05)
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
        self.__initialization()

    def forward(self, x):
        batch_size, chunk_size = x.shape[:2]
        x = self.extractor(x.flatten(end_dim=1))#.flatten(start_dim=1)
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

    def __initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                for name, param in m.named_parameters():
                    nn.init.normal_(param, std=0.05)


class L1(torch.nn.Module):
    """
    https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
    """

    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(
            self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


class OHEMLoss(nn.Module):
    def __init__(
        self,
        batch_size:int,
        init_epoch:int,
        criterion:nn.Module,
    ) -> None:
        super(OHEMLoss, self).__init__()
        self.__batch_size = batch_size
        self.init_epoch = init_epoch
        self.criterion = criterion


    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch:int,
    ) -> torch.Tensor:
        if epoch < self.init_epoch:
            return self.criterion(pred,target)
            
        ohem_loss = F.cross_entropy(
            pred, target, reduction='none', ignore_index=-1)
        sorted_ohem_loss, idx = torch.sort(ohem_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], self.__batch_size)

        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_loss = ohem_loss[keep_idx_cuda]

        return ohem_loss.sum() / keep_num

def visualize_model(
    model:nn.Module,
    batch:torch.Tensor,
    file_name:str='../cnn_transformers'
)->None:
    yhat = model(batch)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(file_name, format="png")
    

def test_OHEM() -> None:
    C = 6
    cls_pred = torch.randn(8, C)
    cls_target = torch.Tensor([1, 1, 2, 3, 5, 3, 2, 1]).type(torch.long)
    criterion = OHEMLoss()
    print(criterion(cls_pred, cls_target))


if __name__ == '__main__':
    """
    torch.Size([4, 1000, 1024])
    https://discuss.pytorch.org/t/finding-model-size/130275
    """
    # test_OHEM()
    flicker_path = '../data/flicker1/'
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
    
    # input_dim = 256  # (4,10,61952)
    # output_dim = 2
    # hidden_dim = 64
    # layer_dim = 2
    # bidirectional = True
    # model = CNN_LSTM(  # model size: 81.642MB
    #     cnn=torchvision.models.vgg16(pretrained=True),
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     hidden_dim=hidden_dim,
    #     layer_dim=layer_dim,
    #     bidirectional=bidirectional,
    # )
    videos = os.listdir(flicker_path)
    test_batch = np.zeros((4, 10, 360, 360, 3))
    for i, video in enumerate(videos[:4]):
        test_batch[i] = skvideo.io.vread(os.path.join(flicker_path, video))
    test_batch = torch.from_numpy(test_batch).permute(0, 1, 4, 2, 3).float()
    visualize_model(model,test_batch)
  


