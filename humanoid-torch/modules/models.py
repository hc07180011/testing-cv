import logging
import json
import torch
import numpy as np

from vit_pytorch import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import OrderedDict
from typing import Tuple


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        config_json:json,
    )->None:
        super().__init__()
        self.x_dim, self.y_dim = config_json["downscale_dim"]
        self.training_dim = config_json["training_dim"]
        self.predicting_dim = config_json["predicting_dim"]
        self.total_channels = self.training_dim + self.predicting_dim
        self.total_interacts = config_json["total_interacts"]
        self.weight_decay = config_json["weight_decay"]
        self.batch_size = config_json["batch_size"]
        self.keep_prob = 0.5
        
        self.in_channels = config_json['channels']
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)

    def build_cnn(
        self,
    )->torch.nn.Sequential:
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,out_channels=16,kernel_size=3,padding='same')
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding='same')
        self.conv3 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding='same')
        self.conv4 = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding='same')
        
        self.pool_heat = torch.nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,padding='same')        


class Humanoid(BaseModel):
    def __init__(
        self,
        config_json:json,
    )->None:
        super().__init__(config_json=config_json)
        
        self.frame_num = config_json["frame_num"]
        
        self.cnn = self.build_cnn()
        self.heatmap = self.build_heatmap()
        self.lstm = torch.nn.LSTM(
            input_size=880,
            hidden_size=880,
            num_layers=1,
            batch_first=True
        )
        
        self.pool5_up = torch.nn.ConvTranspose2d(
            in_channels=, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            groups=1, 
            bias=True, 
            dilation=1, 
            padding_mode='zeros', 
            device=None, 
            dtype=None
        )
        
        self.pool4_up = torch.nn.ConvTranspose2d(
            in_channels=, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            groups=1, 
            bias=True, 
            dilation=1, 
            padding_mode='zeros', 
            device=None, 
            dtype=None
        )
        
        self.pool3_up = torch.nn.ConvTranspose2d(
            in_channels=(64, 22, 40), 
            out_channels=, 
            kernel_size=, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            groups=1, 
            bias=True, 
            dilation=1, 
            padding_mode='zeros', 
            device=None, 
            dtype=None
        )

    def init_hidden(
        self,
        x: torch.Tensor
    ) -> torch.FloatTensor:
        h0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device=x.device
        ).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device=x.device
        ).requires_grad_()
        return h0, c0

    def forward(
        self,
        x:torch.Tensor
    )->torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.max_pool(out)
        
        out = self.relu(self.conv2(out))
        out = self.max_pool(out)
        
        out = self.relu(self.conv3(out))
        out = self.max_pool(out)
        
        pool_heat3 = self.relu(self.pool_heat(out.clone()))
        pool_heat3_in = pool_heat3.reshape((-1,self.frame_num,np.prod([*out.shape[-2:]])))
        pool_heat3 = torch.add(
            self.lstm(pool_heat3_in,self.init_hidden(pool_heat3_in)).reshape((-1,np.prod([*out.shape[-2:]]),1)),
            pool_heat3.reshape(self.batch_size,self.frame_num,*out.shape[-2:],1)[:,self.frame_num - 1,:,:,:]
            )
        
        out = self.relu(self.conv4(out))
        out = self.max_pool(out)
        pool_heat4 = self.relu(self.pool_heat(out.clone()))
        pool_heat4_in = pool_heat4.reshape((-1,self.frame_num,np.prod([*out.shape[-2:]])))
        pool_heat4 = torch.add(
            self.lstm(pool_heat4).reshape((-1,12,20,1)),
            pool_heat4.reshape(self.batch_size,self.frame_num,12,20,1)[:,self.frame_num - 1,:,:,:]
            )
        
        out = self.relu(self.conv4(out))
        out = self.max_pool(out)
        pool_heat5 = self.relu(self.pool_heat(out.clone())).reshape((-1,self.frame_num,np.prod([*out.shape[-2:]])))
        pool_heat5 = torch.add(
            self.lstm(pool_heat5).reshape((-1,6,10,1)),
            pool_heat5.reshape(self.batch_size,self.frame_num,6,10,1)[:,self.frame_num - 1,:,:,:]
        )
        
        return pool3_up,pool_head5
    
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
    
if __name__ == "__main__":
    pass