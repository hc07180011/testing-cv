import logging
import json
import torch
import numpy as np
import torch.nn.functional as F



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
    )->None:
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,out_channels=16,kernel_size=3,padding='same')
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding='same')
        self.conv3 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding='same')
        self.conv4 = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding='same')
        
        self.pool_heat = torch.nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,padding='same')   
    
    def build_lstm(
        self,
    )->None:
        self.lstm1 = torch.nn.LSTM(
            input_size=880,
            hidden_size=880,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm2 = torch.nn.LSTM(
            input_size=220,
            hidden_size=220,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm3 = torch.nn.LSTM(
            input_size=50,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )
    
    def init_hidden(
        self,
        x: torch.Tensor
    ) -> torch.FloatTensor:
        h0 = torch.zeros(
            1,
            x.size(0),
            x.shape[-1],
            device=x.device
        ).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(
            1,
            x.size(0),
            x.shape[-1],
            device=x.device
        ).requires_grad_()
        return h0, c0


class Humanoid(BaseModel):
    def __init__(
        self,
        config_json:json,
    )->None:
        super().__init__(config_json=config_json)
        
        self.frame_num = config_json["frame_num"]
        
        self.build_cnn()
        self.build_lstm()
        
        self.pool5_up_filters = torch.nn.Parameter(torch.randn((5, 11, 1, 1)))
        self.pool4_up_filters = torch.nn.Parameter(torch.randn((11, 22, 1, 1)))
        self.pool3_up_filters = torch.nn.Parameter(torch.randn((22, 180, 1, 1)))

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
        res3, _ = self.lstm1(pool_heat3_in,self.init_hidden(pool_heat3_in))
        pool_heat3 = torch.add(
            res3.reshape((-1,*out.shape[-2:],1)),
            pool_heat3.reshape(1,self.frame_num,*out.shape[-2:],1)[:,self.frame_num - 1,:,:,:]
            ) # TO DO adjust batch size
        
        out = self.relu(self.conv4(out))
        out = self.max_pool(out)
        
        pool_heat4 = self.relu(self.pool_heat(out.clone()))
        pool_heat4_in = pool_heat4.reshape((-1,self.frame_num,np.prod([*out.shape[-2:]])))
        res4,_ = self.lstm2(pool_heat4_in,self.init_hidden(pool_heat4_in))
        pool_heat4 = torch.add(
            res4.reshape((-1,*out.shape[-2:],1)),
            pool_heat4.reshape(1,self.frame_num,*out.shape[-2:],1)[:,self.frame_num - 1,:,:,:]
        )
        
        out = self.relu(self.conv4(out))
        out = self.max_pool(out)
        pool_heat5 = self.relu(self.pool_heat(out.clone()))
        pool_heat5_in = pool_heat5.reshape((-1,self.frame_num,np.prod([*out.shape[-2:]])))
        
        res5,_ = self.lstm3(pool_heat5_in,self.init_hidden(pool_heat5_in))
        pool_heat5 = torch.add(
            res5.reshape((-1,*out.shape[-2:],1)),
            pool_heat5.reshape(1,self.frame_num,*out.shape[-2:],1)[:,self.frame_num - 1,:,:,:]
        )
        
        pool5_up = self.relu(F.conv_transpose2d(
            input=pool_heat5,
            weight=self.pool5_up_filters, 
            output_padding=(1,0), 
            stride=(2,1)
        ))
        pool4_heat_sum = torch.add(pool_heat4,pool5_up)
        pool4_up = self.relu(F.conv_transpose2d(
            input=pool4_heat_sum,
            weight=self.pool4_up_filters, 
            output_padding=(1,0), 
            stride=(2,2)
        ))

        pool3_heat_sum = torch.add(pool_heat3,pool4_up)
        pool3_up = self.relu(F.conv_transpose2d(
            input=pool3_heat_sum,
            weight=self.pool3_up_filters, 
            output_padding=(7,0), 
            stride=(8,1)
        ))
        
        return pool3_up,pool_heat5
    
    def initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(param.data, std=0.05)
            elif isinstance(m, torch.nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'weight_hh' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param.data)


def test_humanoid()->None:
    import json
    import tqdm
    from logger import init_logger
    from losses import HeatMapInteractLoss
    from loader import RicoInteract,prep
    from torch.utils.data import DataLoader

    init_logger()
    config_path = '../config.json'
    config = json.load(open(config_path,'r'))
    model = Humanoid(config_json=config)
    criterion = HeatMapInteractLoss()
    file_generic = 'im_hm_interact'
    ext = 'npz'
    root = '/data/humanoid_train_data'
    dataset = RicoInteract(root=root,file_generic=file_generic,ext=ext,transform=prep)
    dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=1,prefetch_factor=2, pin_memory=True,shuffle=False)
    
    for idx,(inputs,hm,interact) in enumerate(tqdm.tqdm(dataloader)):
        inputs,hm,interact = inputs.squeeze(),hm.squeeze(),interact.squeeze()
        logging.debug(f"{idx}: {inputs.shape} - {hm.shape} - {interact.shape}")
        heatmap_out,interact_out = model(inputs)
        logging.debug(f"{heatmap_out.shape} - {interact_out.shape}")
        loss = criterion(
            x=heatmap_out,
            heatmap_true=hm,
            interact_true=interact
        )
        logging.debug(f'LOSS - {loss}')
        break
    

if __name__ == "__main__":
    test_humanoid()