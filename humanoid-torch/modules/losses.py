import torch
import numpy as np

class HeatMapInteractLoss(torch.nn.Module):
    '''
    https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
    '''
    def __init__(
        self,
    )->None:
        super(HeatMapInteractLoss,self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()
        self.fc = torch.nn.Linear(7,7)
        self.relu = torch.nn.ReLU()
    
    def forward(
        self,
        x:torch.Tensor,
        heatmap_true:torch.Tensor,
        interact_true:torch.Tensor,
    )->torch.Tensor:
        print(x.reshape(-1,180*320).shape)
        print(heatmap_true.reshape(-1,180*320).shape)
        hm_loss = self.ce(
            x.reshape(-1,180*320),
            heatmap_true.reshape(-1,180*320)
            )
        # predict_hm = self.softmax(heatmap.reshape(-1,np.prod(heatmap.shape[1:]))).reshape(-1,*heatmap.shape[1:],1)
        interact_flat = interact_true.reshape(-1,np.prod(interact_true.shape[1:]))
        interact = self.relu(self.fc(interact_flat))
        interact_loss = self.ce(interact_true,interact)
        return  hm_loss + interact_loss

    
    
if __name__ == '__main__':
    pass