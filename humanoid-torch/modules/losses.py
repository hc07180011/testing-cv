import torch
import torch.nn.functional as F
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
        self.fc = torch.nn.Linear(60,7)
        self.relu = torch.nn.ReLU()
    
    def forward(
        self,
        heatmap:torch.Tensor,
        heatmap_true:torch.Tensor,
        interact:torch.Tensor,
        interact_true:torch.Tensor,
    )->torch.Tensor:
        hm_loss = self.ce(
            heatmap.flatten(start_dim=1),
            heatmap_true.flatten(start_dim=1).repeat(heatmap.shape[0],1)
            )
        # predict_hm = self.softmax(heatmap.reshape(-1,np.prod(heatmap.shape[1:]))).reshape(-1,*heatmap.shape[1:],1)
        interact_flat = interact.flatten(start_dim=1)
        interact = self.relu(self.fc(interact_flat))
        interact_loss = self.ce(interact,interact_true)
        return  hm_loss + interact_loss


    
if __name__ == '__main__':
    '''
    2023-03-23 12:06:11,157 heatmap loss: nan
    2023-03-23 12:06:11,320 interact loss: 6.89274e+33
    2023-03-23 12:06:11,621 total loss: nan
    2023-03-23 12:06:12,008 Interacts: (7,)
    2023-03-23 12:06:12,008 Heatmap OUT: (32, 180, 320, 1)
    2023-03-23 12:06:12,008 Heatmap TRUE: (?, 180, 320, 1)
    2023-03-23 12:06:12,008 Interact OUT: (32, 6, 10, 1)
    2023-03-23 12:06:12,008 Interact TRUE: (?, 7)
    '''
    im_outputs = torch.randn(size=(4,1,180,320))
    hm_target = torch.randn(size=(1,1,180,320))
    int_out = torch.randn(size=(1,6,10,1),requires_grad=True)
    int_target = torch.randint(0,7,size=(1,7,)).float()
    criterion = HeatMapInteractLoss()
    print(criterion(im_outputs,hm_target,int_out,int_target))
    pass