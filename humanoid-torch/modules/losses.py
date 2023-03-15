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
        x:torch.Tensor,
        heatmap_true:torch.Tensor,
        interact:torch.Tensor,
        interact_true:torch.Tensor,
    )->torch.Tensor:
        hm_loss = self.ce(
            x.flatten(start_dim=1),
            heatmap_true.flatten(start_dim=1).repeat(x.shape[0],1)
            )
        # predict_hm = self.softmax(heatmap.reshape(-1,np.prod(heatmap.shape[1:]))).reshape(-1,*heatmap.shape[1:],1)
        print(np.prod(interact_true.shape[1:]))
        interact_flat = interact_true.flatten(start_dim=1)
        interact = self.relu(self.fc(interact_flat))
        interact_loss = self.ce(interact_true,interact)
        return  hm_loss + interact_loss






    
if __name__ == '__main__':
    im_outputs = torch.randn(size=(4,1,180,320))
    hm_target = torch.randn(size=(1,1,180,320))
    int_out = torch.randn(0,7,size=(1,60))
    int_target = torch.randint(0,7,size=(1,7))
    criterion = HeatMapInteractLoss()
    print(criterion(im_outputs,hm_target,int_out))
    pass