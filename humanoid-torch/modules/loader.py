import os
import tqdm
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from typing import Callable,Tuple


class RicoInteract(torch.utils.data.Dataset):
    def __init__(
        self,
        root:str,
        file_generic:str,
        ext:str,
        transform:Callable=None,
        target_transform:Callable=None,
    )->None:
        self.__root = root
        self.__file_generic = file_generic
        self.__ext = ext
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(
        self,
        idx:int,
    )->torch.Tensor:
        session = np.load(os.path.join(self.__root,f"{self.__file_generic}_{idx}.{self.__ext}"))
        
        img,hm,interact = session['img'],session['hm'],session['interact']
        if self.transform is not None:
            img, hm, interact = self.transform(img=session['img'],hm=session['hm'],interact=session['interact'])

        return img,hm,interact
    def __len__(
        self,
    )->int:
        return len(os.listdir(self.__root))
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def prep(
    img:np.ndarray,
    hm:np.ndarray,
    interact:np.ndarray,
)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    return torch.from_numpy(img).permute(0,3,1,2),\
        torch.from_numpy(hm),\
            torch.from_numpy(interact)

def test_Rico_interact()->None:
    import logging
    from logger import init_logger
    init_logger()
    
    file_generic = 'im_hm_interact'
    ext = 'npz'
    root = '/data/humanoid_train_data'
    dataset = RicoInteract(root=root,file_generic=file_generic,ext=ext,transform=prep)
    dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=1,prefetch_factor=2, pin_memory=True,shuffle=False)
    for idx,(inputs,mask,targets) in enumerate(tqdm.tqdm(dataloader)):
        logging.debug(f"{idx}: {inputs.shape} - {mask.shape} - {targets.shape}")
        break
        
        
if __name__ == '__main__':
    test_Rico_interact()
    