import torch
from vit_pytorch.vit_3d import ViT
from video_swin_transformer import SwinTransformer3D
from collections import OrderedDict


def load_video_swin() -> torch.nn.Module:
    model = SwinTransformer3D(embed_dim=128,
                              depths=[2, 2, 18, 2],
                              num_heads=[4, 8, 16, 32],
                              patch_size=(2, 4, 4),
                              window_size=(16, 7, 7),
                              drop_path_rate=0.4,
                              patch_norm=True)

    # https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py
    checkpoint = torch.load(
        './checkpoints/swin_base_patch244_window1677_sthv2.pth')

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'backbone' in k:
            name = k[9:]
            new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model
