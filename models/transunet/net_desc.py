"""
TransUNet implementation:
https://github.com/Beckschen/TransUNet
"""
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import numpy as np

from collections import OrderedDict
from ..utils import crop_op

from .vit_seg_model import VisionTransformer as ViT_seg
from .vit_seg_model import CONFIGS as CONFIGS_ViT_seg

class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x

class NetDesc(Net):
    """Initialise DeepLab."""

    def __init__(self, encoder_backbone_name, pretrained=False, input_ch=3, nr_types=None, nr_layers=None, nr_vendors=None, img_size=256, patch_size=16, freeze=False):
        super().__init__()

        img_size=512
        self.nr_types = nr_types
        self.nr_layers = nr_layers
        self.nr_vendors = nr_vendors # added
        config_vit = CONFIGS_ViT_seg[encoder_backbone_name]
        config_vit.n_classes = nr_layers
        config_vit.n_skip = 3
        if encoder_backbone_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / patch_size), int(img_size / patch_size))
        self.model = ViT_seg(config_vit, img_size=img_size, num_classes=nr_layers, nr_domains=nr_vendors, freeze=freeze).cuda()
        if pretrained:
            self.model.load_from(weights=np.load(f'/home/neopath2/OED_Hanya_Adam/ViT_model_weights/imagenet21k/{encoder_backbone_name}.npz'))
    
    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        out, domain_output = self.model(imgs)
        # out = crop_op(out, [92, 92])
        out = crop_op(out, [184, 184])

        out_dict = OrderedDict()
        out_dict['ls'] = out
        if self.nr_vendors is not None:
            out_dict['vendor'] = domain_output

        return out_dict

def create_model(**kwargs):
    return NetDesc(**kwargs)