from mmdet3d.models.builder import MODELS
from mmcv.runner import BaseModule, auto_fp16
import torch.distributed as dist
from collections import OrderedDict
from mmdet.models.backbones.resnet import ResNet

@MODELS.register_module()
class FiveOutputResnet(ResNet):
    def forward(self, x):
        """Forward function."""
        # print('self.deep_stem: ', self.deep_stem)
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            # print('1st: ', x.size())
            x = self.norm1(x)
            # print('2st: ', x.size())
            x = self.relu(x)
            # print('3st: ', x.size())
        outs = []
        outs.append(x)
        x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)