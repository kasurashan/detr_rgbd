# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from net_utils_ours import Fusionmodel, Addmodel   # fusion module

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    #def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
    def __init__(self, backbone: nn.Module, backbone_d: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):  #########
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        #if return_interm_layers:    # if args.masks=True  (segementation)
        if True :  # 그냥 detection도 fused 사용하게 실험
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

            self.body_d = IntermediateLayerGetter(backbone_d, return_layers=return_layers)           
            
            self.FusionBlock_0 = Addmodel(in_channels=256)
            self.FusionBlock_1 = Fusionmodel(in_channels=512)
            self.FusionBlock_2 = Fusionmodel(in_channels=1024)
            self.FusionBlock_3 = Fusionmodel(in_channels=2048)
            
        else:
            return_layers = {'layer4': "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        #self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        
        # for concat (ch4 -> ch3)
        #self.conv_first = nn.Conv2d(4, 3, kernel_size=1).to('cuda')

    def forward(self, tensor_list: NestedTensor): 
        #conv_test = nn.Conv2d(4, 3, kernel_size=1).to('cuda') #######################
        #tensor_list.tensors = conv_test(tensor_list.tensors)
        #print(self.body)
        #tensor_list.tensors = self.conv_first(tensor_list.tensors)  # [batch, channel=4 , h, w]   (depth채널까지 합쳐짐)
        
        xs = OrderedDict()   #output
        xs_original = OrderedDict()   # only rgb output
        
        x_rgb = tensor_list.tensors[:,0:3,:,:]
        x_d = tensor_list.tensors[:,0:1,:,:]
        x_d = torch.cat((x_d,x_d,x_d), dim=1)
        
        #xs = self.body1(tensor_list.tensors)  
        #xs = self.body(tensor_list.tensors[:,0:3,:,:])   
        
        #print(self.body)
        #print(x_rgb.shape)   # [2,3,h,w]
        x_rgb = self.body['conv1'](x_rgb)
        x_rgb = self.body['bn1'](x_rgb)
        x_rgb = self.body['relu'](x_rgb)
        x_rgb = self.body['maxpool'](x_rgb)
        x_rgb = self.body['layer1'](x_rgb)   # [2,256,h/4,w/4]
        #xs['0'] = x_rgb
        xs_original['0'] = x_rgb
        
        x_d = self.body_d['conv1'](x_d)
        x_d = self.body_d['bn1'](x_d)
        x_d = self.body_d['relu'](x_d)
        x_d = self.body_d['maxpool'](x_d)
        x_d = self.body_d['layer1'](x_d)   # [2,256,h/4,w/4]
        
        x_rgb, x_d, x_fused = self.FusionBlock_0(x_rgb, x_d)
        xs['0'] = x_fused
        #xs['0'] = x_rgb   #no fuse
        # print(x_rgb.shape, x_d.shape, x_fused.shape)  # same shapes
        
        
        # #print(self.body['layer2'])
        x_rgb = self.body['layer2'](x_rgb)   # [2, 512, h/8, w/8]
        xs_original['1'] = x_rgb
        x_d = self.body_d['layer2'](x_d)
        x_rgb, x_d, x_fused = self.FusionBlock_1(x_rgb, x_d)
        xs['1'] = x_fused
        
        x_rgb = self.body['layer3'](x_rgb)   # [2, 1024, h/16, w/16]
        xs_original['2'] = x_rgb
        x_d = self.body_d['layer3'](x_d)
        x_rgb, x_d, x_fused = self.FusionBlock_2(x_rgb, x_d)
        xs['2'] = x_fused
        
        
        x_rgb = self.body['layer4'](x_rgb)   # [2, 2048, h/32, w/32]
        xs_original['3'] = x_rgb
        x_d = self.body_d['layer4'](x_d)
        x_rgb, x_d, x_fused = self.FusionBlock_3(x_rgb, x_d)
        xs['3'] = x_fused

        
        #print(xs.items())
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            #print(name, x.shape)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        out1: Dict[str, NestedTensor] = {}
        for name, x in xs_original.items():
            #print(name, x.shape)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out1[name] = NestedTensor(x, mask)


        return out, out1


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        
        # backbone for depth
        backbone_d = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)   #######
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        #super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        super().__init__(backbone, backbone_d, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # xs = self[0](tensor_list)
        
        # out: List[NestedTensor] = []
        # #print(xs, out)
        # pos = []
        # for name, x in xs.items():
        #     out.append(x)
        #     # position encoding
        #     pos.append(self[1](x).to(x.tensors.dtype))

        xs = self[0](tensor_list)
        
        out1: List[NestedTensor] = []
        out2: List[NestedTensor] = []
        #print(xs, out)
        pos = []
        for name, x in xs[0].items():
            out1.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        for name, x in xs[1].items():
            out2.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))


        return out1, out2, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
