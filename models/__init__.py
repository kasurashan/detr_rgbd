# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_Dfuse_Sfuse import build_Dfuse_Sfuse
from .detr_Dfuse_Srgb import build_Dfuse_Srgb
from .detr_Drgb_Sfuse import build_Drgb_Sfuse
from .detr_Drgb_Srgb import build_Drgb_Srgb
from .detr_concat import build_concat

def build_model(args):
    return build(args)

def build_model_Dfuse_Sfuse(args):
    return build_Dfuse_Sfuse(args)

def build_model_Dfuse_Srgb(args):
    return build_Dfuse_Srgb(args)

def build_model_Drgb_Sfuse(args):
    return build_Drgb_Sfuse(args)

def build_model_Drgb_Srgb(args):
    return build_Drgb_Srgb(args)

def build_model_concat(args):
    return build_concat(args)