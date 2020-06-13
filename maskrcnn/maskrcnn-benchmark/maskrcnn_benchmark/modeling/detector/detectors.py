# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .baseline_v1 import Baseline as Baseline1

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "Baseline1": Baseline1}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
