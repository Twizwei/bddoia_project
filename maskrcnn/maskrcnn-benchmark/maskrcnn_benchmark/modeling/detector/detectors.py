# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .baseline_v1 import Baseline as Baseline1
from .baseline_v2 import Baseline as Baseline2
from .baseline_v3 import Baseline as Baseline3

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "Baseline1": Baseline1, 'Baseline2': Baseline2, 'Baseline3':Baseline3}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
