# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import logging
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

# from maskrcnn_benchmark.modeling.poolers import Pooler

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        # resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        # sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )
        #
        # self.pooler = pooler

    def forward(self, images, targets=None, get_feature=False): #TODO: Remember to set get_feature as False if there are any changes later
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # print(type(features))
        logger = logging.getLogger("tmp")
        logger.info(len(features))
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            # if get_feature:
            #     x, result, _ = self.roi_heads(features, targets, targets)
            # else:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
            if get_feature:
                hook_pooler = SimpleHook(self.roi_heads.box.feature_extractor.pooler)
                x, _, _ = self.roi_heads(features, result, targets)
                x = hook_pooler.output.data
                results = {'roi_features':x,
                           'glob_feature':torch.stack(features).squeeze(1), }
                hook_pooler.close()
                return results
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

# Self-made hook
class SimpleHook(object):
    """
    A simple hook function to extract features.
    :return:
    """
    def __init__(self, module, backward=False):
        # super(SimpleHook, self).__init__()
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input_, output_):
        self.input = input_
        self.output = output_

    def close(self):
        self.hook.remove()