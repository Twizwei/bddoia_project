# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import logging
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

# from maskrcnn_benchmark.modeling.poolers import Pooler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from test_all import DrawBbox
import numpy as np

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from .predictor import build_predictor


# def DrawBbox(img, boxlist):
#     plt.imshow(img)
#     currentAxis = plt.gca()
#     # folder = '/data6/SRIP19_SelfDriving/bdd12k/Outputs/'
#     for i in range(boxlist.shape[0]):
#         bbox = boxlist[i]
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
#         # rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')
#         currentAxis.add_patch(rect)
#
#     plt.show()


class Baseline(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(Baseline, self).__init__()
        self.side = cfg.MODEL.SIDE
        self.is_cat = cfg.MODEL.IS_CAT
        print("side:{}".format(self.side))
        print("concat:{}".format(self.is_cat))

        self.training = False
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.predictor = build_predictor(cfg, side=self.side, is_cat=self.is_cat)
        self.count = 0

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

    def forward(self, images, targets=None, get_feature=True): #TODO: Remember to set get_feature as False if there are any changes later
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
        self.count += 1
        # print(self.count)
        images = to_image_list(images)
        # if self.count == 1:
        #     from PIL import Image
        #     # from maskrcnn_benchmark.structures.image_list import to_image_list
        #     import numpy as np
        #     images = Image.open('/data6/SRIP19_SelfDriving/bdd12k/data1/aaddb12e-82dd1431_3.jpg')
        #     images = to_image_list(torch.Tensor(np.array(images).transpose(2, 0, 1)).cuda())
        # print(torch.norm(images.tensors))
        # img = images.tensors.cpu().numpy()[0].transpose(1,2,0)
        # folder = '/data6/SRIP19_SelfDriving/bdd12k/Outputs/'
        # print('Saving img')
        # np.save((folder + str(self.count) + 'img.npy'), img)
        features = self.backbone(images.tensors)
        # print(torch.norm(features[0]))
        # print(features[0])
        # print(type(features))
        logger = logging.getLogger("tmp")
        logger.info(len(features))
        proposals, proposal_losses = self.rpn(images, features, targets)
        # print(proposals)
        # print("Num of proposals: ", proposals[0].bbox.shape)
        # print("Proposal:", proposals[0].bbox.type(torch.int))
        if self.roi_heads:
            # if get_feature:
            #     x, result, _ = self.roi_heads(features, targets, targets)
            # else:
            x, result, _ = self.roi_heads(features, proposals, targets)
            print("x shape",x.shape)
            # print("Num of BBOX: ", result[0].bbox.shape)
            # print("BBOX:", result[0].bbox.type(torch.int))
            # print("Saving BBOX")
            # np.save((folder + str(self.count) + 'box.npy'), result[0].bbox.cpu().numpy())
            # box = result[0].bbox.cpu().numpy()
            # DrawBbox(img, box)
            if get_feature:
                b_size = len(result)
                hook_pooler = SimpleHook(self.roi_heads.box.feature_extractor.pooler)
                x, _, _ = self.roi_heads(features, result, targets)
                x = hook_pooler.output.data
                hook_pooler.close()
                tmp = 0
                preds = []
                preds_reason = []
                # selected_boxes=[]
                for i in range(b_size): # iterate every image in the batch
                    bbox_num = result[i].bbox.shape[0]
                    results = {'roi_features':x[tmp:tmp + bbox_num],
                               'glob_feature':features[0][i].unsqueeze(0), }
                    tmp = bbox_num
                    # hook_selector = SimpleHook(self.predictor.selector)
                    if self.side:
                        pred, pred_reason = self.predictor(results)
                    else:
                        pred = self.predictor(results)

                    # scores = hook_selector.output.data
                    # hook_selector.close()
                    # scores, idx = torch.sort(scores, dim=0, descending=True)
                    # idx = idx.reshape(-1,)
                    # selected_boxes.append(result[i].bbox[idx[:10]])
                    preds.append(pred)
                    # print(self.side)
                    if self.side:
                        preds_reason.append(pred_reason)
                # torch.cuda.empty_cache()
                # print(preds[0])
                # print(torch.stack(preds,dim=0).squeeze(1))
                return (torch.stack(preds,dim=0).squeeze(1), torch.stack(preds_reason,dim=0).squeeze(1)) if self.side else torch.stack(preds,dim=0).squeeze(1)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

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
