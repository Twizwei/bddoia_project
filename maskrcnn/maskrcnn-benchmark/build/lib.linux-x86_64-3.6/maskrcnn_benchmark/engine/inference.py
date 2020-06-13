# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


# def DrawBbox(img, boxlist, folder, k):
#     fig = plt.figure()
#     plt.imshow(img)
#     currentAxis = plt.gca()
#     for i in range(boxlist.shape[0]):
#         bbox = boxlist[i]
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r',
#                                  facecolor='none')
#         # rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')
#         currentAxis.add_patch(rect)
#     print('Saving... ', k)
#     plt.savefig((folder + str(k) + '.jpg'), bbox_inches='tight')
#     plt.clf()



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

def compute_on_dataset(model, data_loader, device, timer=None, get_feature=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    RoIPool = model.roi_heads.box.feature_extractor.pooler
    Backbone = model.backbone
    count = 0

    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        # folder = '/data6/SRIP19_SelfDriving/bdd100k/Outputs/inference/bdd100k_val/images/'
        # plt.imsave((folder + str(i) + '.jpg'), images.tensors.cpu().numpy()[0].transpose(1,2,0).astype(np.uint8))
        # if i == 0:
        #     from PIL import Image
        #     from maskrcnn_benchmark.structures.image_list import to_image_list
        #     import numpy as np
        #     images = Image.open('/data6/SRIP19_SelfDriving/bdd12k/data1/aaddb12e-82dd1431_3.jpg')
        #     images = to_image_list(torch.Tensor(np.array(images).transpose(2,0,1)))

        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                if get_feature:
                    t = [target.to(device) for target in targets]
                    hook_roi = SimpleHook(RoIPool)
                    hook_backbone = SimpleHook(Backbone)
                    output = model(images.to(device), t, get_feature)
                    features_roi = hook_roi.output.data
                    features_backbone = hook_backbone.output[0].data
                    avgpool = nn.AdaptiveAvgPool2d((14,14))
                    features_backbone = avgpool(features_backbone)
                    if(features_roi.shape[0] == 0):
                        features = features_backbone
                    else:
                        features = torch.cat((features_roi, features_backbone),dim=0)
                    folder = "/data6/SRIP19_SelfDriving/bdd100k/features/train/"
                    np.save(folder + str(image_ids[0]) + '.npy',features.cpu().numpy())
                    hook_roi.close()
                else:
                    output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            # print(output.bbox)
            # print(len(output))
            output = [o.to(cpu_device) for o in output]
            # bbox = output[0].bbox.cpu().numpy()
            # np.save((folder + str(i) + '.npy'), bbox)

            # img = images.tensors.cpu().numpy()[0].transpose(1,2,0).astype(np.uint8)
            # DrawBbox(img, bbox, folder, i)
            # print(bbox)
            # print(bbox.shape)
            # print(output)
            # print(type(output))
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        count += 1
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        get_feature=False,
):
    """

    :rtype:
    """
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer, get_feature)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        # torch.save(images, os.path.join(output_folder, "images.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
