from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
config_file = "/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_101_FPN_1x.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.MODEL.WEIGHT = "/data6/SRIP_SelfDriving/Outputs/model_final.pth"

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = Image.open('test2.jpg').convert("RGB")
predictions = coco_demo.run_on_opencv_image(np.array(image))
plt.imshow(predictions)
plt.axis('off')
plt.savefig('prediction.png')
