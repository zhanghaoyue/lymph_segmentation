from torch import nn
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detection_model(num_classes):
    net = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes,
                                                             pretrained=False, pretrained_backbone=True)

    return net
