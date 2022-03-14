import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import OrderedDict

import copy

class SSD(nn.Module):
    def __init__(self, num_classes, discrep = False):
        super().__init__()
        
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        
        self.backbone = model.backbone
        self.rpn = model.rpn
            
        self.roi_heads = model.roi_heads
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        if discrep:
            self.rpn = CustomRPN(self.rpn)
            self.roi_heads = CustomROIHead(self.roi_heads)
       
        
        self.transform = model.transform
    
    def forward(self, images, targets=None, ignore=False):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
                    
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        losses = {}
        
        losses.update(proposal_losses)
        
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator
        losses.update(detector_losses)
        if self.training:
            return losses
        
        return detections
