import torchvision
from torchvision.models.detection.fcos import FCOSHead
import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import OrderedDict

import copy

class FCOS(nn.Module):
    def __init__(self, num_classes, discrep = False):
        super().__init__()
        self.model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
        self.model.head = FCOSHead(self.model.backbone.out_channels, self.model.anchor_generator.num_anchors_per_location()[0], num_classes)
        self.discrep_loss = nn.L1Loss()
        self.discrep = discrep
        
        if discrep:
            self.extra_head = FCOSHead(self.model.backbone.out_channels, self.model.anchor_generator.num_anchors_per_location()[0], num_classes)
    
    def forward(self, images, targets=None):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
           
        
        images, targets = self.model.transform(images, targets)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the fcos heads outputs using the features
        head_outputs = self.model.head(features)
        if self.discrep and self.training:
            extra_head_outputs = self.extra_head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(images, features)
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        

        losses = {}
        detections = []
        if self.training:
            losses = {}

            # compute the losses
            if targets is not None:
                losses = self.model.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
                if self.discrep:
                    extra_losses = self.model.compute_loss(targets, extra_head_outputs, anchors, num_anchors_per_level)
                    for key in extra_losses.keys():
                        losses[key] += extra_losses[key]
                    del extra_losses
            
            if self.discrep:
                for key in head_outputs.keys():
                    if 'reg' not in key:
                        extra_head_outputs[key] = extra_head_outputs[key].sigmoid()
                        head_outputs[key] = head_outputs[key].sigmoid()
                    losses[key + '_discrep'] = self.discrep_loss(extra_head_outputs[key], head_outputs[key].detach())
            
        else:
            # split outputs per level
            split_head_outputs = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.model.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        
        return self.model.eager_outputs(losses, detections)
