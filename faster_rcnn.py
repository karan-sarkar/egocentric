import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import OrderedDict

import copy

class FasterRCNN(nn.Module):
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

class StdDiscrepLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, x, y):
        sentinel = torch.Tensor([0.00001]).to(x.device)
        x1 = (x - y.mean(0)) / (torch.max(y.std(0), sentinel))
        y1 = (y - y.mean(0)) / (torch.max(y.std(0), sentinel))
        return self.loss(x1, y1)
        
class CustomROIHead(nn.Module):
    def __init__(self, roi_head):
        super().__init__()
        self.roi_head = roi_head
        self.extra_predictor = copy.deepcopy(self.roi_head.box_predictor)
        self.discrep_loss = StdDiscrepLoss()
        self.discrep_loss2 = nn.L1Loss()
        self.iter = 0
        
        for layer in self.extra_predictor.children():
            torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
            torch.nn.init.constant_(layer.bias, 0)
    
    
    def forward(self,  features, proposals, image_shapes, targets=None,  ):
        if self.training and targets is not None:
            proposals, matched_idxs, labels, regression_targets = self.roi_head.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.roi_head.box_roi_pool(features, proposals, image_shapes)
        box_features = self.roi_head.box_head(box_features)
        class_logits, box_regression = self.roi_head.box_predictor(box_features)
        extra_class_logits, extra_box_regression = self.extra_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            class_discrep = 0.1 * self.discrep_loss(extra_class_logits.softmax(-1), class_logits.detach().softmax(-1))
            box_discrep = 0.1 * self.discrep_loss2(extra_box_regression, box_regression.detach())
            losses = {'class_discrep': class_discrep, 'box_discrep': box_discrep}
            
            if targets is not None:
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                extra_loss_classifier, extra_loss_box_reg = fastrcnn_loss(extra_class_logits, extra_box_regression, labels, regression_targets)
                losses.update({"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, 
                    "extra_loss_classifier": extra_loss_classifier, "extra_loss_box_reg": extra_loss_box_reg})
        else:
            boxes, scores, labels = self.roi_head.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return (result, losses)
        
class CustomRPN(nn.Module):
    def __init__(self, rpn):
        super().__init__()
        self.rpn = rpn
        self.extra_head = copy.deepcopy(self.rpn.head)
        self.discrep_loss = nn.L1Loss()
        self.iter = 0
        
        for layer in self.extra_head.children():
            torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
            torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]
            
    def forward(self, images, features, targets):
        self.iter += 1
        features = list(features.values())
        objectness, pred_bbox_deltas = self.rpn.head(features)
        extra_objectness, extra_pred_bbox_deltas = self.extra_head(features)
        anchors = self.rpn.anchor_generator(images, features)
        
        
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        extra_objectness, extra_pred_bbox_deltas = concat_box_prediction_layers(extra_objectness, extra_pred_bbox_deltas)
        
        #if self.iter % 2 == 0:
        obj = objectness
        pbd = pred_bbox_deltas
        #else:
            #obj = extra_objectness
            #pbd = extra_pred_bbox_deltas
        
        
        proposals = self.rpn.box_coder.decode(pbd.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn.filter_proposals(proposals, obj, images.image_sizes, num_anchors_per_level)

        losses = {}
        #losses['objectness_discrep'] = self.discrep_loss(extra_objectness.sigmoid(), objectness.detach().sigmoid())
        #losses['bbox_discrep'] = self.discrep_loss(extra_pred_bbox_deltas.sigmoid(), pred_bbox_deltas.detach().sigmoid())
            
            
        if self.training and targets is not None:
            assert targets is not None
            labels, matched_gt_boxes = self.rpn.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.rpn.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.rpn.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            loss_extra_objectness, loss_extra_rpn_box_reg = self.rpn.compute_loss(
                extra_objectness, extra_pred_bbox_deltas, labels, regression_targets
            )
            losses.update({
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_extra_objectness": loss_extra_objectness,
                "loss_extra_rpn_box_reg": loss_extra_rpn_box_reg,
            })
        return boxes, losses

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss