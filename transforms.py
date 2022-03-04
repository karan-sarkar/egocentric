import torch
import torch.nn as nn
import numpy as np

import albumentations as A

class LabelTransform(nn.Module):
    
    def forward(
        self, image, old_targets):
        target = {}
        target["boxes"] = []
        target["labels"] = []
        target["image_id"] = torch.tensor([-1])
        target["area"] = []
        target["iscrowd"] = []
        self.pool = nn.AvgPool2d(16, stride=16)
        
        for old_target in old_targets:
            xmin, ymin, width, height = tuple(old_target['bbox'])           
            target["boxes"].append(torch.FloatTensor([xmin, ymin, width, height]))
            target["labels"].append(torch.LongTensor([old_target['category_id']]))
            target["image_id"] = torch.tensor([old_target["image_id"]])
            target["area"].append(torch.LongTensor([old_target["area"]]))
            target["iscrowd"].append(torch.LongTensor([old_target["iscrowd"]]))
        
        empty = len(target["boxes"]) == 0 
        if empty:
            target["boxes"] = torch.arange(4).unsqueeze(0)
            target["labels"] = torch.tensor([0]).unsqueeze(0)
            target["area"] = torch.tensor([0]).unsqueeze(0)
            target["iscrowd"] = torch.tensor([0]).unsqueeze(0)
        else:
            target["boxes"] = torch.stack(target["boxes"], 0)
            target["labels"] = torch.stack(target["labels"], 0)
            target["area"] = torch.stack(target["area"], 0)
            target["iscrowd"] = torch.stack(target["iscrowd"], 0)
        
        
        return (image, target)

class Albu():
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(
        self, image, target):
        if target is not None:
            transformed = self.transform(image=np.array(image), bboxes = target['boxes'], category_ids = target['labels'])
            target['boxes'] = convert_to_xyxy(transformed['bboxes'])
            target['labels'] = torch.cat(transformed['category_ids'], 0)
        else:
            transformed = self.transform(image=np.array(image))
        return (torch.tensor(transformed['image'].transpose(2, 0, 1)).float() / 255, target)

class Compose():
    def __init__(self, trans):
        self.trans = trans
    
    def __call__(self, image, target):
        current = (image, target)
        for transform in self.trans:
            current = transform(*current)
        return current

def get_transform(train):
    albu = A.Compose([
            #A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields = ['category_ids']))
    if not train:
        albu = A.Compose([], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields = ['category_ids']))
    albu = Albu(albu)
    return Compose([
            LabelTransform(),
            albu,
    ])

def get_unlabeled_transform(train):
    if train:
        albu = A.Compose([
                A.LongestMaxSize(2048),
                #A.RandomCrop(width=450, height=450),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])
    else:
        albu = A.Compose([A.LongestMaxSize(2048)])
    albu = Albu(albu)
    return Compose([
            albu,
    ])

def convert_to_xyxy(boxes):
    boxes = zip(*boxes)
    boxes = [torch.tensor(np.array(arr)) for arr in boxes]
    xmin, ymin, xmax, ymax = boxes
    return torch.stack((xmin, ymin, xmax + xmin, ymax + ymin), dim=1)
            
