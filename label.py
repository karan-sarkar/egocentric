import numpy as np
import torch
import torch.nn as nn
import torchvision

from faster_rcnn import FasterRCNN
from train import train_one_epoch, collate_fn
from video import OpenCVVideo, VideoLabeler
from evaluation import Evaluator, evaluate
from args import read_args
from transforms import get_unlabeled_transform, get_transform

args = read_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 31
cityscapes_categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "rider"}, {"id": 3, "name": "car"}, {"id": 4, "name": "bicycle"}, {"id": 5, "name": "motorcycle"}, {"id": 6, "name": "bus"}, {"id": 7, "name": "truck"}, {"id": 8, "name": "train"}]

model = FasterRCNN(num_classes)
model.to(device)


if args.ckpt is not None:
    mapping = torch.load(args.ckpt)
    model.load_state_dict(mapping['model'])

video = OpenCVVideo('warsaw.mp4', get_unlabeled_transform(True), args.batch, sample=1)
labeler = VideoLabeler('warsaw_labeled.avi', (1720, 1080))
evaluate(model, labeler, video, limit=1000)
