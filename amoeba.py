import numpy as np
import torch
import torch.nn as nn
import torchvision

from faster_rcnn import FasterRCNN
from train import train_one_amoeba_epoch, collate_fn
from video import OpenCVVideo
from evaluation import Evaluator, evaluate
from args import read_args
from transforms import get_unlabeled_transform, get_transform

args = read_args()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 9
cityscapes_categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "rider"}, {"id": 3, "name": "car"}, {"id": 4, "name": "bicycle"}, {"id": 5, "name": "motorcycle"}, {"id": 6, "name": "bus"}, {"id": 7, "name": "truck"}, {"id": 8, "name": "train"}]

dataset = torchvision.datasets.CocoDetection('../cityscapes', '../cityscapes/annotations/instancesonly_filtered_gtFine_train.json', transforms = get_transform(True))
dataset_test = torchvision.datasets.CocoDetection('../cityscapes', '../cityscapes/annotations/instancesonly_filtered_gtFine_val.json', transforms = get_transform(False))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=1,collate_fn=collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch, shuffle=False, num_workers=1,collate_fn=collate_fn)

model = FasterRCNN(num_classes, discrep=True)
model.to(device)


for p in model.parameters():
    p.requires_grad = True
    
d_params = [p for n, p in model.named_parameters() if 'extra' in n]
g_params = [p for n, p in model.named_parameters() if 'extra' not in n]    

g_optimizer = torch.optim.SGD(g_params, lr=0.001,momentum=0.9, weight_decay=0.0005)
d_optimizer = torch.optim.SGD(d_params, lr=0.001,momentum=0.9, weight_decay=0.0005)

if args.ckpt is not None:
    mapping = torch.load(args.ckpt)
    model.load_state_dict(mapping['model'])
    g_optimizer.load_state_dict(mapping['g_opt'])
    d_optimizer.load_state_dict(mapping['d_opt'])

video = OpenCVVideo('warsaw.mp4', get_unlabeled_transform(True), args.batch, sample=1)
num_epochs = args.epoch

for epoch in range(num_epochs):
    train_one_amoeba_epoch(model, (g_optimizer, d_optimizer), (data_loader, video), device, epoch, print_freq=10)
    torch.save({'model': model.state_dict(), 'g_opt': g_optimizer.state_dict(), 'd_opt': d_optimizer.state_dict()}, 'amoeba' + str(epoch) + '.pth') 
    evaluator = Evaluator(cityscapes_categories)
    evaluate(model, evaluator, data_loader_test)
    #video = OpenCVVideo('warsaw.mp4', get_unlabeled_transform(True), args.batch, sample=1)
