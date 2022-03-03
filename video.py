import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision


class OpenCVVideo(object):
    def __init__(self, root, transforms, batch, sample=1):
        self.root = root
        self.capture = cv2.VideoCapture(self.root)
        self.batch = batch
        self.transforms = transforms
        self.sample=sample
        
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.frame_count = int(self.capture.get(cv2. CAP_PROP_FRAME_COUNT))
        self.duration = int(self.frame_count//self.fps)
        
    def __iter__(self):
        return self
    def __len__(self):
        return (self.frame_count // (self.sample * self.batch)) - 1
    def __next__(self):
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.root)
            print('Exited')
            raise StopIteration
        else:
            frames = []
            for _ in range(self.batch):
                for _ in range(self.sample):
                    if not self.capture.isOpened():
                        self.capture = cv2.VideoCapture(self.root)
                        print('Exited')
                        raise StopIteration
                        break
                    ret, frame = self.capture.read()
                if frame is not None:
                    frame = self.transforms(frame, None)[0]
                    frames.append(torch.tensor(frame).float() / 255)
            if len(frames) == 0:
                raise StopIteration
            frames = torch.stack(frames, 0)
            return frames, None
              # Done iterating.
    next = __next__  # python2.x compatibility.
    
class VideoLabeler():
    def __init__(self, filename, size):
        self.video = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, size)
    
    def finalize(self):
        self.video.release()
    
    def accept(self, model, images, targets):
        model.eval()
        with torch.no_grad():
            images = list(image.to(model.device) for image in images)
            outputs = model(images)
            
            images = list(image.cpu() for image in images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            
            for i in range(len(images)):
                im = images[i]
                output = outputs[i]
                image = torchvision.utils.draw_bounding_boxes((im * 255).byte().cpu(), output['boxes'][output['scores'] > 0.5])

                frame = torch.nn.functional.interpolate(image.unsqueeze(0), tuple(reversed(size))) 
                frame = frame.squeeze().cpu().numpy().transpose(1, 2, 0)
                self.video.write(frame)
                del im, frame
            del images, outputs