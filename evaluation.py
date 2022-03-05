from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torch
from tqdm import tqdm

class Evaluator():
    def __init__(self, categories):
        self.categories = categories
        self.gt = {'images': [], 'annotations': [], 'categories': categories}
        self.res =  {'images': [], 'annotations': []}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def finalize(self):
        with open('gt.json', 'w') as outfile:
            json.dump(self.gt, outfile)
        with open('res.json', 'w') as outfile:
            json.dump(self.res['annotations'], outfile)

        cocoGt=COCO('gt.json')
        cocoDt=cocoGt.loadRes('res.json')

        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    
    def accept(self, model, images, targets):
        im = list(image.to(self.device) for image in images)
        images = [im for im, t in zip(im, targets) if t['image_id'].min() >= 0]
        del im
        targets = [t for t in targets if t['image_id'].min() >= 0]
        
        if len(images) == 0:
            return
        with torch.no_grad():
            outputs = model(images)
        
        for i in range(len(outputs)):
            target = targets[i]
            output = outputs[i]

            self.gt['images'].append({'id': int(target['image_id'])})
            self.res['images'].append({'id': int(target['image_id'])})

            for j in range(target['boxes'].size(0)):
                new_gt = {
                   'bbox' : target['boxes'][j].cpu().numpy().tolist(), 'category_id': int(target['labels'][j].view(-1)),
                    'image_id' : int(target['image_id']),  'area' : float(target['area'][j].view(-1)),
                    'iscrowd' : float(target['iscrowd'][j].view(-1)), 'id' : len(self.gt['annotations']),
                }

                self.gt['annotations'].append(new_gt)

            for j in range(output['boxes'].size(0)):
                new_res = {
                    'bbox' : output['boxes'][j].cpu().numpy().tolist(), 'category_id': int(output['labels'][j].view(-1)),
                    'score' : float(output['scores'][j].view(-1)), 'image_id' : int(target['image_id']),
                    'id' : len(self.res['annotations']),
                }

                self.res['annotations'].append(new_res)

        del targets, outputs, images

def evaluate(model, evaluator, data_loader, limit=None):
    model.eval()
    
    with torch.no_grad():
        enum = enumerate(tqdm(data_loader))
        for i in range(len(data_loader)) :
            try:
                _, (images, targets) = next(enum)
            except:
                continue
            evaluator.accept(model, images, targets)
            if limit is not None and i > limit:
                break
    
        evaluator.finalize()


def collate_fn(batch):
    return tuple(zip(*batch))
    
    

    