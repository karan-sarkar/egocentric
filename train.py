from tqdm import tqdm
import torch


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    enum = enumerate(data_loader)
    pbar = tqdm(range(len(data_loader)))
    for _ in pbar:
        try:
            _, (images, targets) = next(enum)
        except:
            continue

        images = list(image.to(device) for image in images)
        images = [im for im, t in zip(images, targets) if t['image_id'].sum() >= 0]
        targets = [t for t in targets if t['image_id'].sum() >= 0]
        if len(images) == 0:
            continue

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        pbar.set_description(loss_string(loss_dict))
        losses = sum(loss for loss in loss_dict.values())


        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def train_one_amoeba_epoch(model, optimizers, data_loaders, device, epoch, print_freq=10):
    model.train()

    
    g_opt, d_opt, c_opt = optimizers
    source_data, target_video = data_loaders 
    
    source_iter = enumerate(source_data)
    
    pbar = tqdm(range(len(source_data)))
    for i in pbar:
        try:
            _, (source_images, source_targets) = next(source_iter)
        except StopIteration:
            break
        except:
            continue
        
        video_batch, _ = next(target_video)
        source_images = [im for im, t in zip(source_images, source_targets) if t['image_id'].sum() >= 0]
        source_targets = [t for t in source_targets if t['image_id'].sum() >= 0]
        descrip = ''
        
        if len(source_images) == 0:
            continue
        
        source_images = list(image.to(device) for image in source_images)
        source_targets = [{k: v.to(device) for k, v in t.items()} for t in source_targets]
        
        loss_dict = model(source_images, source_targets)
        losses = sum(loss for loss in loss_dict.values())
        descrip += loss_string(loss_dict)
        

        g_opt.zero_grad()
        d_opt.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        g_opt.step()
        d_opt.step()
        del loss_dict, losses
        
        loss_dict = model(source_images, source_targets)
        losses = sum(loss for loss in loss_dict.values())
        
        video_batch = video_batch.to(device)
        loss_dict = model(video_batch)
        losses -= sum(loss for loss in loss_dict.values())
        descrip += loss_string(loss_dict)
        
        d_opt.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        d_opt.step()
        del loss_dict, losses
        
        for _ in range(4):
        
            loss_dict = model(source_images, source_targets)
            losses = sum(loss for loss in loss_dict.values())

            video_batch = video_batch.to(device)
            loss_dict = model(video_batch)
            losses = sum(loss for loss in loss_dict.values())

            g_opt.zero_grad()
            c_opt.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            g_opt.step()
            c_opt.step()
            del loss_dict, losses
        

        del source_images, source_targets, video_batch,
        pbar.set_description(descrip)


def loss_string(loss_dict):
    return str({''.join([c for i,c in enumerate(k) if c not in 'aeiou']):round(float(v), 3) for k,v in loss_dict.items() if 'extra' not in k})

def collate_fn(batch):
    return tuple(zip(*batch))

cityscapes_categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "rider"}, {"id": 3, "name": "car"}, {"id": 4, "name": "bicycle"}, {"id": 5, "name": "motorcycle"}, {"id": 6, "name": "bus"}, {"id": 7, "name": "truck"}, {"id": 8, "name": "train"}]
