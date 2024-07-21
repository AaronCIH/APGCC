import torch
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as standard_transforms
from typing import Optional, List
from .dataset import ImageDataset

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(cfg, ):
    #############################################################################
    # SHHA_set: getitem=[img: images, target:{'point', 'image_id', 'label', 'noise_point'}]
    # train_dl: tuple((imgs, points))
    # val_dl: tuple((imgs, points))
    #############################################################################
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    if 'Crop' in cfg.DATALOADER.AUGUMENTATION:
        patch = True
    if 'Flip' in cfg.DATALOADER.AUGUMENTATION:
        flip = True

    train_set = ImageDataset(cfg.DATASETS.DATA_ROOT, train=True, transform=transform, aug_dict=cfg.DATALOADER)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, cfg.SOLVER.BATCH_SIZE, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn_crowd, num_workers=cfg.DATALOADER.NUM_WORKERS)

    # create the validation dataset
    val_set = ImageDataset(cfg.DATASETS.DATA_ROOT, train=False, transform=transform, aug_dict=cfg.DATALOADER)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn_crowd, 
                                 num_workers=cfg.DATALOADER.NUM_WORKERS)

    print("################################################")
    print("# Dataset: ", cfg.DATASETS.DATASET)
    print("# Data_rt:", cfg.DATASETS.DATA_ROOT)
    print("# Train  :", train_set.nSamples)
    print("# Val    :", val_set.nSamples)
    print("################################################")
    return data_loader_train, data_loader_val

def collate_fn_crowd(batch):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved. 
    # re-organize the batch
    batch_new = []
    for b in batch:
        imgs, points = b
        if imgs.ndim == 3:  # batch_size = 1, or gray-scale
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):  # list of imgs and points
            batch_new.append((imgs[i, :, :, :], points[i]))
    batch = batch_new
    batch = list(zip(*batch)) 
    batch[0] = _nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def _max_by_axis_pad(the_list):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved. 
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    block = 128
    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes

def _nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved. 
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor
