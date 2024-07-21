import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class ImageDataset(Dataset):
    def __init__(self, data_root, transform=None, train=False, aug_dict=None):
        self.root_path = data_root
        self.train_lists = "train.list"   # check this
        self.eval_list = "test.list"      # check this
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')
        self.img_map = {}
        self.img_list = []

        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()  # remove head&tail space
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        
        self.nSamples = len(self.img_list)
        self.transform = transform      # normalize, totensor
        self.train = train              # trainable

        self.aug_dict = aug_dict
        if self.aug_dict != None:
            self.patch = 'Crop' in aug_dict.AUGUMENTATION  # enable random crop
            self.flip = 'Flip' in aug_dict.AUGUMENTATION   # enable random flip
            self.upper_bound = aug_dict.UPPER_BOUNDER      # bondary of image size, -1 is not limitation
            self.crop_size = aug_dict.CROP_SIZE      # random crop size
            self.crop_number = aug_dict.CROP_NUMBER  # the number of crop sample
        else:
            self.patch = False        # enable random crop
            self.flip = False         # enable random flip
            self.upper_bound = -1     # bondary of image size, -1 is not limitation
            self.crop_size = 128      # random crop size
            self.crop_number = 4      # the number of crop sample

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
    
        #####################################
        # load image and ground truth
        # imgs, points_array (whole image)
        #####################################
        img, point = load_data((img_path, gt_path), self.train)

        #####################################
        # Data Augumentation
        # Totensor + Normalize
        # Random scale + Random Crop + Random Flipping
        #####################################
        # Apply augumentation: Totensor + Normalize
        if self.transform is not None:   
            img = self.transform(img)
        # Train augumentation: Random scale + Random Crop + Random Flipping
        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            max_size = max(img.shape[1:])
            scale = random.uniform(*scale_range)  

            if max_size > self.upper_bound and self.upper_bound != -1:
                upbound = self.upper_bound / max_size
                scale_range = [upbound-0.1, upbound]
                scale = random.uniform(*scale_range)
            elif self.upper_bound != -1:
                scale_range = [0.7, 1.3]
                scale = random.uniform(*scale_range)
            else:
                scale_range = [0.7, 1.]
                scale = random.uniform(*scale_range)

            if scale * min_size > self.crop_size:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
            
            # random crop augumentaiton 
            if self.patch:
                img, point = random_crop(img, point, num_patch=self.crop_number, crop_size=self.crop_size)
                for i, _ in enumerate(point):  # transfer point to tensor
                    point[i] = torch.Tensor(point[i])

            # random flipping
            if self.flip and random.random() > 0.5:    
                # random flip
                img = torch.Tensor(img[:, :, :, ::-1].copy())
                for i, _ in enumerate(point):
                    point[i][:, 0] = self.crop_size - point[i][:, 0]  
        else:
            max_size = max(img.shape[1:])
            if max_size > self.upper_bound and self.upper_bound!=-1:
                scale = self.upper_bound / max_size
            elif max_size > 2560:
                scale = 2560 / max_size
            else:
                scale = 1.0
            img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            point *= scale

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])    
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id               
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long() 
            target[i]['name'] = os.path.basename(img_path)
        return img, target # target: {'point', 'image_id', 'labels', 'name'}

def load_data(img_gt_path, train):
    ###################################
    # return imgs, points
    ###################################
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])
    return img, np.array(points)

# random crop augumentation
def random_crop(img, den, num_patch=4, crop_size=128):
    half_h, half_w = crop_size, crop_size
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        if len(den) > 0:
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
        else:
            record_den = np.empty((0,2))
            result_den.append(record_den)
    return result_img, result_den