import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sympy import false
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json
import PIL.Image as Image
import random
import warnings
import math
import albumentations as A
from albumentations.core.composition import Compose

warnings.filterwarnings("ignore")
plt.ion()


class StoneData_unimatch(Dataset):
    def __init__(self, args, mode='train', transform_w=None, transform_s=None, labeled=True):
        self.args = args
        self.mode = mode
        json_file_path = args.json_path
        with open(json_file_path, 'r') as file:
            split = json.load(file)

        self.labeled = labeled
        self.images, self.masks = self.get_image_path_list(split, args, mode, labeled=labeled)

        self.images = self.images
        self.masks = self.masks
        if self.args.name == 'test123':
            self.images = self.images[0:196]
            self.masks = self.masks[0:196]

        self.transform_w = transform_w
        self.transform_s = transform_s

        self.transform_norm =  Compose([A.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalization mean (e.g., ImageNet stats)
                std=[0.229, 0.224, 0.225]  # Normalization std (e.g., ImageNet stats)
            )], seed=42)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = np.array(Image.open(img_name))
        if self.mode == 'train' and self.labeled == False:
            #mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask = np.array(Image.open(self.masks[idx]))
        else:
            mask = np.array(Image.open(self.masks[idx]))

        #image = self.zscore_normalization(image)


        augmented_w1 = self.transform_w(image=image, mask=mask)
        image_w1 = augmented_w1['image']
        mask_w1 = augmented_w1['mask']

        augmented_w2 = self.transform_w(image=image, mask=mask)
        image_w2 = augmented_w2['image']
        mask_w2 = augmented_w2['mask']
        #print(np.sum(image_w1 - image_w2))

        augmented_s1 = self.transform_s(image=image_w1, mask=mask_w1)
        image_s1 = augmented_s1['image'].astype(np.float32)

        augmented_s2 = self.transform_s(image=image_w2, mask=mask_w2)
        image_s2 = augmented_s2['image'].astype(np.float32)

        cutmix_box = self.obtain_cutmix_box(image_s2, p=0.5)

        image_w1 = self.transform_norm(image=image_w1)['image']
        image_w2 = self.transform_norm(image=image_w2)['image']
        image_s1 = self.transform_norm(image=image_s1)['image']
        image_s2 = self.transform_norm(image=image_s2)['image']

        image_w1 = np.moveaxis(image_w1, -1, 0)
        image_w2 = np.moveaxis(image_w2, -1, 0)
        image_s1 = np.moveaxis(image_s1, -1, 0)
        image_s2 = np.moveaxis(image_s2, -1, 0)

        mask_w1 = np.expand_dims(mask_w1, axis=2).astype(float)
        mask_w1 = np.moveaxis(mask_w1, -1, 0)

        #return image_w1, image_s1, image_s2, cutmix_box1, cutmix_box2
        sample = {'image_w1': torch.from_numpy(image_w1).type(torch.FloatTensor),
                  'image_w2': torch.from_numpy(image_w2).type(torch.FloatTensor),
                  'image_s1': torch.from_numpy(image_s1).type(torch.FloatTensor),
                  'image_s2': torch.from_numpy(image_s2).type(torch.FloatTensor),
                  'cutmix_box': torch.from_numpy(cutmix_box),
                  'mask_w1': torch.from_numpy(mask_w1).type(torch.FloatTensor),
                  }
        return sample


    def obtain_cutmix_box(self, image, p=0.2, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
        assert image.shape[0] == image.shape[1]
        img_size = image.shape[0]
        mask = np.zeros((img_size, img_size))

        if random.random() > p:
            return mask

        size = np.random.uniform(size_min, size_max) * img_size * img_size
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_w = int(np.sqrt(size / ratio))
            cutmix_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_size)
            y = np.random.randint(0, img_size)
            if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                break
        mask[y:y + cutmix_h, x:x + cutmix_w] = 1
        return mask

    def zscore_normalization(self, data):
        lower_bound = np.percentile(data, 0.5)
        upper_bound = np.percentile(data, 99.5)
        clipped_data = np.clip(data, lower_bound, upper_bound)

        mean = np.mean(clipped_data)
        std = np.std(clipped_data)
        z_score_normalized_data = (clipped_data - mean) / (std + 1e-5)
        return z_score_normalized_data.astype(np.float32)

    def get_image_path_list(self, split, args, mode, labeled=True):
        image_path_list, mask_path_list = [], []
        base_dir = args.inputs

        if mode == 'train':
            if labeled:
                filenames_labeled = split['train']['labeled']
                args.labeled_frames = len(filenames_labeled)
                filenames_unlabeled = split['train']['unlabeled']

                repetition_factor = math.ceil(len(filenames_unlabeled) / len(filenames_labeled))
                filenames_labeled *= repetition_factor
                filenames = filenames_labeled[:len(filenames_unlabeled)]
            else:
                filenames = split['train']['unlabeled']
        else:
            filenames = split['val']

        for filename in filenames:
            folder_name = filename.split('_frame')[0]
            image_name = filename.replace(folder_name + '_', '')

            full_path_image = os.path.join(base_dir, mode, 'images', folder_name, image_name)
            full_path_mask = os.path.join(base_dir, mode, 'masks', folder_name, image_name)

            image_path_list.append(full_path_image)
            if mode == 'train' and labeled == False:
                #mask_path_list.append(None)
                mask_path_list.append(full_path_mask)
            else:
                mask_path_list.append(full_path_mask)

        return image_path_list, mask_path_list




