import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json
import copy
import warnings
import math
import albumentations as A
from albumentations.core.composition import Compose
warnings.filterwarnings("ignore")
plt.ion()


class StoneData_semi(Dataset):
    def __init__(self, args, mode='train', transform=None, labeled=True, sup=False):
        self.args = args
        self.mode = mode
        self.sup = sup
        json_file_path = args.json_path
        with open(json_file_path, 'r') as file:
            split = json.load(file)

        self.labeled = labeled
        if mode != 'test':
            self.images, self.masks = self.get_image_path_list(split, args, mode, labeled=labeled, sup=sup)
        else:
            self.images = sorted(glob.glob(os.path.join(args.inputs, f'{mode}/images/**/*.PNG'), recursive=True))
            self.masks = sorted(glob.glob(os.path.join(args.labels, f'{mode}/masks/**/*.PNG'), recursive=True))


        self.images = self.images
        self.masks = self.masks
        if 'test123' in self.args.name:
            self.images = self.images[0:196]
            self.masks = self.masks[0:196]

        self.transform = transform
        self.transform_norm =  Compose([A.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalization mean (e.g., ImageNet stats)
                std=[0.229, 0.224, 0.225]  # Normalization std (e.g., ImageNet stats)
            )], seed=42)
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = np.array(Image.open(img_name))
        if self.mode == 'train' and self.labeled == False and self.sup == False:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.array(Image.open(self.masks[idx]))


        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        if self.mode == 'train':
            image = self.transform_norm(image=image)['image']

        mask = np.expand_dims(mask, axis=2).astype(float)

        image = np.moveaxis(image, -1, 0)
        mask = np.moveaxis(mask, -1, 0)

        sample = {
                  'image': torch.from_numpy(image).type(torch.FloatTensor),
                  'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                  'name': img_name
                  }

        return sample

    def get_image_path_list(self, split, args, mode, labeled=True, sup=False):
        image_path_list, mask_path_list = [], []
        base_dir = args.inputs

        if mode == 'train':
            if sup:
                filenames = split['train']['labeled']
            else:
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
            if mode == 'train' and labeled == False and sup == False:
                mask_path_list.append(None)
            else:
                mask_path_list.append(full_path_mask)

        return image_path_list, mask_path_list


