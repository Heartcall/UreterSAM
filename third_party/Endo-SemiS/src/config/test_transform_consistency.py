import albumentations as A
import cv2
from albumentations.core.composition import Compose
transform = Compose([
            A.Resize(512, 512),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.25),
        ], seed=42)

transform_s = Compose([
            A.ColorJitter(p=0.25),
            A.CLAHE(p=0.25),
            A.AutoContrast(p=0.25),
            A.MedianBlur(blur_limit=15, p=0.25),
            A.ToGray(num_output_channels=3, p=0.25),
            A.RGBShift(p=0.25),
            A.RandomGamma(p=0.25),
            A.RandomBrightnessContrast(p=0.25)
        ], seed=42)

import numpy as np
image = cv2.imread('/media/hao/mydrive1/arpa-h_tanner_mariana/CAO_data/val/images/0.jpeg')
mask = np.zeros((image.shape[0], image.shape[1]))

image1 = transform_s(image=image)['image']
image2 = transform_s(image=image)['image']
image3 = transform_s(image=image)['image']
image4 = transform_s(image=image)['image']
image5 = transform_s(image=image)['image']


b = np.sum(image1- image2)
a = np.sum(image - image1)
c = np.sum(image2 - image3)
d = np.sum(image2 - image4)
e = np.sum(image2 - image5)


print(1)
