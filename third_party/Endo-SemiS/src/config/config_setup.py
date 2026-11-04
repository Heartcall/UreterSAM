import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.core.composition import Compose
from dataloader.StoneData_semi import StoneData_semi
from dataloader.StoneData_unimatch import StoneData_unimatch
from dataloader.dataset_polygen.polygen_dataset_unimatch import PolyGenData_unimatch
from dataloader.dataset_polygen.polygen_dataset_semi import PolyGenData_semi
import torch.nn as nn
import random, os
import numpy as np
import logging
from torch import optim

def get_net(args, pretrain=False, model=None, net=None, aux_params=None, best=True, attn=None, data='polyp'):
    net = net
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')
    logging.info(f'Building:  {net}')

    if net.lower() == 'unet':
        net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                           in_channels=args.in_channels, classes=args.out_channels, aux_params=aux_params, attn=attn)
    else:
        raise ValueError('Unknown model type: %s', args.net)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    if pretrain:
        if model == 1:
            if best:
                pretrain_path = os.path.join(args.save_dir, 'cp', 'best_net1.pth')
            else:
                pretrain_path = os.path.join(args.save_dir, 'cp', 'last_epoch99_model1.pth')
        else:
            if best:
                pretrain_path = os.path.join(args.save_dir, 'cp', 'best_net2.pth')
            else:
                pretrain_path = os.path.join(args.save_dir, 'cp', 'last_epoch99_model2.pth')

        net.load_state_dict(
            torch.load(pretrain_path, map_location=device)
        )
        logging.info(f'Model{model}  loaded from {pretrain_path}')
    net.to(device=device)
    return net

def add_dropout(net):
    dropout_prob = 0.1
    for block in net.decoder.blocks:
        if isinstance(block.conv1, smp.base.modules.Conv2dReLU):
            block.conv1 = nn.Sequential(
                block.conv1[0],  # Conv2d
                block.conv1[1],  # BatchNorm2d
                nn.Dropout2d(p=dropout_prob, inplace=True),  # Add Dropout
                block.conv1[2]  # ReLU
            )
        if isinstance(block.conv2, smp.base.modules.Conv2dReLU):
            block.conv2 = nn.Sequential(
                block.conv2[0],  # Conv2d
                block.conv2[1],  # BatchNorm2d
                nn.Dropout2d(p=dropout_prob, inplace=True),  # Add Dropout
                block.conv2[2]  # ReLU
            )
    logging.info('added dropout = {}'.format(dropout_prob))
    return net

def get_optimizer_and_scheduler(args, net):
    params = filter(lambda p: p.requires_grad, net.parameters())  # added from
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    return optimizer, scheduler



def get_dataset(args, data='stone', mode=None, labeled=None, batch_sampler=False,
                cps=None, sup=None, w_s=False, unimatch=False, cm=False):

    if mode is None:
        raise ValueError('mode must be specified')


    if mode == 'train':
        transform = Compose([
            A.Resize(args.height, args.width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine((0.7, 1.3), {'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, rotate=(-360, 360), p=0.25),
        ], seed=42)

        transform_s = Compose([
            A.GaussianBlur(p=0.2),
            A.CLAHE(p=0.2),
            A.AutoContrast(p=0.2),
            A.MedianBlur(blur_limit=15, p=0.2),
            A.RandomGamma(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ], seed=42)
    else:
        transform = Compose([
            A.Resize(args.height, args.width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalization mean (e.g., ImageNet stats)
                std=[0.229, 0.224, 0.225]  # Normalization std (e.g., ImageNet stats)
            )], seed=42)
    if data == 'stone':
        if unimatch:
            dataset_kidney_stone = StoneData_unimatch(args, mode=mode, transform_w=transform, transform_s=transform_s, labeled=labeled)
        else:
            dataset_kidney_stone = StoneData_semi(args, mode=mode, transform=transform, labeled=labeled, sup=sup)
    elif data == 'polygen':
        if unimatch:
            dataset_kidney_stone = PolyGenData_unimatch(args, mode=mode, transform_w=transform, transform_s=transform_s,
                                                    labeled=labeled, sup=sup)
        else:
            dataset_kidney_stone = PolyGenData_semi(args, mode=mode, transform=transform, labeled=labeled, sup=sup)
    return dataset_kidney_stone


def init_seeds(seed=42, cuda_deterministic=True):
    from torch.backends import cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True