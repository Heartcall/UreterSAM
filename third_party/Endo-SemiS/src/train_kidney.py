import logging
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torch.utils.data import DataLoader
from util.Dice import dice_coeff
from config.config_args import *
from config.config_setup import add_dropout, get_dataset, init_seeds, get_optimizer_and_scheduler
from util.utils import *
from util.loss_functions import *
from torch.cuda.amp import GradScaler, autocast
from util.train_uncertainty_utils import *
from util.loss_functions import *
from torchmetrics.image import StructuralSimilarityIndexMeasure



def kl_loss_bottleneck(features1, features2):
    if features1.dim() > 2:
        B, C, H, W = features1.shape
        features1 = features1.view(B, C, -1)
        features2 = features2.view(B, C, -1)

    p1 = F.log_softmax(features1, dim=1)
    p2 = F.softmax(features2, dim=1)

    kl_loss = F.kl_div(p1, p2, reduction='batchmean')

    return kl_loss


def train_net(args, net1, net2, net3, dataset_labeled, dataset_unlabeled, valset, save_cp=True):
    n_val, n_train = len(valset), len(dataset_labeled)
    logging.info("Based on: {}  --->   total frames is: {}, labeled frames is: {}".format(args.json_path, n_train,
                                                                                          args.labeled_frames))

    def worker_init_fn(worker_id):
        random.seed(42 + worker_id)

    train_loader_labeled = DataLoader(dataset_labeled, batch_size=args.labeled_batch_size, pin_memory=True,
                                      num_workers=2, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    train_loader_unlabeled = DataLoader(dataset_unlabeled, batch_size=args.unlabeled_batch_size, pin_memory=True,
                                        num_workers=2, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    logging.info(f'''Starting training:
        Epochs:          {args.total_epoch}
        Batch size:      {args.batch_size}
        Labeled b_size:  {args.labeled_batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu').type}
    ''')

    optimizer1, scheduler1 = get_optimizer_and_scheduler(args, net1)
    optimizer2, scheduler2 = get_optimizer_and_scheduler(args, net2)
    optimizer3, scheduler3 = get_optimizer_and_scheduler(args, net3)

    criterion = nn.BCEWithLogitsLoss()

    best_dice1, best_dice2 = 0, 0

    for epoch in range(args.total_epoch):

        train(args, train_loader_labeled, train_loader_unlabeled, net1, net2, net3, criterion,
              optimizer1, optimizer2, optimizer3, scheduler1, scheduler2, scheduler3, epoch)

        mean_dice1, std_dice1, mean_dice_patch1, std_dice_patch1 = validate(net1, val_loader, args.device)
        mean_dice2, std_dice2, mean_dice_patch2, std_dice_patch2 = validate(net2, val_loader, args.device)

        logging.info('')
        logging.info('Model 1, batch-wise validation Dice coeff: {}, std: {}'.format(mean_dice1, std_dice1))
        logging.info('Model 1, batch-wise patch validation Dice coeff: {}, std: {}'.format(mean_dice_patch1, std_dice_patch1))
        logging.info('Model 2, batch-wise validation Dice coeff: {}, std: {}'.format(mean_dice2, std_dice2))
        logging.info('Model 2, batch-wise patch validation Dice coeff: {}, std: {}'.format(mean_dice_patch2, std_dice_patch2))
        logging.info('')

        if save_cp and mean_dice1 > best_dice1:
            save_checkpoint(net1, save_dir=args.save_dir, epoch=epoch, net1=True, best=True)
            best_dice1 = mean_dice1
        if save_cp and mean_dice2 > best_dice2:
            save_checkpoint(net2, save_dir=args.save_dir, epoch=epoch, net2=True, best=True)
            best_dice2 = mean_dice2

        torch.cuda.empty_cache()

    torch.save(net1.state_dict(), os.path.join(args.save_dir, 'cp', 'last_epoch{}_model1.pth'.format(epoch)))
    torch.save(net2.state_dict(), os.path.join(args.save_dir, 'cp', 'last_epoch{}_model2.pth'.format(epoch)))


def validate(net, loader, device):
    dice_list = []
    dice_list_patch = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                input, target = batch['image'].to(device=device, dtype=torch.float32), batch['mask'].to(device=device,
                                                                                                        dtype=torch.float32)
                output = net(input)

                output_prob = torch.sigmoid(output)
                output_pred = (output_prob > 0.5).float()

                target = target.detach().cpu()
                output_pred = output_pred.detach().cpu()

                input_patch, target_patch = split_tensor_into_patches(input), split_tensor_into_patches(target)
                output_patch = net(input_patch)
                output_patch_prob = torch.sigmoid(output_patch)
                output_patch_pred = (output_patch_prob > 0.5).float()

                target_patch = target_patch.detach().cpu()
                output_patch_pred = output_patch_pred.detach().cpu()

                dice_list.append(dice_coeff(output_pred, target).item())
                dice_list_patch.append(dice_coeff(output_patch_pred, target_patch).item())
                pbar.update(1)
            pbar.close()
        return np.mean(dice_list), np.std(dice_list), np.mean(dice_list_patch), np.std(dice_list_patch)


def train(args, train_loader_labeled, train_loader_unlabeled, model1, model2, model3, criterion,
          optimizer1, optimizer2, optimizer3, scheduler1, scheduler2, scheduler3, epoch):
    model1.train()
    model2.train()

    loss1_list_sup, loss1_list_consist, loss2_list_sup, loss2_list_consist = [], [], [], []

    pbar = tqdm(total=len(train_loader_labeled))

    loader = zip(train_loader_labeled, train_loader_unlabeled)

    criterion_pseudo = nn.BCEWithLogitsLoss(reduction='none')
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda(args.device)

    for batch_labeled, batch_unlabeled in loader:
        input_labeled = batch_labeled['image'].to(device=args.device, dtype=torch.float32)
        target_labeled = batch_labeled['mask'].to(device=args.device, dtype=torch.float32)

        input_unlabeled_w1 = batch_unlabeled['image_w1'].to(device=args.device, dtype=torch.float32)
        input_unlabeled_w2 = batch_unlabeled['image_w2'].to(device=args.device, dtype=torch.float32)
        input_unlabeled_s1 = batch_unlabeled['image_s1'].to(device=args.device, dtype=torch.float32)
        input_unlabeled_s2 = batch_unlabeled['image_s2'].to(device=args.device, dtype=torch.float32)
        cutmix_box = batch_unlabeled['cutmix_box'].to(device=args.device).unsqueeze(1)



        with torch.no_grad():
            model1.eval()
            model2.eval()

            pseudo_outputs1 = (torch.sigmoid(model1(input_labeled)).detach() > 0.5).float()
            pseudo_outputs2 = (torch.sigmoid(model2(input_labeled)).detach() > 0.5).float()

            mean_prediction_w1_model1, _, expected_entropy_uncertainty_w1_model1 = estimate_uncertainty(model1, input_unlabeled_w1)
            probability_w1_model1 = mean_prediction_w1_model1
            pseudo_label_w1_model1 = (probability_w1_model1.detach() > 0.5).float()

            mean_prediction_w1_mode2, _, expected_entropy_uncertainty_w1_model2 = estimate_uncertainty(model2, input_unlabeled_w1)
            probability_w1_model2 = mean_prediction_w1_mode2
            pseudo_label_w1_model2 = (probability_w1_model2.detach() > 0.5).float()


            mean_prediction_w2_model1, _, expected_entropy_uncertainty_w2_model1 = estimate_uncertainty(model1, input_unlabeled_w2)
            probability_w2_model1 = mean_prediction_w2_model1
            pseudo_label_w2_model1 = (probability_w2_model1.detach() > 0.5).float()

            mean_prediction_w2_model2, _, expected_entropy_uncertainty_w2_model2 = estimate_uncertainty(model2, input_unlabeled_w2)
            probability_w2_model2 = mean_prediction_w2_model2
            pseudo_label_w2_model2 = (probability_w2_model2.detach() > 0.5).float()

            probability_w1_model1[cutmix_box == 1] = probability_w2_model1[cutmix_box == 1]
            probability_w1_model2[cutmix_box == 1] = probability_w2_model2[cutmix_box == 1]

            pseudo_label_w1_model1[cutmix_box == 1] = pseudo_label_w2_model1[cutmix_box == 1]
            pseudo_label_w1_model2[cutmix_box == 1] = pseudo_label_w2_model2[cutmix_box == 1]

            uncertainty_w1_model1, uncertainty_w2_model1 = expected_entropy_uncertainty_w1_model1, expected_entropy_uncertainty_w2_model1
            uncertainty_w1_model2, uncertainty_w2_model2 = expected_entropy_uncertainty_w1_model2, expected_entropy_uncertainty_w2_model2

            #######################  uncertainty cutmix   #####################################################
            uncertainty_w1_model1[cutmix_box == 1] = uncertainty_w2_model1[cutmix_box == 1]
            threshold1 = get_threshold(uncertainty_w1_model1.detach())
            uncertainty1_binary = (uncertainty_w1_model1 <= threshold1).float()

            uncertainty_w1_model2[cutmix_box == 1] = uncertainty_w2_model2[cutmix_box == 1]
            threshold2 = get_threshold(uncertainty_w1_model2.detach())
            uncertainty2_binary = (uncertainty_w1_model2 <= threshold2).float()

        ################################################################################################################
        input_unlabeled_s1[cutmix_box.expand(input_unlabeled_s1.shape) == 1] = input_unlabeled_s2[cutmix_box.expand(input_unlabeled_s2.shape) == 1]
        input_unlabeled_w1[cutmix_box.expand(input_unlabeled_s1.shape) == 1] = input_unlabeled_w2[cutmix_box.expand(input_unlabeled_s2.shape) == 1]

        model1.train()
        model2.train()



        output1, output1_features = model1(input_labeled, return_features=True)
        loss1 = criterion(output1, target_labeled)

        output2, output2_features = model2(input_labeled, return_features=True)
        loss2 = criterion(output2, target_labeled)
        ########################################################################################################################

        output_w1_model1 = model1(input_unlabeled_w1)
        output_w1_model2 = model2(input_unlabeled_w1)
        output_s1_model1 = model1(input_unlabeled_s1)
        output_s1_model2 = model2(input_unlabeled_s1)
        ####################################################################################

        model1_unsup_w = criterion_pseudo(output_w1_model1, pseudo_label_w1_model2) * uncertainty2_binary # Shape: (B, C, H, W)
        model2_unsup_w = criterion_pseudo(output_w1_model2, pseudo_label_w1_model1) * uncertainty1_binary


        mask_joint = uncertainty_w1_model1 <= uncertainty_w1_model2  # True where model1 is more confident
        probability_w1_joint = torch.where(mask_joint, probability_w1_model1, probability_w1_model2)
        pseudo_label_w1_joint = (probability_w1_joint.detach() > 0.5).float()
        uncertainty_joint = torch.where(mask_joint, uncertainty_w1_model1, uncertainty_w1_model2).detach()


        threshold_joint = get_threshold(uncertainty_joint)
        uncertainty_joint_binary = (uncertainty_joint <= threshold_joint).float()

        model1_unsup_w_s = criterion_pseudo(output_s1_model1, pseudo_label_w1_joint) * uncertainty_joint_binary
        model2_unsup_w_s = criterion_pseudo(output_s1_model2, pseudo_label_w1_joint) * uncertainty_joint_binary


        pseudo_supervision1 = model1_unsup_w.mean() + model1_unsup_w_s.mean() + criterion(output1, pseudo_outputs2)
        pseudo_supervision2 = model2_unsup_w.mean() + model2_unsup_w_s.mean() + criterion(output2, pseudo_outputs1)


        consistency_weight = get_current_consistency_weight(args, epoch)



        KL_supervision1 = F.mse_loss(output1, output2) + 0.5 * (1 - ssim(output1_features[0], output2_features[0]) + kl_loss_bottleneck(output1_features[-1], output2_features[-1]))
        KL_supervision2 = F.mse_loss(output2, output1) + 0.5 * (1 - ssim(output2_features[0], output1_features[0]) + kl_loss_bottleneck(output2_features[-1], output1_features[-1]))


        model1_loss = loss1 + (consistency_weight * pseudo_supervision1) + KL_supervision1 * 0.5
        model2_loss = loss2 + (consistency_weight * pseudo_supervision2) + KL_supervision2 * 0.5

        loss = model1_loss + model2_loss

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward()
        optimizer1.step()
        optimizer2.step()

        loss1_list_sup.append(loss1.item())
        loss1_list_consist.append((consistency_weight * pseudo_supervision1).item())

        loss2_list_sup.append(loss2.item())
        loss2_list_consist.append((consistency_weight * pseudo_supervision2).item())

        pbar.update(1)
    logging.info('===================================================================================')
    logging.info(
        'Epoch: {}, model1 supervised loss: {}, consistency loss: {}'.format(epoch + 1, np.mean(loss1_list_sup),
                                                                             np.mean(loss1_list_consist)))
    logging.info(
        'Epoch: {}, model2 supervised loss: {}, consistency loss: {}'.format(epoch + 1, np.mean(loss2_list_sup),
                                                                             np.mean(loss2_list_consist)))

    pbar.close()

    checkpoint_dict = {
        'epoch': epoch + 1,
        'model1_state_dict': model1.state_dict(),
        'optimizer1_state_dict': optimizer1.state_dict(),
        # 'mean loss': np.mean(loss1_list_sup),
        'scheduler1_state_dict': scheduler1.state_dict(),

        'model2_state_dict': model2.state_dict(),
        'optimizer2_state_dict': optimizer2.state_dict(),
        # 'mean loss': np.mean(loss1_list_sup),
        'scheduler2_state_dict': scheduler2.state_dict(),
    }
    save_checkpoint(model1, net_dict=checkpoint_dict, save_dir=args.save_dir, epoch=epoch, best=False)
    logging.info(f"Epoch {epoch + 1}, learning rate: {scheduler1.get_last_lr()}")
    scheduler1.step()
    scheduler2.step()


if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='train')
    logging.info(os.path.abspath(__file__))
    logging.info(args)

    try:
        assert args.labeled_batch_size == int(args.batch_size / 2)
    except AssertionError:
        print("Warning: labeled_batch_size is not half of batch_size!")

    args.unlabeled_batch_size = args.batch_size - args.labeled_batch_size

    dataset_labeled = get_dataset(args, mode='train', labeled=True)
    dataset_unlabeled = get_dataset(args, mode='train', labeled=False, w_s=False, unimatch=True)
    valset = get_dataset(args, mode='val')


    from models.unet import Unet
    net1 = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=args.in_channels, classes=args.out_channels).cuda(args.device)
    net2 = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=args.in_channels, classes=args.out_channels).cuda(args.device)
    from models.unet_encoder import Unet_encoder
    net3 = Unet_encoder(encoder_name='resnet34', encoder_weights='imagenet', in_channels=args.in_channels, classes=args.out_channels).cuda(args.device)
    net1, net2 = add_dropout(net1), add_dropout(net2)


    logging.info('Models and datasets are loaded')

    logging.info('Training Endo-SemiS...')
    train_net(args, net1=net1, net2=net2, net3=net3, dataset_labeled=dataset_labeled, dataset_unlabeled=dataset_unlabeled,
              valset=valset)
