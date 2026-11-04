import albumentations as A
from albumentations.core.composition import Compose
import torch.nn as nn
from torch.distributed.checkpoint import load_state_dict

from config.config_setup import get_net
from dataloader.StoneData_semi import StoneData_semi
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
from util.Dice import dice_coeff
from config.config_args import *
from config.config_setup import get_net, get_dataset, init_seeds, get_optimizer_and_scheduler
from util.train_uncertainty_utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_fp_fn(pred_mask, gt_mask):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Ensure binary format
    # gt_mask = (gt_mask > 0.5).astype(np.uint8)

    # Compute FP, FN, TP, TN masks
    fp_mask = (pred_mask == 1) & (gt_mask == 0)  # False Positives
    fn_mask = (pred_mask == 0) & (gt_mask == 1)  # False Negatives
    tp_mask = (pred_mask == 1) & (gt_mask == 1)  # True Positives
    tn_mask = (pred_mask == 0) & (gt_mask == 0)  # True Negatives

    # Count values
    fp_count = int(np.sum(fp_mask.astype(np.uint8)))
    fn_count = int(np.sum(fn_mask.astype(np.uint8)))
    tp_count = int(np.sum(tp_mask.astype(np.uint8)))
    tn_count = int(np.sum(tn_mask.astype(np.uint8)))

    # Compute Sensitivity and Specificity
    sensitivity = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1
    specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 1

    return fp_count, fn_count, sensitivity, specificity


def validate(net1, net2, loader, device, activation=nn.Sigmoid(), save=False):
    net1.eval()
    net2.eval()
    dice_list1, dice_list2, dice_list_ensemble = [], [], []
    pred_list1, pred_list2, pred_list_e, gt_list = [], [], [], []
    sen1, spe1 = [], []
    sen2, spe2 = [], []
    sen3, spe3 = [], []
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                #unmod, input, target, names = batch['unmod'], batch['image'], batch['mask'], batch['name']
                # input = input.to(device=device, dtype=torch.float32)
                # target = target.to(device=device, dtype=torch.float32)
                input, target = batch['image'].to(device=device, dtype=torch.float32), batch['mask'].to(device=device, dtype=torch.float32)

                output1 = activation(net1(input))
                output2 = activation(net2(input))

                target = target.detach().cpu()
                output1 = output1.detach().cpu()
                output2 = output2.detach().cpu()

                pred_labels1 = (output1 > 0.5).float()
                pred_labels2 = (output2 > 0.5).float()

                dice_score1 = dice_coeff(pred_labels1, target).item()
                dice_score2 = dice_coeff(pred_labels2, target).item()

                output_ensemble = output1 * 0.5 + output2 * 0.5
                pred_ensemble = (output_ensemble > 0.5).float()
                dice_ensemble = dice_coeff(pred_ensemble, target).item()

                dice_list1.append(dice_score1)
                dice_list2.append(dice_score2)
                dice_list_ensemble.append(dice_ensemble)

                _, _, sensitivity1, specificity1 = compute_fp_fn(pred_labels1, target)
                sen1.append(sensitivity1)
                spe1.append(specificity1)
                _, _, sensitivity2, specificity2 = compute_fp_fn(pred_labels2, target)
                sen2.append(sensitivity2)
                spe2.append(specificity2)
                _, _, sensitivity3, specificity3 = compute_fp_fn(pred_ensemble, target)
                sen3.append(sensitivity3)
                spe3.append(specificity3)
                if save:
                    save_base_dir = '/home/hao/Hao/kidney_stone_results'

                    save_dataset = batch['name'][0].split('/')[-2]
                    save_name = batch['name'][0].split('/')[-1]

                    save_dir_gt = os.path.join(save_base_dir, args.name, 'gt', save_dataset)
                    save_dir_model1 = os.path.join(save_base_dir, args.name, 'model1', save_dataset)
                    save_dir_model2 = os.path.join(save_base_dir, args.name, 'model2', save_dataset)
                    save_dir_model_ensemble = os.path.join(save_base_dir, args.name, 'model_ensemble', save_dataset)

                    os.makedirs(save_dir_gt, exist_ok=True)
                    os.makedirs(save_dir_model1, exist_ok=True)
                    os.makedirs(save_dir_model2, exist_ok=True)
                    os.makedirs(save_dir_model_ensemble, exist_ok=True)


                    gt_save = target.squeeze(0).squeeze(0).numpy().astype('uint8')
                    pred1_save = pred_labels1.squeeze(0).squeeze(0).numpy().astype('uint8')
                    pred2_save = pred_labels2.squeeze(0).squeeze(0).numpy().astype('uint8')
                    ensemble_save = pred_ensemble.squeeze(0).squeeze(0).numpy().astype('uint8')


                    np.save(os.path.join(save_dir_gt, save_name), gt_save)
                    np.save(os.path.join(save_dir_model1, save_name), pred1_save)
                    np.save(os.path.join(save_dir_model2, save_name), pred2_save)
                    np.save(os.path.join(save_dir_model_ensemble, save_name), ensemble_save)

                classification_labels = (target.view(target.size()[0], -1).sum(dim=1) > 0).float().unsqueeze(
                    1).long().item()
                pred_1 = (pred_labels1.view(pred_labels1.size()[0], -1).sum(dim=1) > 0).float().unsqueeze(
                    1).long().item()
                pred_2 = (pred_labels2.view(pred_labels2.size()[0], -1).sum(dim=1) > 0).float().unsqueeze(
                    1).long().item()

                pred_ensemble = (pred_ensemble.view(pred_ensemble.size()[0], -1).sum(dim=1) > 0).float().unsqueeze(
                    1).long().item()
                pred_list1.append(pred_1)
                pred_list2.append(pred_2)
                pred_list_e.append(pred_ensemble)

                gt_list.append(classification_labels)

                pbar.update(1)
            pbar.close()
        accuracy_1 = accuracy_score(gt_list, pred_list1)
        precision_1 = precision_score(gt_list, pred_list1)
        recall_1 = recall_score(gt_list, pred_list1)
        f1 = f1_score(gt_list, pred_list1)

        accuracy_2 = accuracy_score(gt_list, pred_list2)
        precision_2 = precision_score(gt_list, pred_list2)
        recall_2 = recall_score(gt_list, pred_list2)
        f2 = f1_score(gt_list, pred_list2)

        accuracy_3 = accuracy_score(gt_list, pred_list_e)
        precision_3 = precision_score(gt_list, pred_list_e)
        recall_3 = recall_score(gt_list, pred_list_e)
        f3 = f1_score(gt_list, pred_list_e)
        logging.info('Model 1, batch-wise validation accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy_1,
                                                                                                             precision_1,
                                                                                                             recall_1,
                                                                                                             f1))
        logging.info('Model 2, batch-wise validation accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy_2,
                                                                                                             precision_2,
                                                                                                             recall_2,
                                                                                                             f2))

        logging.info('Model 1+2, batch-wise validation accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy_3,
                                                                                                             precision_3,
                                                                                                             recall_3,
                                                                                                             f3))

        logging.info('Model 1, batch-wise validation sensitivity coeff: {}, std: {}, median: {}'.format(np.mean(sen1), np.std(sen1), np.median(sen1)))
        logging.info('Model 2, batch-wise validation sensitivity coeff: {}, std: {}, median: {}'.format(np.mean(sen2), np.std(sen2), np.median(sen2)))
        logging.info('Model 1+2, batch-wise validation specificity coeff: {}, std: {}, median: {}'.format(np.mean(spe3), np.std(spe3), np.median(spe3)))
        return dice_list1, dice_list2, dice_list_ensemble


def test_net(args, net1, net2, dataset, batch_size=1, img_scale=1):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    logging.info(f'''Starting testing:
            Num test:        {len(dataset)}
            Batch size:      {batch_size}
            Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')}
            Images scaling:  {img_scale}
        ''')

    net1.eval()
    net2.eval()

    dice_list1, dice_list2, dice_list_ensemble = validate(net1, net2, test_loader, args.device, activation=nn.Sigmoid(), save=args.save)
    logging.info(
        'Model 1, batch-wise validation Dice coeff: {}, std: {}'.format(np.mean(dice_list1), np.std(dice_list1)))
    logging.info(
        'Model 2, batch-wise validation Dice coeff: {}, std: {}'.format(np.mean(dice_list2), np.std(dice_list2)))
    logging.info('Model ensemble, batch-wise validation Dice coeff: {}, std: {}'.format(np.mean(dice_list_ensemble),
                                                                                        np.std(dice_list_ensemble)))


if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='test')
    logging.info(os.path.dirname(os.path.abspath(__file__)))
    logging.info(args)

    dataset = StoneData_semi(args, mode=args.mode, transform=Compose([Resize(args.height, args.width),
                                                                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                                      ]))
    # net1 = get_net(args, net=args.net1, pretrain=True, model=1)
    # net2 = get_net(args, net=args.net2, pretrain=True, model=2)

    from models.unet import Unet
    net1 = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=args.in_channels, classes=args.out_channels).cuda(args.device)
    net2 = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=args.in_channels, classes=args.out_channels).cuda(args.device)


    checkpoints = f'./checkpoints/{args.name}/cp/best_net1.pth'
    net1.load_state_dict(torch.load(checkpoints))
    net2.load_state_dict(torch.load(checkpoints.replace('best_net1', 'best_net2')))

    logging.info(f'Model{1}  loaded from {checkpoints}')



    logging.info('Models and datasets are loaded')

    test_net(args,
             net1=net1,
             net2=net2,
             dataset=dataset)