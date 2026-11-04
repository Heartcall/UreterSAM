import torch.nn as nn
from dataloader.StoneData_semi import StoneData_semi
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
import numpy as np
import torch
from tqdm import tqdm
from util.Dice import dice_coeff
from config.config_args import *
from config.config_setup import get_net, get_dataset, init_seeds, get_optimizer_and_scheduler
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


def validate_baseline(net1, loader, device, activation=nn.Sigmoid()):
    net1.eval()
    dice_list1 = []
    pred_list1, gt_list = [], []
    sen1, spe1 = [], []
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                input, target = batch['image'].to(device=device, dtype=torch.float32), batch['mask'].to(device=device, dtype=torch.float32)
                output1 = activation(net1(input))

                target = target.detach().cpu()
                output1 = output1.detach().cpu()

                pred_labels1 = (output1 > 0.5).float()

                dice_score1 = dice_coeff(pred_labels1, target).item()
                dice_list1.append(dice_score1)

                _, _, sensitivity1, specificity1 = compute_fp_fn(pred_labels1, target)
                sen1.append(sensitivity1)
                spe1.append(specificity1)

                save_base_dir = '/home/hao/Hao/kidney_stone_results'

                save_dataset = batch['name'][0].split('/')[-2]
                save_name = batch['name'][0].split('/')[-1]
                save_dir_gt = os.path.join(save_base_dir, args.name, 'gt', save_dataset)
                save_dir_model1 = os.path.join(save_base_dir, args.name, 'model1', save_dataset)
                os.makedirs(save_dir_gt, exist_ok=True)
                os.makedirs(save_dir_model1, exist_ok=True)
                gt_save = target.squeeze(0).squeeze(0).numpy().astype('uint8')
                pred1_save = pred_labels1.squeeze(0).squeeze(0).numpy().astype('uint8')
                np.save(os.path.join(save_dir_gt, save_name), gt_save)
                np.save(os.path.join(save_dir_model1, save_name), pred1_save)

                pbar.update(1)
            pbar.close()

        accuracy_1 = accuracy_score(gt_list, pred_list1)
        precision_1 = precision_score(gt_list, pred_list1)
        recall_1 = recall_score(gt_list, pred_list1)
        f1 = f1_score(gt_list, pred_list1)

        logging.info('Model 1, batch-wise validation accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy_1,
                                                                                                             precision_1,
                                                                                                             recall_1,
                                                                                                             f1))
        logging.info('Model 1, batch-wise validation sensitivity coeff: {}, std: {}, median: {}'.format(np.mean(sen1),
                                                                                                        np.std(sen1),
                                                                                                        np.median(
                                                                                                            sen1)))
        logging.info('Model 1, batch-wise validation specificity coeff: {}, std: {}, median: {}'.format(np.mean(spe1),
                                                                                                        np.std(spe1),
                                                                                                        np.median(
                                                                                                            spe1)))
        return dice_list1


def test_net_baseline(args, net1, dataset, batch_size=1):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    logging.info(f'''Starting testing:
            Num test:        {len(dataset)}
            Batch size:      {batch_size}
            Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')}
        ''')

    net1.eval()

    dice_list1 = validate_baseline(net1, test_loader, args.device, activation=nn.Sigmoid())
    logging.info('Model 1, batch-wise validation Dice coeff: {}, std: {}'.format(np.mean(dice_list1), np.std(dice_list1)))




if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='test')
    logging.info(os.path.dirname(os.path.abspath(__file__)))
    logging.info(args)

    dataset = get_dataset(args, data='polygen', mode='test')

    net1 = get_net(args, net=args.net1, pretrain=True, model=1)
    test_net_baseline(args,
             net1=net1,
             dataset=dataset)

