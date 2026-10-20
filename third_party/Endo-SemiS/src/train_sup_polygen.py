import random
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from util.Dice import dice_coeff
from config.config_args import *
from config.config_setup import get_net, get_dataset, init_seeds, get_optimizer_and_scheduler
from util.utils import *
from util.loss_functions import *


def train_net_sup(args, net1, dataset_labeled, valset, save_cp=True):
    n_val, n_train = len(valset), len(dataset_labeled)
    logging.info("Based on: {}  --->   total frames is: {}".format(args.json_path, n_train))

    def worker_init_fn(worker_id):
        random.seed(42 + worker_id)

    train_loader_labeled = DataLoader(dataset_labeled, batch_size=args.batch_size, pin_memory=True,
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

    criterion = nn.BCEWithLogitsLoss()

    best_dice1, best_dice2 = 0, 0

    for epoch in range(args.total_epoch):

        train_sup(args, train_loader_labeled, net1, criterion, optimizer1, epoch)

        mean_dice1, std_dice1 = validate_sup(net1, val_loader, args.device)

        logging.info('')
        logging.info('Model 1, batch-wise validation Dice coeff: {}, std: {}'.format(mean_dice1, std_dice1))
        logging.info(f"Epoch {epoch + 1}, learning rate: {scheduler1.get_last_lr()}")
        logging.info('===================================================================================')
        scheduler1.step()

        if save_cp and mean_dice1 > best_dice1:
            save_checkpoint(net1, args.save_dir, epoch, net1=True, best=True)
            best_dice1 = mean_dice1

        torch.cuda.empty_cache()

    torch.save(net1.state_dict(), os.path.join(args.save_dir, 'cp', 'last_epoch{}_model1.pth'.format(epoch)))


def validate_sup(net, loader, device):
    dice_list = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                input, target = batch['image'].to(device=device, dtype=torch.float32), batch['mask'].to(device=device, dtype=torch.float32)
                output = net(input)

                output_prob = torch.sigmoid(output)
                output_pred = (output_prob > 0.5).float()

                target = target.detach().cpu()
                output_pred = output_pred.detach().cpu()

                dice_list.append(dice_coeff(output_pred, target).item())
                pbar.update(1)
            pbar.close()
        return np.mean(dice_list), np.std(dice_list)


def train_sup(args, train_loader_labeled, model1, criterion, optimizer1, epoch):
    model1.train()

    loss1_list_sup = []

    pbar = tqdm(total=len(train_loader_labeled))

    loader = train_loader_labeled

    for batch_labeled in loader:
        input_labeled = batch_labeled['image'].to(device=args.device, dtype=torch.float32)
        target_labeled = batch_labeled['mask'].to(device=args.device, dtype=torch.float32)

        model1.train()
        output1 = model1(input_labeled)
        loss = criterion(output1, target_labeled)

        optimizer1.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer1.step()
        loss1_list_sup.append(loss.item())
        pbar.update(1)

    logging.info('===================================================================================')
    logging.info('Epoch: {}, model1 supervised loss: {}'.format(epoch, np.mean(loss1_list_sup)))

    pbar.close()


if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='train')
    logging.info(os.path.abspath(__file__))
    logging.info(args)

    dataset_labeled = get_dataset(args, data='polygen', mode='train', labeled=True, sup=True)
    valset = get_dataset(args, data='polygen', mode='val')

    net1 = get_net(args, net=args.net1)
    logging.info('Models and datasets are loaded')

    logging.info('Training PolyGen full supervision...')
    train_net_sup(args, net1=net1, dataset_labeled=dataset_labeled, valset=valset)
