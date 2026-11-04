import argparse
import os
import warnings
import logging
import datetime

parser = argparse.ArgumentParser()


# general setting
parser.add_argument('--labels', type=str, default='/home/hao/Hao/data/kidney_stone_r21_data/')
parser.add_argument('--inputs', type=str, default='/home/hao/Hao/data/kidney_stone_r21_data/')
parser.add_argument('--json_path', type=str, default='/home/hao/Hao/kidney_stone/src/exp1/get_sub_semi_dataset/dataset_info_semi.json')

parser.add_argument('--mode', type=str, default='test')

parser.add_argument('--total_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--labeled_batch_size', type=int, default=24)

parser.add_argument('--labeled_frames', type=int, default=5599) # default number for kidney semi30 setting
parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-5)

parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)

parser.add_argument('--name', dest='name', type=str, default='test123')
parser.add_argument('--base_dir', type=str, default='checkpoints', help='Base dir to save checkpoints and images')

parser.add_argument('--save', action='store_true')

# network setting
parser.add_argument('--net1', type=str, default='unet')
parser.add_argument('--net2', type=str, default='unet')
parser.add_argument('--attn', action='store_true', help='add attention modules')
parser.add_argument('--pretrain', action='store_true')

parser.add_argument('--device', type=int, default=1)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=1)

# consistency setting
parser.add_argument('--consistency_rampup', type=int, default=5)
parser.add_argument('--consistency', type=float, default=0.5)






def setup_logging(args, mode='train'):
    log_dir = os.path.join(args.base_dir, args.name, 'logs', mode)
    os.makedirs(log_dir, exist_ok=True)

    # Get the current time and format it for the log file name
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    mode_prefix = 'training' if mode == 'train' else 'test'
    log_file = os.path.join(log_dir, f'{mode_prefix}_{start_time}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    args.save_dir = os.path.join(args.base_dir, args.name)
    logging.info("Save directory is: {}".format(args.save_dir))
    logging.info(f"Logging setup complete. Logs will be saved in {log_file}")



