import os, torch, logging


def save_checkpoint(net, save_dir, epoch, net1=False, net2=False, net_dict=None, best=False):
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint_dir = os.path.join(save_dir, 'cp')
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    if best:
        if net1:
            torch.save(net.state_dict(), os.path.join(save_checkpoint_dir, 'best_net1.pth'))
            logging.info(f'best Checkpoint net1 {epoch + 1} saved !')
        if net2:
            torch.save(net.state_dict(), os.path.join(save_checkpoint_dir, 'best_net2.pth'))
            logging.info(f'best Checkpoint net2 {epoch + 1} saved !')
    else:
        torch.save(net_dict, os.path.join(save_checkpoint_dir, 'last_net.pth'))
        #logging.info(f'last Checkpoint net {epoch + 1} saved !')



