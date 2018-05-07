import os
import shutil


def get_experiment(name, overwrite, epoch=None):
    log_dir = os.path.join('./log', name)
    save_dir = os.path.join('/media/data/depth-prediction/checkpoints', name)
    if overwrite: # or (os.path.isdir(log_dir) and not os.path.isdir(save_dir)):
        shutil.rmtree(log_dir, ignore_errors=True)
        shutil.rmtree(save_dir, ignore_errors=True)
    if not os.path.isdir(save_dir):
        os.makedirs(log_dir)
        os.makedirs(save_dir)
    save_paths = sorted(os.listdir(save_dir),
                        key=lambda s: int(s.split('.')[0].split('_')[1]))
    if len(save_paths) > 0:
        save_path = save_paths[-1] if epoch is None else 'model_{}.pth'.format(epoch)
        restore_path = os.path.join(save_dir, save_path)
        starting_epoch = int(save_path.split('.')[0].split('_')[1]) + 1
    else:
        restore_path = None
        starting_epoch = 0
    return log_dir, save_dir, restore_path, starting_epoch
