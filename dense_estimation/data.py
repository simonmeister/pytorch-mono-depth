from torch.utils.data import DataLoader


def get_training_loader(dset_class, root, batch_size, out_size,
                        num_threads=1, limit=None, debug=False, shuffle=True):
    dset = dset_class(root, split='train', transform=dset_class.get_transform(True, size=out_size),
                      limit=limit, debug=debug)
    return DataLoader(dset, shuffle=shuffle, batch_size=batch_size, pin_memory=True,
                      num_workers=num_threads)


def get_testing_loader(dset_class, root, batch_size, out_size,
                       num_threads=1, limit=None, debug=False, training=False, shuffle=False):
    dset = dset_class(root, split='test', transform=dset_class.get_transform(training, out_size),
                      limit=limit, debug=debug)
    return DataLoader(dset, shuffle=shuffle, batch_size=batch_size, num_workers=num_threads)
