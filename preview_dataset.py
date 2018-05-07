import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from dense_estimation.datasets.nyu_depth_v2 import NYU_Depth_V2


dset = NYU_Depth_V2("/home/smeister/datasets", split='train',
                    transform=NYU_Depth_V2.get_transform(normalize=False))
#print(dset.compute_image_std(), dset.compute_image_mean())
trainloader = data.DataLoader(dset, batch_size=4)
for i, data in enumerate(trainloader):
    imgs, labels = data
    if i == 0:
        # TODO make_grid is currently broken
        #img = torchvision.utils.make_grid([imgs, labels]).numpy()
        #img = np.transpose(img, (1, 2, 0))
        #img = img[:, :, ::-1]
        #plt.imshow(img)
        print(np.transpose(labels[0, 1].numpy(), (0, 1)))
        plt.imshow(np.transpose(imgs.numpy()[1] , (1, 2, 0)))
        plt.figure()
        plt.imshow(np.transpose(labels[1, 0].numpy(), (0, 1)), cmap='gray')
        plt.figure()
        plt.imshow(np.transpose(labels[1, 1].numpy(), (0, 1)), cmap='gray')
        plt.show()
