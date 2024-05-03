import os
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
import torch

for data_name, call_data in [('MNIST', MNIST), ('CIFAR-10', CIFAR10)]:
    for train_name, train in [('train', True), ('test', False)]:
        
        print(data_name, train_name)
        
        dataset = call_data(download=True, train=train, root='./data')

        if data_name == 'MNIST':
            data = dataset.data.numpy()
            labels = dataset.targets.numpy()
            width_noise = 10
        elif data_name == 'CIFAR-10':
            data = dataset.data
            labels = dataset.targets
            width_noise = 8
        
        
        shape = data.shape
        new_shape = tuple(val + 2 * width_noise if i in (1, 2) else val for i, val in enumerate(shape))

        augmented_data = np.random.randint(0, 256, size=new_shape)
        augmented_data[:, width_noise:new_shape[1]-width_noise, width_noise:new_shape[2]-width_noise] = data

        spurious_data = np.copy(augmented_data)
        spurious_data[:, width_noise:new_shape[1]-width_noise, width_noise:new_shape[2]-width_noise] = np.zeros(shape)

        save_dir = './data/' + os.path.join(data_name, train_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        np.save(save_dir + '/augmented_data.npy', augmented_data)
        np.save(save_dir + '/labels.npy', labels)
        
        if train:
            np.save(save_dir + '/spurious_data.npy', spurious_data)
