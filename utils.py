import numpy as np
import torch
import torchvision
import pandas as pd
import csv
import os


def rf_phi2(a):  # h1 + h2
    vec_phi = np.vectorize(lambda x: x + (x ** 2 - 1) / np.sqrt(2))
    return vec_phi(a)

def rf_phi4(a):  # h1 + h4
    vec_phi = np.vectorize(lambda x: x + (x ** 4 - 6 * x ** 2 + 3) / (np.sqrt(24)))
    return vec_phi(a)


def dev_ntk_phi2(a):  # h0 + h1
    vec_dev_phi = np.vectorize(lambda x: 1 + x)
    return vec_dev_phi(a)

def dev_ntk_phi4(a):  # h0 + h3
    vec_dev_phi = np.vectorize(lambda x: 1 + (1 / np.sqrt(6)) * (x ** 3 - 3 * x))
    return vec_dev_phi(a)


def relu(a):
    vec_relu = np.vectorize(lambda x: x * (x > 0))
    return vec_relu(a)

def dev_relu(a):
    vec_dev_relu = np.vectorize(lambda x: float(x > 0))
    return vec_dev_relu(a)


def rwt(a, b):
    out = []
    if len(a) != len(b):
        print('Error')
        return None
    for i in range(len(a)):
        out.append(np.kron(a[i], b[i]))
    return np.array(out)


def import_in_df(folder, activation):

    data = []
    my_folder = folder + activation

    for filename in os.listdir(my_folder):
        if '.txt' in filename:
            with open(os.path.join(my_folder, filename), 'r') as f:
                reader = csv.reader(f,  delimiter='\t')
                for row in reader:
                    new_row = []
                    for j in range(3):
                        new_row.append(int(row[j]))
                    for j in range(3, 6):
                        new_row.append(float(row[j]))
                    data.append(new_row)

    df = pd.DataFrame(data=data, columns=(['d', 'k', 'N', 'score_train', 'score_test', 'score_spurious']))
    df['activation'] = activation
    
    return df


def import_in_df_copy(folder, activation):

    data = []
    my_folder = folder + activation

    for filename in os.listdir(my_folder):
        if '.txt' in filename:
            with open(os.path.join(my_folder, filename), 'r') as f:
                reader = csv.reader(f,  delimiter='\t')
                for row in reader:
                    new_row = []
                    for j in range(4):
                        new_row.append(int(row[j]))
                    for j in range(4, 5):
                        new_row.append(row[j])
                    for j in range(5, 6):
                        new_row.append(int(row[j]))
                    for j in range(6, 9):
                        new_row.append(float(row[j]))
                    data.append(new_row)

    df = pd.DataFrame(data=data, columns=(['d', 'k', 'N', 'alpha_min_1', 'fmap', 'phi', 'score_train', 'score_test', 'score_spurious']))
    df['activation'] = activation
    
    return df


def load_datasets(dataset, N, class1, class2, Nt, flat=True):
    
    if 'MNIST' in dataset:
        dataset = 'MNIST'
    elif 'CIFAR-10' in dataset:
        dataset = 'CIFAR-10'
    
    train_dir = './data/' + os.path.join(dataset, 'train')
    test_dir = './data/' + os.path.join(dataset, 'test')

    # train
    augmented_data_train = np.load(train_dir + '/augmented_data.npy')
    spurious_data = np.load(train_dir + '/spurious_data.npy')
    labels_train = np.load(train_dir + '/labels.npy')

    # mask
    mask = (labels_train == class1) | (labels_train == class2)

    augmented_data_train = augmented_data_train[mask]
    spurious_data = spurious_data[mask]
    labels_train = labels_train[mask]

    # flatten and center
    if flat:
        augmented_data_train = augmented_data_train.reshape(augmented_data_train.shape[0], -1)
        spurious_data = spurious_data.reshape(spurious_data.shape[0], -1)
    labels_train[labels_train==class1] = -1
    labels_train[labels_train==class2] = 1 
    

    # test
    augmented_data_test = np.load(test_dir + '/augmented_data.npy')
    labels_test = np.load(test_dir + '/labels.npy')

    # mask
    mask = (labels_test == class1) | (labels_test == class2)

    augmented_data_test = augmented_data_test[mask]
    labels_test = labels_test[mask]

    # flatten and center
    if flat:
        augmented_data_test = augmented_data_test.reshape(augmented_data_test.shape[0], -1)
    labels_test[labels_test==class1] = -1
    labels_test[labels_test==class2] = 1
    
    indices = np.random.choice(range(10000), size=N, replace=False, p=None)
    indices_t = np.random.choice(range(2000), size=Nt, replace=False, p=None)


    return augmented_data_train[indices], augmented_data_test[indices_t], spurious_data[indices], labels_train[indices], labels_test[indices_t]
