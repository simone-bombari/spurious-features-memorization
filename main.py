import numpy as np
import argparse
import os
import time

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--activation')
parser.add_argument('--fmap')
parser.add_argument('--i')
parser.add_argument('--dataset')
args = parser.parse_args()

time.sleep(int(args.i))

save_dir = os.path.join(args.dataset, args.fmap, args.activation)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if 'relu' in args.activation:
    phi = relu
    dev_phi = dev_relu
elif 'rf' in args.fmap:
    if 'phi2' in args.activation:
        phi = rf_phi2
    elif 'phi4' in args.activation:
        phi = rf_phi4
elif 'ntk' in args.fmap:
    if 'phi2' in args.activation:
        dev_phi = dev_ntk_phi2
    elif 'phi4' in args.activation:
        dev_phi = dev_ntk_phi4

width_noise_mnist = 10
width_noise_cifar = 8

if args.dataset == 'synthetic':
    d = 1000
elif args.dataset == 'MNIST':
    d = (28 + 2 * width_noise_mnist) ** 2
elif args.dataset == 'CIFAR-10':
    d = 3 * (32 + 2 * width_noise_cifar) ** 2



if 'rf' in args.fmap:
    k = 100000
elif 'ntk' in args.fmap:
    k = 100


Ns = [10, 20, 50, 100, 200, 500, 1000]
Nt = 1000

for N in Ns:
    # Dataset
    if args.dataset == 'synthetic':
        dy = d // 2
        dx = d - dy

        u = np.random.randn(dx) / np.sqrt(dx)

        X = np.random.randn(N, dx)
        Y = np.random.randn(N, dy)
        Z = np.concatenate((X, Y), axis=1)  # training data

        G = np.sign(X @ u)

        Xa = np.random.randn(N, dx)
        Za = np.concatenate((Xa, Y), axis=1)  # spurious data

        Xt = np.random.randn(Nt, dx)
        Yt = np.random.randn(Nt, dy)
        Zt = np.concatenate((Xt, Yt), axis=1)  # test data

        Gt = np.sign(Xt @ u)

    else:
        if args.dataset == 'MNIST':
            class1, class2 = 1, 7
        else:
            class1, class2 = 3, 8  # cats and ships

        Z, Zt, Za, G, Gt = load_datasets(args.dataset, N, class1, class2, Nt, flat=True)


    # Feature Map

    if 'rf' in args.fmap:
        V = np.random.randn(k, d) / np.sqrt(d)
        Phi = phi(Z @ V.transpose())
        Phi_a = phi(Za @ V.transpose())
        Phi_t = phi(Zt @ V.transpose())

    elif 'ntk' in args.fmap:
        W = np.random.randn(k, d) / np.sqrt(d)
        Phi = rwt(Z, dev_phi(Z @ W.transpose()))
        Phi_a = rwt(Za, dev_phi(Za @ W.transpose()))
        Phi_t = rwt(Zt, dev_phi(Zt @ W.transpose()))


    # Solution

    theta = np.linalg.pinv(Phi) @ G


    # Scores

    score_spurious = 0
    for i in range(N):
        if np.inner(Phi_a[i], theta) * G[i] > 0:
            score_spurious += 1
    score_spurious /= N

    score_test = 0
    for i in range(Nt):
        if np.inner(Phi_t[i], theta) * Gt[i] > 0:
            score_test += 1
    score_test /= Nt

    score_train = 0  # sanity check
    for i in range(N):
        if np.inner(Phi[i], theta) * G[i] > 0:
            score_train += 1
    score_train /= N


    with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
        f.write(str(d) + '\t' + str(k) + '\t' + str(N) + '\t' + str(score_train) + '\t' + str(score_test) + '\t' + str(score_spurious) + '\n')
