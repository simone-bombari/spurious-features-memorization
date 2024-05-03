import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import torch.optim as optim
import os
import time
from utils_NN import loss_calculator, compute_loss_accuracy_regressor, compute_loss_accuracy_classifier
from utils_NN import FullyConnected, SmallConvNet, load_full_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--i')
parser.add_argument('--dataset')
parser.add_argument('--save')
parser.add_argument('--net')
args = parser.parse_args()

time.sleep(int(args.i))

save_dir = os.path.join(args.dataset, args.net, args.save)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


noise_width_mnist = 10
noise_width_cifar = 8

if args.dataset == 'synthetic':
    d = 1000
elif args.dataset == 'MNIST':
    noise_width = noise_width_mnist
    d = (28 + 2 * noise_width) ** 2
elif args.dataset == 'CIFAR-10':
    noise_width = noise_width_cifar
    d = 3 * (32 + 2 * noise_width) ** 2


Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
Nt = 10000


for N in Ns:
    # Dataset

    if 'synthetic' in args.dataset:
        dx = d // 2
        dy = d // 2

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
        loss_function = 'MSE'
        compute_loss_accuracy = compute_loss_accuracy_regressor
        
        augmented_train_dataset = TensorDataset(torch.from_numpy(Z).to(torch.float32), torch.from_numpy(G).to(torch.float32))
        spurious_train_dataset = TensorDataset(torch.from_numpy(Za).to(torch.float32), torch.from_numpy(G).to(torch.float32))
        augmented_test_dataset = TensorDataset(torch.from_numpy(Zt).to(torch.float32), torch.from_numpy(Gt).to(torch.float32))
        
    else:
        Z, Zt, Za, G, Gt = load_full_dataset(args.dataset, N, Nt)
        loss_function = 'CEL'
        compute_loss_accuracy = compute_loss_accuracy_classifier
        
        augmented_train_dataset = TensorDataset(torch.from_numpy(Z).to(torch.float32), torch.from_numpy(G))
        spurious_train_dataset = TensorDataset(torch.from_numpy(Za).to(torch.float32), torch.from_numpy(G))
        augmented_test_dataset = TensorDataset(torch.from_numpy(Zt).to(torch.float32), torch.from_numpy(Gt))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, flush=True)

    lr = 0.001
    batch_size = 1024

    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    spurious_loader = DataLoader(spurious_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(augmented_test_dataset, batch_size=batch_size, shuffle=True)

    epochs = 1000
    
    if args.net == 'FC':
        if args.dataset == 'synthetic':
            net = FullyConnected(args.dataset, d)
        else:
            net = FullyConnected(args.dataset, noise_width)
    elif args.net == 'SCN':
        net = SmallConvNet(args.dataset, noise_width)
        
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    for epoch in range(epochs):

        net.train()            
        for input_images, labels in iter(train_loader):
            input_images, labels = input_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(input_images)
            loss = loss_calculator(outputs, labels, loss_function)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % 10 == 9:
            net.eval()
            with torch.no_grad():
                train_loss, train_accuracy = compute_loss_accuracy(train_loader, loss_function, net, device)
                test_loss, test_accuracy = compute_loss_accuracy(test_loader, loss_function, net, device)
                spurious_loss, spurious_accuracy = compute_loss_accuracy(spurious_loader, loss_function, net, device)
                print('Epoch {}:\nTrain loss = {:.5f}\tTrain Accuracy = {:.3f}\nTest loss = {:.5f}\tTest Accuracy = {:.3f}\nspurious loss = {:.5f}\tspurious Accuracy = {:.3f}\n\n'.format(
                       epoch+1, train_loss, train_accuracy, test_loss, test_accuracy, spurious_loss, spurious_accuracy), flush=True)
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])

    net.eval()
    with torch.no_grad():
        train_loss, train_accuracy = compute_loss_accuracy(train_loader, loss_function, net, device)
        test_loss, test_accuracy = compute_loss_accuracy(test_loader, loss_function, net, device)
        spurious_loss, spurious_accuracy = compute_loss_accuracy(spurious_loader, loss_function, net, device)


    with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
        f.write(str(d) + '\t' + '0' + '\t' + str(N) + '\t' + str(train_accuracy) + '\t' + str(test_accuracy) + '\t' + str(spurious_accuracy) + '\n')
