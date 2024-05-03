import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import torch.optim as optim
import os
import time
from utils_NN import loss_calculator, compute_loss_accuracy_regressor, compute_loss_accuracy_classifier
from utils_NN import FullyConnected, SmallConvNet, BigConvNet, load_full_dataset
from torchvision.models import resnet18, resnet50


parser = argparse.ArgumentParser()
parser.add_argument('--i')
parser.add_argument('--dataset')
parser.add_argument('--save')
parser.add_argument('--model')
parser.add_argument('--batch_size')
parser.add_argument('--lr')
args = parser.parse_args()

time.sleep(int(args.i))

save_dir = os.path.join(args.model, args.dataset, args.save + '_' + args.batch_size + '_' + args.lr)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# if args.dataset == 'synthetic':
#     d = 1000
# elif args.dataset == 'MNIST':
#     d = 28 ** 2
# elif args.dataset == 'CIFAR-10':
#     d = 3 * 32 ** 2

d = 3 * 32 ** 2

Ns = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
Nt = 10000

for N in Ns:

    print('-----------------------------', flush=True)
    print('N = {}'.format(N), flush=True)
    print('-----------------------------', flush=True)

    Z, Zt, Za, G, Gt = load_full_dataset('CIFAR-10', N, Nt, folder_name=args.dataset + '/')
    loss_function = 'CEL'
    compute_loss_accuracy = compute_loss_accuracy_classifier

    # permuting because resnet likes [batch size, number of channels, x , y], and now channels are in position 3.
    
    augmented_train_dataset = TensorDataset(torch.from_numpy(Z).to(torch.float32).permute(0, 3, 1, 2), torch.from_numpy(G))
    attack_train_dataset = TensorDataset(torch.from_numpy(Za).to(torch.float32).permute(0, 3, 1, 2), torch.from_numpy(G))
    augmented_test_dataset = TensorDataset(torch.from_numpy(Zt).to(torch.float32).permute(0, 3, 1, 2), torch.from_numpy(Gt))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, flush=True)

    lr = float(args.lr)
    batch_size = int(args.batch_size)

    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    attack_loader = DataLoader(attack_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(augmented_test_dataset, batch_size=batch_size, shuffle=True)

    epochs = 200

    if args.model == 'resnet18':
        net = resnet50(weights=None, num_classes=10).to(device)
    elif args.model == 'resnet50':
        net = resnet50(weights=None, num_classes=10).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

    for epoch in range(epochs):

        net.train()            
        for input_images, labels in iter(train_loader):
            input_images, labels = input_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(input_images)
            loss = loss_calculator(outputs, labels, loss_function)
            loss.backward()
            optimizer.step()

        # scheduler.step()

        if epoch % 10 == 9:
            net.eval()
            with torch.no_grad():
                train_loss, train_accuracy = compute_loss_accuracy(train_loader, loss_function, net, device)
                test_loss, test_accuracy = compute_loss_accuracy(test_loader, loss_function, net, device)
                attack_loss, attack_accuracy = compute_loss_accuracy(attack_loader, loss_function, net, device)
                print('Epoch {}:\nTrain loss = {:.5f}\tTrain Accuracy = {:.3f}\nTest loss = {:.5f}\tTest Accuracy = {:.3f}\nAttack loss = {:.5f}\tAttack Accuracy = {:.3f}\n\n'.format(
                       epoch+1, train_loss, train_accuracy, test_loss, test_accuracy, attack_loss, attack_accuracy), flush=True)
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])

    net.eval()
    with torch.no_grad():
        train_loss, train_accuracy = compute_loss_accuracy(train_loader, loss_function, net, device)
        test_loss, test_accuracy = compute_loss_accuracy(test_loader, loss_function, net, device)
        attack_loss, attack_accuracy = compute_loss_accuracy(attack_loader, loss_function, net, device)


    with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
        f.write(str(d) + '\t' + '0' + '\t' + str(N) + '\t' + str(train_accuracy) + '\t' + str(test_accuracy) + '\t' + str(attack_accuracy) + '\n')
