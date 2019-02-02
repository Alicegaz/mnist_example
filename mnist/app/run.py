from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import visdom
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def load_data(PATH):
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # load the dataset
    train_dataset = datasets.MNIST(root=PATH, train=True,
                                   download=True, transform=train_transform)

    valid_dataset = datasets.MNIST(root=PATH, train=True,
                                   download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # if shuffle == True:
    #        np.random.seed(random_seed)
    #        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 100, 5, 1)
        self.conv2 = nn.Conv2d(100, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, val_loader, optimizer, epochs, log_interval):
    since = time.time()
    model.train()
    if model is None:
        best_model = model.load_state_dict(torch.load('./models/best_model.pt'))
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # torch.save(the_model, './models/best_model.pt')
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                valid_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                # print(output)

        valid_loss /= len(valid_loader.dataset)

        print('\n Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    torch.save(best_model.state_dict(), './models/best_model.pt')


def test(test_loader):
    model = Net()
    model = model.load_state_dict(torch.load('./models/best_model.pt'))
    model.eval()
    test_loss = 0
    correct = 0
    outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            outputs.append(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(output)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, model, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--mode', type=str, default='fit', metavar='N',
                        help='fit or predict')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='fit or predict')
    parser.add_argument('--test_data', type=str, default=None, metavar='N',
                        help='path to the folder of MNIST test data')
    parser.add_argument('--train_data', type=str, default=None, metavar='N',
                        help='path to the folder of MNIST training data')
    args = parser.parse_args()
    if args.train_data is None:
        train_path = './data'
    else:
        train_path = args.train_data
    if args.test_data is None:
        test_path = './data'
    else:
        test_path = args.test_data
    train_loader, valid_loader, test_loader = load_data(train_path, test_path)

    if args.mode == 'fit':
        train_batch_size = 300
        test_batch_size = 200

        lr = 0.01
        momentum = 0.001

        epochs = 10

        log_interval = int(len(test_loader.dataset) / train_batch_size)

        valid_size = 0.2
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        train(None, train_loader, valid_loader, optimizer, args.epochs, log_interval)
    elif args.mode == 'predict':
        test(test_loader)
    if args.mode == 'train':
        train_batch_size = 300
        test_batch_size = 200

        lr = 0.01
        momentum = 0.001

        epochs = 10

        log_interval = int(len(test_loader.dataset) / train_batch_size)

        valid_size = 0.2
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        train(model, train_loader, valid_loader, optimizer, args.epochs, log_interval)



