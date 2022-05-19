# generate toy data
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import sys
import pdb
import os
from torch.utils.data import Subset
from tqdm import tqdm
from nf_resnet_model import NF_ResNet18
from utils import *
from sklearn.metrics import roc_auc_score
import time
import warnings
import pandas as pd

# plot toy data
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


num_samples = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    # random.seed(seed)


seed(42)
batch_size = 512

transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                [0.2023, 0.1994, 0.2010])
        ])


dataset1 = torchvision.datasets.CIFAR10('cifar10Data', train=True, download=True,
                                                transform= torchvision.transforms.Compose([
torchvision.transforms.RandomHorizontalFlip(),
torchvision.transforms.RandomCrop(32, 4),
transform ])   )

dataset1,_2 =torch.utils.data.random_split(dataset1,[int(0.1*len(dataset1)),int(0.9*len(dataset1))])

dataset2 = torchvision.datasets.CIFAR10('cifar10Data', train=False,
                                        transform=transform)
dataset2, _2 = torch.utils.data.random_split(dataset2,
                                                [len(dataset1), len(dataset2)-len(dataset1)])


label_idx = [0, 1, 2, 3, 4, 5]
outlier_label_idx = [6, 7, 8, 9]
trainset = Subset(dataset1, get_classes(dataset1, label_idx))
testset = Subset(dataset2, get_classes(dataset2, label_idx))

testset_outlier = Subset(dataset2, get_classes(dataset2, outlier_label_idx))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128)

outlier_loader = torch.utils.data.DataLoader(testset_outlier, batch_size=128)


model = NF_ResNet18(noise_shape= 8, units=[16, 32, 64]).to(DEVICE)

lr_ = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr_)
lossFun = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[82, 123], last_epoch= -1 )
lossFun = nn.CrossEntropyLoss( reduction='sum')

epochs = 200
Best_Acc = 0
for epoch in range(1, epochs):
    train_loss = 0
    train_correct = 0

    model.train()
   
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)

        ce_loss= lossFun(output, target)
        kl = model.kl(num_samples=10)
        loss = ce_loss +  kl 


        loss.backward()
        optimizer.step()
        train_loss += loss
        pred = output.argmax(dim=1, keepdim=True)  
        train_correct += pred.eq(target.view_as(pred)).sum().item()


    lr_scheduler.step()
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / (len(train_loader.dataset))

    print('Epoch: {} Training Loss = {:.4f}, Train Accuracy =  {:.2f}%, \n'.format(
            epoch, train_loss, train_acc))


    model.eval()

    with torch.no_grad():
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            probs = F.softmax(output, dim=-1)
            pred = probs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            

        print('Epoch: {} , Test Accuracy =  {:.2f}%  \n'.format(
            epoch, 100. * correct / len(test_loader.dataset)))
