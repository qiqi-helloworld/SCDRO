__author__ = 'Qi'
# Created by on 11/18/22.

__author__ = 'Qi'
# Created by on 7/30/22.
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
# import numpy
# from numpy.random import choice
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import copy
import numpy

from models import *
from utils import progress_bar
from _optimizer import PGM
import copy
from parameters import para
import pandas as pd
from datetime import datetime
from resnet20 import resnet20
from resnet import resnet50
from train_alg import train, train_sgd

import ina2018_loader
import data.dataloader as imagenet_LT_load

# import data.LT_Dataset as LT_Dataset


gpuid = para.gpuid
if gpuid > -1:
    torch.cuda.set_device(gpuid)
# device = torch.device('cuda:'+str(gpuid)) if torch.cuda.is_available() else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# Cifar Transform
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if para.data == "Cifar10":
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
elif para.data == "Cifar100":
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
elif para.data == "iNaturalist2019":
    num_classes = 1010
    train_file = '/Users/zhishguo/data/ina/train2019.json'
    val_file = '/Users/zhishguo/data/ina/val2019.json'
    data_root = '/Users/zhishguo/data/ina/images/'
    inaTrainAccBatch = 100
    trainset = ina2018_loader.INAT(data_root, train_file, size=para.size, is_train=True)
    testset = ina2018_loader.INAT(data_root, train_file, size=para.size, is_train=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=para.bsz, \
                                              shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=para.bsz, \
                                             shuffle=True, num_workers=4, pin_memory=True)
elif para.data == "ImageNet_LT":
    ImageNet_root = "/Users/zhishguo/data/imagenet"
    num_classes = 1000
    # if para.alg == "sgd":
    #    shuffle_ = True
    # else:
    #    shuffle_ = False

    trainloader = imagenet_LT_load.load_data(data_root=ImageNet_root, dataset="ImageNet_LT", phase="train", \
                                             batch_size=para.bsz, shuffle=True, num_workers=4)

    testloader = imagenet_LT_load.load_data(data_root=ImageNet_root, dataset="ImageNet_LT", phase="val", \
                                            batch_size=para.bsz, shuffle=True, num_workers=4)
else:
    print("dataset not defined!")
    exit()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# make cifar10 imbalanced
'''
if para.data != "iNaturalist2019" and para.data != "ImageNet_LT":
    idx_deleted = list()
    for delete in range(0, num_classes//2):
        idx0 = [i for i,val in enumerate(trainset.targets) if val==delete]
        full_len = len(idx0)
        idx_deleted = idx_deleted + idx0[0:int(full_len*para.delete_ratio)]

        trainset.data = numpy.delete(trainset.data, idx_deleted, axis=0)
        trainset.targets = numpy.delete(trainset.targets, idx_deleted)
'''

if para.data == "ImageNet_LT":
    n = 115846
else:
    n = len(trainset)
print("Num of data after deletion: " + str(n))
initialWeights = torch.FloatTensor([1.0 / n for i in range(n)])
batchsize = para.bsz
numRounds = n // batchsize + 1
# Model
print('==> Building model..')
# net = VGG('VGG19')
if para.model == 18:
    net = ResNet18(num_classes=num_classes)
elif para.model == 20:
    net = resnet20(num_classes=num_classes)
    # if para.data == "iNaturalist2019":
    #    net = torch.nn.DataParallel(net)
elif para.model == 50:
    net = resnet50(pretrained=True, num_classes=num_classes)
if para.data == "iNaturalist2019" or para.data == "ImageNet_LT":
    net = torch.nn.DataParallel(net)

net = net.cuda()

# torch.save(net.state_dict(), "checkpoint/checkpoint_"+str(epoch)+"_ResNet"+str(para.model)+"_"+str(round(correct/total, 3)))

cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(reduce=False)
# optimizer = optim.PGM(net.parameters(), lr=args.lr, lambd=0, alpha=0, t0=0, weight_decay=1)

global prob
global ptemp
prob = initialWeights.cuda()
uniformWeights = initialWeights.cuda()
ptemp = torch.zeros(n).cuda()

# penalty parameter for the regularizer of entropy lmbda = 2
list1 = []
list2 = []
list3 = []
epoch_list = list()
time_spent_list = list()
train_acc_list = list()
test_acc_list = list()
num_sample_list = list()
global time_spent_on_training
global num_sample
global total_iter
total_iter = 0
# global prob
# global p_average
# p_average = initialWeights.cuda()
time_spent_on_training = 0
num_sample = 0
configs = str(para.data) + "_" + str(para.delete_ratio) + "_" + str(para.alg) + '_ResNet' + str(
    para.model) + '_lr1_' + str(para.lr1) + '_lr2_' + str(para.lr2) + '_E0_' + str(para.E0) \
          + '_gamma_' + str(para.gamma) + '_theta_' + str(para.theta) + '_c_' + str(para.c) + '_bsz_' + str(
    para.bsz) + '_size_' + str(para.size)



# def train_sgd(epoch, lr1, prob, ptemp, uniformWeights, time_spent_on_training, epoch_list, time_spent_list, train_acc_list, net, net0):
def SPD(epoch, lr1, lr2, gamma, theta, net, net0, net_average, count):
    epoch_start_time = datetime.now()
    global ptemp
    global prob
    global uniformWeights
    global time_spent_on_training
    global num_sample
    # global p_average
    ptemp.zero_()
    print('\nEpoch: %d' % epoch)
    print('\nConfigs: %s' % configs)
    randomNumber = torch.rand(numRounds)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for rounds in range(numRounds):
        # print("Rounds:" + str(rounds))
        # if rounds > 100:
        #    break
        #    print("!!!!!!!!!!!!!!!!!1")
        #    break#exit()
        # optimizer.snapshot()
        minibatchIdx = torch.multinomial(prob, batchsize, replacement=False)
        # subsetSampler = torch.utils.data.sampler.SubsetRandomSamplerCustomized(minibatchIdx)
        subsetSampler = torch.utils.data.sampler.SubsetRandomSampler(minibatchIdx)
        train_loader = imagenet_LT_load.load_data(data_root=ImageNet_root, dataset="ImageNet_LT", phase="train", \
                                                     sampler=subsetSampler, batch_size=para.bsz, num_workers=4)

        # exit()
        for batch_idx, tmp_ in enumerate(train_loader):  # Actually one iteration
            if para.data == "iNaturalist2019":
                (inputs, im_id, targets, tax_ids) = tmp_
            elif para.data == "ImageNet_LT":
                (inputs, targets, paths) = tmp_
            else:
                (inputs, targets) = tmp_

            inputs, targets = inputs.cuda(), targets.cuda()
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss2 = criterion2(outputs, targets)
            loss.backward()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            ### update the weights and calculate the training loss (not the actual training loss)
            # optimizer.step()
            PGM(net=net, lr=lr1, gamma=gamma, net0=net0, weight_decay=para.decay)
            num_sample += len(targets)
            prob = n ** (-lr2 * theta) * prob.pow(1 - lr2 * theta) * torch.exp(torch.tensor(-lr2 * theta / n))
            prob[minibatchIdx] = prob[minibatchIdx] * torch.exp(lr2 * loss2)

        if para.divergence == "kl":


        prob /= torch.sum(prob)
        ptemp = (rounds * ptemp + prob) / (rounds + 1)
        # print("prob: " + str(prob) + '\n')
        # ptemp /= torch.sum(ptemp)
        # print("prob:"+str(prob))
        # prob = initialWeights.cuda()
        # ptem = initialWeights.cuda()
        with torch.no_grad():
            for name, param in net.named_parameters():
                net_average[name].data = net_average[name].data * (count / (count + 1.0)) + param.data / (count + 1.0)

        progress_bar(rounds, numRounds, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (rounds + 1), 100. * correct / total, correct, total))

    time_spent_on_training += (datetime.now() - epoch_start_time).total_seconds()
    time_spent_list.append(time_spent_on_training)

    list2.append(correct / total)
    epoch_list.append(epoch)
    train_acc_list.append(correct / total)
    num_sample_list.append(num_sample)

    # prob = ptemp + 1 - 1
    # with torch.no_grad():
    #    for name, param in net.named_parameters():
    #        param.data = net_average[name]



# Calculate Training Error
def trainError(epoch, net):
    # print('\nEpoch: %d' % epoch)
    net.eval()
    train_loss = 0
    correct = 0
    total = 0
    print("\nTraining ACC")
    with torch.no_grad():
        for batch_idx, tmp_ in enumerate(trainloader):
            if batch_idx > 400:
                break
            if para.data == "iNaturalist2019":
                (inputs, im_id, targets, tax_ids) = tmp_
            elif para.data == "ImageNet_LT":
                (inputs, targets, paths) = tmp_
            else:
                (inputs, targets) = tmp_
            inputs, targets = inputs.cuda(), targets.cuda()
            # optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        list2.append(correct / total)
        epoch_list.append(epoch)
        train_acc_list.append(correct / total)
        num_sample_list.append(num_sample)


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("\nTesting ACC")
    with torch.no_grad():
        for batch_idx, tmp_ in enumerate(testloader):
            if batch_idx > 400:
                break
            if para.data == "iNaturalist2019":
                (inputs, im_id, targets, tax_ids) = tmp_
            elif para.data == "ImageNet_LT":
                (inputs, targets, paths) = tmp_
            else:
                (inputs, targets) = tmp_
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_acc_list.append(correct / total)
    df = pd.DataFrame(data={'total_epochs': epoch_list, 'time': time_spent_list, 'numSample': num_sample_list, \
                            'train_acc': train_acc_list, 'test_acc': test_acc_list})
    df.to_csv(para.data + '_history/' + configs + '.csv')
    print('\n')
    torch.save(net.state_dict(), "checkpoint/checkpoint_" + str(epoch) + "_ResNet" + str(para.model) + "_" + str(
        round(correct / total, 3)))


print("new")
total_epoch = 0
net_average = copy.deepcopy(net.state_dict())
for stage in range(1, 10000):
    net0 = copy.deepcopy(net.state_dict())  # Guo
    count = 0
    with torch.no_grad():
        for name, param in net.named_parameters():
            param.data = net_average[name]

        numEpochs = para.E0 * (stage)
        lr1 = para.lr1 / stage
        lr2 = para.lr2 / stage
        # Solve y
        with torch.no_grad():
            # loss2_list = list()
            for t in range(numRounds):
                # minibatchIdx = torch.multinomial(prob, batchsize, replacement=False)
                start_ = min(t * batchsize, n)
                end_ = min((t + 1) * batchsize, n)
                minibatchIdx = list(range(start_, end_))
                subsetSampler = torch.utils.data.sampler.SubsetRandomSampler(minibatchIdx)
                train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=len(minibatchIdx),
                                                           sampler=subsetSampler, shuffle=False, num_workers=0)
                for batch_idx, tmp_ in enumerate(train_loader):
                    if para.data != "iNaturalist2019":
                        (inputs, targets) = tmp_
                        inputs, targets = inputs.cuda(), targets.cuda()
                    else:
                        (inputs, im_id, targets, tax_ids) = tmp_
                        inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = net(inputs)
                    loss2 = criterion2(outputs, targets)
                    # loss2_list += list(loss2)
                    num_sample += len(targets)
                if para.divergence == "kl":
                    prob[minibatchIdx] = torch.exp(loss2 / para.theta - 1.0 / n) / n

        prob /= torch.sum(prob)



    for epoch in range(numEpochs):
        print("Stage: " + str(stage) + "; lr1:" + str(lr1) + "; lr2:" + str(lr2) + "; numEpochs: " + str(
            numEpochs) + "\n")
        train(total_epoch + epoch, lr1, lr2, gamma=para.gamma, theta=para.theta, net=net, net0=net0,
                  net_average=net_average, count=count)

        # trainError(total_epoch+epoch, net)
        test(total_epoch + epoch, net)
    total_epoch += numEpochs



