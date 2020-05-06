#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# protocnn.py                                oN88888UU[[[/;::-.        dP^
# Description Description                   dNMMNN888UU[[[/;:--.   .o@P^
# Description Description                  ,MMMMMMN888UU[[/;::-. o@^
#                                          NNMMMNN888UU[[[/~.o@P^
# Markus Ernst                             888888888UU[[[/o@^-..
#                                         oI8888UU[[[/o@P^:--..
#                                      .@^  YUU[[[/o@^;::---..
#                                    oMP     ^/o@P^;:::---..
#                                 .dMMM    .o@^ ^;::---...
#                                dMMMMMMM@^`       `^^^^
#                               YMMMUP^
#                                ^^
# _____________________________________________________________________________
#
#
# Copyright 2020 Markus Ernst
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# _____________________________________________________________________________


# ----------------
# import libraries
# ----------------

# standard libraries
# -----
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

import time
import math


# custom functions
# -----

from utilities.dataset_handler import dynaMODataset, ToTensor
from utilities.networks.buildingblocks.hopfield import HopfieldNet

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%d %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s ( - %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Encoder and Decoder Classes of the network
# -----------------


class Lenet5(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 8, padding=4)
        self.pool1 = nn.MaxPool2d(4, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 8, padding=4)
        self.pool2 = nn.MaxPool2d(4, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 32*6*6
        self.fc2 = nn.Linear(120, 10)  # 32*6*6

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)

        x = x.view(-1, 16 * 7 * 7)
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), 1)
        # print(x.shape)

        return x


class B_Network(nn.Module):
    def __init__(self):
        super(B_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)

        x = x.view(-1, 32 * 7 * 7)
        # print(x.shape)

        x = F.softmax(self.fc1(x), 1)
        # print(x.shape)

        return x

class BH_Network(nn.Module):
    def __init__(self, i_factor=1, time_steps=2):
        super(BH_Network, self).__init__()
        self.i_factor = i_factor
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.hnet1 = HopfieldNet(32 * 14 * 14)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.hnet2 = HopfieldNet(32 * 7 * 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        b, c, h, w = x.shape
        y = (x > x.mean()).view(b, -1) # reshape here
        self.act1 = y.detach()
        for t in range(self.time_steps):
            y = self.hnet1.step(y)
        y = y.view(b,c,h,w).type(dtype=torch.float32) * 2 - 1 * self.i_factor # reshape again
        x += y

        # layer 2

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        b, c, h, w = x.shape
        y = (x > x.mean()).view(b, -1)
        self.act2 = y.detach()
        for t in range(self.time_steps):
            y = self.hnet2.step(y)
        y = y.view(b,c,h,w).type(dtype=torch.float32) * 2 - 1 * self.i_factor # reshape again
        x += y

        # fc and out
        x = x.view(-1, 32 * 7 * 7)
        x = F.softmax(self.fc1(x), 1)
        return x
# -----------------
# Functions for Training and Evaluation
# -----------------


def train(input_tensor, target_tensor, network, optimizer, criterion):

    optimizer.zero_grad()

    loss = 0

    input_tensor = input_tensor[:, 0:1, :, :]
    network_output = network(input_tensor)
    if len(target_tensor.shape) > 1:
        target_tensor = target_tensor.squeeze()
    loss += criterion(network_output, target_tensor)
    topv, topi = network_output.topk(1)
    accuracy = (topi == target_tensor.unsqueeze(1)).sum(
        dim=0, dtype=torch.float64) / topi.shape[0]

    loss.backward()
    optimizer.step()

    # update the hopfield networks for B-H
    network.hnet1.covariance_update(network.act1)
    network.hnet2.covariance_update(network.act2)

    loss = loss / topi.shape[0]  # average loss per item
    return loss.item(), accuracy.item()


def test(test_loader, network, criterion, epoch):
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_tensor, target_tensor = data
            network_output = network(input_tensor)
            loss += criterion(network_output, target_tensor) / \
                test_loader.batch_size
            topv, topi = network_output.topk(1)
            accuracy = (topi == target_tensor.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]

    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                           loss / i, accuracy), end="\n")
    return loss, accuracy


def trainEpochs(train_loader, test_loader, network, n_epochs, print_every=1000, test_every=1, plot_every=100, learning_rate=0.001):
    plot_losses = []
    print_loss_total = 0
    print_accuracy_total = 0
    plot_loss_total = 0
    len_of_data = len(train_loader)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        if epoch % test_every == 0:
            test_loss, test_accurary = test(test_loader, network, criterion,
                                            epoch)

        start = time.time()
        for i_batch, sample_batched in enumerate(train_loader):

            loss, accuracy = train(sample_batched[0], sample_batched[1],
                                   network, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss
            print_accuracy_total += accuracy

            if i_batch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_accuracy_avg = print_accuracy_total / print_every
                print_accuracy_total = 0
                print(" " * 80 + "\r" +
                      '[Training:] E%d: %s (%d %d%%) %.4f %.4f'
                      % (epoch, timeSince(start, (i_batch + 1) / len_of_data),
                          i_batch, (i_batch + 1) / len_of_data * 100,
                          print_loss_avg, print_accuracy_avg), end="\r")

            if i_batch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)
    plt.show()


# -----------------
# Main Training Loop
# -----------------

# Training dataset
# cnn = Lenet5().to(device)
#cnn = B_Network().to(device)
cnn = BH_Network().to(device)

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.,), (1.,))
                   ])), batch_size=100, shuffle=True, num_workers=0)

# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ])), batch_size=100, shuffle=True, num_workers=0)

trainEpochs(train_loader, test_loader, cnn, n_epochs=10, print_every=1,
            test_every=1)


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
