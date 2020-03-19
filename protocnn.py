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

from datasets.dynaMO.dataset import dynaMODataset, ToTensor
from utilities.convlstm import ConvLSTMCell, ConvLSTM

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

class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 6 * 6, 10) #32*6*6

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)

        x = x.view(-1, 32 * 6 * 6)
        # print(x.shape)

        x = F.softmax(self.fc(x), 1)
        # print(x.shape)

        return x

# -----------------
# Functions for Training and Evaluation
# -----------------



def train(input_tensor, target_tensor, network, optimizer, criterion):

    optimizer.zero_grad()

    loss = 0

    input_tensor = input_tensor[:,0:1,:,:]
    network_output = network(input_tensor)
    if len(target_tensor.shape) > 1:
        target_tensor = target_tensor.squeeze()
    loss += criterion(network_output, target_tensor)
    topv, topi = network_output.topk(1)
    accuracy = (topi == target_tensor.unsqueeze(1)).sum(dim=0, dtype=torch.float64)/topi.shape[0]

    loss.backward()
    optimizer.step()
    return loss.item(), accuracy.item()


def trainEpochs(dataloader, network, n_epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0
    print_accuracy_total = 0
    plot_loss_total = 0
    len_of_data = len(dataloader)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(n_epochs):
        start = time.time()
        for i_batch, sample_batched in enumerate(dataloader):

            loss, accuracy = train(sample_batched['image'], sample_batched['single'], network,
                         optimizer, criterion)
            # loss, accuracy = train(sample_batched[0], sample_batched[1], network,
            #              optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss
            print_accuracy_total += accuracy

            if i_batch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_accuracy_avg = print_accuracy_total / print_every
                print_accuracy_total = 0
                print(" " * 80 + "\r" + '[Training:] E%d: %s (%d %d%%) %.4f %.4f' % (epoch, timeSince(start, (i_batch+1) / len_of_data),
                                             i_batch, (i_batch+1) / len_of_data * 100, print_loss_avg, print_accuracy_avg), end="\r")

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
cnn = StandardCNN().to(device)
dynaMo_transformed = dynaMODataset(
    root_dir='/Users/markus/Research/Code/titan/datasets/dynaMO/image_files/train/',
    transform=transforms.Compose([
        ToTensor()
    ]))

dynaMO_dataloader = DataLoader(dynaMo_transformed, batch_size=100,
                        shuffle=True, num_workers=0, drop_last=True)



trainEpochs(dynaMO_dataloader, cnn, n_epochs=10, print_every=100)


# # Training dataset
# cnn = StandardCNN().to(device)
# mnist_loader = torch.utils.data.DataLoader(
#     datasets.MNIST(root='./', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.,), (1.,))
#                    ])), batch_size=100, shuffle=True, num_workers=4)


#trainEpochs(mnist_loader, cnn, n_epochs=10, print_every=10)

# for i_batch, sample_batched in enumerate(dynaMO_dataloader):
#     print("hi")
#    criterion = nn.NLLLoss()
#criterion(decoder_output, target_tensor[:,di])

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
