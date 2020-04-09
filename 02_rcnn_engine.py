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
from utilities.recconv import BLT_Network, B_Network


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
    loss = loss / topi.shape[0]  # average loss per item
    return loss.item(), accuracy.item()

def train2(input_tensor, target_tensor, network, optimizer, criterion):

    optimizer.zero_grad()

    loss = 0

    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = input_tensor.repeat(1, 3, 1, 1, 1)
    network_output = network(input_tensor)

    for t in range(network_output.shape[1]):
        loss += criterion(network_output[:,t,:], target_tensor)

    topv, topi = network_output[:,t,:].topk(1)
    accuracy = (topi == target_tensor.unsqueeze(1)).sum(
        dim=0, dtype=torch.float64) / topi.shape[0]

    loss.backward()
    optimizer.step()
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


def test2(test_loader, network, criterion, epoch):
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.unsqueeze(1)
            input_tensor = input_tensor.repeat(1, 4, 1, 1, 1)
            network_output = network(input_tensor)
            for t in range(network_output.shape[1]):
                loss += criterion(network_output[:,t,:], target_tensor)
            loss /= test_loader.batch_size
            topv, topi = network_output[:,t,:].topk(1)
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
            test_loss, test_accurary = test2(test_loader, network, criterion,
                                            epoch)

        start = time.time()
        for i_batch, sample_batched in enumerate(train_loader):

            loss, accuracy = train2(sample_batched[0], sample_batched[1],
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
cnn = BLT_Network().to(device)
# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.,), (1.,))
                   ])), batch_size=100, shuffle=True, num_workers=1)

# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ])), batch_size=100, shuffle=True, num_workers=1)

trainEpochs(train_loader, test_loader, cnn, n_epochs=10, print_every=1,
            test_every=1)


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment