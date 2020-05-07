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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import time
import math


# custom functions
# -----
import utilities.helper as helper
from utilities.networks.buildingblocks.hopfield import HopfieldNet
from utilities.networks.buildingblocks.rcnn import BLT_Network, B_Network

from utilities.dataset_handler import ImageFolderLMDB


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

def matplotlib_imshow(img, one_channel=False, cmap='Greys'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        return (fig, ax, ax.imshow(npimg, cmap=cmap))
    else:
        return (fig, ax, ax.imshow(np.transpose(npimg, (1,2,0))))


def checkpoint(epoch, model, experiment_dir, save_every, remove_last=True):
    model_out_path = experiment_dir + "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("[Info:] Checkpoint saved to {}".format(model_out_path, end='\n'))
    if (epoch > 0 and remove_last):
        os.remove(experiment_dir +
                  "model_epoch_{}.pth".format(epoch - save_every))


# cross-platform development

from platform import system
IS_MACOSX = True if system() == 'Darwin' else False
PWD_STEM = "/Users/markus/Research/Code/" if IS_MACOSX else "/home/mernst/git/"

# commandline arguments
# -----

# FLAGS

parser = argparse.ArgumentParser()
parser.add_argument(
     "-t",
     "--testrun",
     # type=bool,
     default=False,
     dest='testrun',
     action='store_true',
     help='reduced dataset configuration on local machine for testing')
parser.add_argument(
     "-c",
     "--config_file",
     type=str,
     default=PWD_STEM +
             'titan/experiments/001_noname_experiment/' +
             'files/config_files/config0.csv',
     help='path to the configuration file of the experiment')
parser.add_argument(
     "-n",
     "--name",
     type=str,
     default='',
     help='name of the run, i.e. iteration1')
parser.add_argument(
     "-r",
     "--restore_ckpt",
     type=bool,
     default=True,
     help='restore model from last checkpoint')

FLAGS = parser.parse_args()


CONFIG = helper.infer_additional_parameters(
    helper.read_config_file(FLAGS.config_file)
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------
# Encoder and Decoder Classes of the network
# -----------------



class BH_Network(nn.Module):
    def __init__(self, i_factor=1, time_steps=2):
        super(BH_Network, self).__init__()
        self.i_factor = i_factor
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.hnet1 = HopfieldNet(32 * 16 * 16) # 32 * 14 * 14 for MNIST
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.hnet2 = HopfieldNet(32 * 8 * 8) # 32 * 7 * 7 for MNIST
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

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
        x = x.view(-1, 32 * 8 * 8)
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
    #network.hnet1.covariance_update(network.act1)
    #network.hnet2.covariance_update(network.act2)

    loss = loss / topi.shape[0]  # average loss per item
    return loss.item(), accuracy.item()

def train_recurrent(input_tensor, target_tensor, network, optimizer, criterion):

    optimizer.zero_grad()

    loss = 0
    time = 3
    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = input_tensor.repeat(1, time, 1, 1, 1)
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
    accuracy = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_tensor, target_tensor = data
            network_output = network(input_tensor)
            loss += criterion(network_output, target_tensor) / \
                test_loader.batch_size
            topv, topi = network_output.topk(1)
            accuracy += (topi == target_tensor.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]

    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                           loss /(i+1), accuracy/(i+1)), end="\n")
    
    
    return loss /(i+1), accuracy/(i+1)

def test_recurrent(test_loader, network, criterion, epoch):
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.unsqueeze(1)
            input_tensor = input_tensor.repeat(1, 4, 1, 1, 1)
            network_output = network(input_tensor)
            for t in range(network_output.shape[1]):
                loss += criterion(network_output[:,t,:], target_tensor)
            # loss /= test_loader.batch_size
            topv, topi = network_output[:,t,:].topk(1)
            accuracy = (topi == target_tensor.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]

    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                       loss /(i+1), accuracy/(i+1)), end="\n")
    return loss /(i+1), accuracy/(i+1)


def trainEpochs(train_loader, test_loader, network, writer, n_epochs, test_every, print_every, plot_every, save_every, learning_rate, output_dir, checkpoint_dir):
    plot_losses = []
    print_loss_total = 0
    print_accuracy_total = 0
    plot_loss_total = 0
    plot_accuracy_total = 0
    
    len_of_data = len(train_loader)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        if epoch % test_every == 0:
            test_loss, test_accurary = test_recurrent(test_loader, network, criterion,
                                            epoch)
            writer.add_scalar('testing/loss', test_loss,
                              epoch * len_of_data)
            writer.add_scalar(
                'testing/accuracy', test_accurary, epoch * len_of_data)
        start = time.time()
        for i_batch, sample_batched in enumerate(train_loader):

            loss, accuracy = train_recurrent(sample_batched[0], sample_batched[1],
                                   network, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss
            print_accuracy_total += accuracy
            plot_accuracy_total += accuracy

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
                plot_accuracy_avg = plot_accuracy_total / plot_every

                plot_losses.append(plot_loss_avg)
                
                writer.add_scalar(
                    'training/loss', plot_loss_avg,
                      epoch * len_of_data + i_batch)
                writer.add_scalar(
                    'training/accuracy', plot_accuracy_avg, epoch * len_of_data + i_batch)
            
                plot_loss_total = 0
                plot_accuracy_total = 0
                
        if epoch % save_every == 0:
                checkpoint(epoch, network, checkpoint_dir + 'network', save_every)


    showPlot(plot_losses)
    plt.show()


# -----------------
# Main Training Loop
# -----------------

# Training network
#network = B_Network().to(device)
#network = BH_Network().to(device)
network = BLT_Network().to(device)

# Datasets
train_dataset = ImageFolderLMDB(
    db_path=CONFIG['input_dir'] + '/dynaMO/data/osmnist2/train.lmdb',
    transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
]))

test_dataset = ImageFolderLMDB(
    db_path=CONFIG['input_dir'] + '/dynaMO/data/osmnist2/test.lmdb',
    transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
]))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=0)


# Tensorboard Writer

output_dir, checkpoint_dir = helper.get_output_directory(CONFIG, FLAGS)
loss_writer = SummaryWriter(output_dir)


trainEpochs(train_loader, test_loader, network, loss_writer, CONFIG['epochs'],
            test_every=CONFIG['test_every'], print_every=CONFIG['write_every'], plot_every=CONFIG['write_every'], save_every=5, learning_rate=CONFIG['learning_rate'], output_dir=output_dir, checkpoint_dir=checkpoint_dir)

# TODO: CONFIG['learning_rate']

torch.save(network.state_dict(), checkpoint_dir + 'network.model')
# TODO: Implement general checkpointing in trainEpochs

# _____________________________________________________________________________

# train_dataset = datasets.MNIST(root='../datasets/', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.,), (1.,))
#                    ]))
# 
# test_dataset = datasets.MNIST(root='../datasets/', train=False, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.,), (1.,))
# ]))


# train_dataset = datasets.ImageFolder(
#     root='../datasets/dynaMO/data/mnist/train/',
#     transform=transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.,), (1.,))
# ]))
# 
# test_dataset = datasets.ImageFolder(
#     root='../datasets/dynaMO/data/mnist/test/',
#     transform=transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.,), (1.,))
# ]))

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
