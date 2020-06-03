#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# May 2020                                       _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# engine.py                                  oN88888UU[[[/;::-.        dP^
# The main file including                   dNMMNN888UU[[[/;:--.   .o@P^
# the training loop                        ,MMMMMMN888UU[[/;::-. o@^
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

#from torchvision import utils, datasets
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
import utilities.visualizer as visualizer
from utilities.networks.buildingblocks.hopfield import HopfieldNet
from utilities.networks.buildingblocks.rcnn import RecConvNet, B_Network

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
    torch.save(model.state_dict(), model_out_path)
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
        self.debug_writer = SummaryWriter(CONFIG['output_dir'])

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        b, c, h, w = x.shape
        
        y = (x > x.mean()).view(b, -1) # reshape here
        # moving_average = x.mean(axis=0) * 0.01  + moving_average * 0.99
        # use moving average as a threshold above
        # mean over all batches!! should be constant or slowly moving
        # TODO: try with constants, separate mean over batch and then exp. avg.
        # TODO: keep track of the scale, the bias/gamma of the BN, maybe turning off the scaling
        self.act1 = y.clone().detach()
        for t in range(self.time_steps):
            y = self.hnet1.step(y)
        y = y.view(b,c,h,w).type(dtype=torch.float32)
        y = y * self.i_factor # y * 2 - 1 * self.i_factor # reshape again
        x += y # think of y as a relu and maybe multiplying then makes sense?
        
        # DEBUG writedown
        helper.print_tensor_info(self.bn1.bias, name='1/bn_bias', writer=self.debug_writer)
        helper.print_tensor_info(x, name='1/x', writer=self.debug_writer)
        helper.print_tensor_info(y, name='1/y', writer=self.debug_writer)


        # layer 2

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        b, c, h, w = x.shape
        y = (x > x.mean()).view(b, -1)
        self.act2 = y.clone().detach()
        for t in range(self.time_steps):
            y = self.hnet2.step(y)
        y = y.view(b,c,h,w).type(dtype=torch.float32)
        y = y = y * self.i_factor # y * 2 - 1 * self.i_factor # reshape again
        x += y

        # DEBUG writedown
        helper.print_tensor_info(self.bn2.bias, name='2/bn_bias', writer=self.debug_writer)
        helper.print_tensor_info(x, name='2/x', writer=self.debug_writer)
        helper.print_tensor_info(y, name='2/y', writer=self.debug_writer)

        # fc and out
        x = x.view(-1, 32 * 8 * 8)
        x = F.softmax(self.fc1(x), 1)
        return x


class BLH_Network(nn.Module):
    def __init__(self, i_factor=1, time_steps=2):
        assert False, 'module not yet implemented'
        super(BLH_Network, self).__init__()

    def forward(self, x):
        return x


# -----------------
# Functions for Training and Evaluation
# -----------------


def train(input_tensor, target_tensor, network, optimizer, criterion):

    optimizer.zero_grad()
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
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
    # network.hnet1.covariance_update(network.act1)
    # network.hnet2.covariance_update(network.act2)
    # look at the patterns that the network stores..
    # think of Hnet as clustering algorithm

    loss = loss / topi.shape[0]  # average loss per item
    return loss.item(), accuracy.item()


def test(test_loader, network, criterion, epoch):
    loss = 0
    accuracy = 0
    confusion_matrix = torch.zeros(
        CONFIG['classes'], CONFIG['classes'], dtype=torch.int64)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, classes = data
            inputs, classes = inputs.to(device), classes.to(device)
            outputs = network(inputs)
            loss += criterion(outputs, classes) / \
                test_loader.batch_size
            topv, topi = outputs.topk(1)
            accuracy += (topi == classes.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]
            
            # confusion matrix construction
            oh_labels = F.one_hot(classes, CONFIG['classes'])
            oh_outputs = F.one_hot(topi, CONFIG['classes']).view(-1,CONFIG['classes'])
            confusion_matrix += torch.matmul(torch.transpose(oh_labels, 0, 1), oh_outputs)
    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                           loss /(i+1), accuracy/(i+1)), end="\n")
    
    
    return loss /(i+1), accuracy/(i+1), confusion_matrix


def train_recurrent(input_tensor, target_tensor, network, optimizer, criterion):
    
    optimizer.zero_grad()
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
    loss = 0
    # TODO: Solve the timestep handling as a function parameter
    timesteps = CONFIG['time_depth'] + 1
    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
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
            

def test_recurrent(test_loader, network, criterion, epoch):
    loss = 0
    accuracy = 0
    confusion_matrix = torch.zeros(
        CONFIG['classes'], CONFIG['classes'], dtype=torch.int64)
    class_probs = []
    class_preds = []

    # TODO: Solve the unroll-timestep handling as a function parameter
    timesteps = CONFIG['time_depth'] + 1 + CONFIG['time_depth_beyond']
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, classes = data
            inputs, classes = inputs.to(device), classes.to(device)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.repeat(1, timesteps, 1, 1, 1)
            
            outputs = network(inputs)
            for t in range(outputs.shape[1]):
                loss += criterion(outputs[:,t,:], classes)
            loss /= test_loader.batch_size
            # outputs at other timesteps?
            topv, topi = outputs[:,t,:].topk(1)
            accuracy += (topi == classes.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]
            
            
            # confusion matrix construction
            oh_labels = F.one_hot(classes, CONFIG['classes'])
            oh_outputs = F.one_hot(topi, CONFIG['classes']).view(-1,CONFIG['classes'])
            confusion_matrix += torch.matmul(torch.transpose(oh_labels, 0, 1), oh_outputs)
            
            # pr curve construction
            class_probs_batch = [F.softmax(el, dim=1) for el in outputs]
            
            class_probs.append(class_probs_batch)
            class_preds.append(topi)
            
    # pr-curves
    test_probs = torch.cat([torch.stack(b) for b in class_probs]).view(-1, CONFIG['classes'])
    test_preds = torch.cat(class_preds).view(-1)

    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                       loss /(i+1), accuracy/(i+1)), end="\n")
    return loss /(i+1), accuracy/(i+1), confusion_matrix, test_probs, test_preds


def evaluate_recurrent(test_loader, network, criterion, epoch):
    #TODO: write a function that evaluates given a test set and
    # returns loss, accuracy, while writing down several properties, like
    # softmax output, hidden representation, etc, maybe look at saturn
    pass

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
            test_loss, test_accurary, cm, test_probs, test_preds = test_recurrent(test_loader, network, criterion, epoch)
            cm_figure = visualizer.cm_to_figure(cm, CONFIG['class_encoding'])
            writer.add_scalar('testing/loss', test_loss,
                              epoch * len_of_data)
            writer.add_scalar(
                'testing/accuracy', test_accurary, epoch * len_of_data)
            writer.add_figure('testing/confusionmatrix', cm_figure, global_step=epoch * len_of_data, close=True, walltime=None)
            for i in range(CONFIG['classes']):
                visualizer.add_pr_curve_tensorboard(CONFIG['class_encoding'], i, test_probs, test_preds, writer, global_step=epoch * len_of_data)
            writer.close()
        start = time.time()
        for i_batch, sample_batched in enumerate(train_loader):

            loss, accuracy = train_recurrent(sample_batched[0], sample_batched[1],
                                   network, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss
            print_accuracy_total += accuracy
            plot_accuracy_total += accuracy

            if (epoch * len_of_data + i_batch) % print_every == 0:
                divisor = 1 if (epoch * len_of_data + i_batch) // print_every == 0 else print_every
                print_loss_avg = print_loss_total / divisor
                print_loss_total = 0
                print_accuracy_avg = print_accuracy_total / divisor
                print_accuracy_total = 0
                print(" " * 80 + "\r" +
                      '[Training:] E%d: %s (%d %d%%) %.4f %.4f'
                      % (epoch, timeSince(start, (i_batch + 1) / len_of_data),
                          i_batch, (i_batch + 1) / len_of_data * 100,
                          print_loss_avg, print_accuracy_avg), end="\r")

            if (epoch * len_of_data + i_batch) % plot_every == 0:
                divisor = 1 if (epoch * len_of_data + i_batch) // plot_every == 0 else plot_every
                plot_loss_avg = plot_loss_total / divisor
                plot_loss_total = 0
                plot_accuracy_avg = plot_accuracy_total / divisor
                plot_accuracy_total = 0

                plot_losses.append(plot_loss_avg)
                
                writer.add_scalar(
                    'training/loss', plot_loss_avg,
                      epoch * len_of_data + i_batch)
                writer.add_scalar(
                    'training/accuracy', plot_accuracy_avg, epoch * len_of_data + i_batch)
                writer.close()
                
            
        if epoch % save_every == 0:
                checkpoint(epoch, network, checkpoint_dir + 'network', save_every)
    
    writer.close()


# -----------------
# Main Training Loop
# -----------------

# Training network
# network = B_Network().to(device)
#network = BH_Network().to(device)
network = RecConvNet(CONFIG['connectivity'], kernel_size=(3,3), n_features=32).to(device)

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=1)


# Tensorboard Writer

output_dir, checkpoint_dir = helper.get_output_directory(CONFIG, FLAGS)
loss_writer = SummaryWriter(output_dir)


trainEpochs(train_loader, test_loader, network, loss_writer, CONFIG['epochs'],
            test_every=CONFIG['test_every'], print_every=CONFIG['write_every'], plot_every=CONFIG['write_every'], save_every=5, learning_rate=CONFIG['learning_rate'], output_dir=output_dir, checkpoint_dir=checkpoint_dir)


torch.save(network.state_dict(), checkpoint_dir + 'network.model')
# TODO: Better Checkpointing and saving



# _____________________________________________________________________________

# train_dataset = datasets.MNIST(root=CONFIG['input_dir'] + '/dynaMO/', train=True, download=True, transform=transforms.Compose([
    #        transforms.ToTensor(),
    #        transforms.Normalize((0.,), (1.,))
    #    ]))
    # 
    # test_dataset = datasets.MNIST(root=CONFIG['input_dir'] + '/dynaMO', train=False, transform=transforms.Compose([
        #        transforms.ToTensor(),
        #        transforms.Normalize((0.,), (1.,))
        #    ]))

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
