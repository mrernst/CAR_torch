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

# for MNIST
from torchvision import utils, datasets

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
from utilities.networks.buildingblocks.rcnn import RecConvNet, CAM

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

class RandomData(Dataset):
    
    def __init__(self, length=45, timesteps=4, constant_over_time=False, transform=None):
        self.length = length
        self.timesteps = timesteps
        self.transform = transform
        if constant_over_time:
            input_tensor = torch.randint(255,[length, 1, 32, 32], dtype=torch.float)
            input_tensor = input_tensor.unsqueeze(1)
            self.data = input_tensor.repeat(1, timesteps, 1, 1, 1)
        else:
            self.data = torch.randint(255,[length, timesteps, 1, 32, 32], dtype=torch.float)
        self.labels = torch.randint(9, [length], dtype=torch.float)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

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
# Functions for Training, Testing and Evaluation
# -----------------


def train_recurrent(input_tensor, target_tensor, network, optimizer, criterion):
    
    optimizer.zero_grad()
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
    loss = 0
    # TODO: Solve the timestep handling as a function parameter
    timesteps = CONFIG['time_depth'] + 1
    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
    outputs, _ = network(input_tensor)
    
    for t in range(outputs.shape[1]):
        loss += criterion(outputs[:,t,:], target_tensor)
        
    topv, topi = outputs[:,t,:].topk(1)
    accuracy = (topi == target_tensor.unsqueeze(1)).sum(
        dim=0, dtype=torch.float64) / topi.shape[0]
        
    loss.backward()
    optimizer.step()
    loss = loss / topi.shape[0]  # average loss per item
   
    return loss.item(), accuracy.item()   
    
def test_recurrent(test_loader, network, criterion, epoch):
    loss = 0
    accuracy = 0
    
    confusion_matrix = visualizer.ConfusionMatrix(n_cls=CONFIG['classes'])
    precision_recall = visualizer.PrecisionRecall(n_cls=CONFIG['classes'])

    # TODO: Solve the unroll-timestep handling as a function parameter
    timesteps = CONFIG['time_depth'] + 1 + CONFIG['time_depth_beyond']
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, classes = data
            inputs, classes = inputs.to(device), classes.to(device)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.repeat(1, timesteps, 1, 1, 1)
            
            outputs, _ = network(inputs)
            for t in range(outputs.shape[1]):
                loss += criterion(outputs[:,t,:], classes)
            loss /= test_loader.batch_size
            # outputs at other timesteps?
            topv, topi = outputs[:,t,:].topk(1)
            accuracy += (topi == classes.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]
            
            
            # update confusion matrix
            confusion_matrix.update(outputs[:,-1,:].cpu(), classes.cpu())
               
            # update pr curves
            precision_recall.update(outputs[:,-1,:].cpu(), classes.cpu())

    #visual_prediction = visualizer.plot_classes_preds(outputs[:,-1,:].cpu(), inputs.cpu(), classes.cpu(), CONFIG['class_encoding'], CONFIG['image_channels'])
    visual_prediction = None
    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                       loss /(i+1), accuracy/(i+1)), end="\n")
    return loss /(i+1), accuracy/(i+1), confusion_matrix, precision_recall, visual_prediction


def test_cam(test_loader, network):
    
    cam = CAM(network)
    loss = 0
    accuracy = 0
    
    confusion_matrix = visualizer.ConfusionMatrix(n_cls=CONFIG['classes'])
    precision_recall = visualizer.PrecisionRecall(n_cls=CONFIG['classes'])

    # TODO: Solve the unroll-timestep handling as a function parameter
    timesteps = CONFIG['time_depth'] + 1 + CONFIG['time_depth_beyond']
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, classes = data
            inputs, classes = inputs.to(device), classes.to(device)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.repeat(1, timesteps, 1, 1, 1)
            
            outputs, cam_dict = cam(inputs)
            cam_dict['maps'] = torch.stack(cam_dict['maps'])
            # print(cam_dict['maps'].shape)
            import matplotlib.pyplot as plt
            # stacked = np.reshape(cam_dict['maps'][:,0,:,:], [32*4, 32])
            # stacked = np.concatenate([inputs[0,0,0,:,:], stacked], axis=0)
            # plt.imshow(stacked)
            # plt.show()
            fig, ax = visualizer.saliencymap_to_figure(cam_dict['maps'], inputs[0,0,0,:,:])
            plt.show()
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
            test_loss, test_accurary, cm, pr, vp = test_recurrent(test_loader, network, criterion, epoch)
            writer.add_scalar('testing/loss', test_loss,
                              epoch * len_of_data)
            writer.add_scalar(
                'testing/accuracy', test_accurary, epoch * len_of_data)
            cm.to_tensorboard(writer, CONFIG['class_encoding'], epoch)
            cm.print_misclassified_objects(CONFIG['class_encoding'], 5)
            pr.to_tensorboard(writer, CONFIG['class_encoding'], epoch)
            #writer.add_figure('predictions vs. actuals', vp, epoch)
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
    

network = RecConvNet(CONFIG['connectivity'], kernel_size=CONFIG['kernel_size'], n_features=CONFIG['n_features']).to(device)



# Datasets
# train_dataset = ImageFolderLMDB(
#     db_path=CONFIG['input_dir'] + '/dynaMO/data/osmnist2/train.lmdb',
#     transform=transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.,), (1.,))
# ]))
# 
# test_dataset = ImageFolderLMDB(
#     db_path=CONFIG['input_dir'] + '/dynaMO/data/osmnist2/test.lmdb',
#     transform=transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.,), (1.,))
# ]))

train_dataset = datasets.MNIST(root=CONFIG['input_dir'] + '/dynaMO/data/mnist/', train=True, download=True, transform=transforms.Compose([
           transforms.CenterCrop(32),
           transforms.ToTensor(),
           transforms.Normalize((0.,), (1.,)),
       ]))
    
test_dataset = datasets.MNIST(root=CONFIG['input_dir'] + '/dynaMO/data/mnist/', train=False, transform=transforms.Compose([
           transforms.CenterCrop(32),
           transforms.ToTensor(),
           transforms.Normalize((0.,), (1.,)),
       ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=1)



# test_cam(test_loader, network)


output_dir, checkpoint_dir = helper.get_output_directory(CONFIG, FLAGS)
loss_writer = SummaryWriter(output_dir)

trainEpochs(train_loader, test_loader, network, loss_writer, CONFIG['epochs'],
            test_every=CONFIG['test_every'], print_every=CONFIG['write_every'], plot_every=CONFIG['write_every'], save_every=5, learning_rate=CONFIG['learning_rate'], output_dir=output_dir, checkpoint_dir=checkpoint_dir)


torch.save(network.state_dict(), checkpoint_dir + 'network.pt')



# evaluation of network (to be outsourced at some point)
# -----

# redefine model
# torch.load model
# evaluate network

# _____________________________________________________________________________



# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
