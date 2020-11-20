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

from torch.utils.data import Dataset, DataLoader, Subset
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
import utilities.afterburner as afterburner
import utilities.helper as helper
import utilities.visualizer as visualizer
from utilities.networks.buildingblocks.rcnn import RecConvNet, CAM
from utilities.dataset_handler import StereoImageFolderLMDB, StereoImageFolder


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
     default='i1',
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
# Functions for Checkpointing, Training, Testing and Evaluation
# -----------------

def checkpoint(epoch, model, optimizer, ckpt_dir, save_every, remove_last=True):
    # write dict with model and optimizer parameters
    state = {
        'epoch': epoch + 1, 'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_out_path = ckpt_dir + "checkpoint_epoch_{}.pt".format(epoch)
    #torch.save(model.state_dict(), model_out_path)
    torch.save(state, ckpt_out_path)

    #print("[INFO] Checkpoint saved to {}".format(ckpt_out_path, end='\n'))
    #print("[INFO] Checkpoint saved.".format(ckpt_out_path, end='\n'))

    if (epoch > 0 and remove_last):
        try:
            os.remove(ckpt_dir +
                  "checkpoint_epoch_{}.pt".format(epoch - save_every))
        except(FileNotFoundError):
            print('[INFO] ' +
                  "Old checkpoint_epoch_{}.pt could not be found/deleted".format(epoch - save_every))


def load_checkpoint(model, optimizer, ckpt_dir):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    
    def sort_key(string):
        return int(string.split('.')[0].split('_')[-1])
    
    
    list_of_ckpts = [f for f in os.listdir(ckpt_dir) if '.pt' in f]
    list_of_ckpts.sort(key=sort_key, reverse=True)
    
    if len(list_of_ckpts) > 0:
        final_checkpoint = list_of_ckpts[0]
        checkpoint = torch.load(os.path.join(ckpt_dir, final_checkpoint), map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("[INFO] Loaded checkpoint '{}' (continue: epoch {})"
                  .format(final_checkpoint, checkpoint['epoch']))
    else:
        #print("[INFO] No checkpoint found at '{}'".format(ckpt_dir))
        print("[INFO] No checkpoint found, starting from scratch")


    return model, optimizer, start_epoch


def train_recurrent(input_tensor, target_tensor, network, optimizer, criterion, timesteps, stereo):
    
    optimizer.zero_grad()
    if stereo:
        input_tensor = torch.cat(input_tensor, dim=1)
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
    loss = 0
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
    
def test_recurrent(test_loader, network, criterion, epoch, timesteps, stereo):
    loss = 0
    accuracy = 0
    
    confusion_matrix = visualizer.ConfusionMatrix(n_cls=network.fc.out_features)
    precision_recall = visualizer.PrecisionRecall(n_cls=network.fc.out_features)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_tensor, target_tensor = data
            if stereo:
                input_tensor = torch.cat(input_tensor, dim=1)
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            input_tensor = input_tensor.unsqueeze(1)
            input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
            
            outputs, _ = network(input_tensor)
            topv, topi = outputs[:,-1,:].topk(1)
            # other timesteps?
            for t in range(outputs.shape[1]):
                loss += criterion(outputs[:,t,:], target_tensor) / topi.shape[0]

            accuracy += (topi == target_tensor.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]
            
            
            # update confusion matrix
            confusion_matrix.update(outputs[:,-1,:].cpu(), target_tensor.cpu())
               
            # update pr curves
            precision_recall.update(outputs[:,-1,:].cpu(), target_tensor.cpu())
    
    visual_prediction = visualizer.plot_classes_preds(outputs[:,-1,:].cpu(), input_tensor[:,-1,:,:,:].cpu(), target_tensor.cpu(), CONFIG['class_encoding'])
    #visual_prediction = None
    print(" " * 80 + "\r" + '[Testing:] E%d: %.4f %.4f' % (epoch,
                                                       loss /(i+1), accuracy/(i+1)), end="\n")
    return loss /(i+1), accuracy/(i+1), confusion_matrix, precision_recall, visual_prediction


def evaluate_recurrent(dataset, network, batch_size, criterion, timesteps, stereo, projector=False):
    # create a random but deterministic order for the dataset (important for bc)
    torch.manual_seed(1234)
    shuffled_dataset = Subset(dataset, torch.randperm(len(dataset)).tolist())
    eval_loader = DataLoader(shuffled_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    def show(img):
        import matplotlib.pyplot as plt
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()
    
    loss = 0
    accuracy = 0
    list_of_output_tensors = []
    list_of_bc_values = []
    

    
    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            input_tensor, target_tensor = data
            if stereo:
                input_tensor = torch.cat(input_tensor, dim=1)
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            # Show a grid of images
            # show(torchvision.utils.make_grid(input_tensor, padding=8))
            input_tensor = input_tensor.unsqueeze(1)
            input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
            
            outputs, _ = network(input_tensor)
            topv, topi = outputs[:,-1,:].topk(1)
            # other timesteps?
            for t in range(outputs.shape[1]):
                loss += criterion(outputs[:,t,:], target_tensor) / topi.shape[0]
            
            accuracy += (topi == target_tensor.unsqueeze(1)).sum(
                dim=0, dtype=torch.float64) / topi.shape[0]
            
            list_of_bc_values.append(torch.eq(
                torch.argmax(outputs[:,-1,:], 1),
                target_tensor))
            list_of_output_tensors.append(F.softmax(outputs, dim=-1))
    
        bc_values = torch.cat(list_of_bc_values, 0).type(torch.int8)
        output_values = torch.cat(list_of_output_tensors, 0)
        
        print(" " * 80 + "\r" + '[Evaluation:] E%d: %.4f %.4f' % (-1,
           loss /(i+1), accuracy/(i+1)), end="\n")
        
    evaluation_data = \
    {'boolean_classification': np.array(bc_values.cpu()),
     'softmax_output': np.array(output_values.cpu())}    
    
    embedding_data = None # TODO implement at some point
    return evaluation_data, embedding_data


def test_cams(test_loader, network, timesteps, stereo):
    
    cam = CAM(network)
    loss = 0
    accuracy = 0
    
    confusion_matrix = visualizer.ConfusionMatrix(n_cls=network.fc.out_features)
    precision_recall = visualizer.PrecisionRecall(n_cls=network.fc.out_features)

    # TODO: Solve the unroll-timestep handling as a function parameter
    #timesteps = CONFIG['time_depth'] + 1 + CONFIG['time_depth_beyond']

    with torch.no_grad():
        cam_list = []
        target_list = []
        topk_prob_list = []
        topk_pred_list = []
        
        for i, data in enumerate(test_loader):
            
            input_tensor, target_tensor = data
            if stereo:
                input_tensor = torch.cat(input_tensor, dim=1)
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            input_tensor = input_tensor.unsqueeze(1)
            input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
            
            outputs, (cams, topk_prob, topk_pred) = cam(input_tensor)            
            cam_list.append(cams)
            target_list.append(target_tensor)
            topk_prob_list.append(topk_prob)
            topk_pred_list.append(topk_pred)
            
            if i > 50:
                break
        
        visualizer.show_cam_samples(cams, input_tensor, target_tensor, topk_prob, topk_pred, n_samples=5)
        # lists to tensors
        cams = torch.cat(cam_list, dim=0)
        target_tensor = torch.cat(target_list, dim=0)
        topk_prob = torch.cat(topk_prob_list, dim=0)
        topk_pred = torch.cat(topk_pred_list, dim=0)
        
        visualizer.show_cam_means(cams, target_tensor, topk_prob, topk_pred)
        
        # filter correct predictions - best topk at last timestep = target
        correct_indices = (target_tensor == topk_pred[:, -1, 0])
        # show means for correct predictions
        visualizer.show_cam_means(
            cams[correct_indices],
            target_tensor[correct_indices],
            topk_prob[correct_indices],
            topk_pred[correct_indices]
            )
        # visual_prediction = visualizer.plot_classes_preds(outputs[:,-1,:].cpu(), input_tensor[:,-1,:,:,:].cpu(), target_tensor.cpu(), CONFIG['class_encoding'], CONFIG['image_channels'])
    pass

def trainEpochs(train_loader, test_loader, network, optimizer, criterion, writer, start_epoch, n_epochs, test_every, print_every, log_every, save_every, learning_rate, lr_decay, lr_cosine, lr_decay_rate, lr_decay_epochs, output_dir, checkpoint_dir):
    plot_losses = []
    print_loss_total = 0
    print_accuracy_total = 0
    plot_loss_total = 0
    plot_accuracy_total = 0
    
    len_of_data = len(train_loader)
        
    for epoch in range(start_epoch, n_epochs):
        if epoch % test_every == 0:
            test_loss, test_accurary, cm, pr, vp = test_recurrent(test_loader, network, criterion, epoch, CONFIG['time_depth'] + 1 + CONFIG['time_depth_beyond'], CONFIG['stereo'])
            
            writer.add_scalar('testing/loss', test_loss,
                              epoch * len_of_data)
            writer.add_scalar(
                'testing/accuracy', test_accurary, epoch * len_of_data)
            network.log_stats(writer, epoch * len_of_data)
            
            cm.to_tensorboard(writer, CONFIG['class_encoding'], epoch)
            #cm.print_misclassified_objects(CONFIG['class_encoding'], 5)
            pr.to_tensorboard(writer, CONFIG['class_encoding'], epoch)
            writer.add_figure('predictions vs. actuals', vp, epoch)
            writer.close()
            
        start = time.time()
        if lr_decay:
            helper.adjust_learning_rate(
                learning_rate,
                lr_cosine,
                lr_decay_rate,
                n_epochs,
                lr_decay_epochs,
                optimizer,
                epoch)
        for i_batch, sample_batched in enumerate(train_loader):
            loss, accuracy = train_recurrent(
                sample_batched[0], sample_batched[1],
                network, optimizer, criterion, CONFIG['time_depth'] + 1, CONFIG['stereo'])

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

            if (epoch * len_of_data + i_batch) % log_every == 0:
                divisor = 1 if (epoch * len_of_data + i_batch) // log_every == 0 else log_every
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
                checkpoint(epoch, network, optimizer, checkpoint_dir + 'network', save_every)
    
    if start_epoch < n_epochs:
        # final test after training; do not test if restarting from the same epoch
        test_loss, test_accurary, cm, pr, vp = test_recurrent(test_loader, network, criterion, n_epochs, CONFIG['time_depth'] + 1 + CONFIG['time_depth_beyond'], CONFIG['stereo'])
        
        writer.add_scalar('testing/loss', test_loss,
                          n_epochs * len_of_data)
        writer.add_scalar(
            'testing/accuracy', test_accurary, n_epochs * len_of_data)
        network.log_stats(writer, n_epochs * len_of_data)
        
        cm.to_tensorboard(writer, CONFIG['class_encoding'], n_epochs)
        cm.print_misclassified_objects(CONFIG['class_encoding'], 5)
        pr.to_tensorboard(writer, CONFIG['class_encoding'], n_epochs)
        writer.add_figure('predictions vs. actuals', vp, n_epochs)
        writer.close()
    
    checkpoint(n_epochs, network, optimizer, checkpoint_dir + 'network', save_every)


# -----------------
# Main Program
# -----------------

# state_dict = torch.load('/Users/markus/Research/Code/titan/datasets/BLT3_osmnist2r_ep100.pt', map_location=torch.device('cpu'))
# 
# 
# BLT3_osfmnist2r = '/home/mernst/git/titan/experiments/010_osfmnist2r_rcnn_comparison/data/config12/i1/BLT3_2l_fm1_d1.0_l20.0_bn1_bs500_lr0.001/osfmnist2r_2occ_Xp/32x32x1_grayscale_onehot/checkpoints/networkmodel_epoch_100.pt'
# 
# BLT3_osfmnist2c = '/home/mernst/git/titan/experiments/009_osfmnist2c_rcnn_comparison/data/config12/i1/BLT3_2l_fm1_d1.0_l20.0_bn1_bs500_lr0.001/osfmnist2c_2occ_Xp/32x32x1_grayscale_onehot/checkpoints/networkmodel_epoch_100.pt'
# 
# BLT3_osmnist2r = '/home/mernst/git/titan/experiments/008_osmnist2r_rcnn_comparison/data/config6/i1/BLT3_2l_fm1_d1.0_l20.0_bn1_bs500_lr0.001/osmnist2r_2occ_Xp/32x32x1_grayscale_onehot/checkpoints/networkmodel_epoch_100.pt'
# 
# BLT3_osmnist2c = '/home/mernst/git/titan/experiments/007_osmnist2c_rcnn_comparison/data/config6/i1/BLT3_2l_fm1_d1.0_l20.0_bn1_bs500_lr0.001/osmnist2c_2occ_Xp/32x32x1_grayscale_onehot/checkpoints/networkmodel_epoch_100.pt'
# 
# B_osmnist2r = '/home/mernst/git/titan/experiments/008_osmnist2r_rcnn_comparison/data/config0/i1/B0_2l_fm1_d1.0_l20.0_bn1_bs500_lr0.001/osmnist2r_2occ_Xp/32x32x1_grayscale_onehot/checkpoints/networkmodel_epoch_100.pt'
# 
# B_osfmnist2r = '/home/mernst/git/titan/experiments/010_osfmnist2r_rcnn_comparison/data/config0/i1/B0_2l_fm1_d1.0_l20.0_bn1_bs500_lr0.001/osfmnist2r_2occ_Xp/32x32x1_grayscale_onehot/checkpoints/networkmodel_epoch_100.pt'
# 
# 
# # state_dict = torch.load(BLT3_osmnist2c, map_location=torch.device('cpu'))
# 
# network.load_state_dict(state_dict)
# # network.eval() # network evaluation, does not work for recurrent models because of BN


# input transformation

if CONFIG['color'] == 'grayscale':
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ])
    test_transform = train_transform
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ])
    test_transform = train_transform


if CONFIG['dataset'] == 'mnist':
    train_dataset = datasets.MNIST(root=CONFIG['input_dir'], train=True,
    transform=transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ,]),
    download=True)
    test_dataset = datasets.MNIST(root=CONFIG['input_dir'], train=False,
    transform=transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))
    ,]),
    download=True)
else:
    # Datasets LMDB Style
    try:
        train_dataset = StereoImageFolderLMDB(
            db_path=CONFIG['input_dir'] + '/{}/{}_train.lmdb'.format(CONFIG['dataset'], CONFIG['dataset']),
            stereo=CONFIG['stereo'],
            transform=train_transform
            )
        
        test_dataset = StereoImageFolderLMDB(
            db_path=CONFIG['input_dir'] + '/{}/{}_test.lmdb'.format(CONFIG['dataset'], CONFIG['dataset']),
            stereo=CONFIG['stereo'],
            transform=test_transform
            )
    except:
        print('[INFO] No LMDB-file available, using standard folder instead')
        # Datasets direct import
        train_dataset = StereoImageFolder(
            root_dir=CONFIG['input_dir'] + '/{}'.format(CONFIG['dataset']),
            train=True,
            stereo=CONFIG['stereo'],
            transform=train_transform
            )
            
        test_dataset = StereoImageFolder(
            root_dir=CONFIG['input_dir'] + '/{}'.format(CONFIG['dataset']),
            train=False,
            stereo=CONFIG['stereo'],
            transform=test_transform
            )



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batchsize'], shuffle=True, num_workers=4)


output_dir, checkpoint_dir = helper.get_output_directory(CONFIG, FLAGS)
stats_writer = SummaryWriter(output_dir)


# configure network
network = RecConvNet(
    CONFIG['connectivity'],
    kernel_size=CONFIG['kernel_size'], input_channels=CONFIG['image_channels'],
    n_features=CONFIG['n_features'],
    num_layers=CONFIG['network_depth'], 
    num_targets=CONFIG['classes']
    ).to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(network.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['l2_lambda'])

if FLAGS.restore_ckpt:
    network, optimizer, start_epoch = load_checkpoint(network, optimizer, checkpoint_dir)
else:
    start_epoch = 0


# training loop

trainEpochs(
        train_loader, test_loader, network, optimizer, criterion,
        writer=stats_writer,
        start_epoch=start_epoch,
        n_epochs=CONFIG['epochs'],
        test_every=CONFIG['test_every'],
        print_every=CONFIG['write_every'],
        log_every=CONFIG['write_every'],
        save_every=CONFIG['test_every'],
        learning_rate=CONFIG['learning_rate'],
        lr_decay=CONFIG['lr_decay'],
        lr_cosine=CONFIG['lr_cosine'], 
        lr_decay_rate=CONFIG['lr_decay_rate'],
        lr_decay_epochs=CONFIG['lr_decay_epochs'],
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir
    )



# evaluation and afterburner
# -----

evaluation_data, embedding_data = evaluate_recurrent(test_dataset, network, CONFIG['batchsize'], criterion, CONFIG['time_depth'] + 1, CONFIG['stereo'])

essence = afterburner.DataEssence()
essence.distill(path=output_dir, evaluation_data=evaluation_data,
                embedding_data=None)  # embedding_data (save space)
essence.write_to_file(filename=CONFIG['output_dir'] +
                      FLAGS.config_file.split('/')[-1].split('.')[0] +
                      '{}'.format(FLAGS.name) + '.pkl')
essence.plot_essentials(CONFIG['output_dir'].rsplit('/', 2)[0] +
                        '/visualization/' +
                        FLAGS.config_file.split('/')[-1].split('.')[0] +
                        '{}'.format(FLAGS.name) + '.pdf')



# _____________________________________________________________________________



# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
