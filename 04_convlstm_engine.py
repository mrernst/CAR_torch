#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# protoengine.py                             oN88888UU[[[/;::-.        dP^
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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
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


# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap='Greys')
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))

def matplotlib_imshow(img, one_channel=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        return (fig, ax, ax.imshow(npimg, cmap='Greys'))
    else:
        return (fig, ax, ax.imshow(np.transpose(npimg, (1,2,0))))


def checkpoint(epoch, model, experiment_dir, save_every, remove_last=True):
    model_out_path = experiment_dir + "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("[Info:] Checkpoint saved to {}".format(model_out_path, end='\n'))
    if (epoch > 0 and remove_last):
        os.remove(experiment_dir +
                  "model_epoch_{}.pth".format(epoch - save_every))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Encoder and Decoder Classes of the network
# -----------------


class DecoderNetwork(ConvLSTM):
    """docstring for DecoderNetwork."""

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(DecoderNetwork, self).__init__(
            input_dim, hidden_dim, kernel_size, num_layers,
            batch_first=batch_first, bias=bias,
            return_all_layers=return_all_layers)
        self.input_conv = nn.Conv2d(1, hidden_dim, (5, 5), padding=2)
        self.one_by_one = nn.Conv2d(hidden_dim*num_layers, 1, (1, 1))

    def forward(self, input_tensor, hidden_state=None,
                use_teacher_forcing=True):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        time_output_list = []
        # copy hidden state from encoder onto the input_tensor

        # ----
        # improvisation to get the same shapes with a convolution
        seq_len = input_tensor.size(1)
        input_tensor_list = []
        for t in range(seq_len):
            input_tensor_list.append(self.input_conv(input_tensor[:, t, :, :, :]))
        input_tensor = torch.stack(input_tensor_list, dim=1)
        # ----

        input_tensor = torch.cat(
            [hidden_state[0][0].unsqueeze(1), input_tensor], dim=1)
        # make current input the hidden state
        cur_layer_input = input_tensor[:, 0, :, :, :]
        seq_len = input_tensor.size(1)
        # print(input_tensor.shape)
        # print(cur_layer_input.shape)
        for t in range(seq_len):
            output_inner = []
            for layer_idx in range(self.num_layers):
                if (layer_idx == 0) and use_teacher_forcing:
                    cur_layer_input = input_tensor[:, t, :, :, :]
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                                 cur_state=[h, c])
                hidden_state[layer_idx] = h, c
                cur_layer_input = h

                output_inner.append(h)
                # print(h.shape)
            stacked_output = torch.cat(output_inner, dim=1)
            time_output_list.append(self.one_by_one(stacked_output))

        return torch.stack(time_output_list, dim=1)


# -----------------
# Functions for Training and Evaluation
# -----------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length):
    #encoder_hidden = encoder.initHidden(input_tensor.size(0))

    # split data
    input_tensor, ground_truth = torch.split(input_tensor.unsqueeze(2), input_tensor.shape[1]//2, dim=1)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    accuracy = 0
    encoder_output, encoder_hidden = encoder(input_tensor)

    teacher_forcing_ratio = .5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoder_output = decoder(ground_truth, encoder_hidden, use_teacher_forcing)
    decoder_output = decoder_output[:,:-1,:,:,:]
    loss += criterion(decoder_output, ground_truth)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), accuracy, torch.stack([decoder_output[0], ground_truth[0]], dim=0)


def trainEpochs(dataloader, encoder, decoder, writer, n_epochs, max_length,
                print_every=1, plot_every=100, save_every=5, learning_rate=0.01,
                experiment_dir='./experiments/protoengine_experiment_1/data/config0/'):
    plot_losses = []
    print_loss_total = 0
    print_accuracy_total = 0
    plot_loss_total = 0
    plot_accuracy_total = 0

    len_of_data = len(dataloader)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    for epoch in range(n_epochs):
        start = time.time()
        for i_batch, sample_batched in enumerate(dataloader):

            loss, accuracy, sample = train(sample_batched['image'], sample_batched['target'], encoder, decoder,
                                   encoder_optimizer, decoder_optimizer, criterion, max_length)

            print_loss_total += loss
            plot_loss_total += loss
            print_accuracy_total += accuracy
            plot_accuracy_total += accuracy

            if i_batch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_accuracy_avg = print_accuracy_total / print_every
                print_accuracy_total = 0
                print(" " * 80 + "\r" + '[Training:] E%d: %s (%d %d%%) %.4f %.4f' % (epoch, timeSince(start, (i_batch + 1) / len_of_data),
                                                                                     i_batch, (i_batch + 1) / len_of_data * 100, print_loss_avg, print_accuracy_avg), end="\r")

            if i_batch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_accuracy_avg = plot_accuracy_total / plot_every

                plot_losses.append(plot_loss_avg)
                writer.add_scalar('training loss', plot_loss_avg,
                                  epoch * len(dataloader) + i_batch)
                writer.add_scalar(
                    'training accuracy', plot_accuracy_avg, epoch * len(dataloader) + i_batch)
                b,t,c,h,w = sample.shape
                #print(sample.view([b,c,h,t*w]).shape)
                img_grid = torchvision.utils.make_grid(sample.view([b,c,t*h,w]), nrow=2, normalize=True, scale_each=True)
                # npimg = img_grid.detach().numpy()
                # plt.imshow(np.transpose(npimg, (1,2,0)))
                # plt.show()
                writer.add_image('sample', img_grid, epoch * len(dataloader) + i_batch)
                plot_loss_total = 0
                plot_accuracy_total = 0
        if epoch % save_every == 0:
            checkpoint(epoch, encoder, experiment_dir + 'encoder_', save_every)
            checkpoint(epoch, decoder, experiment_dir + 'decoder_', save_every)

    writer.close()


def evaluate(encoder, decoder, sample, predict_for=None, use_teacher_forcing=False):
    with torch.no_grad():

        #encoder_hidden = encoder.initHidden(input_tensor.size(0))

        # split data
        input_tensor, ground_truth = torch.split(sample.unsqueeze(2), sample.shape[1]//2, dim=1)


        loss = 0
        accuracy = 0
        encoder_output, encoder_hidden = encoder(input_tensor)

        b, t, c, h, w = ground_truth.shape
        if predict_for:
            ground_truth = torch.zeros([b, predict_for, c, h, w])
        else:
            predict_for = t
        decoder_output = decoder(ground_truth, encoder_hidden, use_teacher_forcing)
        #decoder_output = decoder_output[:,:-1,:,:,:]
        #loss += criterion(decoder_output, ground_truth)
        plt.ion()
        fig, ax, im = matplotlib_imshow(input_tensor[0,0,0:1,:,:], one_channel=True)

        for step in range(t):
            img = input_tensor[0,step,0:1,:,:]
            img = (img.mean(dim=0) / 2 + 0.5).numpy()
            im.set_array(img)
            fig.canvas.draw()
            plt.pause(.05)
        for step in range(predict_for):
            img = decoder_output[0,step,0:1,:,:]
            img = (img - img.min())/(img.max() - img.min())*255.
            img = (img.mean(dim=0) / 2 + 0.5).numpy()
            im.set_array(img)
            fig.canvas.draw()
            plt.pause(.05)

    return decoder_output


# -----------------
# Main Training Loop
# -----------------

loss_writer = SummaryWriter(
    './experiments/convlstm_experiment_1/data/config0/')

max_length = 20
UNITS=64
LAYERS=2

encoder = ConvLSTM(input_dim=1, hidden_dim=UNITS, kernel_size=(5, 5), num_layers=LAYERS,
                   batch_first=True, bias=True,
                   return_all_layers=True).to(device)
predictor = DecoderNetwork(input_dim=UNITS, hidden_dim=UNITS, kernel_size=(5, 5), num_layers=LAYERS,
                           batch_first=True, bias=True,
                           return_all_layers=True).to(device)

# encoder.load_state_dict(torch.load('./experiments/convlstm_experiment_1/data/config0/models/encoder.model', map_location=torch.device('cpu')))
# encoder.eval()
# predictor.load_state_dict(torch.load('./experiments/convlstm_experiment_1/data/config0/models/predictor.model', map_location=torch.device('cpu')))
# predictor.eval()
# out = evaluate(encoder, predictor, torch.randn([1,16,32,32]))

dynaMo_transformed = dynaMODataset(
    root_dir='./datasets/dynaMO/image_files/train/',
    transform=transforms.Compose([
        ToTensor()
    ]))

dynaMO_dataloader = DataLoader(dynaMo_transformed, batch_size=50,
                               shuffle=True, num_workers=0, drop_last=True)

trainEpochs(dynaMO_dataloader, encoder, predictor, loss_writer,
            n_epochs=10, max_length=max_length, print_every=10, plot_every=100)


torch.save(encoder.state_dict(), './experiments/convlstm_experiment_1/data/config0/models/encoder.model')
torch.save(predictor.state_dict(), './experiments/convlstm_experiment_1/data/config0/models/predictor.model')



# this is just for evaluation purposes..
# for general checkpointing more needs to be saved, i.e.
# ----
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             ...
#             }, PATH)
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
#
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval()
# # - or -
# model.train()
# ----
# you can also save multiple models into one file:
# ----
# torch.save({
#             'modelA_state_dict': modelA.state_dict(),
#             'modelB_state_dict': modelB.state_dict(),
#             'optimizerA_state_dict': optimizerA.state_dict(),
#             'optimizerB_state_dict': optimizerB.state_dict(),
#             ...
#             }, PATH)
# modelA = TheModelAClass(*args, **kwargs)
# modelB = TheModelBClass(*args, **kwargs)
# optimizerA = TheOptimizerAClass(*args, **kwargs)
# optimizerB = TheOptimizerBClass(*args, **kwargs)
#
# checkpoint = torch.load(PATH)
# modelA.load_state_dict(checkpoint['modelA_state_dict'])
# modelB.load_state_dict(checkpoint['modelB_state_dict'])
# optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
# optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
#
# modelA.eval()
# modelB.eval()
# # - or -
# modelA.train()
# modelB.train()


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
