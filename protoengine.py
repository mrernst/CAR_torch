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

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

def checkpoint(epoch, model, experiment_dir, save_every, remove_last=True):
    model_out_path = experiment_dir + "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("[Info:] Checkpoint saved to {}".format(model_out_path, end='\n'))
    if (epoch > 0 and remove_last):
        os.remove(experiment_dir + "model_epoch_{}.pth".format(epoch-save_every))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Encoder and Decoder Classes of the network
# -----------------

class EncoderRNN(nn.Module):
    """docstring for EncoderRNN."""

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # variables
        self.hidden_size = hidden_size

        # modules
        self.linear = nn.Linear(input_size, hidden_size, bias=True)
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        input = input.view(input.shape[0], -1)
        linear = self.linear(input)
        linear = linear.view(input.shape[0], 1, -1)
        output = linear
        output, hidden = self.gru1(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttentionDecoderRNN(nn.Module):
    """docstring for AttentionDecoderRNN."""

    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        # variables
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # modules
        self.linear = nn.Linear(self.output_size, self.hidden_size, bias=True)
        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #print('#########################')
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        embedded = torch.nn.functional.one_hot(input, 10).float()
        embedded = self.linear(embedded)
        embedded = self.dropout(embedded)
        #print('embedded', embedded.shape)
        #print('hidden', hidden.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:,0], hidden[0]), 1)), dim=1)
        #print('attn_weights', attn_weights.shape)
        #print('encoder_outputs', encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(
            1), encoder_outputs)
        #print('attn_applied', attn_applied.shape)
        output = torch.cat((embedded[:,0], attn_applied[:,0]), 1)
        output = self.attn_combine(output).unsqueeze(1)
        #print('output', output.shape)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #print('output', output.shape)
        output = F.log_softmax(self.out(output[:,0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# -----------------
# Functions for Training and Evaluation
# -----------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden(input_tensor.size(0))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    encoder_outputs = torch.zeros(input_tensor.size(0),
        input_length, encoder.hidden_size, device=device)

    loss = 0
    accuracy = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[:, ei, :, :], encoder_hidden)

        encoder_outputs[:,ei] = encoder_output[:,0]

    #decoder_input = torch.tensor([[0]], device=device)
    #decoder_input = torch.zeros(input_tensor.shape[0],10, device=device)
    decoder_input = torch.zeros(input_tensor.shape[0],1, dtype=torch.int64, device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:,di])
            decoder_input = target_tensor[:,di:di+1]
            topv, topi = decoder_output.topk(1)
            accuracy += (topi == target_tensor[:,di:di+1]).sum(dim=0, dtype=torch.float64)


    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi #topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[:,di])
            accuracy += (topi == target_tensor[:,di:di+1]).sum(dim=0, dtype=torch.float64)
            #print(topi[0], target_tensor[0,di:di+1])
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, accuracy / (target_tensor.shape[0] * target_tensor.shape[1])


def trainEpochs(dataloader, encoder, decoder, writer, n_epochs, max_length,
                print_every=1000, plot_every=100, save_every=5, learning_rate=0.01,
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
    for epoch in range(n_epochs):
        start = time.time()
        for i_batch, sample_batched in enumerate(dataloader):

            loss, accuracy = train(sample_batched['image'], sample_batched['target'], encoder, decoder,
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
                print(" " * 80 + "\r" + '[Training:] E%d: %s (%d %d%%) %.4f %.4f' % (epoch, timeSince(start, (i_batch+1) / len_of_data),
                                             i_batch, (i_batch+1) / len_of_data * 100, print_loss_avg, print_accuracy_avg), end="\r")

            if i_batch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_accuracy_avg = plot_accuracy_total / plot_every

                plot_losses.append(plot_loss_avg)
                writer.add_scalar('training loss', plot_loss_avg, epoch * len(dataloader) + i_batch)
                writer.add_scalar('training accuracy', plot_accuracy_avg, epoch * len(dataloader) + i_batch)

                plot_loss_total = 0
                plot_accuracy_total = 0
        if epoch % save_every == 0:
            checkpoint(epoch, encoder, experiment_dir + 'encoder_', save_every)
            checkpoint(epoch, decoder, experiment_dir + 'decoder_', save_every)

    writer.close()


def evaluate(encoder, decoder, sample):
    with torch.no_grad():
        input_tensor = sample
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(batch_size,
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[:, ei, :, :], encoder_hidden)

            encoder_outputs[:,ei] += encoder_output[:,0]

        decoder_input = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)

        decoder_hidden = encoder_hidden
        decoded_digits = []

        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            decoded_digits.append(topi)
            decoder_input = topi #topi.squeeze().detach()

    return decoded_digits, decoder_attentions[:di + 1]




def showAttention():
    pass


def evaluateAndShowAttention():
    pass


# -----------------
# Main Training Loop
# -----------------

loss_writer = SummaryWriter('./experiments/protoengine_experiment_1/data/config0/')

hidden_size = 256
max_length = 16
encoder1 = EncoderRNN(32*32, hidden_size).to(device)
attn_decoder1 = AttentionDecoderRNN(
    hidden_size, output_size=10,  max_length=max_length, dropout_p=0.1).to(device)


dynaMo_transformed = dynaMODataset(
    root_dir='./datasets/dynaMO/image_files/train/',
    transform=transforms.Compose([
        ToTensor()
    ]))

dynaMO_dataloader = DataLoader(dynaMo_transformed, batch_size=100,
                        shuffle=True, num_workers=0, drop_last=True)

trainEpochs(dynaMO_dataloader, encoder1, attn_decoder1, loss_writer,
            n_epochs=101, max_length=max_length, print_every=100)

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
