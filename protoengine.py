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
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torchvision import transforms, utils
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
SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    """docstring for EncoderRNN."""

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # variables
        self.hidden_size = hidden_size

        # modules
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size, bias=True)
        self.gru1 = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.linear(input.view(1, -1)).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru1(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


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
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(
            0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# -----------------
# Functions for Training and Evaluation
# -----------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        print(input_tensor[:, ei, :, :].shape, input_tensor[:, ei, :, :].dtype)
        encoder_output, encoder_hidden = encoder(
            input_tensor[:, ei, :, :], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:,di])
            decoder_input = target_tensor[:,di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[:,di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(dataloader, encoder, decoder, n_iters, max_length, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['target'].size())

        loss = train(sample_batched['image'], sample_batched['target'], encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion, max_length)

        print_loss_total += loss
        plot_loss_total += loss

        if i_batch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if i_batch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluate():
    pass


def showAttention():
    pass


def evaluateAndShowAttention():
    pass


# -----------------
# Main Training Loop
# -----------------

hidden_size = 256
max_length = 16
encoder1 = EncoderRNN(32*32, hidden_size).to(device)
attn_decoder1 = AttentionDecoderRNN(
    hidden_size, 3,  max_length=max_length, dropout_p=0.1).to(device)


dynaMo_transformed = dynaMODataset(
    root_dir='/Users/markus/Research/Code/titan/datasets/dynaMO/image_files/test/',
    transform=transforms.Compose([
        ToTensor()
    ]))

dynaMO_dataloader = DataLoader(dynaMo_transformed, batch_size=1,
                        shuffle=True, num_workers=4)

trainIters(dynaMO_dataloader, encoder1, attn_decoder1, 75000, max_length=max_length, print_every=5000)
# for i_batch, sample_batched in enumerate(dynaMO_dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['target'].size())
#     print(sample_batched['image'][:,0,:,:].size())
#     print(sample_batched['image'][:,0,:,:].view(1,1,-1).size())
#     print(sample_batched['image'].dtype)
#     print(sample_batched['target'].dtype)


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
