#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# March 2020                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# rc.py                                      oN88888UU[[[/;::-.        dP^
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
import torch.nn as nn
import torch.nn.functional as F
import torch

# custom functions
# -----

# -----------------
# Network Constructor
# -----------------

def constructor(name,
                configuration_dict,
                is_training,
                keep_prob,
                custom_net_parameters=None):
    """
    constructor takes a name, a configuration dict the booleans is_training,
    keep_prob and optionally a custom_net_parameters dict and returns
    an initialized NetworkClass instance.
    """
    assert False, 'not yet implemented for pytorch'

    def get_net_parameters(configuration_dict):
        net_param_dict = {}
        receptive_pixels = 3  # 3
        n_features = 32
        feature_multiplier = configuration_dict['feature_multiplier']

        if "F" in configuration_dict['connectivity']:
            n_features = 64
        if "K" in configuration_dict['connectivity']:
            receptive_pixels = 5  # 5

        net_param_dict["activations"] = [bb.lrn_relu, bb.lrn_relu, tf.identity]
        net_param_dict["conv_filter_shapes"] = [
            [receptive_pixels, receptive_pixels,
                configuration_dict['image_channels'], n_features],
            [receptive_pixels, receptive_pixels, n_features,
                configuration_dict['feature_multiplier'] * n_features]
                                               ]
        net_param_dict["bias_shapes"] = [
            [1, configuration_dict['image_height'],
                configuration_dict['image_width'], n_features],
            [1, int(np.ceil(configuration_dict['image_height']/2)),
                int(np.ceil(configuration_dict['image_width']/2)),
                configuration_dict['feature_multiplier']*n_features],
            [1, configuration_dict['classes']]]
        net_param_dict["ksizes"] = [
            [1, 2, 2, 1], [1, 2, 2, 1]]
        net_param_dict["pool_strides"] = [[1, 2, 2, 1], [1, 2, 2, 1]]
        net_param_dict["topdown_filter_shapes"] = [
            [receptive_pixels, receptive_pixels, n_features,
                feature_multiplier * n_features]]
        net_param_dict["topdown_output_shapes"] = [
            [configuration_dict['batchsize'],
                configuration_dict['image_height'],
                configuration_dict['image_width'],
                n_features]]

        net_param_dict["global_weight_init_mean"] = \
            configuration_dict['global_weight_init_mean']
        net_param_dict["global_weight_init_std"] = \
            configuration_dict['global_weight_init_std']

        return net_param_dict

    if custom_net_parameters:
        net_parameters = custom_net_parameters
    else:
        net_parameters = get_net_parameters(configuration_dict)

    # copy necessary items from configuration
    net_parameters['connectivity'] = configuration_dict['connectivity']
    net_parameters['batchnorm'] = configuration_dict['batchnorm']

    return NetworkClass(name, net_parameters, is_training, keep_prob)



# -----------------
# RC Classes
# -----------------


class RecConvCell(nn.Module):
    """docstring for RCCell."""

    def __init__(self, connectivity, input_channels, output_channels, output_channels_above, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_channels: int
            Number of channels of input tensor.
        output_channels: int
            Number of channels of the output tensor.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(RecConvCell, self).__init__()
        
        self.connectivity = connectivity
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_channels_above = output_channels_above

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.bottomup = nn.Conv2d(in_channels=self.input_channels,
                                  out_channels=self.output_channels,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

        self.lateral = nn.Conv2d(in_channels=self.output_channels,
                                 out_channels=self.output_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=self.bias)

        self.topdown = nn.ConvTranspose2d(in_channels=self.output_channels_above,
                                          out_channels=self.output_channels,
                                          kernel_size=self.kernel_size,
                                          stride=(2,2),
                                          padding=1,
                                          output_padding=1,
                                          groups=1,
                                          bias=True,
                                          dilation=1,
                                          padding_mode='zeros')

        self.bn_b = nn.BatchNorm2d(self.output_channels)
        self.bn_l = nn.BatchNorm2d(self.output_channels)
        self.bn_t = nn.BatchNorm2d(self.output_channels)
        self.bn_all = nn.BatchNorm2d(self.output_channels)

        self.activation = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=1.0)


    def forward(self, b_input, l_input, t_input):
        b_conv = self.bn_b(self.bottomup(b_input))

        if 'BLT' in self.connectivity:
            l_conv = self.bn_l(self.lateral(l_input))
            t_conv = self.bn_t(self.topdown(t_input))
            next_state = self.lrn(self.activation(b_conv + l_conv + t_conv))
        elif 'BL' in self.connectivity:
            l_conv = self.bn_l(self.lateral(l_input))
            next_state = self.lrn(self.activation(b_conv + l_conv))

        elif 'BT' in self.connectivity:
            t_conv = self.bn_t(self.topdown(t_input))
            next_state = self.lrn(self.activation(b_conv + t_conv))
        else:
            next_state = self.lrn(self.activation(b_conv))
            
        return next_state

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        l = torch.zeros(batch_size,
                            self.output_channels, height, width,
                            device=self.bottomup.weight.device)
        td = torch.zeros(batch_size,
                    self.output_channels_above, height//2, width//2,
                    device=self.topdown.weight.device)
        return (l, td)




class RecConv(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, connectivity, input_dim, hidden_dim, kernel_size,
        num_layers, batch_first=False, bias=True, pooling=True):
        super(RecConv, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers + 1)
        if not len(kernel_size) == len(hidden_dim) - 1 == num_layers:
            raise ValueError('Inconsistent list length.')

        if not(pooling) and num_layers > 1:
            raise ValueError('Multiple layers without pooling are not supported')
        # TODO: implement multiple layers without pooling

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.pooling = pooling

        cell_list = []
        pooling_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(RecConvCell(connectivity=connectivity,
                                         input_channels=cur_input_dim,
                                         output_channels=self.hidden_dim[i],
                                         output_channels_above=self.hidden_dim[i+1],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        if self.pooling:
            self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, input_tensor, hidden_state=None):
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
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w),
                                             pooling=self.pooling)
        
        seq_len = input_tensor.size(1)
        output_inner = []
    
        for t in range(seq_len):
            layer_output_list = []
            cur_layer_input = input_tensor[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                l, td = hidden_state[layer_idx]
                
                cur_layer_input = self.cell_list[layer_idx](
                    b_input=cur_layer_input,
                    l_input=l,
                    t_input=td)
                    
                layer_output_list.append(cur_layer_input)

                if self.pooling:
                    cur_layer_input = self.maxpool(cur_layer_input)



            # update hidden states
            for layer_idx in range(self.num_layers - 1):
                hidden_state[layer_idx] = (layer_output_list[layer_idx], layer_output_list[layer_idx+1])
            hidden_state[self.num_layers - 1] = (layer_output_list[self.num_layers - 1], hidden_state[self.num_layers - 1][-1])

            output_inner.append(cur_layer_input)

        return torch.stack(output_inner, dim=1)

    def _init_hidden(self, batch_size, image_size, pooling):
        init_states = []
        if pooling:
            init_states.append(self.cell_list[0].init_hidden(batch_size, image_size))
            for i in range(1, self.num_layers):
                init_states.append(self.cell_list[i].init_hidden(batch_size, tuple(dim//(2*i) for dim in image_size)))
        else:
            for i in range(self.num_layers):
                init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class RecConvNet(nn.Module):
    def __init__(self, connectivity, kernel_size, input_channels=1, n_features=32, num_layers=2):
        super(RecConvNet, self).__init__()
        self.rcnn = RecConv(connectivity, input_channels, n_features, kernel_size, num_layers, batch_first=True)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.rcnn(x)
        seq_len = x.size(1)
        output_list = []
        for t in range(seq_len):
            input = x[:, t, :, :, :].view(x.shape[0], 32 * 8 * 8)
            output_list.append(F.softmax(self.fc(input), 1))
        x = torch.stack(output_list, dim=1)
        return x


class B_Network(nn.Module):
    def __init__(self, kernel=3, filters=32):
        self.kernel = kernel
        self.filters = filters
        super(B_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, self.filters, self.kernel, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.conv2 = nn.Conv2d(self.filters, self.filters, self.kernel, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.bn2 = nn.BatchNorm2d(self.filters)
        self.fc1 = nn.Linear(self.filters * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # print(x.shape)

        x = x.view(-1, 32 * 8 * 8)
        # print(x.shape)

        x = F.softmax(self.fc1(x), 1)
        # print(x.shape)

        return x



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

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
