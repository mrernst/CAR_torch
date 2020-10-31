#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# helper.py                                  oN88888UU[[[/;::-.        dP^
# various helper functions                  dNMMNN888UU[[[/;:--.   .o@P^
# to manage project parameters             ,MMMMMMN888UU[[/;::-. o@^
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
import torch
import numpy as np

import csv
import os
import sys
import errno
import importlib

# custom functions
# -----

import utilities.dataset_handler as dataset_handler
# import visualizer as visualizer


# helper functions
# -----

def mkdir_p(path):
    """
    mkdir_p takes a string path and creates a directory at this path if it
    does not already exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def largest_indices(arr, n):
    """
    Returns the n largest indices from a numpy array.
    """
    flat_arr = arr.flatten()
    indices = np.argpartition(flat_arr, -n)[-n:]
    indices = indices[np.argsort(-flat_arr[indices])]
    return np.unravel_index(indices, arr.shape)


def print_misclassified_objects(cm, encoding, n_obj=5):
    """
    prints out the n_obj misclassified objects given a
    confusion matrix array cm.
    """
    np.fill_diagonal(cm, 0)
    maxind = largest_indices(cm, n_obj)
    most_misclassified = encoding[maxind[0]]
    classified_as = encoding[maxind[1]]
    print('most misclassified:', most_misclassified)
    print('classified as:', classified_as)
    pass


def print_tensor_info(tensor, name=None, writer=None):
    """
    Takes a torch.tensor and returns name and shape for debugging purposes
    """

    if writer:
        writer.add_scalar(name + '/min', tensor.min(), global_step=global_step)
        writer.add_scalar(name + '/max', tensor.max(), global_step=global_step)
        writer.add_scalar(name + '/std', tensor.type(torch.float).std(), global_step=global_step)
        writer.add_scalar(name + '/mean', tensor.type(torch.float).mean(), global_step=global_step)
    else:
        name = name if name else tensor.names
        text = "[DEBUG] name = {}, shape = {}, dtype = {}, device = {} \n" + \
        "\t min = {}, max = {}, std = {}, mean = {}"
        print(text.format(name, list(tensor.shape), tensor.dtype, tensor.device.type,
        tensor.min(), tensor.max(), tensor.type(torch.float).std(), tensor.type(torch.float).mean()))
        
    pass


def infer_additional_parameters(configuration_dict):
    """
    infer_additional_parameters takes a dict configuration_dict and infers
    additional parameters on the grounds of dataset etc.
    """
    # define correct network parameters
    # -----

    # read the number of layers from the network file

    if ('ycb' in configuration_dict['dataset']):
        configuration_dict['image_height'] = 240
        configuration_dict['image_width'] = 320
        configuration_dict['image_channels'] = 3
        configuration_dict['classes'] = 80
        configuration_dict['class_encoding'] = tfrecord_handler.OSYCB_ENCODING
        if configuration_dict['downsampling'] == 'ds2':
            configuration_dict['image_height'] //= 2
            configuration_dict['image_width'] //= 2
        elif configuration_dict['downsampling'] == 'ds4':
            configuration_dict['image_height'] //= 4
            configuration_dict['image_width'] //= 4
    elif (('mnist' in configuration_dict['dataset']) and
          not('os' in configuration_dict['dataset'])):
        configuration_dict['image_height'] = 28
        configuration_dict['image_width'] = 28
        configuration_dict['image_channels'] = 1
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = np.array(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    elif 'cifar10' in configuration_dict['dataset']:
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 3
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = np.array(
            ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
             'frog', 'horse', 'ship', 'truck'])
    else:
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 1
        configuration_dict['classes'] = 10
        configuration_dict['class_encoding'] = np.array(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    if 'fashion' in configuration_dict['dataset']:
        configuration_dict['class_encoding'] = np.array(
            ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    elif 'kuzushiji' in configuration_dict['dataset']:
        pass  # unicode characters not supported
    
    if configuration_dict['color'] == 'grayscale':
        configuration_dict['image_channels'] = 1
    if configuration_dict['stereo']:
        configuration_dict['image_channels'] *= 2
    
    
    # to crop the images
    # store the original values
    configuration_dict['image_height_input'] = \
        configuration_dict['image_height']
    configuration_dict['image_width_input'] = \
        configuration_dict['image_width']
    
    # change the image height and image width if the network is supposed
    
    if configuration_dict['cropped'] or configuration_dict['augmented']:
        configuration_dict['image_height'] = \
            configuration_dict['image_width']\
            // 10 * 4
        configuration_dict['image_width'] = \
            configuration_dict['image_width']\
            // 10 * 4
    else:
        pass
    
    if 'F' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 64
        configuration_dict['network_depth'] = 2
    elif 'K' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (5,5)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 2
    elif 'D' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 4
    else:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 2
    
    
    # overwrite the default time_depth if network is not recurrent
    if configuration_dict['connectivity'] in ['B', 'BK', 'BF', 'BD']:
        configuration_dict['time_depth'] = 0
        configuration_dict['time_depth_beyond'] = 0
    return configuration_dict


def read_config_file(path_to_config_file):
    """
    read_config_file takes a string path_to_config_file and returns a
    dict config_dict with all the keys and values from the csv file.
    """
    config_dict = {}
    with open(path_to_config_file) as config_file:
        csvReader = csv.reader(config_file)
        for key, value in csvReader:
            config_dict[key] = value

    config_dict['config_file'] = path_to_config_file
    return convert_config_types(config_dict)


def convert_config_types(config_dictionary):
    for key, value in config_dictionary.items():
        try:
            if '.' in value:
                config_dictionary[key] = float(value)
            elif ('True' in value) or ('False' in value):
                config_dictionary[key] = value.lower() in \
                    ("yes", "true", "t", "1")
            elif 'None' in value:
                config_dictionary[key] = None
            else:
                config_dictionary[key] = int(value)
        except(ValueError, TypeError):
            pass
    return config_dictionary


def get_output_directory(configuration_dict, flags):
    """
    get_output_directory takes a dict configuration_dict and established the
    directory structure for the configured experiment. It returns paths to
    the checkpoints and the writer directories.
    """

    cfg_name = flags.config_file.split('/')[-1].split('.')[0]
    writer_directory = '{}{}/{}/'.format(
        configuration_dict['output_dir'], cfg_name,
        flags.name)

    # architecture string
    architecture_string = ''
    architecture_string += '{}{}_{}l_fm{}_d{}_l2{}'.format(
        configuration_dict['connectivity'],
        configuration_dict['time_depth'],
        configuration_dict['network_depth'],
        configuration_dict['feature_multiplier'],
        configuration_dict['keep_prob'],
        configuration_dict['l2_lambda'])

    if configuration_dict['batchnorm']:
        architecture_string += '_bn1'
    else:
        architecture_string += '_bn0'
    architecture_string += '_bs{}'.format(configuration_dict['batchsize'])
    if configuration_dict['decaying_lrate']:
        architecture_string += '_lr{}-{}-{}'.format(
            configuration_dict['lr_eta'],
            configuration_dict['lr_delta'],
            configuration_dict['lr_d'])
    else:
        architecture_string += '_lr{}'.format(
            configuration_dict['learning_rate'])

    # data string
    data_string = ''
    if ('ycb' in configuration_dict['dataset']):
        data_string += "{}_{}occ_{}p".format(
            configuration_dict['dataset'],
            configuration_dict['n_occluders'],
            configuration_dict['occlusion_percentage'])
    else:
        data_string += "{}_{}occ_Xp".format(
            configuration_dict['dataset'],
            configuration_dict['n_occluders'])

    # format string
    format_string = ''
    format_string += '{}x{}x{}'.format(
        configuration_dict['image_height'],
        configuration_dict['image_width'],
        configuration_dict['image_channels'])
    format_string += "_{}_{}".format(
        configuration_dict['color'],
        configuration_dict['label_type'])

    writer_directory += "{}/{}/{}/".format(architecture_string,
                                           data_string, format_string)

    checkpoint_directory = writer_directory + 'checkpoints/'

    # make sure the directories exist, otherwise create them
    mkdir_p(checkpoint_directory)
    mkdir_p(checkpoint_directory + 'evaluation/')

    return writer_directory, checkpoint_directory



def compile_list_of_train_summaries():
    pass


def compile_list_of_test_summaries():
    pass


def compile_list_of_image_summaries():
    pass


def compile_list_of_additional_summaries():
    pass


def get_and_merge_summaries():
    pass


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
