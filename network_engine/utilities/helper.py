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
import math

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
        configuration_dict['image_height'] = 32
        configuration_dict['image_width'] = 32
        configuration_dict['image_channels'] = 3
        configuration_dict['classes'] = 79
        configuration_dict['class_encoding'] = np.array([
            '005_tomato_soup_can',
            '072-a_toy_airplane',
            '065-g_cups',
            '063-b_marbles',
            '027_skillet',
            '036_wood_block',
            '013_apple',
            '073-e_lego_duplo',
            '028_skillet_lid',
            '017_orange',
            '070-b_colored_wood_blocks',
            '015_peach',
            '048_hammer',
            '063-a_marbles',
            '073-b_lego_duplo',
            '035_power_drill',
            '054_softball',
            '012_strawberry',
            '065-b_cups',
            '072-c_toy_airplane',
            '062_dice',
            '040_large_marker',
            '044_flat_screwdriver',
            '037_scissors',
            '011_banana',
            '009_gelatin_box',
            '014_lemon',
            '016_pear',
            '022_windex_bottle',
            '065-c_cups',
            '072-d_toy_airplane',
            '073-a_lego_duplo',
            '065-e_cups',
            '003_cracker_box',
            '065-f_cups',
            '070-a_colored_wood_blocks',
            '073-g_lego_duplo',
            '033_spatula',
            '043_phillips_screwdriver',
            '055_baseball',
            '073-d_lego_duplo',
            '029_plate',
            '052_extra_large_clamp',
            '021_bleach_cleanser',
            '065-a_cups',
            '019_pitcher_base',
            '018_plum',
            '065-h_cups',
            '065-j_cups',
            '065-d_cups',
            '025_mug',
            '032_knife',
            '065-i_cups',
            '026_sponge',
            '071_nine_hole_peg_test',
            '004_sugar_box',
            '056_tennis_ball',
            '038_padlock',
            '053_mini_soccer_ball',
            '059_chain',
            '061_foam_brick',
            '058_golf_ball',
            '006_mustard_bottle',
            '073-f_lego_duplo',
            '031_spoon',
            '051_large_clamp',
            '072-b_toy_airplane',
            '050_medium_clamp',
            '072-e_toy_airplane',
            '042_adjustable_wrench',
            '010_potted_meat_can',
            '024_bowl',
            '073-c_lego_duplo',
            '007_tuna_fish_can',
            '008_pudding_box',
            '057_racquetball',
            '030_fork',
            '002_master_chef_can',
            '077_rubiks_cube'
        ])
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
    
    # if configuration_dict['cropped'] or configuration_dict['augmented']:
    #     configuration_dict['image_height'] = \
    #         configuration_dict['image_width']\
    #         // 10 * 4
    #     configuration_dict['image_width'] = \
    #         configuration_dict['image_width']\
    #         // 10 * 4
    # else:
    #     pass
    
    if 'F' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 64
        configuration_dict['network_depth'] = 2
    elif 'K' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (5,5)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 2
    elif 'Kx' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (6,6)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 2
    elif 'D' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 4
    elif 'GLM' in configuration_dict['connectivity']:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 1
    else:
        configuration_dict['kernel_size'] = (3,3)
        configuration_dict['n_features'] = 32
        configuration_dict['network_depth'] = 2
    
    
    # overwrite the default time_depth if network is not recurrent
    if configuration_dict['connectivity'] in ['B', 'BK', 'BKx', 'BF', 'BD', 'GLM']:
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
            elif ',' in value:
                mult_values = config_dictionary[key].split(',')
                config_dictionary[key] = list([])
                for v in mult_values:
                    config_dictionary[key].append(int(v))
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
    if configuration_dict['lr_decay']:
        if configuration_dict['lr_cosine']:
            architecture_string += '_lr{}-{}-{}'.format(
                configuration_dict['learning_rate'],
                configuration_dict['lr_decay_rate'],
                'cos')
        else:
            architecture_string += '_lr{}-{}-{}'.format(
                configuration_dict['learning_rate'],
                configuration_dict['lr_decay_rate'],
                str(configuration_dict['lr_decay_epochs']).strip(
                    '[]').replace(', ', ','))
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


def adjust_learning_rate(learning_rate, cosine, lr_decay_rate, epochs, lr_decay_epochs, optimizer, epoch):
    lr = learning_rate
    if cosine:
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        if steps > 0:
            lr = lr * (lr_decay_rate ** steps)
            #lr = learning_rate * (lr_decay_rate ** (1/d * epoch)) ?

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(warm, warm_epochs, warmup_from, warmup_to, epoch, batch_id, total_batches, optimizer):
    if warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
