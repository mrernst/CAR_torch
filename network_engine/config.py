#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# config.py                                  oN88888UU[[[/;::-.        dP^
# set and get experiment parameters         dNMMNN888UU[[[/;:--.   .o@P^
#                                          ,MMMMMMN888UU[[/;::-. o@^
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

# custom functions
# -----
from platform import system
IS_MACOSX = True if system() == 'Darwin' else False
PWD_STEM = "/Users/markus/Research/Code/" if IS_MACOSX else "/home/mernst/git/"


# --------------------------
# main experiment parameters
# --------------------------

def get_par():
    """
    Get main parameters.
    For each experiment, change these parameters manually for different
    experiments.
    """

    par = {}

    par['exp_name'] = ["mnist_rcnn"]
    # par['name'] must be defined as a FLAG to engine, b/c it resembles the
    # iteration number that gets passed by the sbatch script
    # TODO: add documentation i.e. parameter possibilities
    par['dataset'] = ["mnist"] #osmnist2 #ycb1_single

    par['n_occluders'] = [0] #2
    par['occlusion_percentage'] = [0]
    par['label_type'] = ["onehot"] #["onehot"]
    par['connectivity'] = ['BLT', 'BL', 'BT']#['B', 'BF', 'BK', 'BT', 'BL', 'BLT'] # ['BD', 'BT', 'BL', 'BLT'] # ['B', 'BF', 'BK', 'BD', 'BT', 'BL', 'BLT'] #['BLT']
    par['BLT_longrange'] = [0]
    par['time_depth'] = [3, 10, 20]
    par['time_depth_beyond'] = [0]
    par['feature_multiplier'] = [1]
    par['keep_prob'] = [1.0]

    par['stereo'] = [False]
    par['downsampling'] = ['ds4'] #fine
    par['color'] = ['grayscale'] #color
    par['cropped'] = [False]
    par['augmented'] = [False]

    par['write_every'] = [500] # 500
    par['test_every'] = [1] # 5
    par['buffer_size'] = [600000] #[600000]
    par['verbose'] = [False]
    par['visualization'] = [False] #False
    par['projector'] = [False]

    par['batchsize'] = [500] #500
    par['epochs'] = [25]
    par['learning_rate'] = [0.003]

    return par

# ----------------------------
# auxiliary network parameters
# ----------------------------


def get_aux():
    """
    Get auxiliary parameters.
    These auxiliary parameters do not have to be changed manually for the most
    part. Configure once in the beginning of setup.
    """

    aux = {}
    aux['wdir'] = ["{}titan/".format(PWD_STEM)]
    aux['input_dir'] = ["{}titan/datasets/".format(PWD_STEM)]
    # aux['input_dir'] = ["/home/aecgroup/aecdata/Textures/occluded/datasets/"]
    aux['output_dir'] = ["{}titan/experiments/".format(PWD_STEM)]
    # aux['output_dir'] = ["/home/aecgroup/aecdata/Results_python/markus/experiments/"]
    aux['network_module'] = ["utilities.networks.proto_net"]
    aux['norm_by_stat'] = [False]
    aux['training_dir'] = [""] # "all"
    aux['validation_dir'] = [""] # ""
    aux['test_dir'] = [""] # ""
    aux['evaluation_dir'] = [""] # ""

    aux['decaying_lrate'] = [False]
    aux['lr_eta'] = [0.1]
    aux['lr_delta'] = [0.1]
    aux['lr_d'] = [40.]
    aux['l2_lambda'] = [0.]
    aux['batchnorm'] = [True]
    aux['global_weight_init_mean'] = ['None'] #[1.0, 0.0]
    aux['global_weight_init_std'] = ['None']
    # Info: None-Values have to be strings b/c of csv text conversion

    aux['iterations'] = [1] # 5
    return aux


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
