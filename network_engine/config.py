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
# Copyright 2021 Markus Ernst
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
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

    par['exp_name'] = ["noname_experiment"]
    # par['name'] must be defined as a FLAG to engine, b/c it resembles the
    # iteration number that gets passed by the sbatch script
    # TODO: add documentation i.e. parameter possibilities
    par['dataset'] = ["osmnist2r_reduced"] #osmnist2 #ycb1_single
    par['n_occluders'] = [2] #2
    par['occlusion_percentage'] = [0]
    par['label_type'] = ["onehot"] #["onehot"]
    par['connectivity'] = ['B', 'BF', 'BK', 'BD', 'BT', 'BL', 'BLT'] #['B', 'BF', 'BK', 'BT', 'BL', 'BLT'] # ['BD', 'BT', 'BL', 'BLT'] # ['B', 'BF', 'BK', 'BD', 'BT', 'BL', 'BLT'] #['BLT']
    par['BLT_longrange'] = [0]
    par['time_depth'] = [3]
    par['time_depth_beyond'] = [0]
    par['feature_multiplier'] = [1]
    par['keep_prob'] = [1.0]

    par['batchnorm'] = [True]
    par['stereo'] = [False]
    par['color'] = ['grayscale'] #color
    # par['cropped'] = [False]
    # par['augmented'] = [False]

    par['write_every'] = [100] # 500
    par['test_every'] = [5] # 5
    par['buffer_size'] = [600000] #[600000]
    par['verbose'] = [False]
    par['visualization'] = [False] #False
    par['projector'] = [False]

    par['batchsize'] = [500] #500
    par['epochs'] = [100]
    par['learning_rate'] = [0.004]

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
    aux['norm_by_stat'] = [False]
    aux['training_dir'] = [""] # "all"
    aux['validation_dir'] = [""] # ""
    aux['test_dir'] = [""] # ""
    aux['evaluation_dir'] = [""] # ""

    aux['lr_decay'] = [True]
    aux['lr_cosine'] = [False]
    aux['lr_decay_epochs'] = ['90,'] # ['60, 75, 90'] # ['90,']
    aux['lr_decay_rate'] = [0.1]
    aux['l2_lambda'] = [0.] # 0.0005
    # old parameters for Spoerer Like decay
    # aux['lr_eta'] = [0.1]
    # aux['lr_delta'] = [0.1]
    # aux['lr_d'] = [40.]
    aux['global_weight_init_mean'] = ['None'] #[1.0, 0.0]
    aux['global_weight_init_std'] = ['None']
    # Info: None-Values have to be strings b/c of csv text conversion
    
    #aux['num_workers'] = [4]
    aux['iterations'] = [1] # 5
    return aux


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
