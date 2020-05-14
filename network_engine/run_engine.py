#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# run_engine.py                              oN88888UU[[[/;::-.        dP^
# setup and initialization                  dNMMNN888UU[[[/;:--.   .o@P^
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
import numpy as np
import os
import errno
import shutil
import csv
import itertools
import argparse


# custom functions
# -----
from config import get_par, get_aux


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


# commandline arguments
# -----


parser = argparse.ArgumentParser()
parser.add_argument(
     "-tbp",
     "--tensorboard_port",
     type=int,
     default=6006,
     help='port for tensorboard monitoring')
parser.add_argument(
     "-gpus",
     "--number_of_gpus",
     type=int,
     default=1,
     help='activate gpu acceleration')
parser.add_argument(
     "-nodes",
     "--number_of_nodes",
     type=int,
     default=1,
     help='spread jobs on more nodes')
parser.add_argument(
     "-mem",
     "--memory",
     type=int,
     default=20,
     help='memory to be reserved (GB)')
args = parser.parse_args()


class SbatchDocument(object):
    """docstring for SbatchDocument."""

    def __init__(self, paths_to_config_files, files_dir, experiment_name,
                 iterations):
        super(SbatchDocument, self).__init__()
        self.experiment_name = experiment_name
        self.files_dir = files_dir
        self.iterations = iterations
        self.write_to_file(paths_to_config_files)

    def gen_sbatch(self, paths_to_config_files):
        """
        gen_sbatch takes a parameter dict and a auxiliary_parameters dict and
        generates a sbatch file for the corresponding experiment
        """

        bash_array = '('
        for p in paths_to_config_files:
            bash_array += '"' + p + '" '
        bash_array = bash_array.strip()
        bash_array += ')'

        # 10GB for mnist 15 for osmnist
        header = \
            "#!/bin/bash \n" + \
            "# \n" + \
            "#SBATCH --nodes=1 \n" + \
            "#SBATCH --ntasks-per-node=1 \n" + \
            "#SBATCH --cpus-per-task=4 \n" + \
            "#SBATCH --time=700:00:00 \n" + \
            "#SBATCH --mem={}GB \n".format(args.memory) + \
            "#SBATCH --reservation triesch-shared \n" + \
            "#SBATCH --partition=sleuths \n" + \
            "#SBATCH --job-name={} \n".format(self.experiment_name) + \
            "#SBATCH --mail-type=END \n" + \
            "#SBATCH --mail-user=mernst@fias.uni-frankfurt.de \n" + \
            "#SBATCH --output={}slurm_output/{}_slurm_%j.out \n".format(
                self.files_dir, self.experiment_name) + \
            "#SBATCH --array=0-{}%{} \n".format(len(paths_to_config_files)-1,
                                                args.number_of_nodes)
        if args.number_of_gpus > 0:
            header += "#SBATCH --gres=gpu:rtx2080ti:{} \n\n".format(args.number_of_gpus)
        else:
            header += "\n"

        middle = \
            'config_array={} \n'.format(bash_array) + \
            'j=$((SLURM_ARRAY_TASK_ID)) \n' + \
            'for i in `seq 1 1 {}` \n'.format(self.iterations) + \
            'do \n' + \
            '    echo "iteration $i" \n' + \
            '    echo "job $j" \n' + \
            '    srun python3 engine.py' + \
            ' --config_file ${config_array[$j]}' + \
            ' --name i$i\n' + \
            'done \n'

        footer = \
            '# --- end of experiment --- \n'
        # TODO: think about management for afterburner. Lookup if experiment is
        # still running. Then proceed with afterburner. -> self.config_dir

        return (header + middle + footer)

    def write_to_file(self, paths_to_config_files):
        file = open("{}/run_experiment.sbatch".format(self.files_dir), "w")
        file.write(self.gen_sbatch(paths_to_config_files))
        file.close()
        pass

    def run_sbatch(self):
        os.chdir(self.files_dir)
        os.system("sbatch run_experiment.sbatch")
        print("sbatch {}run_experiment.sbatch".format(self.files_dir))
        print("[INFO] running {} on cluster".format(self.experiment_name))
        pass


class ExperimentEnvironment(object):
    """docstring for ExperimentEnvironment."""

    def __init__(self, parameters):
        super(ExperimentEnvironment, self).__init__()
        self.parameters = parameters
        self.working_directory = self.parameters['wdir'][0]
        self.output_directory = self.parameters["output_dir"][0]
        self.experiment_name = self.parameters["exp_name"][0]

        self.est_folder_structure()
        self.copy_experiment_files()

    def update_parameters(self):
        self.parameters['output_dir'] = [self.data_dir]
        return self.parameters

    def est_folder_structure(self):
        """
        est_folders establishes a folder structure given the parameters
        of the configuration
        """

        list_of_previous_experiment_folders = os.listdir(self.output_directory)
        try:
            list_of_previous_experiment_folders.sort()
            list_of_previous_experiment_folders.remove(".DS_Store")
        except(ValueError):
            pass
        if len(list_of_previous_experiment_folders) == 0:
            experiment_number = 1
        else:
            experiment_number = \
                int(list_of_previous_experiment_folders[-1].split('_')[0]) + 1

        self.experiment_dir = self.output_directory + \
            "{0:0=3d}".format(experiment_number) + \
            "_{}/".format(self.experiment_name)

        self.config_dir = self.experiment_dir + "files/config_files/"
        self.files_dir = self.experiment_dir + "files/"
        self.data_dir = self.experiment_dir + "data/"
        self.visualization = self.experiment_dir + "visualization/"

        self.slurm_dir = self.files_dir + "slurm_output/"
        mkdir_p(self.experiment_dir)
        mkdir_p(self.data_dir)
        mkdir_p(self.visualization)
        mkdir_p(self.slurm_dir)
        mkdir_p(self.config_dir)

    def copy_experiment_files(self):
        shutil.copyfile(self.working_directory +
                        "/network_engine/engine.py", self.files_dir +
                        "engine.py")
        shutil.copyfile(self.working_directory +
                        "/network_engine/config.py", self.files_dir +
                        "config.py")
        shutil.copytree(self.working_directory +
                        "/network_engine/utilities", self.files_dir +
                        "utilities",
                        ignore=shutil.ignore_patterns('tsne', '__pycache__',
                                                      '*.pyc', 'tmp*'))
        pass


class ExperimentConfiguration(object):
    """docstring for ExperimentConfiguration."""

    def __init__(self, parameters):
        super(ExperimentConfiguration, self).__init__()
        self.infer_additional_parameters(parameters)

    def infer_additional_parameters(self, parameters):
        # this was implemented as a helper function and is not part of the
        # run_engine file
        self.parameters = parameters
        pass

    def generate_single_configurations(self):
        number_of_configs = np.prod([len(v) for v in self.parameters.values()])
        a = list(self.parameters.values())
        combinations = list(itertools.product(*a))
        return list(self.parameters.keys()), combinations

    def write_config_files(self, path_to_config_folder=''):
        paths_to_config_files = []
        keys, cfs = self.generate_single_configurations()
        for i in range(len(cfs)):
            w = csv.writer(open(path_to_config_folder +
                                "config{}.csv".format(i), "w"))
            for j in range(len(keys)):
                w.writerow([keys[j], cfs[i][j]])
            paths_to_config_files.append(
                path_to_config_folder + "config{}.csv".format(i))

        return paths_to_config_files


# ------------
# main program
# ------------


if __name__ == '__main__':

    par, aux = get_par(), get_aux()
    par.update(aux)

    # generate main experiment structure
    # -----

    environment = ExperimentEnvironment(par)
    par = environment.update_parameters()
    # generate configuration files
    # -----
    config = ExperimentConfiguration(par)
    config_paths = config.write_config_files(environment.config_dir)

    # generate sbatch file
    # -----

    sbatch_file = SbatchDocument(config_paths, environment.files_dir,
                                 environment.experiment_name,
                                 par['iterations'][0])
    sbatch_file.run_sbatch()

    # start a tensorboard instance to monitor experiment
    # -----
    os.system("screen -dmS tb_monitor")
    os.system("screen -S tb_monitor -p 0 -X stuff \
        'tf14\n '")
    os.system("screen -S tb_monitor -p 0 -X stuff \
        'tensorboard --logdir {} --port {}\n '".format(
        environment.data_dir, args.tensorboard_port))
    print("[INFO] monitoring {} in tensorboard at port {}".format(
        environment.experiment_name, args.tensorboard_port))

# _____________________________________________________________________________
# Description:
#
# This program is supposed to create an folder structure for the experiment,
# copy the necessary files to that environment, create an sbatch file and run
# the file given the environment-variables predefined in an external document.
# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
