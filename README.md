# CAR_torch
## Occluded Object Recognition Codebase

<p align="center">
  <img src="https://github.com/mrernst/CAR_torch/blob/master/img/OSCAR_fmnist.png" width="375">

CAR_torch stands for Convolutional Architectures with Recurrence, pytorch implementation. It is the codebase used for the journal publication "Recurrent Processing Improves Occluded Object Recognition and Gives Rise to Perceptual Hysteresis" [1]. 
If you make use of this code please cite as follows:
 

[1] **Ernst, M. R., Burwick, T., & Triesch, J. (2021). Recurrent Processing Improves Occluded Object Recognition and Gives Rise to Perceptual Hysteresis. In Journal of Vision**


## Getting started with the repository

* Download the OSCAR version 2 datasets from Zenodo and put the in their respective folders in /datasets [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4085133.svg)](https://doi.org/10.5281/zenodo.4085133)
* Configure the config.py file
* Start an experiment on a slurm cluster using run_engine.py or on your local machine with engine.py

### Prerequisites

* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikitlearn](http://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [pytorch](https://www.pytorch.org/)


### Directory structure

```bash
.
├── datasets                          
│   ├── cifar10                      # CIFAR10
│   ├── cifar100                     # CIFAR100
│   ├── mnist                        # MNIST
│   ├── dynaMO                       # Dynamic Occluded MNIST (experimental)
│   ├── osfashionmnist2c             # OS-fashion-MNIST
│   ├── osmnist2c                    # OS-MNIST
│   ├── osfashionmnist2r             # OS-fashion-MNIST
│   ├── osmnist2r                    # OS-MNIST
│   ├── osycb2                       # OS-YCB
├── network_engine                    
│   ├── utilities             		    
│   │   ├── afterburner.py            # Combines experiment files post-hoc
│   │   ├── dataset_handler.py        # Pytorch dataloaders for different datasets
│   │   ├── helper.py                 # Helper functions
│   │   ├── metrics.py                # Distance metrics for high-dim. analysis
│   │   ├── publisher.py              # Create 'Paper-ready' plots      
│   │   ├── visualizer.py             # Visualization functions
│   │   ├── networks
│   │   │   ├── buildingblocks
│   │   │   │   ├── rcnn.py           # Dynamic network for recurrent networks
│   │   │   │   ├── convlstm.py       # Convolutional LSTM Networks (experimental)
│   ├── engine.py                     # Main Program
│   ├── config.py             		  # Experiment Parameters 
│   ├── run_engine.py                 # Setup and Run Experiments
├── experiments                       # Experiment saves
├── LICENSE                           # MIT License
├── README.md                         # ReadMe File
└── requirements.txt                  # conda/pip requirements
```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite SORN folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.9](https://www.python.org/downloads/release/python-395/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `recurrentnetworks`) and install the [required packages](requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.7`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
