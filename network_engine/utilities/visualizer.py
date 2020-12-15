#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# June 2020                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# visualizer.py                              oN88888UU[[[/;::-.        dP^
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
import torch
import numpy as np

import sys, os, re
import itertools
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import offsetbox, patches
from matplotlib.markers import MarkerStyle
import seaborn as sns


import scipy.optimize as opt
from PIL import Image
from textwrap import wrap
from math import sqrt

# van der Maaten TSNE implementations
import utilities.tsne.bhtsne as bhtsne
import utilities.tsne.tsne as tsne



# ----------------
# create anaglyphs
# ----------------


# anaglyph configurations
# -----

_magic = [0.299, 0.587, 0.114]
_zero = [0, 0, 0]
_ident = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]


true_anaglyph = ([_magic, _zero, _zero], [_zero, _zero, _magic])
gray_anaglyph = ([_magic, _zero, _zero], [_zero, _magic, _magic])
color_anaglyph = ([_ident[0], _zero, _zero],
                  [_zero, _ident[1], _ident[2]])
half_color_anaglyph = ([_magic, _zero, _zero],
                       [_zero, _ident[1], _ident[2]])
optimized_anaglyph = ([[0, 0.7, 0.3], _zero, _zero],
                      [_zero, _ident[1], _ident[2]])
methods = [true_anaglyph, gray_anaglyph, color_anaglyph, half_color_anaglyph,
           optimized_anaglyph]


def anaglyph(npimage1, npimage2, method=half_color_anaglyph):
    """
    anaglyph takes to numpy arrays of shape [H,W,C] and optionally a anaglyph
    method and returns a resulting PIL Image and a numpy composite.

    Example usage:
        im1, im2 = Image.open("left-eye.jpg"), Image.open("right-eye.jpg")

        ana, _ = anaglyph(im1, im2, half_color_anaglyph)
        ana.save('output.jpg', quality=98)
    """
    m1, m2 = [np.array(m).transpose() for m in method]

    if (npimage1.shape[-1] == 1 and npimage2.shape[-1] == 1):
        im1, im2 = np.repeat(npimage1, 3, -1), np.repeat(npimage2, 3, -1)
    else:
        im1, im2 = npimage1, npimage2

    composite = np.matmul(im1, m1) + np.matmul(im2, m2)
    result = Image.fromarray(composite.astype('uint8'))

    return result, composite


# ---------------------
# make custom colormaps
# ---------------------


def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


# ---------------------
# make custom image annotations
# ---------------------

# helper function to make markers
def makeMarker(image, zoom=.65):
    return offsetbox.OffsetImage(image,zoom=zoom, cmap='Greys_r')

class ImageAnnotations3D():
    def __init__(self, xyz, imgs, ax3d,ax2d):
        self.xyz = xyz
        self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
        for s,im in zip(self.xyz, self.imgs):
            x,y = self.proj(s)
            self.annot.append(self.image(im,[x,y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

        self.funcmap = {"button_press_event" : self.ax3d._button_press,
                        "motion_notify_event" : self.ax3d._on_move,
                        "button_release_event" : self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                        for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1, 
            calculate position in 2D in ax2 """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=0.5)
        im.image.axes = self.ax3d
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                            xycoords='data', boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->"), frameon=True)
        self.ax2d.add_artist(ab)
        return ab

    def update(self,event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                        np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s,ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)





class ConfusionMatrix(object):
        """Holds and updates a confusion matrix given the networks
        outputs"""
        def __init__(self, n_cls):
            self.n_cls = n_cls
            self.reset()
    
        def reset(self):
            self.val = torch.zeros(self.n_cls, self.n_cls, dtype=torch.float32)
    
        def update(self, batch_output, batch_labels):
            _, topi = batch_output.topk(1)
            oh_labels = torch.nn.functional.one_hot(batch_labels, self.n_cls)
            oh_outputs = torch.nn.functional.one_hot(topi, self.n_cls).view(-1, self.n_cls)
            self.val += torch.matmul(torch.transpose(oh_labels, 0, 1), oh_outputs)
    
        def print_misclassified_objects(self, encoding, n_obj=5):
            """
            prints out the n_obj misclassified objects given a
            confusion matrix array cm.
            """
            cm = self.val.numpy()
            encoding = np.array(encoding)
            
            np.fill_diagonal(cm, 0)
            maxind = self.largest_indices(cm, n_obj)
            most_misclassified = encoding[maxind[0]]
            classified_as = encoding[maxind[1]]
            print('most misclassified:', most_misclassified)
            print('classified as:', classified_as)
            pass
        
        def largest_indices(self, arr, n):
            """
            Returns the n largest indices from a numpy array.
            """
            flat_arr = arr.flatten()
            indices = np.argpartition(flat_arr, -n)[-n:]
            indices = indices[np.argsort(-flat_arr[indices])]
            return np.unravel_index(indices, arr.shape)
            
        def to_figure(self, labels, title='Confusion matrix',
                         normalize=False,
                         colormap='Oranges'):
            """
            Parameters:
                confusion_matrix                : Confusionmatrix Array
                labels                          : This is a list of labels which will
                                                  be used to display the axis labels
                title='confusion matrix'        : Title for your matrix
                tensor_name = 'MyFigure/image'  : Name for the output summary tensor
                normalize = False               : Renormalize the confusion matrix to
                                                  ones
                colormap = 'Oranges'            : Colormap of the plot, Oranges fits
                                                  with tensorboard visualization
        
        
            Returns:
                summary: TensorFlow summary
        
            Other items to note:
                - Depending on the number of category and the data , you may have to
                  modify the figsize, font sizes etc.
                - Currently, some of the ticks dont line up due to rotations.
            """
            cm = self.val
            if normalize:
                cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm, copy=True)
                cm = cm.astype('int')
        
            np.set_printoptions(precision=2)
        
            fig = mpl.figure.Figure(
                figsize=(14, 10), dpi=90, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(cm, cmap=colormap)
            fig.colorbar(im)
        
            classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
                       for x in labels]
            classes = ['\n'.join(wrap(l, 40)) for l in classes]
        
            tick_marks = np.arange(len(classes))
        
            ax.set_xlabel('Predicted', fontsize=7)
            ax.set_xticks(tick_marks)
            c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()
        
            ax.set_ylabel('True Label', fontsize=7)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, fontsize=4, va='center')
            ax.yaxis.set_label_position('left')
            ax.yaxis.tick_left()
        
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i, format(cm[i, j], '.0f') if cm[i, j] != 0 else '.',
                        horizontalalignment="center", fontsize=6,
                        verticalalignment='center', color="black")
            fig.set_tight_layout(True)
            return fig
        
        def to_tensorboard(self, writer, class_encoding, global_step):
            
            writer.add_figure('confusionmatrix', self.to_figure(class_encoding), global_step=global_step)
            
            writer.close()


class PrecisionRecall(object):
    """Holds and updates values for precision and recall"""
    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.probabilities = []
        self.predictions = []
        #self.labels = []

    def update(self, batch_output, batch_labels):
        _, topi = batch_output.topk(1)
        class_probs_batch = [torch.nn.functional.softmax(el, dim=0) for el in batch_output]
        
        self.probabilities.append(class_probs_batch)
        self.predictions.append(torch.flatten(topi))
        #self.labels.append(batch_labels)
    
    def to_tensorboard(self, writer, class_encoding, global_step):
        '''
        Takes in a "the class_encoding" i.e. from 0 to 9 and plots the corresponding precision-recall curves to tensorboard
        '''
        
        probs = torch.cat([torch.stack(b) for b in self.probabilities]).view(-1, self.n_cls)
        preds = torch.cat(self.predictions).view(-1)
        #labels = torch.cat(self.labels).view(-1)
        
        for class_index, class_name in enumerate(class_encoding):
            
            # subset = np.where(labels == class_index)
            # sub_probs = probs[subset[0]]
            # sub_preds = preds[subset[0]]
            # 
            # ground_truth = sub_preds == class_index
            # probability = sub_probs[:, class_index]
            
            ground_truth = preds == class_index
            probability = probs[:, class_index]

            
            writer.add_pr_curve(class_encoding[class_index],
                ground_truth,
                probability,
                global_step=global_step)
        
        writer.close()


    
def images_to_probs(output, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    
    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(output, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    _,channels,height,width = images.shape
    
    one_channel = True if channels in [1, 2] else False
    stereo = True if (channels % 2) == 0 else False
    
    preds, probs = images_to_probs(output, images)
    # plot the images in the batch, along with predicted and true labels
    fig = mpl.figure.Figure(
        figsize=(12, 12), dpi=90, facecolor='w', edgecolor='k')
    total_imgs = len(images) if len(images) < 10 else 10
    for idx in np.arange(total_imgs):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        img = images[idx]
        if stereo:
            #img = img.view(channels//2,height*2,width)
            img1, img2 = torch.split(img, channels//2)
            img = torch.cat([img1,img2], dim=1)
        elif one_channel:
            img = img.mean(dim=0)
            img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()

        if one_channel:
            if len(npimg.shape) > 2:
                npimg = np.transpose(npimg, (1, 2, 0))[:,:,0]
            ax.imshow(npimg, cmap="Greys")
        else:
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
                    
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"), fontsize=6)
    
    return fig


# saliency maps or class activation mapping
# -----

def saliencymap_to_figure(smap, pic, alpha=0.5):
    """
    saliencymap_to_figure takes a saliency map smap, a picture pic and an
    optional value for alpha and returns a matplotlib figure containing the 
    picture overlayed with the saliency map with transparency alpha.
    """
    number_of_maps = smap.shape[0]
    fig, axes = plt.subplots(smap.shape[0],smap.shape[1])
    
    for i in range(number_of_maps):
        for j in range(smap.shape[1]):
            classmap_answer = smap[i, j, :, :]
            axes[i,j].imshow(pic, cmap="Greys")
            axes[i,j].imshow(classmap_answer, cmap=mpl.cm.jet, alpha=alpha,
                      interpolation='nearest', vmin=0, vmax=1)
            axes[i,j].axis('off')

    return fig, axes



def show_cam_samples(cams, pics, targets, probs, preds, alpha=0.5, n_samples=5):
    """
    cams (b,t,n_classes,h,w)   Class Activation Maps
    pics (b,t,n_channels,h,w)  Input Images for Recurrent Network
    targets (b)                Target Vectors
    probs (b,t,topk)           Probabilities for Each output
    preds (b,t,topk)           Predictions of the Network
    alpha                      Transparency of the heatmap overlay
    """
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    b,t,c,h,w = pics.shape
   
    all_cams = cams
    all_pics = pics
    all_targets = targets
    all_probs = probs
    all_preds = preds
    
    for n in range(n_samples):
        fig, axes = plt.subplots(3,t+1, figsize=(12,12))
        cams = all_cams[n]
        pics = all_pics[n]
        targets = all_targets[n]
        probs = all_probs[n]
        preds = all_preds[n]
        
        for i in range(t):
            axes[0,i].imshow(pics[i,0,:,:], cmap="Greys")
            im = axes[0,i].imshow(cams[i,preds[i,0],:,:], cmap=mpl.cm.jet, alpha=alpha,
                  interpolation='nearest')#, vmin=0, vmax=1)
            axes[0,i].set_title('t{}: tar/pred ({}/{})'.format(i, targets, preds[i,0]))
            axes[0,i].axis('off')
            divider = make_axes_locatable(axes[0,i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        axes[0, -1].imshow(pics[-1,0,:,:], cmap="Greys")
        min, max = (cams[-1,preds[-1,0],:,:] - cams[0,preds[0,0],:,:]).min(), (cams[-1,preds[-1,0],:,:] - cams[0,preds[0,0],:,:]).max()
        # zero_pos = (-min) / (-min + max)
        # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
        im = axes[0, -1].imshow(cams[-1,preds[-1,0],:,:] - cams[0,preds[0,0],:,:], cmap=mpl.cm.seismic, alpha=alpha, interpolation='nearest', vmin=-max, vmax=max)
        axes[0, -1].set_title('Delta t')
        axes[0, -1].axis('off')
        divider = make_axes_locatable(axes[0,-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
        for i in range(t):
            axes[1,i].imshow(pics[i,0,:,:], cmap="Greys")
            im = axes[1,i].imshow(cams[i,preds[-1,0],:,:], cmap=mpl.cm.jet, alpha=alpha,
                  interpolation='nearest')#, vmin=0, vmax=1)
            axes[1,i].set_title('t{}: tar/pred ({}/{})'.format(i, targets, preds[i,0]))
            axes[1,i].axis('off')
            divider = make_axes_locatable(axes[1,i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        axes[1, -1].imshow(pics[-1,0,:,:], cmap="Greys")
        min, max = (cams[-1,preds[-1,0],:,:] - cams[0,preds[-1,0],:,:]).min(), (cams[-1,preds[-1,0],:,:] - cams[0,preds[-1,0],:,:]).max()
        # zero_pos = (-min) / (-min + max)
        # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
        im = axes[1, -1].imshow(cams[-1,preds[-1,0],:,:] - cams[0,preds[-1,0],:,:], cmap=mpl.cm.seismic, alpha=alpha, interpolation='nearest', vmin=-max, vmax=max)
        axes[1, -1].set_title('Delta t')
        axes[1, -1].axis('off')
        divider = make_axes_locatable(axes[1,-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
        for i in range(t):
            axes[2,i].imshow(pics[i,0,:,:], cmap="Greys")
            im = axes[2,i].imshow(cams[i,targets,:,:], cmap=mpl.cm.jet, alpha=alpha,
                  interpolation='nearest')#, vmin=0, vmax=1)
            axes[2,i].set_title('t{}: tar/pred ({}/{})'.format(i, targets, preds[i,0]))
            axes[2,i].axis('off')
            divider = make_axes_locatable(axes[2,i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        axes[2, -1].imshow(pics[-1,0,:,:], cmap="Greys")
        min, max = (cams[-1,targets,:,:] - cams[0,targets,:,:]).min(), (cams[-1,targets,:,:] - cams[0,targets,:,:]).max()
        # zero_pos = (-min) / (-min + max)
        # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
        im = axes[2, -1].imshow(cams[-1,targets,:,:] - cams[0,targets,:,:], cmap=mpl.cm.seismic, alpha=alpha, interpolation='nearest', vmin=-max, vmax=max)
        axes[2, -1].set_title('Delta t')
        axes[2, -1].axis('off')
        divider = make_axes_locatable(axes[2,-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
        # for i in range(1,t):
        #     for j in range(1,i+1):
        #         axes[j,i].imshow(pics[i,0,:,:], cmap="Greys")
        #         axes[j,i].imshow(cams[i,0,:,:] - cams[i-j,0,:,:], cmap=mpl.cm.seismic, alpha=alpha,
        #               interpolation='nearest')#, vmin=-1, vmax=1)
        #         axes[j,i].set_title('t{}-t{}'.format(i,i-j))
        #         
        # 
        # for j in range(1,t):
        #     for i in range(0,j):
        #         axes[j,i].axis('off')
        
        plt.show()
    pass

def show_cam_means(cams, targets, probs, preds):
    """
    cams (b,t,n_classes,h,w)   Class Activation Maps
    targets (b)                Target Vectors
    probs (b,t,topk)           Probabilities for Each output
    preds (b,t,topk)           Predictions of the Network
    alpha                      Transparency of the heatmap overlay
    """
    
    b,t,n_classes,h,w = cams.shape
    
    # topk output
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, preds[batch, timestep, 0],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams1 = torch.mean(torch.stack(uber_cam, dim=1), dim=0)
    
    # last predicition evolution
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, preds[batch, -1, 0],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams2 = torch.mean(torch.stack(uber_cam, dim=1), dim=0)
    
    # target evolution
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, targets[batch],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams3 = torch.mean(torch.stack(uber_cam, dim=1), dim=0)
    
    
    fig, axes = plt.subplots(3,t+1, figsize=(12,12))
    
    
    cams = cams[0]
    targets = targets[0]
    probs = probs[0]
    preds = preds[0]
    alpha = 1.0
    
    for i in range(t):
        im = axes[0,i].imshow(cams1[i], cmap=mpl.cm.jet, alpha=alpha,
              interpolation='nearest', vmin=-cams1[i].max(), vmax=cams1[i].max())
        axes[0,i].set_title('t{}: tar/pred ({}/{})'.format(i, targets, preds[i,0]))
        axes[0,i].axis('off')
        divider = make_axes_locatable(axes[0,i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
    min, max = (cams1[-1] - cams1[0]).min(), (cams1[-1] - cams1[0]).max()
    # zero_pos = (-min) / (-min + max)
    # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
    im = axes[0, -1].imshow(cams1[-1] - cams1[0], cmap=mpl.cm.seismic, alpha=alpha, interpolation='nearest', vmin=-max, vmax=max)
    axes[0, -1].set_title('Delta t')
    axes[0, -1].axis('off')
    divider = make_axes_locatable(axes[0,-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for i in range(t):
        im = axes[1,i].imshow(cams2[i], cmap=mpl.cm.jet, alpha=alpha,
              interpolation='nearest', vmin=-cams2[i].max(), vmax=cams2[i].max())
        axes[1,i].set_title('t{}: tar/pred ({}/{})'.format(i, targets, preds[i,0]))
        axes[1,i].axis('off')
        divider = make_axes_locatable(axes[1,i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    min, max = (cams2[-1] - cams2[0]).min(), (cams2[-1] - cams2[0]).max()
    # zero_pos = (-min) / (-min + max)
    # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
    im = axes[1, -1].imshow(cams2[-1] - cams2[0], cmap=mpl.cm.seismic, alpha=alpha, interpolation='nearest', vmin=-max, vmax=max)
    axes[1, -1].set_title('Delta t')
    axes[1, -1].axis('off')
    divider = make_axes_locatable(axes[1,-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for i in range(t):
        im = axes[2,i].imshow(cams3[i], cmap=mpl.cm.jet, alpha=alpha,
              interpolation='nearest', vmin=-cams3[i].max(), vmax=cams3[i].max())
        axes[2,i].set_title('t{}: tar/pred ({}/{})'.format(i, targets, preds[i,0]))
        axes[2,i].axis('off')
        divider = make_axes_locatable(axes[2,i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    min, max = (cams3[-1] - cams3[0]).min(), (cams3[-1] - cams3[0]).max()
    # zero_pos = (-min) / (-min + max)
    # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
    im = axes[2, -1].imshow(cams3[-1] - cams3[0], cmap=mpl.cm.seismic, alpha=alpha, interpolation='nearest', vmin=-max, vmax=max)
    axes[2, -1].set_title('Delta t')
    axes[2, -1].axis('off')
    divider = make_axes_locatable(axes[2,-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.show()
    pass


def plot_tsne_timetrajectories(representations, imgs, targets, points=1000, show_stimuli=False, show_indices=False, N='all', savefile='./../trained_models/tsnesave.npy', overwrite=False):
    
    # Constants, maybe become variables later
    N_UNOCC = 10
    
    # reduce dataset for plotting
    representations = representations[-points:]
    targets = targets[-points:]
    
    markers = ["o","v","s","D","H"]
    classes = set(targets.numpy())
    
    # plotting
    markersizes = [10,30] #,10,30]
    alpha=1.0
    colors = sns.color_palette("colorblind", len(classes))
    points,time,feature,height,width = representations.shape
    
    
    representations = representations.view(points,time,-1)
    

    
    # old way where we would combine the representations into one data reduction
    representations = representations.view(points*time,-1).numpy()
    # restore or save tsne model
    if os.path.exists(savefile):
        projected_data = np.load(savefile)
        print('[INFO] Loaded tsne-file at {}'.format(savefile))
    else:
        projected_data = np.zeros([1])
    if (projected_data.shape[0] != representations.shape[0]) or overwrite:
        projected_data = bhtsne.run_bh_tsne(representations, no_dims=2, perplexity=25, verbose=True, use_pca=False, initial_dims=representations[-1], max_iter=1000) #10000
        np.save(savefile, projected_data)
    
    projected_data = projected_data.reshape(points, time, -1)
    
    x_data = projected_data[:-N_UNOCC,:,0] # (index, time)
    y_data = projected_data[:-N_UNOCC,:,1] # (index, time)
    tar = targets[:-N_UNOCC]
    
    x_data_u = projected_data[-N_UNOCC:,:,0] # (index, time)
    y_data_u = projected_data[-N_UNOCC:,:,1] # (index, time)
    tar_u = targets[-N_UNOCC:]



    fig, axes = plt.subplots(2,2, sharex=False, sharey=False, figsize=(9,6))
    for pltnr, ax in enumerate([axes[0,0],axes[0,1]]):
        for ti in [0,3]: #range(4)
            for (i, cla) in enumerate(classes):
                xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
                yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
                ax.scatter(xc,yc,c=colors[i], label=str(int(cla)), marker=markers[ti], alpha=alpha, s=markersizes[pltnr])
    
        
        ax.scatter([0], [0], c='white', label=' ')
    
        # unoccluded trajectories
        for ti in range(time): 
            for (i,cla) in enumerate(sorted((set(tar_u.numpy())))):
                xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
                yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
                ax.scatter(xc,yc,c='black', marker=markers[ti], alpha=alpha, s=markersizes[pltnr], label='$t_{}$'.format(ti))
        for i in range(N_UNOCC):
            ax.plot(x_data_u[i,:], y_data_u[i,:], color='black', linestyle='-', alpha=alpha)
    
    # bottom plots
    grays = ['lightgray'] * len(classes)
    fills = ['none'] * len(classes)
    
    colorset = [sns.color_palette(grays),sns.color_palette(grays)]
    marker_fills = [fills,fills]
    allhighlights = [[3,8],[9]] # TODO: find out what this does!
    for pltnr in range(len(allhighlights)):
        for hl in allhighlights[pltnr]:
            colorset[pltnr][hl] = colors[hl]
            marker_fills[pltnr][hl] = 'full'
    
    for pltnr,ax in enumerate([axes[1,0],axes[1,1]]):
        
        if N=='all':
            n_indices = range(len(projected_data))
        elif isinstance(N, int):
            min_N_ind = N//len(set(targets.numpy()))
            n_indices = []
            for cla in classes:
                ind = np.where(tar == cla)[0]
                n_indices += list(np.random.choice(ind, min(min_N_ind, len(ind)), replace=False))
            n_indices += list(np.random.randint(0,len(projected_data)-N_UNOCC,N-len(n_indices)))
            n_indices = np.array(n_indices)
        elif (pltnr==1):
            n_indices = [N[-1]]
        else:
            n_indices = N[:-1]
            
        
        if show_stimuli:
            artists = []
            for ti in range(time):
                for x0, y0, i in zip(projected_data[:,ti,0][n_indices], projected_data[:,ti,1][n_indices], n_indices):
                    # adapt the center of arrows intelligently
                    if ((y0 < 0) and (x0>5)) :
                        xc,yc = x0-5,y0+25 #x0+10,y0+20
                    elif ((y0 < 0) and (x0<5)):
                        xc,yc = x0-5,y0+40
                    elif (y0 > 0):
                        xc,yc = x0-20,y0-30
                    else:
                        xc,yc = x0,y0
                    
                    #calculate scaling factor c
                    c = np.sqrt((1800./(xc**2 + yc**2))) # (-30., 30.)
                  
                    ab = offsetbox.AnnotationBbox(makeMarker(imgs[i,ti,0], zoom=0.65*32./len(imgs[i,ti,0])), (x0, y0), xybox=(c*xc, c*yc), xycoords='data', boxcoords="offset points",
                                      pad=0.3,bboxprops=dict(color=colorset[pltnr][int(targets[i])]) , arrowprops=dict(arrowstyle=patches.ArrowStyle("->", head_length=.2, head_width=.1)), frameon=True)
                    # ab2 = offsetbox.AnnotationBbox(makeMarker(tile_tensor_lowres[i], zoom=0.65*32./len(tile_tensor_lowres[i])), (x0, y0), xybox=(c*xc-30, c*yc), xycoords='data', boxcoords="offset points",
                    #                   pad=0.3,bboxprops=dict(color=colorset[pltnr][int(all_classes[i])]), frameon=True)
                  
                    if show_indices:
                        ax.annotate('{}'.format(i), xy=(x0, y0), xytext=(x0, y0), zorder=-1)
                    if len(n_indices) >= 25:
                        ab.zorder=-1
                    
                    artists.append(ax.add_artist(ab))
                    # if i == 622:
                    #   artists.append(ax.add_artist(ab2))
                
        for ti in [0,3]: #range(4)
            for (i, cla) in enumerate(classes):
                xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
                yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
                ax.scatter(xc,yc,c=colorset[pltnr][i], label=str(int(cla)), marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][i]), alpha=alpha, s=markersizes[0])
                
        ax.scatter([0], [0], c='white', label=' ')
        
        # unoccluded trajectories
        for ti in range(time): 
            for (i,cla) in enumerate(sorted((set(tar_u.numpy())))):
                xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
                yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
                ax.scatter(xc,yc,c='lightgray', marker=markers[ti], alpha=alpha, s=markersizes[0], label='$t_{}$'.format(ti)) #label='${}_{}$'.format(cla[0],cla[1])
        for i in range(N_UNOCC):
            ax.plot(x_data_u[i,:], y_data_u[i,:], color='lightgray', linestyle='-', alpha=alpha)
        
        # plot unoccluded trajectories as highlights
        for ti in range(time):
            for (i,cla) in enumerate(sorted((set(tar_u.numpy())))):
                xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
                yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
                ax.scatter(xc,yc,c='lightgray', marker=markers[ti], alpha=alpha, s=markersizes[0])
        
        for i in allhighlights[pltnr]:
            xd = x_data_u[i,:]
            yd = y_data_u[i,:]
            ax.plot(xd, yd, color='black', linestyle='-', alpha=alpha)
            for j in range(len(xd)):
                ax.scatter(xd[j],yd[j],c='black', marker=markers[j], alpha=alpha, s=markersizes[0],zorder=9999)
    
    handles, labels = axes[0,1].get_legend_handles_labels()
    #handles = handles[:10] + handles[-5:]
    handles = handles[:10] + handles[20:21] + [handles[-20+i*10] for i in range(time)]

    #labels = labels[:10] + labels[-5:]
    labels = labels[:10] + labels[20:21] + [labels[-20+i*10] for i in range(time)]


    

    axes[0,1].legend(handles, labels, title='class label', loc='center left', bbox_to_anchor=(1, 0), frameon=False)
    bottom, top = plt.ylim()
    
    
    
    # general setup
    
    axes[0,0].set_ylabel('t-SNE dimension 1')
    axes[1,0].set_ylabel('t-SNE dimension 1')
    axes[1,0].set_xlabel('t-SNE dimension 2')
    axes[1,1].set_xlabel('t-SNE dimension 2')
    
    
    ax_in = axes[0,1]
    ax_in.set_xlim([x_data_u[5,-1] - 7, x_data_u[5,-1] + 7])
    ax_in.set_ylim([y_data_u[5,-1] - 7, y_data_u[5,-1] + 7])
    
    # Create a Rectangle patch
    rect = patches.Rectangle((ax_in.get_xlim()[0],ax_in.get_ylim()[0]),ax_in.get_xlim()[1]-ax_in.get_xlim()[0],ax_in.get_ylim()[1]-ax_in.get_ylim()[0],linewidth=1,edgecolor='black',facecolor='none')
    # Add the patch to the Axes
    axes[0,0].add_patch(rect)
    # Annotate
    axes[0,0].annotate('B', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-2,+1]), weight='bold')
    for n, ax in enumerate(axes.flatten()):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], weight='bold', transform=ax.transAxes, size=18)
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    
       
    plt.show()

    pass
    
def plot_tsne_timetrajectories2(representations, imgs, targets, points=1000, show_stimuli=True, show_indices=True, N=10, savefile='./../trained_models/tsnesave', overwrite=False):
    
    # Constants, maybe become variables later
    N_UNOCC = 10
    
    # reduce dataset for plotting
    representations = representations[-points:]
    targets = targets[-points:]
    imgs = imgs[-points:]
    
    targets = targets.numpy()
    
    markers = ["o","v","s","D","H"]
    classes = [0,1,2,3,4,5,6,7,8,9] #set(targets.numpy())
    

    
    # plotting
    markersizes = [10,30] #,10,30]
    alpha=1.0
    colors = sns.color_palette("colorblind", len(classes))
    points,time,feature,height,width = representations.shape
    
    
    representations = representations.view(points,time,-1)
    

    
    # new way where we calculate tsne for each timestep seperately
    # restore or save tsne model
    if os.path.exists(savefile + '.npy'):
        projection = np.load(savefile + '.npy')
        print('[INFO] Loaded tsne-file at {}'.format(savefile))
    else:
        projection = np.zeros([1])
    if (projection.shape[0] == 1) or overwrite:
        projection = np.zeros([points, time, 2])
        for ti in range(time):
            time_rep = representations[:,ti,:].numpy()
            projected_data = bhtsne.run_bh_tsne(time_rep, no_dims=2, perplexity=25, verbose=True, use_pca=False, initial_dims=time_rep[-1], max_iter=1000) #10000
            projection[:,ti,:] = projected_data
    else:
        targets = np.load(savefile + '_targets.npy')
        
        np.save(savefile + '.npy', projection)
        np.save(savefile + '_targets.npy', targets)
    
    projected_data = projection
    
    x_data = projected_data[:-N_UNOCC,:,0] # (index, time)
    y_data = projected_data[:-N_UNOCC,:,1] # (index, time)
    tar = targets[:-N_UNOCC]
    
    x_data_u = projected_data[-N_UNOCC:,:,0] # (index, time)
    y_data_u = projected_data[-N_UNOCC:,:,1] # (index, time)
    tar_u = targets[-N_UNOCC:]
    
    
    
    for i, t in enumerate(tar_u):
        print(t)
        plt.imshow(imgs[-N_UNOCC+i][0,0,:,:])
        plt.show()
    
    
    x_spread = x_data.std()
    for ti in range(time):
        x_data[:,ti] = x_data[:,ti] + ti * x_spread*6
        x_data_u[:,ti] = x_data_u[:,ti] + ti * x_spread*6

    
    fig, ax = plt.subplots(figsize=(9,6))
    for (i, cla) in enumerate(classes):
        xc = [p for (j,p) in enumerate(x_data[:,-1]) if tar[j]==cla]
        yc = [p for (j,p) in enumerate(y_data[:,-1]) if tar[j]==cla]
        ax.scatter(xc,yc,c=colors[i], label=str(int(cla)), marker=markers[3], alpha=alpha, s=markersizes[0])
    plt.show()


    fig, axes = plt.subplots(2,2, sharex=False, sharey=False, figsize=(9,6))
    for pltnr, ax in enumerate([axes[0,0],axes[0,1]]):
        for ti in range(time): #range(4) #[0,3]
            for (i, cla) in enumerate(classes):
                xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
                yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
                ax.scatter(xc,yc,c=colors[i], label=str(int(cla)), marker=markers[ti], alpha=alpha, s=markersizes[pltnr])
    
        
        ax.scatter([0], [0], c='white', label=' ')
    
        # unoccluded trajectories
        for ti in range(time): 
            for (i,cla) in enumerate(sorted((set(tar_u)))):
                xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
                yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
                ax.scatter(xc,yc,c=colors[i], marker=markers[ti], alpha=alpha, s=markersizes[pltnr], edgecolor='black', label='$t_{}$'.format(ti))
        # for i in range(N_UNOCC):
        #     ax.plot(x_data_u[i,:], y_data_u[i,:], color='black', linestyle='-', alpha=alpha)
    
    # bottom plots
    grays = ['lightgray'] * len(classes)
    fills = ['none'] * len(classes)
    
    colorset = [sns.color_palette(grays),sns.color_palette(grays)]
    marker_fills = [fills,fills]
    allhighlights = [[3,8],[9]] # TODO: find out what this does!
    for pltnr in range(len(allhighlights)):
        for hl in allhighlights[pltnr]:
            colorset[pltnr][hl] = colors[hl]
            marker_fills[pltnr][hl] = 'full'
    
    for pltnr,ax in enumerate([axes[1,0],axes[1,1]]):
        
        if N=='all':
            n_indices = range(len(projected_data))
        elif isinstance(N, int):
            min_N_ind = N//len(set(targets))
            n_indices = []
            for cla in classes:
                ind = np.where(tar == cla)[0]
                n_indices += list(np.random.choice(ind, min(min_N_ind, len(ind)), replace=False))
            n_indices += list(np.random.randint(0,len(projected_data)-N_UNOCC,N-len(n_indices)))
            n_indices = np.array(n_indices)
        elif (pltnr==1):
            n_indices = [N[-1]]
        else:
            n_indices = N[:-1]
            
        
        if show_stimuli:
            artists = []
            for ti in range(time):
                for x0, y0, i in zip(projected_data[:,ti,0][n_indices], projected_data[:,ti,1][n_indices], n_indices):
                    # adapt the center of arrows intelligently
                    if ((y0 < 0) and (x0>5)) :
                        xc,yc = x0-5,y0+25 #x0+10,y0+20
                    elif ((y0 < 0) and (x0<5)):
                        xc,yc = x0-5,y0+40
                    elif (y0 > 0):
                        xc,yc = x0-20,y0-30
                    else:
                        xc,yc = x0,y0
                    
                    #calculate scaling factor c
                    c = np.sqrt((1800./(xc**2 + yc**2))) # (-30., 30.)
                  
                    ab = offsetbox.AnnotationBbox(makeMarker(imgs[i,ti,0], zoom=0.65*32./len(imgs[i,ti,0])), (x0, y0), xybox=(c*xc, c*yc), xycoords='data', boxcoords="offset points",
                                      pad=0.3,bboxprops=dict(color=colorset[pltnr][int(targets[i])]) , arrowprops=dict(arrowstyle=patches.ArrowStyle("->", head_length=.2, head_width=.1)), frameon=True)
                    # ab2 = offsetbox.AnnotationBbox(makeMarker(tile_tensor_lowres[i], zoom=0.65*32./len(tile_tensor_lowres[i])), (x0, y0), xybox=(c*xc-30, c*yc), xycoords='data', boxcoords="offset points",
                    #                   pad=0.3,bboxprops=dict(color=colorset[pltnr][int(all_classes[i])]), frameon=True)
                  
                    if show_indices:
                        ax.annotate('{}'.format(i), xy=(x0, y0), xytext=(x0, y0), zorder=-1)
                    if len(n_indices) >= 25:
                        ab.zorder=-1
                    
                    artists.append(ax.add_artist(ab))
                    # if i == 622:
                    #   artists.append(ax.add_artist(ab2))
                
        for ti in range(time): #[0,3]: #range(4)
            for (i, cla) in enumerate(classes):
                xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
                yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
                ax.scatter(xc,yc,c=colorset[pltnr][i], label=str(int(cla)), marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][i]), alpha=alpha, s=markersizes[0])
                
        ax.scatter([0], [0], c='white', label=' ')
        
        # unoccluded trajectories
        for ti in range(time): 
            for (i,cla) in enumerate(sorted((set(tar_u)))):
                xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
                yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
                ax.scatter(xc,yc,c='lightgray', marker=markers[ti], alpha=alpha, s=markersizes[0], label='$t_{}$'.format(ti)) #label='${}_{}$'.format(cla[0],cla[1])
        # for i in range(N_UNOCC):
        #     ax.plot(x_data_u[i,:], y_data_u[i,:], color='lightgray', linestyle='-', alpha=alpha)
        
        # plot unoccluded trajectories as highlights
        for ti in range(time):
            for (i,cla) in enumerate(sorted((set(tar_u)))):
                xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
                yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
                ax.scatter(xc,yc,c='lightgray', marker=markers[ti], alpha=alpha, s=markersizes[0])
        
        for i in allhighlights[pltnr]:
            xd = x_data_u[i,:]
            yd = y_data_u[i,:]
            # ax.plot(xd, yd, color='black', linestyle='-', alpha=alpha)
            for j in range(len(xd)):
                ax.scatter(xd[j],yd[j],c='black', marker=markers[j], alpha=alpha, s=markersizes[0],
                zorder=9999)
    
    handles, labels = axes[0,1].get_legend_handles_labels()
    #handles = handles[:10] + handles[-5:]
    handles = handles[:10] + handles[20:21] + [handles[-20+i*10] for i in range(time)]

    #labels = labels[:10] + labels[-5:]
    labels = labels[:10] + labels[20:21] + [labels[-20+i*10] for i in range(time)]


    

    axes[0,1].legend(handles, labels, title='class label', loc='center left', bbox_to_anchor=(1, 0), frameon=False)
    bottom, top = plt.ylim()
    
    
    
    # general setup
    
    axes[0,0].set_ylabel('t-SNE dimension 1')
    axes[1,0].set_ylabel('t-SNE dimension 1')
    axes[1,0].set_xlabel('t-SNE dimension 2')
    axes[1,1].set_xlabel('t-SNE dimension 2')
    
    
    ax_in = axes[0,1]
    ax_in.set_xlim([x_data_u[5,-1] - 7, x_data_u[5,-1] + 7])
    ax_in.set_ylim([y_data_u[5,-1] - 7, y_data_u[5,-1] + 7])
    
    # Create a Rectangle patch
    rect = patches.Rectangle((ax_in.get_xlim()[0],ax_in.get_ylim()[0]),ax_in.get_xlim()[1]-ax_in.get_xlim()[0],ax_in.get_ylim()[1]-ax_in.get_ylim()[0],linewidth=1,edgecolor='black',facecolor='none')
    # Add the patch to the Axes
    axes[0,0].add_patch(rect)
    # Annotate
    axes[0,0].annotate('B', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-2,+1]), weight='bold')
    for n, ax in enumerate(axes.flatten()):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], weight='bold', transform=ax.transAxes, size=18)
    
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    
       
    plt.show()

    pass


def plot_relative_distances(representations, targets, distance_metric):
    
    # define representations
    # calculate relative distances with distance metric
    pass

# ---------------------
# image transformations
# ---------------------


def normalize(x, inp_max=1, inp_min=-1):
    """
    normalize takes and input numpy array x and optionally a minimum and
    maximum of the output. The function returns a numpy array of the same
    shape normalized to values beween inp_max and inp_min.
    """
    normalized_digit = (inp_max - inp_min) * (x - x.min()
                                              ) / (x.max() - x.min()) + inp_min
    return normalized_digit


class MidPointNorm(mpl.colors.Normalize):
    """
    MidPointNorm inherits from Normalize. It is a class useful for
    visualizations with a bidirectional color-scheme. It chooses
    the middle of the colorbar to be in the middle of the data distribution.
    """

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0)  # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            # First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat > 0] /= abs(vmax - midpoint)
            resdat[resdat < 0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if mpl.cbook.iterable(value):
            val = np.ma.asarray(value)
            val = 2 * (val - 0.5)
            val[val > 0] *= abs(vmax - midpoint)
            val[val < 0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return val * abs(vmin - midpoint) + midpoint
            else:
                return val * abs(vmax - midpoint) + midpoint


# -----------------------
# activations and filters
# -----------------------

# TODO add functions to visualize activations and filters

# -----------------------------
# sprite images for tensorboard
# -----------------------------


def create_sprite_image(images):
    """
    create_sprite_image returns a sprite image consisting of images passed as
    argument. Images should be count x width x height
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]

    # get image channels
    if len(images.shape) > 3:
        channels = images.shape[3]
    else:
        channels = 1

    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.zeros((img_h * n_plots, img_w * n_plots, channels))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                            j * img_w:(j + 1) * img_w] = this_img

    # built in support for stereoscopic images
    if (channels == 2) or (channels == 6):
        _, spriteimage = anaglyph(
            spriteimage[:, :, :channels // 2],
            spriteimage[:, :, channels // 2:])

    return spriteimage


def save_sprite_image(savedir, raw_images):
    sprite_image = create_sprite_image(raw_images)
    if sprite_image.shape[2] == 1:
        plt.imsave(savedir, sprite_image[:, :, 0], cmap='gray_r')
    else:
        plt.imsave(savedir, sprite_image.astype(np.uint8))


# -----------------
# tensorboard specific
# -----------------

def add_pr_curve_tensorboard(class_enc, class_index, test_probs, test_preds, writer, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(class_enc[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
