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
import torch.nn.functional as F
import numpy as np

import sys, os, re
import itertools
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import offsetbox, patches
from matplotlib.markers import MarkerStyle
import seaborn as sns
import pandas as pd


import scipy.optimize as opt
import scipy.stats as st
from PIL import Image
from textwrap import wrap
from math import sqrt

import utilities.metrics as metrics

# van der Maaten TSNE implementations
try:
    import utilities.tsne.bhtsne as bhtsne
    import utilities.tsne.tsne as tsne
except ImportError:
    pass

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


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# ---------------------
# define a 2D gaussian distribution
# ---------------------

def full_twoD_Gaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = pos
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def twoD_Gaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = pos
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + np.abs(amplitude)*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()
    

def plot_distribution(measure, ax, lab=None, xlabel=''):
    sns.distplot(measure, fit=st.norm, kde=False, label=lab, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.axvline(x=measure.mean(),
        ymin=0.0, ymax = 5, linewidth=1, color='gray')
    ax.axvline(x=measure.mean()-measure.std(),
        ymin=0.0, ymax = 5, linewidth=1, color='gray', linestyle='--')
    ax.axvline(x=measure.mean()+measure.std(),
        ymin=0.0, ymax = 5, linewidth=1, color='gray', linestyle='--')

# ---------------------
# make custom image annotations
# ---------------------

# helper function to make markers
def makeMarker(image, zoom=.65):
    """
    makeMarker takes an image (height, weight) numpy array and optionally a zoom factor
    and returns a matplotlib offsetbox containing an image for plotting
    """
    return offsetbox.OffsetImage(image,zoom=zoom, cmap='Greys')

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
        """
        From a 3D point in axes ax1, 
        calculate position in 2D in ax2 
        """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self,arr,xy):
        """
        Place an image (arr) as annotation at position xy
        """
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
        """
        Holds and updates a confusion matrix object given the networks
        outputs
        """
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
    """
    Holds and updates values for precision and recall object
    """
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


# -----------------
# sensitivity of concentration
# -----------------

def plot_concentration_mass(target_percentage, occluder_percentage, overlap_percentage, background_percentage, filename):
    target_percentage *= 100
    occluder_percentage *= 100
    overlap_percentage *= 100
    background_percentage *= 100
    
    fig, ax = plt.subplots(figsize=(4,4))
    ax.errorbar(np.arange(0,4), target_percentage.mean(axis=0), yerr=target_percentage.std(axis=0), xerr=None, fmt='o-', label='target')
    ax.errorbar(np.arange(0,4), occluder_percentage.mean(axis=0), yerr=occluder_percentage.std(axis=0), xerr=None, fmt='o-', label='occluder')
    ax.errorbar(np.arange(0,4), overlap_percentage.mean(axis=0), yerr=overlap_percentage.std(axis=0), xerr=None, fmt='o-', label='overlap')
    ax.errorbar(np.arange(0,4), background_percentage.mean(axis=0), yerr=background_percentage.std(axis=0), xerr=None, fmt='o-', label='background')
    ax.set_xlabel("timesteps")
    ax.set_ylabel("percentage")
    ax.legend()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['$t_0$','$t_1$','$t_2$','$t_3$'])
    plt.savefig(filename)
    plt.close()
    
    # reform into a pandas dataframe
    points, _ = target_percentage.shape
    
    concentration_df = pd.DataFrame(
        np.hstack([
        
        np.vstack([background_percentage[:,0], np.repeat(0, points), np.repeat('background', points)]),
        np.vstack([background_percentage[:,0], np.repeat(1, points), np.repeat('background', points)]),
        np.vstack([background_percentage[:,0], np.repeat(2, points), np.repeat('background', points)]),
        np.vstack([background_percentage[:,0], np.repeat(3, points), np.repeat('background', points)])
        ,
        np.vstack([occluder_percentage[:,0], np.repeat(0, points), np.repeat('occluder', points)]),
        np.vstack([occluder_percentage[:,1], np.repeat(1, points), np.repeat('occluder', points)]),
        np.vstack([occluder_percentage[:,2], np.repeat(2, points), np.repeat('occluder', points)]),
        np.vstack([occluder_percentage[:,3], np.repeat(3, points), np.repeat('occluder', points)])
        ,
        np.vstack([overlap_percentage[:,0], np.repeat(0, points), np.repeat('overlap', points)]),
        np.vstack([overlap_percentage[:,1], np.repeat(1, points), np.repeat('overlap', points)]),
        np.vstack([overlap_percentage[:,2], np.repeat(2, points), np.repeat('overlap', points)]),
        np.vstack([overlap_percentage[:,3], np.repeat(3, points), np.repeat('overlap', points)])
        ,
        np.vstack([target_percentage[:,0], np.repeat(0, points), np.repeat('target', points)]),
        np.vstack([target_percentage[:,1], np.repeat(1, points), np.repeat('target', points)]),
        np.vstack([target_percentage[:,2], np.repeat(2, points), np.repeat('target', points)]),
        np.vstack([target_percentage[:,3], np.repeat(3, points), np.repeat('target', points)])
        ]).T, columns=['data', 'timestep', 'type'])
    concentration_df = concentration_df.explode('data')
    concentration_df['data'] = concentration_df['data'].astype('float')
    concentration_df = concentration_df.explode('timestep')
    concentration_df['timestep'] = concentration_df['timestep'].astype('int')
    with sns.axes_style("ticks"):
        sns.set_context("paper", font_scale=1.0, )#rc={"lines.linewidth": 0.5})
        fig, ax = plt.subplots(figsize=(4,4))
        palette = sns.color_palette("colorblind")
        palette = [sns.color_palette("colorblind")[7]] + sns.color_palette("colorblind")
        sns.set_palette(palette)
        sns.boxplot(data=concentration_df, x='timestep', y='data',
        hue='type', showfliers = False, ax=ax
        )
        sns.despine(offset=10, trim=True)
        ax.set_xticklabels(['$t_0$','$t_1$','$t_2$','$t_3$'], fontsize=12)
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Time step')
        # from statannot import add_stat_annotation
        # add_stat_annotation(ax, data=concentration_df, x='timestep', y='data', hue='type',
        #     box_pairs=[
        #         # ((0,1),(1,1)),
        #         # ((1,1),(2,1)),
        #         # ((0,2),(1,2)),
        #         # ((1,2),(2,2)),
        #         # ((2,1),(3,1)),
        #         ((1, 'target'),(2, 'occluder')),
        #         ((1, 'target'),(2, 'target')),
        #     ], test='t-test_ind', text_format='star', loc='inside', verbose=2)
        
        plt.show()
    

# -----------------
# class activation mapping
# -----------------

def quantify_pixel_importance(cams, preds, percentage=0.25):
    b,t,n_classes,h,w = cams.shape
    
    
    # normalize cams to a probability distribution over pixels
    # -----
    # offset by minimum of each upsampled activation map
    min_val, min_args = torch.min(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams -= torch.unsqueeze(min_val, dim=-1)
    ## or take the absolute value?
    #cams = torch.abs(cams)
    # normalize by the sum of each upsampled activation map
    sum_val = torch.sum(cams, dim=[-2,-1], keepdim=True)
    cams /= sum_val
    
    pixel_array = []
    for timestep in range(t):
        pixels_per_timestep = []
        for batch in range(b):
            pixels = cams[batch, timestep, preds[batch, timestep, 0],:,:].view(h*w)
            pixels = torch.cumsum(torch.sort(pixels, 0, descending=True)[0], 0)
            pixels = np.argmax(pixels>percentage)
            pixels_per_timestep.append(pixels)
        pixels_per_timestep = torch.stack(pixels_per_timestep, 0)
        pixel_array.append(pixels_per_timestep)
    pixel_array = np.array(torch.stack(pixel_array, dim=1))
    #print(np.mean(pixel_array,0))
    
    fig, ax = plt.subplots()
    for timestep in range(t):
        plot_distribution(pixel_array[:,timestep], ax, lab='$t={}$'.format(timestep))
    ax.legend()
    ax.set_title('pixels accounting for {}% class output mass'.format(int(percentage*100)))
    plt.show()


    pass

def plot_cam_fourier_space(predicted_cams,imnr=None):
    b,t,h,w = predicted_cams.shape
    
    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(9, 11))
    # fft of the heatmap
    for timestep in range(t):
        if imnr:
            image = predicted_cams[imnr, timestep,:,:]
        else:
            image = torch.mean(predicted_cams, dim=0)[timestep,:,:]
        freq = np.fft.fft2(image)
        freq = np.abs(freq)
        freq[0,0] = np.min(freq)
    
        ax[0,timestep].hist(freq.ravel(), bins=100)
        ax[0,timestep].set_title('hist(freq)', fontsize=7)
        ax[1,timestep].hist(np.log(freq).ravel(), bins=100)
        ax[1,timestep].set_title('hist(log(freq))', fontsize=7)
        ax[2,timestep].imshow(freq, interpolation="none")
        ax[2,timestep].set_title('freq', fontsize=7)
        ax[3,timestep].imshow(np.fft.fftshift(freq), interpolation="none")
        ax[3,timestep].set_title('freq', fontsize=7)
        ax[4,timestep].imshow(np.log(freq), interpolation="none")
        ax[4,timestep].set_title('log(freq)', fontsize=7)
        ax[5,timestep].imshow(image, interpolation="none")
        ax[5,timestep].set_title('image', fontsize=7)
    plt.show()
    
    pass


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
        cbar = plt.colorbar(im, cax=cax)
    
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
    
    # prepare real data according to different properties
    # i.e. target prediction, current prediction, final prediction
    
    # topk output (current)
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, preds[batch, timestep, 0],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams1 = torch.mean(torch.stack(uber_cam, dim=1), dim=0)
    
    # last prediction evolution (final)
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, preds[batch, -1, 0],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams2 = torch.mean(torch.stack(uber_cam, dim=1), dim=0)
    
    # target evolution (target)
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
    
def plot_cam_samples(cams, pics, targets, probs, preds, filename, list_of_indices=[948,614,541], alpha=0.5):
    """
    cams (b,t,n_classes,h,w)   Class Activation Maps
    pics (b,t,n_channels,h,w)  Input Images for Recurrent Network
    targets (b,n_occ)          NHot Target Vectors
    probs (b,t,topk)           Probabilities for Each output
    preds (b,t,topk)           Predictions of the Network
    alpha                      Transparency of the heatmap overlay
    """
    # generate a rocket-like colormap with alpha values
    # -----
    # get colormap
    from matplotlib.colors import LinearSegmentedColormap
    ncolors = 256
    color_array = plt.get_cmap('rocket')(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rocket_alpha',colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    
    
    b,t,n_classes,h,w = cams.shape
    n_rows = len(list_of_indices)
    # Create x and y indices for gaussian2D data
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    x, y = np.meshgrid(x, y)
    n_hot_targets = targets if len(targets.shape) > 1 else torch.stack([targets,targets,targets],-1)
    targets = targets[:,0] if len(targets.shape) > 1 else targets
    print("[INFO] showing CAMS for indices {}".format(list_of_indices))
    # normalize cams to 0,1
    # -----
    # offset by minimum of each upsampled activation map
    min_val, min_args = torch.min(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams -= torch.unsqueeze(min_val, dim=-1)
    # divide by the maximum of each activation map
    max_val, max_args = torch.max(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams /= torch.unsqueeze(max_val, dim=-1)
    # normalize by the sum of each upsampled activation map
    #sum_val = torch.sum(cams, dim=[-2,-1], keepdim=True)
    #cams /= sum_val
    if n_rows > 3:
        grid_dict = dict(left=0.05,right=0.875, bottom=0.04, top=0.973)
    else:
        grid_dict = dict(left=0.05,right=0.875)

    fig, ax = plt.subplots(n_rows,t+1, gridspec_kw=grid_dict, figsize=(6.4, 4.8/3.0*n_rows))

    for row,ind in enumerate(list_of_indices):
        for ti in range(t):
            current_image = pics[ind,ti,0,:,:]
            threshold_map = np.array(cams[ind,ti,preds[ind,ti,0],:,:]>0.2, dtype=np.float)
            # get the cams, threshold them to output 1 in case > 0.2
            threshold_cam = cams[ind,ti,preds[ind,ti,0],:,:].numpy().copy()
            threshold_cam[threshold_cam>0.2] = 1.0
            threshold_cam /= 0.2
            
            faded_image = current_image * threshold_cam #cams[ind,ti,preds[ind,ti,0],:,:]
            
            #ax[row,ti].imshow(current_image, cmap="Greys", interpolation="none")
            ax[row,ti].imshow(faded_image, cmap="Greys", interpolation="none")

            min, max = (cams[ind,ti,preds[ind,ti,0],:,:]).min(), (cams[ind,ti,preds[ind,ti,0],:,:]).max()
            min, max = 0.0, 1.0
            #im = ax[row,ti].imshow(cams[ind,ti,preds[ind,ti,0],:,:], cmap="rocket",    alpha=alpha, interpolation='nearest', vmin=min, vmax=max)
            im = ax[row,ti].imshow(threshold_map, cmap="rocket_alpha", alpha=alpha, interpolation='nearest', vmin=min, vmax=max)
            
            def contour_rect_slow(im):
                """Clear version"""
            
                pad = np.pad(im, [(1, 1), (1, 1)])  # zero padding
            
                im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
                im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
            
                lines = []
            
                for ii, jj in np.ndindex(im0.shape):
                    if im0[ii, jj] == 1:
                        lines += [([ii-.5, ii-.5], [jj-.5, jj+.5])]
                    if im1[ii, jj] == 1:
                        lines += [([ii-.5, ii+.5], [jj-.5, jj-.5])]
            
                return lines
            
            
            # lines = contour_rect_slow(threshold_map)
            # for line in lines:
            #     ax[row,ti].plot(line[1], line[0], color='r', alpha=1, linewidth=0.5)
            
            #ax[row,ti].contour(threshold_map, levels=[1.0], colors='red', linewidths=[0.5],
            #    extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])
            
            g = metrics.gini(np.array(cams[ind,ti,preds[ind,ti,0],:,:]))

            ax[row,ti].set_xlabel('{}|{} [{},{}]\n g={:0.3f}'.format(preds[ind,ti,0], n_hot_targets[ind,0], n_hot_targets[ind,1], n_hot_targets[ind,2], g), fontsize=12)
            ax[row,ti].set_yticks([])
            ax[row,ti].set_xticks([])
            
            initial_guess = (5.,16,16,5,5,0.,10.)
            cdat = cams[ind,ti,preds[ind,ti,0],:,:].numpy()
            x_init, y_init = np.where(cdat == np.amax(cdat))  
            initial_guess_2 = (np.amax(cdat),x_init[0],y_init[0],1,1,0.,3.)
            #popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), np.reshape(cams[ind,ti,preds[ind,ti,0],:,:],h*w), p0=initial_guess, maxfev=50000)
            #ax[row,ti].contour(x, y, (twoD_Gaussian((x, y), *popt)).reshape(32, 32), [.35,.4,.45,.5], colors='w',linewidths=.5)
        # divider = make_axes_locatable(ax[row,-2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = fig.colorbar(im, cax=cax, ticks=[min, (min+max)/2., max])
        # cax.tick_params(axis='both', which='major', labelsize=7)
        # # enlarge axis by 5%
        # box = ax[row,-2].get_position()
        # ax[row,-2].set_position([box.x0, box.y0, box.width * 1.07, box.height * 1.07])
        
        # Delta T Plot
        ax[row, -1].imshow(pics[ind,-1,0,:,:], cmap="Greys")
        min, max = (cams[ind,-1,preds[ind,-1,0],:,:] - cams[ind,0,preds[ind,0,0],:,:]).min(), (cams[ind,-1,preds[ind,-1,0],:,:] - cams[ind,0,preds[ind,0,0],:,:]).max()
        min, max = -0.5, +0.5
        # zero_pos = (-min) / (-min + max)
        # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
        im = ax[row, -1].imshow(cams[ind,-1,preds[ind,-1,0],:,:] - cams[ind,0,preds[ind,0,0],:,:], cmap="icefire", alpha=alpha if alpha > 0.75 else 0.75, interpolation='nearest', vmin=-max, vmax=max)
        divider = make_axes_locatable(ax[row,-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[min, (min+max)/2., max])
        cax.tick_params(axis='both', which='major', labelsize=10)
        # enlarge axis by 10%
        box = ax[row,-1].get_position()
        if n_rows > 3:
            y_offset = 0.0045
        else:
            y_offset = 0.014
        ax[row,-1].set_position([box.x0 + 0.025, box.y0 - y_offset , box.width * 1.12, box.height * 1.12]) #box.y0 - 0.014
        
        ax[row,-1].set_yticks([])
        ax[row,-1].set_xticks([])
        

    
    for ti in range(t):
        ax[0,ti].annotate('$t_{}$'.format(ti), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
        fontsize=12, ha='center', va='bottom',
        bbox=dict(boxstyle='square', fc='white', ec='white'),
        #arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0)
        )
    ax[0,-1].annotate('$\Delta t$'.format(ti), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
    fontsize=12, ha='center', va='bottom',
    bbox=dict(boxstyle='square', fc='white', ec='white'),
    #arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0)
    )
    
    ax_in = ax[0,1]

    ax[0,0].annotate('A', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-10,-12]), weight='bold', fontsize=24)
    
    ax[-1,0].text(18, 48, '3|3 [8,5] = output:3 | target:3 [occluder1:8, occluder2:5]',
    fontsize=12, horizontalalignment='left',
    verticalalignment='center')


    plt.savefig(filename, dpi=300, format='pdf')
    plt.close()

def plot_cam_samples_alt(cams, pics, targets, probs, preds, filename, list_of_indices=[948,614,541], alpha=0.5):
    """
    cams (b,t,n_classes,h,w)   Class Activation Maps
    pics (b,t,n_channels,h,w)  Input Images for Recurrent Network
    targets (b,n_occ)          NHot Target Vectors
    probs (b,t,topk)           Probabilities for Each output
    preds (b,t,topk)           Predictions of the Network
    alpha                      Transparency of the heatmap overlay
    """
    # generate a rocket-like colormap with alpha values
    # -----
    # get colormap
    from matplotlib.colors import LinearSegmentedColormap
    ncolors = 256
    color_array = plt.get_cmap('rocket')(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rocket_alpha',colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    
    
    b,t,n_classes,h,w = cams.shape
    n_rows = len(list_of_indices)
    # Create x and y indices for gaussian2D data
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    x, y = np.meshgrid(x, y)
    n_hot_targets = targets if len(targets.shape) > 1 else torch.stack([targets,targets,targets],-1)
    targets = targets[:,0] if len(targets.shape) > 1 else targets
    print("[INFO] showing CAMS for indices {}".format(list_of_indices))
    # normalize cams to 0,1
    # -----
    # offset by minimum of each upsampled activation map
    min_val, min_args = torch.min(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams -= torch.unsqueeze(min_val, dim=-1)
    # divide by the maximum of each activation map
    max_val, max_args = torch.max(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams /= torch.unsqueeze(max_val, dim=-1)
    # normalize by the sum of each upsampled activation map
    #sum_val = torch.sum(cams, dim=[-2,-1], keepdim=True)
    #cams /= sum_val
    if n_rows > 3:
        grid_dict = dict(left=0.05,right=0.875, bottom=0.04, top=0.973)
    else:
        grid_dict = dict(left=0.05,right=0.875, bottom=0.12)

    fig, ax = plt.subplots(n_rows,t, gridspec_kw=grid_dict, figsize=(6.4 - 0.5, 5.2/3.0*n_rows))
    
    for row,ind in enumerate(list_of_indices):
        current_image = pics[ind,0,0,:,:]
        ax[row,0].imshow(current_image, cmap="Greys", interpolation="none")
        ax[row,0].set_xlabel('{} [{},{}]'.format(n_hot_targets[ind,0], n_hot_targets[ind,1], n_hot_targets[ind,2]), fontsize=12)
        ax[row,0].set_xlabel
        ax[0,0].annotate('stimulus', xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
        fontsize=12, ha='center', va='bottom',
        bbox=dict(boxstyle='square', fc='white', ec='white'),
        )
        
        ti = 0
        min, max = (cams[ind,ti,preds[ind,ti,0],:,:]).min(), (cams[ind,ti,preds[ind,ti,0],:,:]).max()
        im = ax[row,1].imshow(cams[ind,ti,preds[ind,ti,0],:,:], cmap="rocket",    alpha=1.0, interpolation='nearest', vmin=min, vmax=max)
        ax[0,1].annotate('$t_{}$'.format(0), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
        fontsize=12, ha='center', va='bottom',
        bbox=dict(boxstyle='square', fc='white', ec='white'),
        #arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0)
        )
        g = metrics.gini(np.array(cams[ind,ti,preds[ind,ti,0],:,:]))
        ax[row,1].set_xlabel('pred.: {}, g={:0.3f}'.format(preds[ind,ti,0], g), fontsize=12)
        
        ti = 3
        min, max = (cams[ind,ti,preds[ind,ti,0],:,:]).min(), (cams[ind,ti,preds[ind,ti,0],:,:]).max()
        im = ax[row,2].imshow(cams[ind,ti,preds[ind,ti,0],:,:], cmap="rocket",    alpha=1.0, interpolation='nearest', vmin=min, vmax=max)
        ax[0,2].annotate('$t_{}$'.format(3), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
        fontsize=12, ha='center', va='bottom',
        bbox=dict(boxstyle='square', fc='white', ec='white'),
        #arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0)
        )
        g = metrics.gini(np.array(cams[ind,ti,preds[ind,ti,0],:,:]))
        ax[row,2].set_xlabel('pred.: {}, g={:0.3f}'.format(preds[ind,ti,0], g), fontsize=12)
        for ti in range(t):
            ax[row,ti].set_yticks([])
            ax[row,ti].set_xticks([])
        # color bar
        min, max = 0.0, 1.0
        divider = make_axes_locatable(ax[row,2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[min, max])
        cax.tick_params(axis='both', which='major', labelsize=10)
        # enlarge axis by 10%
        box = ax[row,2].get_position()
        if n_rows > 3:
            y_offset = 0.0040
        else:
            y_offset = 0.0055
        ax[row,2].set_position([box.x0, box.y0 - y_offset, box.width * 1.07, box.height * 1.07]) #box.y0 - 0.014
        
        
        # Delta T Plot
        ax[row, -1].imshow(pics[ind,-1,0,:,:], cmap="Greys")
        min, max = (cams[ind,-1,preds[ind,-1,0],:,:] - cams[ind,0,preds[ind,0,0],:,:]).min(), (cams[ind,-1,preds[ind,-1,0],:,:] - cams[ind,0,preds[ind,0,0],:,:]).max()
        min, max = -0.5, +0.5
        # zero_pos = (-min) / (-min + max)
        # zero_centered_cmap = make_cmap([(0,0,255),(255,255,255),(255,0,0)], position=[0, zero_pos, 1], bit=True)
        im = ax[row, -1].imshow(cams[ind,-1,preds[ind,-1,0],:,:] - cams[ind,0,preds[ind,0,0],:,:], cmap="icefire", alpha=alpha if alpha > 0.75 else 0.75, interpolation='nearest', vmin=-max, vmax=max)
        divider = make_axes_locatable(ax[row,-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[min, (min+max)/2., max])
        cax.tick_params(axis='both', which='major', labelsize=10)
        # enlarge axis by 10%
        box = ax[row,-1].get_position()
        if n_rows > 3:
            y_offset = 0.0040
        else:
            y_offset = 0.0055
        ax[row,-1].set_position([box.x0 + 0.025, box.y0 - y_offset , box.width * 1.07, box.height * 1.07]) #box.y0 - 0.014
        
        ax[row,-1].set_yticks([])
        ax[row,-1].set_xticks([])
        
    
    ax[0,-1].annotate('$\Delta t$'.format(ti), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
    fontsize=12, ha='center', va='bottom',
    bbox=dict(boxstyle='square', fc='white', ec='white'),
    #arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0)
    )
    
    ax_in = ax[0,1]

    ax[0,0].annotate('A', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-8,-8]), weight='bold', fontsize=24)
    
    ax[-1,0].text(20, 44, '3 [8,5] = target:3 [occluder1:8, occluder2:5]',
    fontsize=12, horizontalalignment='left',
    verticalalignment='center')


    plt.savefig(filename, dpi=300, format='pdf')
    plt.close()



def plot_cam_means(cams, targets, probs, preds):
    b,t,n_classes,h,w = cams.shape
    n_rows = 3
    # Create x and y indices for gaussian2D data
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    x, y = np.meshgrid(x, y)
    
    quantify_pixel_importance(cams, preds, percentage=0.25)
    quantify_pixel_importance(cams, preds, percentage=0.5)

    # normalize cams to 0,1
    # -----
    # offset by minimum of each upsampled activation map
    min_val, min_args = torch.min(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams -= torch.unsqueeze(min_val, dim=-1)
    # divide by the maximum of each activation map
    max_val, max_args = torch.max(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
    cams /= torch.unsqueeze(max_val, dim=-1)
    
    # prepare real data according to different properties
    # i.e. target prediction, current prediction, final prediction
    
    # topk output (current)
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, preds[batch, timestep, 0],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams_current = torch.stack(uber_cam, dim=1)
    #plot_cam_fourier_space(cams_current, imnr=1)
    #plot_cam_fourier_space(cams_current)
    
    # last prediction evolution (final)
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, preds[batch, -1, 0],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams_final = torch.stack(uber_cam, dim=1)
    
    # target evolution (target)
    uber_cam = []
    for timestep in range(t):
        topk_cam = []
        for batch in range(b):
            topk_cam.append(cams[batch, timestep, targets[batch],:,:])
        topk_cam = torch.stack(topk_cam, 0)
        uber_cam.append(topk_cam)
    cams_target = torch.stack(uber_cam, dim=1)
    
    cams_by_row = [cams_current, cams_final, cams_target]
    
    fig, ax = plt.subplots(n_rows, t+1)
    for row in range(n_rows):
        dict_of_metric = {}
        popt_list = []
        for ti in range(t):
            dict_of_metric[ti] = []
            list_of_data = []
            list_of_fitted_data = []
            initial_guess = (3.,15,14,5,5,0.,10.) # handcrafted parameters
            for j in range(b):
                # insert real data here
                data_noisy = cams_by_row[row][j,ti,:,:].numpy()
                x_init, y_init = np.where(data_noisy == np.amax(data_noisy))   
                initial_guess_2 = (np.amax(data_noisy),x_init[0],y_init[0],5,5,0.,10.)
                data_noisy = np.reshape(data_noisy,h*w)
                list_of_data.append(data_noisy)
                try:
                    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess, maxfev=50000)
                except RuntimeError:
                    print('[INFO] RuntimeError, reset initial guess')
                    #plt.imshow(np.reshape(data_noisy,[32,32]))
                    #plt.show()
                    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess_2, maxfev=50000)
                popt_list.append(popt)
                initial_guess = popt
                # decide whether the metric is supposed to be the maximum or the intermediate
                dict_of_metric[ti].append(np.sqrt(popt[3]**2 + popt[4]**2))
                #dict_of_metric[ti].append(max(popt[3], popt[4]))
            for popt in popt_list:
                list_of_fitted_data.append(twoD_Gaussian((x, y), *popt))
            popt_list = []

            ax[row,ti].imshow(np.mean(list_of_data,0).reshape(32, 32), cmap="rocket", origin='upper',
                extent=(x.min(), x.max(), y.min(), y.max()))
            ax[row,ti].contour(x, np.flip(y), np.mean(list_of_fitted_data,0).reshape(32, 32), [.35,.4,.45,.5], colors='gray',linewidths=.25, linestyles='dashed')
            
            popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), np.mean(list_of_data,0), p0=(1.,15,14,1,1,0.,3.), maxfev=50000)
            ax[row,ti].contour(x, np.flip(y), (twoD_Gaussian((x, y), *popt)).reshape(32, 32), [.35,.4,.45,.5], colors='w',linewidths=.25)

            #ax[row,ti].axis('off')
            ax[row,ti].set_yticks([])
            ax[row,ti].set_xticks([])
            
            ax[0,0].set_ylabel('current')
            ax[1,0].set_ylabel('final') 
            ax[2,0].set_ylabel('target')
            
            ax[0,ti].annotate('$t_{}$'.format(ti), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
                        fontsize=9, ha='center', va='bottom',
                        bbox=dict(boxstyle='square', fc='white', ec='white'),
                        arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0))    
            
            #print(row, ti)
        
        
        for ti in range(t):
        # T-Test mit Bonferroni Korrektur nach Benjamini Hochberg
            qstar = 0.05
            pval = np.ones([t, t])
            stval = np.ones([t, t])
            significance_table = np.zeros([t, t])
            for k in range(t):
                for j in range(t):
                    if k != j and k > j:
                        stval[k, j], pval[k, j] = st.ttest_ind(dict_of_metric[k], dict_of_metric[j], equal_var=False)
    
            #print(np.round(pval, 4))
            #print(np.round(stval, 4))
    
            sorted_pvals = np.sort(pval[pval < 1])
            bjq = np.arange(1, len(sorted_pvals) + 1) / \
                len(sorted_pvals) * qstar
    
            for k in range(t):
                for j in range(t):
                    if k != j and k > j:
                        if pval[k, j] in sorted_pvals[sorted_pvals - bjq <= 0]:
                            significance_table[k, j] = 1
                        else:
                            significance_table[k, j] = 0
    
    
    
            # Shrink current axis by 20%
            box = ax[row,-1].get_position()
            ax[row,-1].set_position([box.x0 + 0.01, box.y0 + 0.02, box.width * 0.8, box.height * 0.8])
        
            #ax = plt.axes([0.75, 0.5, .10, .10])
            ax[row,-1].set_zorder(-1)
            ax[row,-1].matshow(significance_table, cmap='Greys')
    
            ax[row,-1].set_xticklabels(['','$t_0$','$t_1$','$t_2$'], fontsize=7)#fontsize=65)
            ax[row,-1].tick_params(labelbottom=True, labeltop=False,
                          right=False, top=False)
            ax[row,-1].set_yticklabels(['$t_0$','$t_1$','$t_2$','$t_3$'], fontsize=7)#fontsize=65)
            ax[row,-1].set_xlim([-0.5, t - 1 - 0.5])
            ax[row,-1].set_ylim([t - 0.5, -0.5 + 1])
            ax[row,-1].spines['top'].set_visible(False)
            ax[row,-1].spines['right'].set_visible(False)
    
    ax[-1,-1].add_patch(patches.Rectangle((0.0, 2.5+4), 1, 1, fill='black',
                           color='black', alpha=1, clip_on=False))
    
    ax[-1,-1].text(0.0, 6.4 - 1.0+4, 'Significant difference \n(two-sided t-test, \nexpected FDR=0.05)',
            fontsize=7, horizontalalignment='left',
            verticalalignment='center')
    
    # Create a Rectangle patch
    ax_in = ax[0,1]
    
    rect = patches.Rectangle((ax_in.get_xlim()[0],ax_in.get_ylim()[0]),ax_in.get_xlim()[1]-ax_in.get_xlim()[0],ax_in.get_ylim()[1]-ax_in.get_ylim()[0],linewidth=1,edgecolor='black',facecolor='none')
    # Add the patch to the Axes
    ax[0,0].add_patch(rect)
    # Annotate
    ax[0,0].annotate('B', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-24,+12]), weight='bold', fontsize=24)
    
    plt.show()
    

def plot_cam_means2(cams_list, targets, probs, preds, filename):
    
    b,t,n_classes,h,w = cams_list[0].shape
    n_rows = 3
    # Create x and y indices for gaussian2D data
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    x, y = np.meshgrid(x, y)
    

    # normalize cams to 0,1 and preprocessing
    # -----
    cams_final = []
    for i, cams in enumerate(cams_list):
        # offset by minimum of each upsampled activation map
        min_val, min_args = torch.min(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
        cams -= torch.unsqueeze(min_val, dim=-1)
        # divide by the maximum of each activation map
        max_val, max_args = torch.max(cams.view(b,t,n_classes,h*w), dim=-1, keepdim=True)
        cams /= torch.unsqueeze(max_val, dim=-1)
        
        # last prediction evolution (final)
        uber_cam = []
        for timestep in range(t):
            topk_cam = []
            for batch in range(b):
                topk_cam.append(cams[batch, timestep, preds[i][batch, -1, 0],:,:])
            topk_cam = torch.stack(topk_cam, 0)
            uber_cam.append(topk_cam)
        cams_final.append(torch.stack(uber_cam, dim=1))
    
    # prepare real data according to different properties
    # i.e. target prediction, current prediction, final prediction
    
    # # topk output (current)
    # uber_cam = []
    # for timestep in range(t):
    #     topk_cam = []
    #     for batch in range(b):
    #         topk_cam.append(cams[batch, timestep, preds[batch, timestep, 0],:,:])
    #     topk_cam = torch.stack(topk_cam, 0)
    #     uber_cam.append(topk_cam)
    # cams_current = torch.stack(uber_cam, dim=1)
    # #plot_cam_fourier_space(cams_current, imnr=1)
    # #plot_cam_fourier_space(cams_current)
    
    # # last prediction evolution (final)
    # uber_cam = []
    # for timestep in range(t):
    #     topk_cam = []
    #     for batch in range(b):
    #         topk_cam.append(cams[batch, timestep, preds[batch, -1, 0],:,:])
    #     topk_cam = torch.stack(topk_cam, 0)
    #     uber_cam.append(topk_cam)
    # cams_final1 = torch.stack(uber_cam, dim=1)
    
    # # target evolution (target)
    # uber_cam = []
    # for timestep in range(t):
    #     topk_cam = []
    #     for batch in range(b):
    #         topk_cam.append(cams[batch, timestep, targets[batch],:,:])
    #     topk_cam = torch.stack(topk_cam, 0)
    #     uber_cam.append(topk_cam)
    # cams_target = torch.stack(uber_cam, dim=1)
    
    cams_by_row = cams_final
    row_x_shift = [-8,8,0]
    row_y_shift = [8,-8,0]
    fig, ax = plt.subplots(n_rows, t+1, gridspec_kw={
           'width_ratios': [1, 1, 1, 1, 1],
           'height_ratios': [1, 1, 1], 'left':0.05, 'right':0.875},)
    for row in range(n_rows):
        dict_of_metric = {}
        popt_list = []
        initial_guess = (3.,16.+row_x_shift[row],16.+row_y_shift[row],2.,2.,0.,0.) # handcrafted parameters
        
        # TODO: Transfer upper and lower bound to plot_cam_means function
        lower_bound = (0,0,0,0,0.,0.,-100.)
        upper_bound = (100,32,32,32.,32.,2*np.pi,100.)
        min = np.mean(cams_by_row[row].numpy(), axis=0).min()
        max = np.mean(cams_by_row[row].numpy(), axis=0).max()
        ginis = np.zeros([b, t])
        for ti in range(t):
            dict_of_metric[ti] = []
            list_of_data = []
            list_of_fitted_data = []
            
            for j in range(b):
                # insert real data here
                data_noisy = cams_by_row[row][j,ti,:,:].numpy()
                ginis[j,ti] = metrics.gini(data_noisy)
                x_init, y_init = np.where(data_noisy == np.amax(data_noisy))
                initial_guess_2 = (3.,x_init[0],y_init[0],5,5,0.,10.)
                data_noisy = np.reshape(data_noisy,h*w)
                list_of_data.append(data_noisy)
                # *** fitting gaussians
            #     try:
            #         popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess, bounds=(lower_bound, upper_bound), maxfev=50000)
            #     except RuntimeError:
            #         print('[INFO] RuntimeError, reset initial guess')
            #         #plt.imshow(np.reshape(data_noisy,[32,32]))
            #         #plt.show()
            #         popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess_2, bounds = (lower_bound, upper_bound), maxfev=50000)
            #     popt_list.append(popt)
            #     # update initial_guess intelligently
            #     if (j % 100) == 0:
            #         initial_guess = np.mean(popt_list, axis=0)
            #         print('[INFO] {}/{} gaussian fits done'.format(j, b))
            #     # initial_guess = popt
            #     # decide whether the metric is supposed to be the maximum or the intermediate
            #     dict_of_metric[ti].append(np.sqrt(popt[3]**2 + popt[4]**2))
            #     #dict_of_metric[ti].append(max(popt[3], popt[4]))
            # for popt in popt_list:
            #     list_of_fitted_data.append(twoD_Gaussian((x, y), *popt))
            # popt_list = []
            # ***
            im = ax[row,ti].imshow(np.mean(list_of_data,0).reshape(32, 32), cmap="rocket", origin='upper', vmin=min, vmax=max, extent=(x.min(), x.max(), y.min(), y.max()))
            #ax[row,ti].contour(x, np.flip(y), np.mean(list_of_fitted_data,0).reshape(32, 32), levels=[.35,.4,.45,.5], colors='w',linewidths=.25, linestyles='dashed')
            
            popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), np.mean(list_of_data,0), p0=initial_guess, bounds=(lower_bound, upper_bound), maxfev=50000)
            ax[row,ti].contour(x, np.flip(y), (twoD_Gaussian((x, y), *popt)).reshape(32, 32), levels=[.35,.4,.45,.5], colors='w',linewidths=.25)

            #ax[row,ti].axis('off')
            ax[row,ti].set_yticks([])
            ax[row,ti].set_xticks([])
            

            
            ax[0,ti].annotate('$t_{}$'.format(ti), xy=(0.5, 1.10), xytext=(0.5, 1.10), xycoords='axes fraction', 
                        fontsize=12, ha='center', va='bottom',
                        bbox=dict(boxstyle='square', fc='white', ec='white'),
                        #arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(3), lw=1.0)
                    )    
            
            #print(row, ti)
        
        
            # plot marker where target is supposed to be
            ax[0,0].set_ylabel('bottom-left', fontsize=12)
            ax[0,ti].plot(16-row_y_shift[0], 16-row_y_shift[0], marker="+", color='black', markersize=5.0, markeredgewidth=.25)

            ax[1,0].set_ylabel('top-right', fontsize=12)
            ax[1,ti].plot(16+row_y_shift[0], 16+row_y_shift[0], marker="+", color='black', markersize=5.0, markeredgewidth=.25)

            ax[2,0].set_ylabel('center', fontsize=12)
            ax[2,ti].plot(16, 16, marker="+", color='black', markersize=5.0, markeredgewidth=.25)
            
            #ax[row,ti].set_xlabel('g={:0.3f}'.format(np.mean(ginis)))
            print('[INFO] row: {} ti: {} - gini coefficient mean: {:0.3f}, std: {:0.3f}'.format(row, ti, np.mean(ginis, axis=0)[ti], np.std(ginis, axis=0)[ti]))
        
        # T-Test mit Bonferroni Korrektur nach Benjamini Hochberg
        qstar = 0.05
        pval = np.ones([t, t])
        stval = np.ones([t, t])
        significance_table = np.zeros([t, t])
        for k in range(t):
            for j in range(t):
                if k != j and k > j:
                    stval[k, j], pval[k, j] = st.ttest_ind(dict_of_metric[k], dict_of_metric[j], equal_var=False)

        print(np.round(pval, 4))
        print(np.round(stval, 4))

        sorted_pvals = np.sort(pval[pval < 1])
        bjq = np.arange(1, len(sorted_pvals) + 1) / \
            len(sorted_pvals) * qstar

        for k in range(t):
            for j in range(t):
                if k != j and k > j:
                    if pval[k, j] in sorted_pvals[sorted_pvals - bjq <= 0]:
                        significance_table[k, j] = 1
                    else:
                        significance_table[k, j] = 0
        
        print(significance_table)

        divider = make_axes_locatable(ax[row,-2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[min, (min+max)/2., max])
        cax.tick_params(axis='both', which='major', labelsize=10)
        cax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # enlarge axis by 10%
        box = ax[row,-2].get_position()
        ax[row,-2].set_position([box.x0, box.y0 - 0.014, box.width * 1.12, box.height * 1.12])
        
        ax[row,-1].set_zorder(-1)
        
        
        # Shrink current axis by some amount%
        box = ax[row,-1].get_position()
        ax[row,-1].set_position([box.x0 + 0.125, box.y0 + 0.02, box.width * 0.85, box.height * 0.85])

        
        ax[row,-1].errorbar(np.arange(t), ginis.mean(axis=0), yerr=ginis.std(axis=0), xerr=None, fmt='o-', color='black')
        #ax[-1,-1].set_xlabel("timesteps")
        #ax[row,-1].set_ylabel("$g_c$")
        ax[0,-1].set_title("$g_c$")
        ax[row,-1].spines['top'].set_visible(False)
        ax[row,-1].spines['right'].set_visible(False)
        ax[row, -1].tick_params(axis='y', right=False, left=True, labelleft=True, labelright=False)
        ax[row, -1].set_xticks(np.arange(t))
        ax[row, -1].set_yticks([0, 0.1, 0.2, 0.3])
        ax[row, -1].set_ylim([0.,0.3])
        ax[row, -1].set_xlim([-0.25,3.25])
        ax[row, -1].set_xticklabels(['$t_0$','$t_1$','$t_2$','$t_3$'])
        
        # significance table (deprecated)
        # 
        # # Shrink current axis by 50%
        # box = ax[row,-1].get_position()
        # ax[row,-1].set_position([box.x0 + 0.1, box.y0 + 0.02, box.width * 0.5, box.height * 0.5])

        # #ax = plt.axes([0.75, 0.5, .10, .10])
        # ax[row,-1].set_zorder(-1)
        # ax[row,-1].matshow(significance_table, cmap='Greys')
        # ax[row,-1].set_xticklabels(['','$t_0$','$t_1$','$t_2$'], fontsize=10)#fontsize=65)
        # ax[row,-1].tick_params(labelbottom=True, labeltop=False,
        #               right=False, top=False)
        # ax[row,-1].set_yticklabels(['$t_0$','$t_1$','$t_2$','$t_3$'], fontsize=10)#fontsize=65)
        # ax[row,-1].set_xlim([-0.5, t - 1 - 0.5])
        # ax[row,-1].set_ylim([t - 0.5, -0.5 + 1])
        # ax[row,-1].spines['top'].set_visible(False)
        # ax[row,-1].spines['right'].set_visible(False)
    
    #ax[-1,-1].add_patch(patches.Rectangle((0.0 - 1.5, 2.5+2.5), 1, 1, fill='black',
    #   color='black', alpha=1, clip_on=False))
    
    #ax[-1,-1].text(0.0, 6.4 - 1.0 + 1, 'Significant difference \n(two-sided t-test, \nexpected FDR=0.05)',
    #fontsize=7, horizontalalignment='left',
    #verticalalignment='center')

    
    # fig, ax = plt.subplots()
    # for ti in range(t):
    #     plot_distribution(np.array(dict_of_metric[ti]), ax, lab='$t={}$'.format(ti))
    # ax.legend()
    # ax.set_title('sigma metric')
    # plt.show()
    
    # Create a Rectangle patch
    ax_in = ax[0,1]
    
    #rect = patches.Rectangle((ax_in.get_xlim()[0],ax_in.get_ylim()[0]),ax_in.get_xlim()[1]-ax_in.get_xlim()[0],ax_in.get_ylim()[1]-ax_in.get_ylim()[0],linewidth=1,edgecolor='black',facecolor='none')
    # Add the patch to the Axes
    #ax[0,0].add_patch(rect)
    # Annotate
    ax[0,0].annotate('B', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-10,+12]), weight='bold', fontsize=24)
    
    plt.savefig(filename, dpi=300, format='pdf')
    plt.close()

# -----------------
# tsne and softmax output functions
# -----------------

def plot_tsne_timetrajectories(representations, imgs, targets, points=1000, show_stimuli=False, show_indices=False, N='all', savefile='./../trained_models/tsnesave.npy', overwrite=False):
    """plot_tsne_timetrajectories is deprecated, use plot_tsne_evolution instead"""
    
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


def plot_tsne_evolution(representations, imgs, targets, show_stimuli=False, show_indices=False, N='all', savefile='./../trained_models/tsnesave', overwrite=False):
    
    # hack to mitigate output
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    
     
    # Constants, maybe become variables later
    N_UNOCC = 1000
    
    targets = targets.numpy()
    classes = [0,1,2,3,4,5,6,7,8,9]    

    markers = ["o","v","s","D","H"]
    # same markers for all timesteps
    markers = ["o","o","o","o","o"]
    
    markersizes = [10,10] #[10,30]
    alpha=1.0
    colors = sns.color_palette("colorblind", len(classes))
    points,time,feature,height,width = representations.shape
    
    
    representations = representations.view(points,time,-1)
    

    # learn tsne embedding
    # -----
    
    
    # we calculate tsne for each timestep seperately
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
        np.save(savefile + '.npy', projection)
        np.save(savefile + '_targets.npy', targets)
    else:
        targets = np.load(savefile + '_targets.npy')
        

    
    projected_data = projection
    
    x_data = projected_data[:-N_UNOCC,:,0] # (index, time)
    y_data = projected_data[:-N_UNOCC,:,1] # (index, time)
    tar = targets[:-N_UNOCC]
    
    x_data_u = projected_data[-N_UNOCC:,:,0] # (index, time)
    y_data_u = projected_data[-N_UNOCC:,:,1] # (index, time)
    tar_u = targets[-N_UNOCC:]
    
    # calculate unoccluded centroids
    
    x_data_uc = np.zeros([len(classes),time])
    y_data_uc = np.zeros([len(classes), time])
    tar_uc = np.zeros(len(classes))
    for ti in range(time): 
        for (i, cla) in enumerate(classes):
            x_data_uc[i,ti] = np.mean([p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla])
            y_data_uc[i,ti]= np.mean([p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla])
            tar_uc[i] = cla
    
    
    # show images for sanity check    
    # for i, t in enumerate(tar_u[-10:]):
    #     print(t)
    #     plt.imshow(imgs[-N_UNOCC+i][0,0,:,:])
    #     plt.show()
    # 
    
    # plot of last timestep
    # fig, ax = plt.subplots(figsize=(9,6))
    # for (i, cla) in enumerate(classes):
    #     xc = [p for (j,p) in enumerate(x_data[:,-1]) if tar[j]==cla]
    #     yc = [p for (j,p) in enumerate(y_data[:,-1]) if tar[j]==cla]
    #     ax.scatter(xc,yc,c=colors[i], label=str(int(cla)), marker=markers[3], alpha=alpha, s=markersizes[0])
    # plt.show()


    # start of the plots
    # -----
    
    # shift data according to timestep
    x_spread = x_data.std()
    for ti in range(time):
        x_data[:,ti] = x_data[:,ti] + ti * x_spread*6 #6
        x_data_u[:,ti] = x_data_u[:,ti] + ti * x_spread*6
        x_data_uc[:,ti] = x_data_uc[:,ti] + ti * x_spread*6

        
    fig, axes = plt.subplots(2,2, sharex=False, sharey=False, figsize=(18,6))
    for pltnr, ax in enumerate([axes[0,0],axes[0,1]]):
        for ti in range(time):
            for (i, cla) in enumerate(classes):
                xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
                yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
                ax.scatter(xc,yc,c=colors[i], label=str(int(cla)), marker=markers[ti], alpha=alpha, s=markersizes[pltnr])        
            
            bracket_ypos = 1.05*y_data.max()
            data_cloud = np.concatenate([x_data[:,ti], x_data_u[:,ti]])
            bracket_xpos = data_cloud.mean()
            bracket_width = data_cloud.std()/3
            ax.annotate('$t_{}$'.format(ti), xy=(bracket_xpos, bracket_ypos), xytext=(bracket_xpos, bracket_ypos), xycoords='data', 
            fontsize=9, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white', ec='white'),
            arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(bracket_width), lw=1.0))
        
        ax.scatter([0], [0], c='white', label=' ')
        ax.axis('off')
    
    # only second plot gets the unoccluded markers
    # unoccluded trajectories
    for ti in range(time): 
        for (i, cla) in enumerate(classes):
            xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
            yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
            ax.scatter(xc,yc,c=colors[i], marker=markers[ti], alpha=alpha, s=markersizes[pltnr], edgecolor='black', label='$t_{}$'.format(ti))
    
    # start of the bottom plots
    # -----
    
    grays = ['lightgray'] * len(classes)
    fills = ['none'] * len(classes)
    colorset = [sns.color_palette("colorblind", len(classes)), sns.color_palette(grays)]
    marker_fills = [fills,fills]
    allhighlights = [[3,8],[9]]
    
    for pltnr in range(len(allhighlights)):
        for hl in allhighlights[pltnr]:
            colorset[pltnr][hl] = colors[hl]
            marker_fills[pltnr][hl] = 'full'
    
    for pltnr, ax in enumerate([axes[1,0],axes[1,1]]):       
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
        ax.axis('off')
    
    ax, pltnr = axes[1,0], 0
    for ti in range(time):
        for (i, cla) in enumerate(classes):
            # rest of the data
            xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
            yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][i], label=str(int(cla)), marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][i]), alpha=alpha, s=markersizes[0])
            # unoccluded centroids
            xc = [p for (j,p) in enumerate(x_data_uc[:,ti]) if tar_uc[j]==cla]
            yc = [p for (j,p) in enumerate(y_data_uc[:,ti]) if tar_uc[j]==cla]
            ax.scatter(xc,yc,c=colors[i], marker=markers[ti], alpha=alpha, s=markersizes[pltnr], edgecolor='black', label='$t_{}$'.format(ti))

    
            # for i in allhighlights[pltnr]:
            #     xd = x_data_u[i,:]
            #     yd = y_data_u[i,:]
            #     for j in range(len(xd)):
            #         ax.scatter(xd[j],yd[j],c='black', marker=markers[j], alpha=alpha, s=markersizes[0],
            #         zorder=9999)
        
    ax, pltnr = axes[1,1], 1
    for ti in range(time):
        for (i, cla) in enumerate(classes):
            # unoccluded data
            xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
            yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][i], label=str(int(cla)), marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][i]), alpha=alpha, s=markersizes[0])
            # rest of the data
            xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
            yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][i], label=str(int(cla)), marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][i]), alpha=alpha, s=markersizes[0])    
    
    
    handles, labels = axes[0,1].get_legend_handles_labels()
    handles = handles[:10] + handles[40:41] # + [handles[-40+i*10] for i in range(time)]
    labels = labels[:10] + labels[40:41] + [labels[-40+i*10] for i in range(time)]

    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], marker=markers[i], color='w', label='', markerfacecolor='black', markersize=6) for i in range(time)]    

    axes[0,1].legend(handles, labels, title='class label', loc='center left', bbox_to_anchor=(1, 0), frameon=False)
    bottom, top = plt.ylim()
    
    
    
    # general setup
    
    axes[0,0].set_ylabel('t-SNE dimension 1')
    axes[1,0].set_ylabel('t-SNE dimension 1')
    axes[1,0].set_xlabel('t-SNE dimension 2')
    axes[1,1].set_xlabel('t-SNE dimension 2')
    
    
    # ax_in = axes[0,1]
    # ax_in.set_xlim([x_data_u[5,-1] - 7, x_data_u[5,-1] + 7])
    # ax_in.set_ylim([y_data_u[5,-1] - 7, y_data_u[5,-1] + 7])
    # 
    # # Create a Rectangle patch
    # rect = patches.Rectangle((ax_in.get_xlim()[0],ax_in.get_ylim()[0]),ax_in.get_xlim()[1]-ax_in.get_xlim()[0],ax_in.get_ylim()[1]-ax_in.get_ylim()[0],linewidth=1,edgecolor='black',facecolor='none')
    # # Add the patch to the Axes
    # axes[0,0].add_patch(rect)
    # Annotate
    # axes[0,0].annotate('B', xy=(ax_in.get_xlim()[0],ax_in.get_ylim()[1]), xytext=np.array([ax_in.get_xlim()[0],ax_in.get_ylim()[1]])+np.array([-2,+1]), weight='bold')
    
    for n, ax in enumerate(axes.flatten()):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], weight='bold', transform=ax.transAxes, size=18)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.show()
    pass


def plot_tsne_evolution2(representations, imgs, targets, show_stimuli=True, show_indices=True, N=25, savefile='./../trained_models/tsnesave', overwrite=False, filename='tsne.pdf'):
    
    # hack to mitigate output
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    
     
    # Constants, maybe become variables later
    N_UNOCC = 1000
    
    targets = targets.numpy()
    classes = [0,1,2,3,4,5,6,7,8,9]    

    markers = ["o","v","s","D","H"]
    # same markers for all timesteps
    markers = ["o","o","o","o","o"]
    
    markersizes = [3,3] #[10,30]
    alpha=1.0
    colors = sns.color_palette("colorblind", len(classes))
    points,time,feature,height,width = representations.shape
    
    
    representations = representations.view(points,time,-1)
    

    # learn tsne embedding
    # -----
    
    
    # we calculate tsne for each timestep seperately
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
        np.save(savefile + '.npy', projection)
        np.save(savefile + '_targets.npy', targets)
    else:
        targets = np.load(savefile + '_targets.npy')
        

    
    projected_data = projection
    
    x_data = projected_data[:-N_UNOCC,:,0] # (index, time)
    y_data = projected_data[:-N_UNOCC,:,1] # (index, time)
    tar = targets[:-N_UNOCC]
    
    x_data_u = projected_data[-N_UNOCC:,:,0] # (index, time)
    y_data_u = projected_data[-N_UNOCC:,:,1] # (index, time)
    tar_u = targets[-N_UNOCC:]
    
    # calculate unoccluded centroids
    
    x_data_uc = np.zeros([len(classes),time])
    y_data_uc = np.zeros([len(classes), time])
    tar_uc = np.zeros(len(classes))
    for ti in range(time): 
        for (i, cla) in enumerate(classes):
            x_data_uc[i,ti] = np.mean([p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla])
            y_data_uc[i,ti]= np.mean([p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla])
            tar_uc[i] = cla

    # start of the plots
    # -----
    
    # shift data according to timestep
    x_spread = x_data.std()
    for ti in range(time):
        x_data[:,ti] = x_data[:,ti] + ti * x_spread*6 #6
        x_data_u[:,ti] = x_data_u[:,ti] + ti * x_spread*6
        x_data_uc[:,ti] = x_data_uc[:,ti] + ti * x_spread*6

        
    fig, axes = plt.subplots(3,1, sharex=False, sharey=False, figsize=(9,6))
    for pltnr, ax in enumerate([axes[0],axes[1]]):
        for ti in range(time):
            for (i, cla) in enumerate(classes):
                xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
                yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
                ax.scatter(xc,yc,c=colors[i], label=str(int(cla)), marker=markers[ti], alpha=alpha, s=markersizes[pltnr])
        
        ax.axis('off')
    
    # only second plot gets the unoccluded markers
    # unoccluded trajectories
    for ti in range(time): 
        for (i, cla) in enumerate(classes):
            xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
            yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
            ax.scatter(xc,yc,c=colors[i], marker=markers[ti], alpha=alpha, s=markersizes[pltnr], edgecolor='black', linewidth=0.5, zorder=10)
    
    # start of the bottom plots
    # -----
    
    grays = ['lightgray'] * len(classes)
    fills = ['none'] * len(classes)
    colorset = [sns.color_palette("colorblind", len(classes)), sns.color_palette(grays)]
    marker_fills = [fills,fills]
    allhighlights = [[3,8],[3]]
    
    for pltnr in range(len(allhighlights)):
        for hl in allhighlights[pltnr]:
            colorset[pltnr][hl] = colors[hl]
            marker_fills[pltnr][hl] = 'full'
    
        
    ax, pltnr = axes[2], 1
    for ti in range(time):
        for (i, cla) in enumerate(classes):
            # rest of the data
            xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
            yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][cla], marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][cla]), alpha=alpha, s=markersizes[0])  
            # unoccluded data
            # xc = [p for (j,p) in enumerate(x_data_u[:,ti]) if tar_u[j]==cla]
            # yc = [p for (j,p) in enumerate(y_data_u[:,ti]) if tar_u[j]==cla]
            # ax.scatter(xc,yc,c=colorset[pltnr][i], label=str(int(cla)), marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][i]), alpha=alpha, s=markersizes[0])
            # unoccluded centroids
            xc = [p for (j,p) in enumerate(x_data_uc[:,ti]) if tar_uc[j]==cla]
            yc = [p for (j,p) in enumerate(y_data_uc[:,ti]) if tar_uc[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][cla], marker=markers[ti], alpha=alpha, s=markersizes[pltnr], edgecolor='black', linewidth=0.5, zorder=10)
        for (i, cla) in enumerate(allhighlights[pltnr]):
            # rest of the data
            xc = [p for (j,p) in enumerate(x_data[:,ti]) if tar[j]==cla]
            yc = [p for (j,p) in enumerate(y_data[:,ti]) if tar[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][cla], marker=MarkerStyle(marker=markers[ti], fillstyle=marker_fills[pltnr][cla]), alpha=alpha, s=markersizes[0])
            # unoccluded centroids
            xc = [p for (j,p) in enumerate(x_data_uc[:,ti]) if tar_uc[j]==cla]
            yc = [p for (j,p) in enumerate(y_data_uc[:,ti]) if tar_uc[j]==cla]
            ax.scatter(xc,yc,c=colorset[pltnr][cla], marker=markers[ti], alpha=alpha, s=markersizes[pltnr], edgecolor='black', linewidth=0.5, zorder=10)
            
    # plot markers
    # -----
    
    for pltnr, ax in enumerate(axes[1:]):       
        if N=='all':
            n_indices = range(len(projected_data))
        elif isinstance(N, int):
            min_N_ind = N//len(set(targets))
            n_indices = []
            for cla in classes:
                ind = np.where(targets == cla)[0]
                n_indices += list(np.random.choice(ind, min(min_N_ind, len(ind)), replace=False))
            n_indices += list(np.random.randint(0,len(projected_data)-N_UNOCC,N-len(n_indices)))
            n_indices = np.array(n_indices)
        elif (pltnr==0):
            n_indices = N[0]
        else:
            n_indices = N[-1]
            
        
        if show_stimuli:
            artists = []
            for ti in range(time):
                for x0, y0, i in zip(projected_data[:,ti,0][n_indices], projected_data[:,ti,1][n_indices], n_indices):
                    # adapt the center of arrows intelligently
                    if ((y0 < 0) and (x0>5)) :
                        xc,yc = x0 - ti * x_spread * 6 ,y0 + 10 #x0+10,y0+20
                    elif ((y0 < 0) and (x0<5)):
                        xc,yc = x0 - ti * x_spread * 6 ,y0 + 10
                    elif (y0 > 0):
                        xc,yc = x0 - ti * x_spread * 6 ,y0
                    else:
                        xc,yc = x0 - ti * x_spread * 6, y0
                    xc,yc = x0 - ti * x_spread * 4, y0


                    #calculate scaling factor c
                    c = np.sqrt((1500./(xc**2 + yc**2))) # (-30., 30.)
                    xy_box = np.array([c*xc, c*yc])
                    
                    # handcrafted rearrangement
                    if i in [516] and ti == 0:
                        xy_box = xy_box - np.array([0,15])
                    elif i in [1629, 516] and ti == 1:
                        xy_box = xy_box + np.array([0,15])
                    elif i in [909] and ti == 1:
                        xy_box = xy_box - np.array([10,0])
                    elif i in [909] and ti == 2:
                        xy_box = xy_box + np.array([30,0])
                    elif i in [226] and ti == 2:
                        xy_box = xy_box + np.array([10,-10])
                    elif i in [1629, 516] and ti == 3:
                        xy_box = xy_box - np.array([20,13])
                    else:
                        pass
                    # handcrafted annotation
                    if i in [516] and ti == 0:
                        ax.annotate('{}'.format('3 [2,0]'), xy=(x0, y0), xytext=xy_box + np.array([-5,25]), zorder=-1)
                    elif i in [1629] and ti == 0:
                        ax.annotate('{}'.format('2'), xy=(x0, y0), xytext=xy_box + np.array([-20,18]), zorder=-1)
                    elif i in [909] and ti == 0:
                        ax.annotate('{}'.format('3 [2,8]'), xy=(x0, y0), xytext=xy_box + np.array([-10,-23]), zorder=-1)
                    elif i in [226] and ti == 0:
                        ax.annotate('{}'.format('2 [9,8]'), xy=(x0, y0), xytext=xy_box + np.array([0,-27]), zorder=-1)
                    else:
                        pass
                    
                    
                    ab = offsetbox.AnnotationBbox(makeMarker(imgs[i,ti,0], zoom=0.65*32./len(imgs[i,ti,0])), (x0, y0), xybox=xy_box, xycoords='data', boxcoords="offset points",
                                      pad=0.1,bboxprops=dict(color=colorset[pltnr][int(targets[i])]) , arrowprops=dict(arrowstyle=patches.ArrowStyle("->", head_length=.2, head_width=.1)), frameon=True)
                    
                    
                  
                    if show_indices:
                        ax.annotate('{}'.format(i), xy=(x0, y0), xytext=(x0, y0), zorder=-1)
                    if len(n_indices) >= 100:
                        ab.zorder=-1
                    else:
                        ab.zorder=99
                    
                    artists.append(ax.add_artist(ab))
                    # if i == 622:
                    #   artists.append(ax.add_artist(ab2))
        ax.axis('off')

    
    # general setup
    
    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles[:10]
    labels = labels[:10]

    axes[0].legend(handles, labels, title='class label', loc='center left', bbox_to_anchor=(1, 0), frameon=False)
    bottom, top = plt.ylim()
    
    for ti in range(time):
        bracket_ypos = 1.10*y_data.max()
        data_cloud = np.concatenate([x_data[:,ti], x_data_u[:,ti]])
        bracket_xpos = data_cloud.mean()
        bracket_width = data_cloud.std()/3
        axes[0].annotate('$t_{}$'.format(ti), xy=(bracket_xpos, bracket_ypos), xytext=(bracket_xpos, bracket_ypos), xycoords='data', 
        fontsize=10, ha='center', va='bottom',
        bbox=dict(boxstyle='square', fc='white', ec='white'),
       # arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.25, angleB=0'.format(bracket_width), lw=1.0)
        )
        
    
    
    
    axes[0].set_ylabel('t-SNE dimension 1')
    axes[1].set_ylabel('t-SNE dimension 1')
    axes[2].set_ylabel('t-SNE dimension 1')
    axes[2].set_xlabel('t-SNE dimension 2')
    
    
    for n, ax in enumerate(axes.flatten()):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], weight='bold', transform=ax.transAxes, size=18)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.savefig(filename, dpi=300, format='pdf')
    plt.show()
    pass




def plot_relative_distances(representations, nhot_targets, representations_unocc, onehot_targets_unocc, filename):
    
    classes = 10
    n_occ = 2

        
    points,time,feature,height,width = representations.shape
    representations = representations.view(points,time,-1).numpy()
    representations_unocc = representations_unocc.view(points,time,-1).numpy()
    
    _,_,dim = representations.shape
    _,n_targets = nhot_targets.shape
    
    
    # get the centroid of the un-occluded representation for each class
    # and timestep
    centroid_unocc = np.zeros([classes, time, dim]) # (10,4,32)
    for (i,cla) in enumerate(range(classes)):
        r_sortedbyclass = np.array([p for (j,p) in enumerate(representations_unocc) if onehot_targets_unocc[j]==cla])
        centroid_unocc[cla,:,:] = np.mean(r_sortedbyclass, axis=0)

    
    # calculate the distances for the two occluder-centroids to the representation
    # of the occluded digits
    
    sim = metrics.Similarity(minimum=0.001)
    
    distances = np.zeros([points,time,n_targets])
    relative_distances = np.zeros([points,time,n_targets-1])
    

    for ti in range(time):
        for i,(a,b,c) in enumerate(nhot_targets):
            distances[i,ti,0] = sim.fractional_distance(
                representations[i,ti], centroid_unocc[a,ti])
            distances[i,ti,1] = sim.fractional_distance(
                representations[i,ti], centroid_unocc[b,ti])
            distances[i,ti,2] = sim.fractional_distance(
                representations[i,ti], centroid_unocc[c,ti])
    
    # calculate relative distances relative_distance = d_zur_8 / 0.5(d_zur_8 + d_zur_2)
    # for ti in range(time):
    #     relative_distances[:,ti,0] = distances[:,ti,0] / (0.5*(distances[:,ti,0] + distances[:,ti,1]))
    #     relative_distances[:,ti,1] = distances[:,ti,0] / (0.5*(distances[:,ti,0] + distances[:,ti,2]))
    
    # alternative relative distances
    for ti in range(time):
        relative_distances[:,ti,0] = distances[:,ti,0] / distances[:,ti,1]
        relative_distances[:,ti,1] = distances[:,ti,0] / distances[:,ti,2]
    
    # create distribution plots
    # -----
    
    fig, ax = plt.subplots()
    for ti in range(time):
        plot_distribution(relative_distances[:,ti,0], ax, lab='$t={}$'.format(ti))
    ax.legend()
    ax.set_title('relative distance target, occluder 1')
    # plt.savefig('A.pdf')
    plt.show()
    
    fig, ax = plt.subplots()
    for ti in range(time):
        plot_distribution(relative_distances[:,ti,1], ax, lab='$t={}$'.format(ti))
    ax.legend()
    ax.set_title('relative distance target, occluder 2')
    # plt.savefig('B.pdf')
    ax.annotate('B', xy=(ax.get_xlim()[0],ax.get_ylim()[1]), xytext=np.array([ax.get_xlim()[0],ax.get_ylim()[1]])+np.array([-7,+2]), weight='bold', fontsize=24)
    plt.show()
    

    fig, ax = plt.subplots()
    plot_distribution(distances[:,0,0], ax, lab='$target,t=0$')
    plot_distribution(distances[:,1,0], ax, lab='$target,t=1$')
    plot_distribution(distances[:,2,0], ax, lab='$target,t=2$')
    plot_distribution(distances[:,3,0], ax, lab='$target,t=3$')
    ax.set_title('absolute distance target to stimulus')
    ax.legend()
    # plt.savefig('C.pdf')
    plt.show()
    
    fig, ax = plt.subplots()
    plot_distribution(distances[:,0,1], ax, lab='$occ1,t=0$')
    plot_distribution(distances[:,1,1], ax, lab='$occ1,t=1$')
    plot_distribution(distances[:,2,1], ax, lab='$occ1,t=2$')
    plot_distribution(distances[:,3,1], ax, lab='$occ1,t=3$', xlabel='absolute distance')
    ax.set_title('absolute distance occluder 1 to stimulus')
    ax.legend()
    # plt.savefig('C.pdf')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(4.5, 3.4),
                           gridspec_kw=dict(bottom=0.15, left=0.15))
    reldist_df = pd.DataFrame(
    np.hstack([
    np.vstack([relative_distances[:,0,0], np.repeat(0, points), np.repeat(1, points)]),
    np.vstack([relative_distances[:,1,0], np.repeat(1, points), np.repeat(1, points)]),
    np.vstack([relative_distances[:,2,0], np.repeat(2, points), np.repeat(1, points)]),
    np.vstack([relative_distances[:,3,0], np.repeat(3, points), np.repeat(1, points)])
    ,
    np.vstack([relative_distances[:,0,1], np.repeat(0, points), np.repeat(2, points)]),
    np.vstack([relative_distances[:,1,1], np.repeat(1, points), np.repeat(2, points)]),
    np.vstack([relative_distances[:,2,1], np.repeat(2, points), np.repeat(2, points)]),
    np.vstack([relative_distances[:,3,1], np.repeat(3, points), np.repeat(2, points)])
    
    ]).T, columns=['data', 'timestep', 'occluder'])
    sns.violinplot(data=reldist_df, y='data', x='timestep', palette='Greys', hue='occluder', split=True, ax=ax)
    ax.axhline(y=relative_distances[:,0,0].mean(), xmin=0, xmax=5, color='black', linestyle='--')
    #ax.axhline(y=0), xmin=0, xmax=5, color='black', linestyle='--')
    #ax.set_title('Relative distance - unoccluded target, unoccluded occluder')
    ax.set_xticklabels(['$t_0$','$t_1$','$t_2$','$t_3$'], fontsize=12)
    ax.set_ylabel('Relative distance')
    ax.set_xlabel('Time step')
    ax.legend(ax.get_legend_handles_labels()[0],['$d_{rel,1}$', '$d_{rel,2}$'],frameon=True, facecolor='white', edgecolor='white', framealpha=0.0, loc='upper right')
    # from statannot import add_stat_annotation
    # add_stat_annotation(ax, data=reldist_df, x='timestep', y='data', hue='occluder',
    #     box_pairs=[
    #         # ((0,1),(1,1)),
    #         # ((1,1),(2,1)),
    #         # ((0,2),(1,2)),
    #         # ((1,2),(2,2)),
    #         ((2,1),(3,1)),
    #         ((2,2),(3,2)),
    #     ], test='Kolmogorov-Smirnov-ls', text_format='star', loc='inside', verbose=2)
    ax.annotate('B', xy=(ax.get_xlim()[0],ax.get_ylim()[1]), xytext=np.array([ax.get_xlim()[0],ax.get_ylim()[1]])+np.array([-0.75,0.25]), weight='bold', fontsize=16)
    
    plt.savefig(filename, dpi=300, format='pdf')
    
    pass

def plot_softmax_output(network_output, targets, images, filename):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    batchsize,time,classes = network_output.shape
    
    softmax_output = torch.zeros([batchsize, time, classes])
    for ti in range(time):
        # calculate softmaxover time
        softmax_output[:,ti,:] = F.softmax(network_output[:,ti,:], 1)        
    
    
    # find interesting samples
    # -----
    
    correct_t0 = []
    for i in range(batchsize):
        if (np.argmax(network_output[i, 0, :]) == int(targets[i])):
            correct_t0.append(i)


    correct = []
    for i in range(batchsize):
        if (np.argmax(network_output[i, -1, :]) == int(targets[i])):
            correct.append(i)

    revised = []
    for i in range(batchsize):
        if (np.argmax(network_output[i, 0, :]) != np.argmax(network_output[i, -1, :])) and (np.argmax(network_output[i, -1, :]) == int(targets[i])):
            revised.append(i)

    reinforced = []
    for i in range(batchsize):
        if (np.argmax(network_output[i, -2, :]) == np.argmax(network_output[i, -1, :])) and (np.argmax(network_output[i, -3, :]) == np.argmax(network_output[i, -1, :])) and (np.argmax(network_output[i, -1, :]) == int(targets[i])):
            reinforced.append(i)

    destroyed = []
    for i in range(batchsize):
        if (np.argmax(network_output[i, -1, :]) != np.argmax(network_output[i, 0, :])) and (np.argmax(network_output[i, 0, :]) == int(targets[i])):
            destroyed.append(i)
    
    print('[INFO] softmax output stats:')
    print('\t correct:\t {}, percentage: {}'.format(
            len(correct), len(correct)/batchsize))
    print('\t revised:\t {}, of all: {}, of correct: {}'.format(
            len(revised), len(revised)/batchsize, np.round(len(revised)/len(correct), 3)))
    print('\t reinforced:\t {}, of all: {}, of correct: {}'.format(
            len(reinforced), len(reinforced)/batchsize, np.round(len(reinforced)/len(correct), 3)))

    print('\t destroyed:\t {}, of all: {}, of false: {}, of correct_t0: {}'.format(
            len(destroyed), len(destroyed)/batchsize, np.round(len(destroyed)/(batchsize-len(correct)), 3), np.round(len(destroyed)/(len(correct_t0)), 3)))
    
    # look at interesting cases
    # -----
    
    for j in revised[30:30]:#range(55,60,1):
        fig, ax = plt.subplots()
        for ti in range(time):
            ax.plot(softmax_output[j,ti,:], label='$t_{}$'.format(ti))        
        
        ax.set_yscale('log')
        ax.set_ylim([1e-8,3])
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(frameon=True, facecolor='white', edgecolor='white', framealpha=1.0, bbox_to_anchor=(1, .5), loc='center left')
        ab = offsetbox.AnnotationBbox(makeMarker(images[j,0,0], zoom=2.0), (.835, .775), xycoords='figure fraction', boxcoords="offset points",
                        pad=0.3, frameon=True)
        ax.axvline(x=targets[j], ymin=0, ymax=2, color='black', linestyle='--')
        ax.add_artist(ab)
        ax.set_ylabel('Softmax output (probability)')
        ax.set_xlabel('Class')
        print('showing image no.', j)
        ax.text(-0.1, 1.01, '{}'.format(j), weight='bold',
                      transform=ax.transAxes, size=40)
        plt.xticks(np.arange(0, classes, step=classes//10))
    plt.show()
    
    
    # calculate mean output
    # -----
    pointsofclass = {}
    for (i, cla) in enumerate(range(classes)):
        pointsofclass[i] = [p for (j, p) in enumerate(
            softmax_output.numpy()) if int(targets[j]) == cla]
        
    mean_output = np.zeros([classes, time, classes])
    error_output = np.zeros([classes, time, classes])

    for cla in range(classes):
        try:
            mean_output[cla, :, :] = np.array(pointsofclass[cla]).mean(axis=0)
            error_output[cla, :, :] = np.array(pointsofclass[cla]).std(
                axis=0) / np.sqrt(len(pointsofclass[cla]))
        except:
            print('error, class {} not found'.format(cla))
            mean_output[cla, :, :] = 0
            error_output[cla, :, :] = 0

    lstylist = ['-', '--', ':', '-.']
    markerlist = ['o', 'v', 's', 'D']
    fillstylelist = ['full', 'full', 'full', 'full']
    
    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(9, 3.5),
                             gridspec_kw=dict(bottom=0.15,))
                                 # wspace=0.0, hspace=0.3,
                                 # top=0.90,
                                 # bottom=0.075,
                                 # left=0.05,
                                 # right=0.95),))
    for j, ax in enumerate(axes.flatten()):
        for ti in range(time):
            ax.plot(mean_output[j, ti, :], label='$t_{}$'.format(ti), marker=markerlist[ti], markersize=3)
            ax.fill_between(np.arange(
                0, classes, 1), mean_output[j, ti, :] + error_output[j, ti, :], mean_output[j, ti, :] - error_output[j, ti, :], alpha=0.25)

        # logscale!
        ax.set_yscale('log')
        ax.set_ylim([1e-4, 1])
        ax.axvline(x=int(j),
                   ymin=0, ymax=2, color='black', linestyle='--')
        # ax.add_artist(ab)
        
        # add titles with targets
        if j > 4:
            t_x, t_y = 0.025, 0.
        else:
            t_x, t_y = 0.875, 0.
        ax.text( 
        # position text relative to Figure
        t_x, t_y, '{}'.format(j),
        ha='left', va='bottom',
        transform=ax.transAxes, fontsize=14, color=sns.color_palette("colorblind", 10)[j])
        #ax.add_patch(patches.Circle((1,1),1), ha='left', va='top', transform=ax.transAxes)

    axes[0, 4].legend(frameon=True, facecolor='white', edgecolor='white',
                      framealpha=1.0, bbox_to_anchor=(1.0, .72), loc='center left', title='time step')
    axes[0, 0].set_ylabel('Softmax output')
    axes[1, 0].set_ylabel('Softmax output')
    for i in range(5):
        axes[1, i].set_xlabel('Class')



    plt.xticks(np.arange(0, classes, step=classes//10))
    #plt.suptitle('Mean softmax output over candidates for each target')
    #plt.savefig('mean_softmax.pdf')
    axes[0,0].annotate('B', xy=(axes[0,0].get_xlim()[0],axes[0,0].get_ylim()[1]), xytext=np.array([axes[0,0].get_xlim()[0],axes[0,0].get_ylim()[1]])+np.array([-9,+2]), weight='bold', fontsize=16)
    plt.savefig('{}B.pdf'.format(filename), dpi=300, format='pdf')
    #plt.show()
    
    
#     fig, axes = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(14, 10))
#     for j, ax in enumerate(axes.flatten()):
#         j += 150
#         for ti in range(time):
#             ax.plot(softmax_output[j, ti, :], label='$t_{}$'.format(ti), color=colors[ti], marker=markerlist[ti], markersize=3)
# 
#         ax.set_yscale('log')
#         ax.set_ylim([1e-9, 5])
#         ab = offsetbox.AnnotationBbox(makeMarker(images[j,0,0], zoom=1), (.9, .1), xycoords='axes fraction', boxcoords="offset points",
#                                       pad=0.3, frameon=True)
#         ax.axvline(x=targets[j],
#                    ymin=0, ymax=2, color='black', linestyle='--')
#         ax.add_artist(ab)
#         ax.set_ylabel('Softmax output')
#         ax.set_xlabel('Class')
# 
#     axes[0, 4].legend(frameon=True, facecolor='white', edgecolor='white',
#                       framealpha=1.0, bbox_to_anchor=(1.0, .8), loc='center left')
# 
#     plt.xticks(np.arange(0, classes, step=classes//10))
#     plt.suptitle('Softmax output for specific candidates')
#     #plt.savefig('specific_candidates.pdf')
#     plt.show()
    
    
    
#     lstylist = ['-', '--', ':', '-.']
#     markerlist = ['o', 'v', 's', 'D']
#     fillstylelist = ['full', 'full', 'full', 'full']
#     candlist = [87, 128, 206] 
#     meanlist = [2, 3, 4]
#     fig, axes = plt.subplots(2, 3, sharex=False, sharey='row', figsize=(12, 8))
#     
#     for j, ax in zip(candlist, axes[0]):
#         for ti in range(time):
#             ax.plot(softmax_output[j, ti, :], label='$t_{}$'.format(
#                 ti), linewidth=3, marker=markerlist[ti], markersize=7.0, fillstyle=fillstylelist[ti], color=colors[ti])
# 
#         ax.set_yscale('log')
#         ax.set_ylim([1e-10, 5])
#         ab = offsetbox.AnnotationBbox(makeMarker(images[j,0,0], zoom=1.6), (.805, .16), xycoords='axes fraction', boxcoords="offset points",
#                                       pad=0.3, frameon=True)
#         ax.axvline(x=targets[j], ymin=0,
#                    ymax=2, color='black', linestyle='--', linewidth=2)
#         ax.add_artist(ab)
#         #ax.set_ylabel('Softmax output')
#         ax.set_xticks(np.arange(0, classes, step=classes//10))
#         # ax.set_xlabel('Class')
#     for j, ax in zip(meanlist, axes[1]):
#         for ti in range(time):
#             #ax.plot(mean_output[j,t,:], label='$t_{}$'.format(t), linewidth=3, marker=markerlist[t], markersize=7.0, fillstyle=fillstylelist[t], color=colors[t])
#             # ax.bar(np.arange(0,10,1),output_data[0,i,j,:],label='$t_{}$'.format(i))
# 
#             # fill between error
#             #ax.fill_between(np.arange(0,10,1), mean_output[j,t,:]+error_output[j,t,:], mean_output[j,t,:]-error_output[j,t,:], alpha=0.25)
#             # small outline for errors
#             # ax.plot(mean_output[j,t,:]+error_output[j,t,:], linewidth=1, color=colors[t]) #, linestyle=lstylist[t])
#             # ax.plot(mean_output[j,t,:]-error_output[j,t,:], linewidth=1, color=colors[t]) #, linestyle=lstylist[t])
#             ax.errorbar(np.arange(0, classes, 1), mean_output[j, ti, :], label='$t_{}$'.format(
#                 ti), linewidth=3, marker=markerlist[ti], markersize=7.0, fillstyle=fillstylelist[ti], color=colors[ti], yerr=error_output[j, ti, :])
# 
#         # switch on or off logscale
#         ax.set_yscale('log')
#         ax.set_ylim([1e-4, 3])
# 
# 
#         ax.axvline(x=int(j), ymin=0, ymax=2, color='black',
#                    linestyle='--', linewidth=2)
#         # ax.add_artist(ab)
#         #ax.set_ylabel('Softmax output')
#         ax.set_xlabel('Class label')
#         ax.set_xticks(np.arange(0, classes, step=classes//10))
# 
#     axes[0, 2].legend(frameon=True, facecolor='white', edgecolor='white', framealpha=1.0,
#                       bbox_to_anchor=(1., .725), loc='center left', title='time step')
#     #axes[0,1].set_title('Specific candidates')
#     #axes[1,1].set_title('Mean softmax output per class (2,4,8)')
#     axes[0, 0].set_ylabel('Softmax output')
#     axes[1, 0].set_ylabel('Softmax output')
# 
#     axes[0, 0].text(-0.57, 1.03, 'A', weight='bold',
#                     transform=axes[0, 0].transAxes, size=40)
#     axes[1, 0].text(-0.57, 1.03, 'B', weight='bold',
#                     transform=axes[1, 0].transAxes, size=40)
# 
#     plt.subplots_adjust(left=None, bottom=None, right=0.88,
#                         top=0.935, wspace=None, hspace=None)
#     # plt.savefig('os_softmax33avg.ps')
#     # plt.savefig('os_softmax33avg.pdf')
#     plt.show()
    
    
    #candlist = [87, 128, 206, 24, 33]
    candlist = [55, 49, 330, 313, 342]
    #candlist = [342, 340, 339, 330, 313, 285, 264,]

    fig, axes = plt.subplots(1, 5, sharex=False, sharey='row', figsize=(9, 2))

    for j, ax in zip(candlist, axes):
        for ti in range(time):
            ax.plot(softmax_output[j, ti, :], label='$t_{}$'.format(
                ti), marker=markerlist[ti], markersize=3.0, fillstyle=fillstylelist[ti], color=colors[ti])

        ax.set_yscale('log')
        ax.set_ylim([1e-10, 5])
        ab = offsetbox.AnnotationBbox(makeMarker(images[j,0,0], zoom=0.60), (.85, .15), xycoords='axes fraction', boxcoords="offset points",
                                      pad=0.3, frameon=True)
        ax.axvline(x=targets[j], ymin=0,
                   ymax=2, color='black', linestyle='--')
        ax.add_artist(ab)
        ax.set_xticks(np.arange(0, classes, step=classes//10))
        ax.set_xlabel('Class')
        
        # add titles with targets
        t_x, t_y = 0.025, 0.
        ax.text( 
        # position text relative to Figure
        t_x, t_y, '{}'.format(targets[j]),
        ha='left', va='bottom',
        transform=ax.transAxes, fontsize=14, color=sns.color_palette("colorblind", 10)[targets[j]])


    axes[4].legend(frameon=True, facecolor='white', edgecolor='white', framealpha=1.0,
                      bbox_to_anchor=(1., .725), loc='center left', title='time step')
    axes[0].set_ylabel('Softmax output')


    axes[0].annotate('A', xy=(axes[0].get_xlim()[0],axes[0].get_ylim()[1]), xytext=np.array([axes[0].get_xlim()[0],axes[0].get_ylim()[1]])+np.array([-9,+2]), weight='bold', fontsize=16)

    plt.subplots_adjust(left=None, bottom=0.25, right=None,
                        top=0.85, wspace=None, hspace=None)
    plt.savefig('{}A.pdf'.format(filename), dpi=300, format='pdf')
    #plt.show()
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
