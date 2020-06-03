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
import numpy as np

from PIL import Image
from textwrap import wrap
import matplotlib as mpl
import re
import itertools
from math import sqrt
import matplotlib.pyplot as plt


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



# saliency maps or class activation mapping
# -----

def saliencymap_to_figure(smap, pic, alpha=0.5):
    """
    saliencymap_to_tfsummary takes a saliency map smap, a picture pic and an
    optional value for alpha and returns a tf summary containing the picture
    overlayed with the saliency map with transparency alpha.
    Add tfplot.figure.to_summary(fig, tag=tag) to plot to summary
    """
    number_of_maps_per_axis = int(np.floor(np.sqrt(smap.shape[0])))
    fig = mpl.figure.Figure(figsize=(
        number_of_maps_per_axis, number_of_maps_per_axis), dpi=90,
        facecolor='w', edgecolor='k')

    for i in range(np.square(number_of_maps_per_axis)):
        classmap_answer = smap[i, :, :, 0]
        vis = list(
            map(lambda x: ((x - x.min()) / (x.max() - x.min())),
                classmap_answer))
        ori = (pic[i, :, :, 0] + 1.) / 2.
        ax = fig.add_subplot(number_of_maps_per_axis,
                             number_of_maps_per_axis, (i + 1))
        ax.imshow(ori, cmap="Greys")
        ax.imshow(vis, cmap=mpl.cm.jet, alpha=alpha,
                  interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')

    return fig


# confusion matrix
# -----

def cm_to_figure(confusion_matrix, labels, title='Confusion matrix',
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
    cm = confusion_matrix
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

    ax.set_xlabel('True Label', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Predicted', fontsize=7)
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


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
