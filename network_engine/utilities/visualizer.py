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
    _,_,channels,height,width = images.shape
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
            img = img.view(1,channels//2,height*2,width)
            print(img.shape)
        if one_channel:
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
