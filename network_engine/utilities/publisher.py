#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# July 2021	                                     _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# publisher.py                               oN88888UU[[[/;::-.        dP^
# create paperready                         dNMMNN888UU[[[/;:--.   .o@P^
# figures and plots		                  ,MMMMMMN888UU[[/;::-. o@^
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision

import os
import sys

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# statsmodels for ANOVA
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

from scipy.stats import shapiro, levene, ks_2samp, ttest_ind

# custom functions
# -----
import utilities.visualizer as visualizer
from utilities.networks.buildingblocks.rcnn import RecConvNet, CAM
from utilities.dataset_handler import StereoImageFolderLMDB, StereoImageFolder, AffineTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------
# Helper Functions
# ----------------


def eta_squared(aov):
	aov['eta_sq'] = 'NaN'
	aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
	return aov
def omega_squared(aov):
	mse = aov['sum_sq'][-1]/aov['df'][-1]
	aov['omega_sq'] = 'NaN'
	aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
	return aov
	

def show(img):
	npimg = img.numpy()
	#print(np.transpose(npimg, (1,2,0)).shape)
	plt.matshow(np.transpose(npimg, (1,2,0))[:,:,0], interpolation='nearest', cmap='viridis')


def generate_hidden_representation(test_loader, network, timesteps, stereo):
	loss = 0
	accuracy = 0
	feature_list = []
	input_list = []
	target_list = []
	output_list = []
	classification_list = []
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			input_tensor, target_tensor = data
			if stereo:
				input_tensor = torch.cat(input_tensor, dim=1)
			input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
			input_tensor = input_tensor.unsqueeze(1)
			input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
			
			outputs, features = network(input_tensor)
			# get features after GAP
			features = features.mean(dim=[-2,-1], keepdim=True) #global average pooling

			# topv, topi = outputs[:,-1,:].topk(1)
			# accuracy += (topi == target_tensor.unsqueeze(1)).sum(
			#     dim=0, dtype=torch.float64) / topi.shape[0]
			
			feature_list.append(features)
			input_list.append(input_tensor)
			target_list.append(target_tensor)
			output_list.append(outputs)
		features = torch.cat(feature_list, dim=0)
		inputs = torch.cat(input_list, dim=0)
		targets = torch.cat(target_list, dim=0)
		outputs = torch.cat(output_list, dim=0)

	return features, inputs, targets, outputs

def generate_class_activation(test_loader, network, timesteps, stereo):
	cam = CAM(network)
	loss = 0
	accuracy = 0
	cam_list = []
	input_list = []
	target_list = []
	output_list = []
	topk_prob_list = []
	topk_pred_list = []

	# TODO: Solve the unroll-timestep handling as a function parameter
	#timesteps = configuration_dict['time_depth'] + 1 + configuration_dict['time_depth_beyond']

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			input_tensor, target_tensor = data
			if stereo:
				input_tensor = torch.cat(input_tensor, dim=1)
			input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
			input_tensor = input_tensor.unsqueeze(1)
			input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
			
			outputs, (cams, topk_prob, topk_pred) = cam(input_tensor)            
			cam_list.append(cams)
			input_list.append(input_tensor)
			target_list.append(target_tensor)
			output_list.append(outputs)
			topk_prob_list.append(topk_prob)
			topk_pred_list.append(topk_pred)
	
	class_activations = torch.cat(cam_list, dim=0)
	inputs = torch.cat(input_list, dim=0)
	targets = torch.cat(target_list, dim=0)
	outputs = torch.cat(output_list, dim=0)
	topk_probabilities = torch.cat(topk_prob_list, dim=0)
	topk_predictions = torch.cat(topk_pred_list, dim=0)
	
	return class_activations, inputs, targets, outputs, topk_probabilities, topk_predictions


def compare_concentration_mass(rgb_loader, test_loader, network, timesteps, stereo):
	cam = CAM(network)
	with torch.no_grad():
		target_percentages = []
		target_pixel_percentages = []
		occluder_percentages = []
		occluder_pixel_percentages = []
		overlap_percentages = []
		overlap_pixel_percentages = []
		background_percentages = []
		background_pixel_percentages = []

		for i, (data, rgb) in enumerate(zip(test_loader, rgb_loader)):
			input_tensor, target_tensor = data
			if stereo:
				input_tensor = torch.cat(input_tensor, dim=1)
			input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
			input_tensor = input_tensor.unsqueeze(1)
			input_tensor = input_tensor.repeat(1, timesteps, 1, 1, 1)
			
			outputs, (cams, topk_prob, topk_pred) = cam(input_tensor)
			
			b,t,n_classes,h,w = cams.shape
			
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
			
			
			mass_perc_on_target = np.zeros([b, timesteps])
			mass_perc_on_target_pixel = np.zeros([b, timesteps])

			mass_perc_on_occluder = np.zeros([b, timesteps])
			mass_perc_on_occluder_pixel = np.zeros([b, timesteps])
			
			mass_perc_on_background = np.zeros([b, timesteps])
			mass_perc_on_background_pixel = np.zeros([b, timesteps])
			
			mass_perc_on_overlap = np.zeros([b, timesteps])
			mass_perc_on_overlap_pixel = np.zeros([b, timesteps])

			# R: target, G: occluder, B: overlap, None: background) 
			# R and B: targets
			tar_pixels = rgb[0][0][:, 0, :, :]# + rgb[0][0][:, 2, :, :]
			# G and B: occluders
			occ_pixels = rgb[0][0][:, 1, :, :]# + rgb[0][0][:, 2, :, :]
			#
			ovl_pixels = rgb[0][0][:, 2, :, :]
			# None
			background_pixels = (-1)*(rgb[0][0][:, 0, :, :] + rgb[0][0][:, 1, :, :] + rgb[0][0][:, 2, :, :]) + 1
			
			
			
			for ti in range(timesteps):
				for ind in range(b):
					# filter out incorrect classifications
					#if topk_pred[ind,-1,0] == target_tensor[ind, 0]:
					if True:
						c = cams[ind, ti, topk_pred[ind,ti,0], :, :]
						
						total_mass = c.sum()
						mass_perc_on_target[ind, ti] = (c[tar_pixels[ind] > 0].sum() / total_mass)
						mass_perc_on_target_pixel[ind, ti] = mass_perc_on_target[ind, ti] / (tar_pixels[ind] > 0).sum()
	
						mass_perc_on_occluder[ind, ti] = (c[occ_pixels[ind] > 0].sum() / total_mass)
						mass_perc_on_occluder_pixel[ind, ti] = mass_perc_on_occluder[ind, ti] / (occ_pixels[ind] > 0).sum()

						mass_perc_on_background[ind, ti] = (c[background_pixels[ind] > 0].sum() / total_mass)
						mass_perc_on_background_pixel[ind, ti] = mass_perc_on_background[ind, ti] / (background_pixels[ind] > 0).sum()
						
						mass_perc_on_overlap[ind, ti] = (c[ovl_pixels[ind] > 0].sum() / total_mass)
						mass_perc_on_overlap_pixel[ind, ti] = mass_perc_on_overlap[ind, ti] / (ovl_pixels[ind] > 0).sum()
					else:
						pass

			target_percentages.append(mass_perc_on_target.copy())
			target_pixel_percentages.append(mass_perc_on_target_pixel.copy())
			occluder_percentages.append(mass_perc_on_occluder.copy())
			occluder_pixel_percentages.append(mass_perc_on_occluder_pixel.copy())
			background_percentages.append(mass_perc_on_background.copy())
			background_pixel_percentages.append(mass_perc_on_background_pixel.copy())
			overlap_percentages.append(mass_perc_on_overlap.copy())
			overlap_pixel_percentages.append(mass_perc_on_overlap_pixel.copy())

		
		target_percentages = np.concatenate(target_percentages, axis=0)
		target_pixel_percentages = np.concatenate(target_pixel_percentages, axis=0)
		occluder_percentages = np.concatenate(occluder_percentages, axis=0)
		occluder_pixel_percentages = np.concatenate(occluder_pixel_percentages, axis=0)
		background_percentages = np.concatenate(background_percentages, axis=0)
		background_pixel_percentages = np.concatenate(background_pixel_percentages, axis=0)
		overlap_percentages = np.concatenate(overlap_percentages, axis=0)
		overlap_pixel_percentages = np.concatenate(overlap_pixel_percentages, axis=0)
		
		
		return target_percentages, target_pixel_percentages, occluder_percentages, occluder_pixel_percentages, overlap_percentages, overlap_pixel_percentages, background_percentages, background_pixel_percentages
	


# ----------------
# Analysis Functions
# ----------------

def fig_cam(network, test_transform, configuration_dict, sample_size, random_seed):
	np.random.seed(random_seed)
	
	test_dataset = StereoImageFolder(
		root_dir=configuration_dict['input_dir'] + '/{}'.format(configuration_dict['dataset']),
		train=False,
		stereo=configuration_dict['stereo'],
		transform=test_transform,
		nhot_targets=True
		)

	rep_sample = list(np.random.choice(range(len(test_dataset)), size=sample_size, replace=False)) 
	test_subset = torch.utils.data.Subset(test_dataset, rep_sample)
	test_loader = torch.utils.data.DataLoader(test_subset, batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)


	cams, img, tar, out, topk_prob, topk_pred = generate_class_activation(test_loader, network, configuration_dict['time_depth'] + 1, configuration_dict['stereo'])
	
	# filter correct predictions - best topk at last timestep = target
	correct_indices = (tar[:,0] == topk_pred[:, -1, 0])
	# show means for correct predictions
	# visualizer.plot_cam_means(
	#     cams[correct_indices],
	#     tar[correct_indices,0],
	#     topk_prob[correct_indices],
	#     topk_pred[correct_indices]
	#     )    
	

	#visualizer.plot_cam_samples(cams, img, tar, topk_prob, topk_pred, list_of_indices=[948,614,541], filename='{}/fig8a_cam_samples.pdf'.format(configuration_dict['visualization_dir']))
	#visualizer.plot_cam_samples_alt(cams, img, tar, topk_prob, topk_pred, list_of_indices=[948,614,541], filename='{}/fig8a_cam_samples_alt.pdf'.format(configuration_dict['visualization_dir']))
	# np.random.choice(np.arange(1000),10)
	# visualizer.plot_cam_samples(cams, img, tar, topk_prob, topk_pred, list_of_indices=[972, 51, 205, 227, 879, 538, 112, 741, 309, 289])
	# visualizer.plot_cam_samples_alt(cams, img, tar, topk_prob, topk_pred, list_of_indices=[972, 51, 205, 227, 879, 538, 112, 741])
	# 
	# for i in range(10):
	#     visualizer.plot_cam_samples_alt(cams, img, tar, topk_prob, topk_pred, list_of_indices=list(np.random.choice(np.arange(1000),8)))
	
	
	c3,t3,prob3,pred3 = [],[],[],[]
	for ds in ['osmnist2rf_br_reduced','osmnist2rf_tl_reduced','osmnist2rf_c_reduced']:
		
		test_dataset = StereoImageFolder(
			root_dir=configuration_dict['input_dir'] + '/{}'.format(ds),
			train=False,
			stereo=configuration_dict['stereo'],
			transform=test_transform,
			nhot_targets=True
			)
		
		# delete subset generation from final evaluation
		test_subset = torch.utils.data.Subset(test_dataset, rep_sample)
		
		test_loader = torch.utils.data.DataLoader(test_subset,
		#test_loader = torch.utils.data.DataLoader(test_dataset,
			batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)
		
		cams, img, tar, out, topk_prob, topk_pred = generate_class_activation(test_loader, network, configuration_dict['time_depth'] + 1, configuration_dict['stereo'])
		c3.append(cams)
		t3.append(tar)
		prob3.append(topk_prob)
		pred3.append(topk_pred)
	
	gini_coefficients = visualizer.plot_cam_means2(c3, t3, prob3, pred3, filename='{}/fig8b_cam_means.pdf'.format(configuration_dict['visualization_dir']))
	
	# ANOVA of gini-coefficients
	
	dataframe = pd.DataFrame({'id':np.arange(gini_coefficients.shape[0]), 't0':gini_coefficients[:,0], 't1':gini_coefficients[:,1], 't2':gini_coefficients[:,2], 't3':gini_coefficients[:,3]})
	
	melted_df = pd.melt(dataframe, id_vars=['id'])
	melted_df.rename(columns = {'variable':'timesteps'}, inplace=True)
	melted_df['condition'] = np.repeat('background', melted_df.shape[0])
	
	
	# # normality assumption
	# for i in range(gini_coefficients.shape[1]):
	# 	stat, p = shapiro(gini_coefficients[:,i])
	# 	print('Wilk-Shapiro: t{}: statistics={}, p={}'.format(i, stat, p))
	# 	# interpret
	# 	alpha = 0.05
	# 	if p > alpha:
	# 		print('Sample looks Gaussian (fail to reject H0)')
	# 	else:
	# 		print('Sample does not look Gaussian (reject H0)')
	# 
	# # homogeneity of variance assumption
	# stat, p = levene(gini_coefficients[:,0], gini_coefficients[:,1], gini_coefficients[:,2],gini_coefficients[:,3], center='median')
	# print('Levene: statistics={}, p={}'.format(stat, p))
	# stat, p = levene(gini_coefficients[:,0], gini_coefficients[:,1], gini_coefficients[:,2],gini_coefficients[:,3], center='mean')
	# print('Levene: statistics={}, p={}'.format(stat, p))
# 
	# if p > alpha:
	# 	print('Groups have equal variance (fail to reject H0)')
	# else:
	# 	print('Groups do not have equal variance (reject H0)')
	# # standard ANOVA
	# print('[INFO] Standard one-way ANOVA')
	# melted_df.boxplot('value', by='timesteps')
	# plt.show()
	# 
	# linear_model= ols('value ~ timesteps', data=melted_df).fit()
	# #print(linear_model.summary())
	# aov_table = sm.stats.anova_lm(linear_model, typ=2)
	# eta_squared(aov_table)
	# omega_squared(aov_table)
	# print(aov_table.round(4))
	# 
	# from statsmodels.stats.oneway import anova_oneway
	# 
	# print('[INFO] Repeated Measures one-way ANOVA')
	# aovrm = AnovaRM(melted_df, 'value', 'id', within=['timesteps'])
	# res = aovrm.fit()
	# print(res)
	# 
	# (F(3,2997)=890.7, p=0.000)
	
	# posthoc
	# tukey HSD or multiple t-tests with bonferroni-correction
	
	pass
	



def fig_softmax_and_tsne(network, test_transform, configuration_dict, sample_size, random_seed):
	np.random.seed(random_seed)
	
	
	test_dataset = StereoImageFolder(
		root_dir=configuration_dict['input_dir'] + '/{}'.format(configuration_dict['dataset']),
		train=False,
		stereo=configuration_dict['stereo'],
		transform=test_transform,
		nhot_targets=True
		)

	rep_sample = list(np.random.choice(range(len(test_dataset)), size=sample_size, replace=False)) 
	test_subset = torch.utils.data.Subset(test_dataset, rep_sample)
	test_loader = torch.utils.data.DataLoader(test_subset, batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)

	feat, img, tar, out = generate_hidden_representation(test_loader, network, configuration_dict['time_depth'] + 1, configuration_dict['stereo'])
	
	
	# prepare tsne analysis
	# -----
	
	test_dataset_unoccluded = StereoImageFolder(
		root_dir=configuration_dict['input_dir'] + '/{}'.format('osmnist2_0occ'),
		train=False,
		stereo=configuration_dict['stereo'],
		transform=test_transform
		)
	
	test_subset_unoccluded = torch.utils.data.Subset(test_dataset_unoccluded, rep_sample)
	test_loader_unoccluded = torch.utils.data.DataLoader(test_subset_unoccluded, batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)
	
	featu, imgu, taru, _ = generate_hidden_representation(test_loader_unoccluded, network, configuration_dict['time_depth'] + 1, configuration_dict['stereo'])
	

	# hand the data to the visualization functions
	# -----
	highlights=[[1629,226],[516,909]] #[[1672,812,1629,226],[516,909]]
	visualizer.plot_tsne_evolution2(
	    torch.cat([feat,featu], dim=0),
	    torch.cat([img,imgu], dim=0),
	    torch.cat([tar[:,0],taru], dim=0),
	    show_indices=False, N=highlights,
	    overwrite=False, filename='{}/fig6_tsne.pdf'.format(configuration_dict['visualization_dir']))
	
	visualizer.plot_softmax_output(out, tar[:,0], img, filename='{}/fig5_softmax'.format(configuration_dict['visualization_dir']))
	visualizer.plot_relative_distances(feat, tar, featu, taru, filename='{}/fig7_distance'.format(configuration_dict['visualization_dir']))



def first_layer_network_filters(network, test_transform, configuration_dict, sample_size, random_seed):
	np.random.seed(random_seed)
	
	
	test_dataset = StereoImageFolder(
		root_dir=configuration_dict['input_dir'] + '/{}'.format(configuration_dict['dataset']),
		train=False,
		stereo=configuration_dict['stereo'],
		transform=test_transform,
		nhot_targets=True
		)

	rep_sample = list(np.random.choice(range(len(test_dataset)), size=sample_size, replace=False))
	
	test_subset = torch.utils.data.Subset(test_dataset, rep_sample)
	test_loader = torch.utils.data.DataLoader(test_subset, batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)


	fl_weights = network.rcnn.cell_list[0].bottomup.weight.detach()
	
	# im_left = Image.open('/Users/markus/Desktop/JOVstuff/grating.png')
	# im_left = ImageOps.grayscale(im_left)
	# im_left = torch.from_numpy(np.array(im_left, dtype=np.float32))
	# 
	# im_right = Image.open('/Users/markus/Desktop/JOVstuff/grating_shift1.png')
	# im_right = ImageOps.grayscale(im_right)
	# im_right = torch.from_numpy(np.array(im_right, dtype=np.float32))
	# 
	# im_tensor = torch.stack([im_left, im_right], dim=0)
	# im_tensor = torch.unsqueeze(im_tensor, 0)

	# you need a batch of useful data to make sure the bn-values are useful
	# im_tensor = im_tensor.repeat(1, configuration_dict['time_depth'] + 1, 1, 1, 1)
	
	#combine it with the first X samples of the testloader
	testbatch, _ = next(iter(test_loader))
	if configuration_dict['stereo']:
		testbatch = torch.cat(testbatch, dim=1)
	testbatch = testbatch.unsqueeze(1)
	testbatch = testbatch.repeat(1, configuration_dict['time_depth'] + 1, 1, 1, 1)
	# testbatch[0] = im_tensor
	

	activations = network.rcnn.return_activations(testbatch)
	imgno = 0
	for i in range(0,1):
		for ti in range(configuration_dict['time_depth'] + 1):
			show(torchvision.utils.make_grid(activations[ti][0][i].reshape(32,1,32,32), padding=2, normalize=False, scale_each=False))
			#plt.savefig('/Users/markus/Desktop/activations/{:03d}.png'.format(imgno))
			imgno += 1
			#plt.close()
			plt.show()
		
		# for ti in range(configuration_dict['time_depth'] + 1):
		#     show(torchvision.utils.make_grid(activations[ti][1][i].reshape(32,1,16,16), padding=2, normalize=False, scale_each=False))
		#     plt.show()

	if configuration_dict['stereo']:
		show(torchvision.utils.make_grid(fl_weights.reshape(32,1,6,3), padding=2, normalize=True, scale_each=True))
	else:
		show(torchvision.utils.make_grid(fl_weights.reshape(32,1,3,3), padding=2, normalize=True, scale_each=True))
	plt.show()




def fig_concentration(network, test_transform, configuration_dict, sample_size, random_seed):
	# set random seed
	np.random.seed(random_seed)
	# get concentration of sensititivy (COS) dataset
	cos_dataset = StereoImageFolder(
		root_dir=configuration_dict['input_dir'] + '/concentration_of_sensitivity/osmnist2r/',
		train=False,
		stereo=configuration_dict['stereo'],
		transform=test_transform,
		nhot_targets=True
		)
	
	rep_sample = list(np.random.choice(range(len(cos_dataset)), size=sample_size, replace=False))

	cos_subset = torch.utils.data.Subset(cos_dataset, rep_sample)
	cos_loader = torch.utils.data.DataLoader(cos_subset,
			batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)

	# get rgb information dataset
	# rgb dataset needs a transform without grayscale
	rgb_dataset = StereoImageFolder(
		root_dir=configuration_dict['input_dir'] + '/concentration_of_sensitivity/osmnist2r_rgb/',
		train=False,
		stereo=configuration_dict['stereo'],
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.,), (1.,))]),
		nhot_targets=True
		)
	rgb_subset = torch.utils.data.Subset(rgb_dataset, rep_sample)
	rgb_loader = torch.utils.data.DataLoader(rgb_subset,
			batch_size=configuration_dict['batchsize'], shuffle=False, num_workers=4)
	
	
	
	tp, tpp, op, opp, ovlp, ovlpp, bp, bpp = compare_concentration_mass(rgb_loader, cos_loader, network, configuration_dict['time_depth'] + 1, configuration_dict['stereo'])
	
	
	visualizer.plot_concentration_mass(tp, op, ovlp, bp, filename='{}/fig8c_percentage.pdf'.format(configuration_dict['visualization_dir']))
	visualizer.plot_concentration_mass(tpp, opp, ovlpp, bpp, filename='{}/fig8c_pixelpercentage.pdf'.format(configuration_dict['visualization_dir']))

	pass

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
