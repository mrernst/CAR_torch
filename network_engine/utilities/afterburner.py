#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# November 2020                                   _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# afterburner.py                             oN88888UU[[[/;::-.        dP^
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
import pandas as pd

import sys
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as mplstyle

from matplotlib.ticker import MaxNLocator
from matplotlib import collections
from matplotlib.patches import Rectangle
from scipy import stats
from collections import namedtuple
from cycler import cycler

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_multiplexer

# custom functions
# -----

def read_single_summary(path_to_tfevent, chist=0, img=0, audio=0, scalars=0,
						hist=0):
	ea = event_accumulator.EventAccumulator(path_to_tfevent, size_guidance={
			event_accumulator.COMPRESSED_HISTOGRAMS: chist,
			event_accumulator.IMAGES: img,
			event_accumulator.AUDIO: audio,
			event_accumulator.SCALARS: scalars,
			event_accumulator.HISTOGRAMS: hist,
											})
	ea.Reload()
	ea.Tags()
	return ea


def read_multiple_runs(path_to_project, chist=0, img=0, audio=0, scalars=0,
					   hist=0):
	# use with event_multiplexer (multiplexes different events together
	# useful for retraining I guess...)
	em = event_multiplexer.EventMultiplexer(size_guidance={
		event_accumulator.COMPRESSED_HISTOGRAMS: chist,
		event_accumulator.IMAGES: img,
		event_accumulator.AUDIO: audio,
		event_accumulator.SCALARS: scalars,
		event_accumulator.HISTOGRAMS: hist,
	})
	em.AddRunsFromDirectory(path_to_project)
	# load data
	em.Reload()
	return em


def convert_em_to_df(multiplexer):
	# this needs to be better and be able to cope with different scales
	# sort into training and testing/network
	df_dict = {}
	
	if len(multiplexer.Runs()) == 1:
		# figure out separate runs progressively
		entries = {}
		for run in multiplexer.Runs().keys():
			for tag in multiplexer.Runs()[run]["scalars"]:
				if tag.split('/')[0] not in entries.keys():
					entries[tag.split('/')[0]] = []
				entries[tag.split('/')[0]].append(tag)
		
		for run in entries:
			run_df = pd.DataFrame()
			for tag in entries[run]:
				tag_df = pd.DataFrame(multiplexer.Scalars(list(multiplexer.Runs().keys())[0], tag))
				tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
				run_df[tag] = tag_df.value
				run_df["step"] = tag_df.step
			df_dict[run] = run_df

	else:
		for run in multiplexer.Runs().keys():
			# create fresh empty dataframe
			run_df = pd.DataFrame()
			for tag in multiplexer.Runs()[run]["scalars"]:
				tag_df = pd.DataFrame(multiplexer.Scalars(run, tag))
				tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
				run_df[tag] = tag_df.value
				run_df["step"] = tag_df.step
			df_dict[run] = run_df

	return df_dict


# ----------------
# Afterburner Classes
# ----------------


class DataEssence(object):
	"""docstring for DataEssence."""

	def __init__(self):
		super(DataEssence, self).__init__()

	def write_to_file(self, filename):
		with open(filename, 'wb') as output:  # Overwrites any existing file.
			pickle.dump(self.essence, output, pickle.HIGHEST_PROTOCOL)

	def read_from_file(self, filename):
		with open(filename, 'rb') as input:
			self.essence = pickle.load(input)

	def distill(self, path, evaluation_data, embedding_data=None):
		self.essence = self._read_tfevents(path)

		# needless_keys = [l for l in list(self.essence.keys()) if '/' in l]
		# for key in needless_keys:
		# 	self.essence.pop(key, None)

		self.essence['evaluation'] = evaluation_data
		if embedding_data:
			self.essence['embedding'] = embedding_data

	def _read_tfevents(self, path):
		em = read_multiple_runs(path)
		df = convert_em_to_df(em)
		return df

	def plot_essentials(self, savefile):
		# start figure
		fig, axes = plt.subplots(4, 3, sharex='all', figsize=[7, 9])
		# plot images onto figure
		self._plot_traintest_lcurve(axes)
		# self._plot_timebased_lcurve(axes) # TODO: not functional for TITAN
		self._plot_parameter_lcurve(fig)

		# save figure
		fig.suptitle(savefile.rsplit('/')[-1])
		fig.savefig(savefile)
		pass

	def _plot_traintest_lcurve(self, axes):
		training_acc = \
			self.essence['training']['training/accuracy'].tolist()
		training_partacc = \
			self.essence['training']['training/accuracy']\
			.tolist()
		training_loss = \
			self.essence['training']['training/loss'].tolist()
		training_steps = \
			self.essence['training']['step'].tolist()

		testing_acc = \
			self.essence['testing']['testing/accuracy'].tolist()
		testing_partacc = \
			self.essence['testing']['testing/accuracy'].tolist()
		testing_loss = \
			self.essence['testing']['testing/loss'].tolist()
		testing_steps = \
			self.essence['testing']['step'].tolist()

		axes[0, 0].plot(training_steps, training_acc)
		axes[0, 0].plot(testing_steps, testing_acc)
		axes[0, 0].set_title('accuracy')

		axes[0, 2].plot(training_steps, training_partacc)
		axes[0, 2].plot(testing_steps, testing_partacc)
		axes[0, 2].set_title('partial accuracy')

		axes[0, 1].plot(training_steps, training_loss)
		axes[0, 1].plot(testing_steps, testing_loss)
		axes[0, 1].set_title('loss')

		pass

	def _plot_timebased_lcurve(self, axes):
		# TODO: not yet functional, but we also don't have tensorboard logs for other timesteps
		sorted_list_of_keys = self.essence['testing'].keys().tolist()
		sorted_list_of_keys.sort()
		timedepth = int(sorted_list_of_keys[-1].rsplit('_')[-1])

		sorted_list_of_keys = \
			[key for key in sorted_list_of_keys if 'testtime_beyond' in key]
		sorted_list_of_keys = \
			[key.rsplit('_', 1)[0] for key in sorted_list_of_keys]
		sorted_list_of_keys = list(set(sorted_list_of_keys))
		sorted_list_of_keys.sort()
		for i in range(len(sorted_list_of_keys)):
			for j in range(timedepth + 1):
				axes[1, i].plot(self.essence['testing']['step'].tolist(),
								self.essence['testing']['{}_{}'.format(
									sorted_list_of_keys[i], j)].tolist(),
								label=sorted_list_of_keys[i].split('/')[-1])
			axes[1, i].set_title(sorted_list_of_keys[i].split('/')[-1])
		pass

	def _plot_parameter_lcurve(self, fig):
		sorted_list_of_keys = self.essence['network'].keys().tolist()
		sorted_list_of_keys.sort()
		sorted_list_of_keys = \
			[key for key in sorted_list_of_keys if 'network' in key]
		sorted_list_of_keys = \
			[key.rsplit('/', 1)[0].split('/', 1)[1] for key in sorted_list_of_keys]
		sorted_list_of_keys = list(set(sorted_list_of_keys))
		sorted_list_of_keys = \
			[key for key in sorted_list_of_keys if '_weights' in key]
		sorted_list_of_keys.sort()
		# for i in range(len(sorted_list_of_keys)):
		for i, ax in enumerate(fig.axes[6:]):
			if i < len(sorted_list_of_keys):
				ax.plot(
					self.essence['network']['step'].tolist(),
					self.essence['network']['network/{}/mean'.format(
						sorted_list_of_keys[i])].tolist(),
					label=sorted_list_of_keys[i].split('/')[1],
					color='tab:blue')
				ax.plot(
					self.essence['network']['step'].tolist(),
					self.essence['network']['network/{}/median'.format(
						sorted_list_of_keys[i])].tolist(),
					label=sorted_list_of_keys[i].split('/')[1],
					color='tab:blue',
					linestyle='dashed')
				ax.plot(
					self.essence['network']['step'].tolist(),
					self.essence['network']['network/{}/min'.format(
						sorted_list_of_keys[i])].tolist(),
					label=sorted_list_of_keys[i].split('/')[1],
					color='tab:blue')
				ax.plot(
					self.essence['network']['step'].tolist(),
					self.essence['network']['network/{}/max'.format(
						sorted_list_of_keys[i])].tolist(),
					label=sorted_list_of_keys[i].split('/')[1],
					color='tab:blue')
				ax.fill_between(
					self.essence['network']['step'],
					self.essence['network']['network/{}/mean'.format(
						sorted_list_of_keys[i])] +
					self.essence['network']['network/{}/std'.format(
						sorted_list_of_keys[i])],
					self.essence['network']['network/{}/mean'.format(
						sorted_list_of_keys[i])] -
					self.essence['network']['network/{}/std'.format(
						sorted_list_of_keys[i])],
					color='tab:blue',
					alpha=0.3
				)
				ax.set_title(
					sorted_list_of_keys[i])
		pass

	# TODO: write these visualizer functions
	def visualize_softmax_output(self):
		pass

	def vizualize_tsne_analysis(self):
		pass


class EssenceCollection(object):
	"""docstring for EssenceCollection."""

	def __init__(self, remove_files=False):
		super(EssenceCollection, self).__init__()
		path_to_file = os.path.realpath(__file__)
		self.path_to_experiment = path_to_file.rsplit('/', 3)[0]
		self.collection = self.collect_data_essences(
			self.path_to_experiment, remove_files)

	def collect_data_essences(self, path_to_experiment, remove_files):
		# gather and read all files in files/essence/
		collection = {}
		essence = DataEssence()
		path_to_data = path_to_experiment + "/data/"
		for file in os.listdir(path_to_data):
			if file.endswith(".pkl") and file.startswith("conf"):
				config_name = file.split('.')[0].rsplit('i', 1)[0]
				iteration_number = file.split('.')[0].rsplit('i', 1)[-1]
				essence.read_from_file(os.path.join(path_to_data, file))
				if config_name in collection.keys():
					collection[config_name][iteration_number] = essence.essence
				else:
					collection[config_name] = {}
					collection[config_name][iteration_number] = essence.essence
				# delete file
				if remove_files:
					os.remove(os.path.join(path_to_data, file))
					print("[INFO] File '{}' deleted.".format(file))
				else:
					pass
		return collection

	def write_to_file(self, filename):
		with open(filename, 'wb') as output:  # Overwrites any existing file.
			pickle.dump(self.collection, output, pickle.HIGHEST_PROTOCOL)

	def read_from_file(self, filename):
		with open(filename, 'rb') as input:
			self.collection = pickle.load(input)

	def _plot_barplot_comparison(self, filename):

		configurations_to_plot = "all"

		# mplstyle.use('default')

		rects = []
		n_groups = len(self.collection.keys())
		n_iterations = 100  # high number so that true value is smaller
		for configuration in self.collection.keys():
			n_iterations = min(n_iterations,
							   len(self.collection[configuration].keys()))

		n_bars = 1
		data = np.zeros([n_iterations, n_bars, n_groups])
		means = np.zeros([n_bars, n_groups])
		stderror = np.zeros([n_bars, n_groups])
		mcnemar_data = {}
		ev_it = 1

		list_of_configurations = list(self.collection.keys())
		list_of_configurations.sort()

		for j, configuration in enumerate(list_of_configurations):
			list_of_iterations = list(
				self.collection['config{}'.format(j)].keys())
			list_of_iterations.sort()

			for it, iteration in enumerate(list_of_iterations):
				data[it, 0, j] = 1 - \
					self.collection[configuration][iteration]['testing']['testtime/partial_accuracy'].values[-1]
				mcnemar_data[it, 0, j] = \
					self.collection[configuration][iteration]['evaluation']['boolean_classification']

		chancelevel = 1 - 1. / 10.

		means = np.mean(data, axis=0)
		stderror = np.std(data, axis=0) / np.sqrt(n_iterations)

		# printout
		# -----
		print((means).round(3))
		print((stderror).round(3))

		fig, ax = plt.subplots(figsize=[8.0 / 6 * n_groups, 6.0])

		index = np.arange(n_groups)
		bar_width = 0.8

		opacity = 1
		error_config = {'ecolor': '0.3'}

		current_palette = sns.dark_palette(
			sns.color_palette('colorblind')[0], 5, reverse=True)
		current_palette = sns.dark_palette(
			sns.color_palette('Set3')[0], 5, reverse=True)
		current_palette = sns.dark_palette(
			'LightGray', 5, reverse=True)
		current_palette = sns.dark_palette(
			'Black', 5, reverse=False)

		for k in range(n_bars):
			rects.append(ax.bar(index + k * bar_width, means[k], bar_width,
								alpha=opacity,
								yerr=stderror[k], error_kw=error_config,
								label='{}'.format('configuration'),
								linewidth=0, color=current_palette[k]))
			# , edgecolor=current_palette[(k+1)],
			# hatch=hatches[k]))

		ax.set_xlabel('Network configuration')
		ax.set_ylabel('Error rate')
		ax.set_title('experiment: {}'.format(
			self.path_to_experiment.rsplit('/')[-1]), fontsize=10)
		ax.set_ylim([0, ax.get_ylim()[1] * 1])  # *1.25])
		ax.set_xticks(index + bar_width * (n_bars / 2. - 0.5))
		# ax.set_xticklabels([arch[:-1] for arch in ARCHARRAY])
		ax.legend(frameon=True, facecolor='white',
				  edgecolor='white', framealpha=1.0)
		ax.grid(axis='y', zorder=0, alpha=0.5)
		ax.set_axisbelow(True)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=True)  # labels along the bottom edge are off
		ax.spines['left'].set_visible(False)

		ax.axhline(y=chancelevel, xmin=0, xmax=6, color='black')
		ax.text(0.2, chancelevel + 0.005, r'chancelevel')

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		# Put a legend to the right of the current axis
		ax.legend(frameon=True, facecolor='white', edgecolor='white',
				  framealpha=1.0, bbox_to_anchor=(1, .9), loc='center left')

		# mcnemar's table
		for i in range(n_bars):
			qstar = 0.05
			pval = np.ones([n_groups * n_bars, n_groups * n_bars])
			chi_table = np.ones([n_groups * n_bars, n_groups * n_bars])
			significance_table = np.zeros(
				[n_groups * n_bars, n_groups * n_bars])
			vertexlist = []
			for k in range(n_groups * n_bars):
				for j in range(n_groups * n_bars):
					if k != j and k > j:
						mcnemar_table = 2 * \
							mcnemar_data[(ev_it - 1), i, k] - \
							mcnemar_data[(ev_it - 1), i, j]
						# i.e. nhot encoding
						if (len(mcnemar_data[(ev_it - 1), i, k].shape) > 1):
							ak = np.sum(
								np.array(
									[e for e in mcnemar_table == 1.]), axis=1)
							bk = np.sum(
								np.array(
									[e for e in mcnemar_table == 2.]), axis=1)
							ck = np.sum(
								np.array(
									[e for e in mcnemar_table == -1.]), axis=1)
							dk = np.sum(
								np.array(
									[e for e in mcnemar_table == 0]), axis=1)
							chi_sq_denom = np.sum(
								((bk - ck) / mcnemar_table.shape[-1]))**2
							chi_sq_count = np.sum(
								((bk - ck) / mcnemar_table.shape[-1])**2)
							chi_sq = chi_sq_denom / chi_sq_count
						else:
							a = mcnemar_table[mcnemar_table == 1.].shape[0]
							b = mcnemar_table[mcnemar_table == 2.].shape[0]
							c = mcnemar_table[mcnemar_table == -1.].shape[0]
							d = mcnemar_table[mcnemar_table == 0].shape[0]
							chi_sq = ((b - c)**2) / (b + c)
						# stats.chi2.pdf(chi_sq , 1) #
						pval[k, j] = 1 - stats.chi2.cdf(chi_sq, 1)
						chi_table[k, j] = chi_sq

			print(np.round(pval, 4))
			print(np.round(chi_table, 4))
			sorted_pvals = np.sort(pval[pval < 1])
			bjq = np.arange(1, len(sorted_pvals) + 1) / \
				len(sorted_pvals) * qstar

			for k in range(n_groups * n_bars):
				for j in range(n_groups * n_bars):
					if k != j and k > j:
						if pval[k, j] in sorted_pvals[sorted_pvals - bjq <= 0]:
							significance_table[k, j] = 1

							if j < 3 and k > 2 and (means[i][j] < means[i][k]):
								vertexlist.append(
									[[j + 0.25, k - 0.5],
									 [j + 0.5, k - 0.25],
									 [j + 0.5, k - 0.5]])
						else:
							significance_table[k, j] = 0

			ax = plt.axes([0.8, 0.3, .2, .2])
			ax.matshow(significance_table, cmap='Greys')
			# ax.set_xticklabels(, fontsize=8)#fontsize=65)
			ax.tick_params(labelbottom='on', labeltop='off',
						   top='off', right='off')
			# ax.set_yticklabels(, fontsize=8)#fontsize=65)
			ax.set_xlim([-0.5, n_groups * n_bars - 1 - 0.5])
			ax.set_ylim([n_groups * n_bars - 0.5, -0.5 + 1])
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)

			ax.axhline(y=2.5, xmin=-0.5, xmax=2.5, color='white', linewidth=1)
			ax.axvline(x=2.5, ymin=-0.5, ymax=2.5, color='white', linewidth=1)
			for vertices in vertexlist:
				pc = collections.PolyCollection(
					(vertices,), color='white', edgecolor="none")
				ax.add_collection(pc)

			ax.add_patch(Rectangle((-2.5, 6.9), 1, 1, fill='black',
								   color='black', alpha=1, clip_on=False))
			ax.text(-2.5, 9.4, 'Significant difference \n(two-sided McNemar \
				test, \nexpected FDR=0.05)',
					fontsize=8, horizontalalignment='left',
					verticalalignment='center')
			# plt.show()
			plt.savefig(filename)

			fig, ax = plt.subplots(figsize=(12, 12))
			ax.matshow(significance_table, cmap='Greys')
			# ax.set_xticklabels(
			#     [''] + [arch[:-1] for arch in ARCHARRAY], fontsize=65)
			ax.tick_params(labelbottom='on', labeltop='off',
						   top='off', right='off')
			# ax.set_yticklabels(
			#     [arch[:-1] for arch in ARCHARRAY], fontsize=65)
			ax.set_xlim([-0.5, n_groups * n_bars - 1 - 0.5])
			ax.set_ylim([n_groups * n_bars - 0.5, -0.5 + 1])
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.axhline(y=2.5, xmin=-0.5, xmax=2.5, color='white')
			ax.axvline(x=2.5, ymin=-0.5, ymax=2.5, color='white')
			for vertices in vertexlist:
				pc = collections.PolyCollection(
					(vertices,), color='white', edgecolor="none")
				ax.add_collection(pc)
			# ax.set_ylabel('Network Architecture')
		plt.tight_layout()
		plt.savefig(filename.rsplit('.', 1)[0] + '_sigtable.pdf')





if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config", type=str)
	
	args = parser.parse_args()
	
	print('[INFO] afterburner running, collecting data')
	ess_coll = EssenceCollection(remove_files=True)

	ess_coll._plot_barplot_comparison(
		ess_coll.path_to_experiment +
		'/visualization/{}_comparision.pdf'.format(
			ess_coll.path_to_experiment.rsplit('/')[-1]))
	ess_coll.write_to_file(
		ess_coll.path_to_experiment +
		'/data/{}.pkl'.format(
			ess_coll.path_to_experiment.rsplit('/')[-1]))
	

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
