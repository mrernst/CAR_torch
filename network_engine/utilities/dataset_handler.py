#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020     	                              _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# dataset_handler.p                          oN88888UU[[[/;::-.        dP^
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


# standard libraries
# -----
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# custom functions
# -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard Usecase
# -----
# For the standard usecase of having just one class you can use the built-in
# torchvision.datasets.ImageFolder dataset


class dynaMODataset(Dataset):
	"""Dynamic Occluded MNIST Dataset"""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.paths_to_samples = []
		self.height = 32
		self.width = 32
		# move through the filestructure to get a list of all images
		uberclasses = os.listdir(self.root_dir)
		try:
			uberclasses.remove('.DS_Store')
		except(ValueError):
			pass
		for cla in uberclasses:
			class_folder = os.path.join(self.root_dir, cla)

			filenames = os.listdir(class_folder)
			try:
				filenames.remove('.DS_Store')
			except(ValueError):
				pass
			for name in filenames:
				self.paths_to_samples.append(
					os.path.join(self.root_dir, cla, name))


	def __len__(self):
		return len(self.paths_to_samples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = self.paths_to_samples[idx]
		image_array = io.imread(img_name)
		image_array = image_array.reshape(self.height, self.width, -1, order='F')

		target = []
		for t in self.paths_to_samples[idx].rsplit('_', 1)[-1].rsplit('.')[0]:
			target.append(int(t))
		target = np.array(target, dtype=np.uint8)
		sample = {'image': image_array, 'target': target}

		if self.transform:
			sample = self.transform(sample)

		return sample


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, target = sample['image'], sample['target']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1)) / (255./2) - 1 # crude normalization
		# return {'image': torch.from_numpy(image).type(torch.float32),
		#         'target': torch.from_numpy(target).type(torch.int64),
		#         'single': torch.from_numpy(target[-1:]).type(torch.int64)} #transforms.functional.normalize(tensor, 0, 1, inplace=False)
		return {'image': torch.tensor(image, dtype=torch.float32, device=device),
				'target': torch.tensor(target, dtype=torch.int64, device=device),
				'single': torch.tensor(target[-1], dtype=torch.int64, device=device)}

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment