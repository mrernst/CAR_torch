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

import numpy as np

import os
import sys
import six
import string

import lmdb
import pickle
import msgpack
import tqdm
import pyarrow as pa

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

from skimage import io, transform
from PIL import Image


# custom functions
# -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard Usecase
# -----
# For the standard usecase of having just one class you can use the built-in
# torchvision.datasets.ImageFolder dataset


class dynaMODataset(Dataset):
	"""Dynamic Occluded MNIST Dataset"""

	def __init__(self, root_dir, transform=None, target_transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.target_transform = target_transform
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
		
		if self.transform is not None:
			image_array = self.transform(image_array)

		if self.target_transform is not None:
			target = self.target_transform(target)

		sample = {'image': image_array, 'target': target}

		return sample

class ToSingle(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, target_sequence):
		return torch.tensor(target_sequence[-1], dtype=torch.int64, device=device)


class ToTimeSeries(object):
	"""Convert given data in sample to timeseries."""
	def __init__(self, height=32, width=32):
		self.height = height
		self.width = width
	def __call__(self, image):
		image_array = np.array(image)
		image_array = image_array.reshape(self.height, self.width, -1, order='F')
		return image_array

		
class StereoImageFolder(Dataset):
	"""Modified ImageFolder Structure to Import Stereoscopic Data"""

	def __init__(self, root_dir, train, stereo=False, loader=pil_loader, transform=None, target_transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir + '/train/left/' if train else root_dir + '/test/left/'
		self.transform = transform
		self.target_transform = target_transform
		self.paths_to_left_samples = []
		self.paths_to_right_samples = []
		self.height = 32
		self.width = 32
		self.loader = loader
		self.stereo = stereo
		
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
				self.paths_to_left_samples.append(
					os.path.join(self.root_dir, cla, name))
					
		
		for item in self.paths_to_left_samples:
			self.paths_to_right_samples.append(item.split('left')[0] + 'right' + item.split('left')[1])

	def __len__(self):
		return len(self.paths_to_left_samples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = self.paths_to_left_samples[idx]
		image = self.loader(img_name)
		
		
		target = []
		for t in self.paths_to_left_samples[idx].rsplit('_', 1)[-1].rsplit('/')[0]:
			target.append(int(t))
		target = np.array(target, dtype=np.uint8)
		
		if self.target_transform is not None:
			target = self.target_transform(target)


		if self.stereo:
			image_l = image
			image_r = self.loader(self.paths_to_right_samples[idx])
			
			if self.transform is not None:
				image_l = self.transform(image_l)
				image_r = self.transform(image_r)
			
			sample = [(image_l, image_r), target]


		else:
			if self.transform is not None:
				image = self.transform(image)
			
			sample = [image, target]

		return sample




class ImageFolderLMDB(Dataset):
	def __init__(self, db_path, transform=None, target_transform=None):
		self.db_path = db_path
		self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
							 readonly=True, lock=False,
							 readahead=False, meminit=False)
		with self.env.begin(write=False) as txn:
			# self.length = txn.stat()['entries'] - 1
			self.length =pa.deserialize(txn.get(b'__len__'))
			self.keys= pa.deserialize(txn.get(b'__keys__'))

		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		img, target = None, None
		env = self.env
		with env.begin(write=False) as txn:
			byteflow = txn.get(self.keys[index])
		unpacked = pa.deserialize(byteflow)

		# load image
		imgbuf = unpacked[0]
		buf = six.BytesIO()
		buf.write(imgbuf)
		buf.seek(0)
		img = Image.open(buf).convert('RGB')

		# load label
		target = unpacked[1]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target


	def __len__(self):
		return self.length

	def __repr__(self):
		return self.__class__.__name__ + ' (' + self.db_path + ')'


class StereoImageFolderLMDB(Dataset):
	def __init__(self, db_path, stereo=True, transform=None, target_transform=None):
		self.db_path = db_path
		self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
							 readonly=True, lock=False,
							 readahead=False, meminit=False)
		with self.env.begin(write=False) as txn:
			# self.length = txn.stat()['entries'] - 1
			self.length =pa.deserialize(txn.get(b'__len__'))
			self.keys= pa.deserialize(txn.get(b'__keys__'))
	
		self.stereo = stereo
		self.transform = transform
		self.target_transform = target_transform
	
	def __getitem__(self, index):
		img_l, img_r, target = None, None, None
		
		env = self.env
		with env.begin(write=False) as txn:
			byteflow = txn.get(self.keys[index])
		unpacked = pa.deserialize(byteflow)

		# load image
		imgbuf = unpacked[0][0]
		buf = six.BytesIO()
		buf.write(imgbuf)
		buf.seek(0)
		img_l = Image.open(buf).convert('RGB')
		
		# load label
		target = unpacked[1]
		
		if self.target_transform is not None:
			target = self.target_transform(target)
			

		if self.stereo:
			imgbuf = unpacked[0][1]
			buf = six.BytesIO()
			buf.write(imgbuf)
			buf.seek(0)
			img_r = Image.open(buf).convert('RGB')

			if self.transform is not None:
				img_l = self.transform(img_l)
				img_r = self.transform(img_r)
			
			
			return (img_l, img_r), target
		else:
			if self.transform is not None:
				img_l = self.transform(img_l)
			
			return img_l, target


def raw_reader(path):
	with open(path, 'rb') as f:
		bin_data = f.read()
	return bin_data

def pil_loader(path):
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')

def dumps_pyarrow(obj):
	"""
	Serialize an object.

	Returns:
		Implementation-dependent bytes-like object
	"""
	return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, name="train", write_frequency=5000, num_workers=16):
	directory = os.path.expanduser(os.path.join(dpath, name))
	print("Loading dataset from %s" % directory)
	dataset = ImageFolder(directory, loader=raw_reader)
	data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

	lmdb_path = os.path.join(dpath, "%s.lmdb" % name)
	isdir = os.path.isdir(lmdb_path)

	print("Generate LMDB to %s" % lmdb_path)
	db = lmdb.open(lmdb_path, subdir=isdir,
				   map_size=1099511627776,
				   readonly=False,
				   meminit=False, map_async=True)
	
	print(len(dataset), len(data_loader))
	txn = db.begin(write=True)
	for idx, data in enumerate(data_loader):
		# print(type(data), data)
		image, label = data[0]
		txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
		if idx % write_frequency == 0:
			print("[%d/%d]" % (idx, len(data_loader)))
			txn.commit()
			txn = db.begin(write=True)

	# finish iterating through dataset
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
	with db.begin(write=True) as txn:
		txn.put(b'__keys__', dumps_pyarrow(keys))
		txn.put(b'__len__', dumps_pyarrow(len(keys)))

	print("Flushing database ...")
	db.sync()
	db.close()


def stereofolder2lmdb(dpath, name, write_frequency=5000, num_workers=16):
	directory = os.path.expanduser(dpath)
	print("Loading dataset from %s" % directory)
	
	for train_bool in [True, False]:
		finalname = name + '_train' if train_bool else name + '_test'
		dataset = StereoImageFolder(directory, stereo=True, train=train_bool, loader=raw_reader)
		data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)
	
		lmdb_path = os.path.join(dpath, "%s.lmdb" % finalname)
		isdir = os.path.isdir(lmdb_path)
	
		print("Generate LMDB to %s" % lmdb_path)
		db = lmdb.open(lmdb_path, subdir=isdir,
					   map_size=1099511627776,
					   readonly=False,
					   meminit=False, map_async=True)
		
		print(len(dataset), len(data_loader))
		txn = db.begin(write=True)
		for idx, data in enumerate(data_loader):
			# print(type(data), data)
			(image_l, image_r), label = data[0]
			txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(((image_l, image_r), label)))
			if idx % write_frequency == 0:
				print("[%d/%d]" % (idx, len(data_loader)))
				txn.commit()
				txn = db.begin(write=True)
	
		# finish iterating through dataset
		txn.commit()
		keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
		with db.begin(write=True) as txn:
			txn.put(b'__keys__', dumps_pyarrow(keys))
			txn.put(b'__len__', dumps_pyarrow(len(keys)))
	
		print("Flushing database ...")
		db.sync()
		db.close()



class RandomData(Dataset):
	
	def __init__(self, length=45, timesteps=4, constant_over_time=False, transform=None):
		self.length = length
		self.timesteps = timesteps
		self.transform = transform
		if constant_over_time:
			input_tensor = torch.randint(255,[length, 1, 32, 32], dtype=torch.float)
			input_tensor = input_tensor.unsqueeze(1)
			self.data = input_tensor.repeat(1, timesteps, 1, 1, 1)
		else:
			self.data = torch.randint(255,[length, timesteps, 1, 32, 32], dtype=torch.float)
		self.labels = torch.randint(9, [length], dtype=torch.float)
		
	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.data[idx]
		label = self.labels[idx]
		
		if self.transform:
			image = self.transform(image)

		return image, label




if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--folder", type=str)
	parser.add_argument('-n', '--name', type=str, default="dataset")
	parser.add_argument('-p', '--procs', type=int, default=20)
	parser.add_argument( "-os", "--stereo", type=bool, default=False)
	
	
	args = parser.parse_args()
	
	if args.stereo:
		stereofolder2lmdb(args.folder, name=args.name)
	else:
		folder2lmdb(args.folder, num_workers=args.procs, name=args.name)

	
# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
