import numpy as np
import torch
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn import preprocessing, decomposition, naive_bayes, linear_model, metrics

from dataset_handler import StereoImageFolder

if __name__ == "__main__":
	
	batch_size = 100
	dataset_list = ['osmnist2c','osmnist2r','osfmnist2c','osfmnist2r','osycb']
	for ds in dataset_list:
		for stereo_boolean in [False, True]:
			train_set = StereoImageFolder(
				#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
				#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
				root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
				train=True,
				stereo=stereo_boolean,
				transform=transforms.Compose([
					transforms.Grayscale(),
					transforms.ToTensor(),
				])
				)
			
			train_loader = DataLoader(
			dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)
			
			test_set = StereoImageFolder(
			#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
			#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=False,
			stereo=stereo_boolean,
			transform=transforms.Compose([
				transforms.Grayscale(),
				transforms.ToTensor(),
			])
			)
			
			
			test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
			
			# accumulate the dataset into memory
			trainset_data = []
			testset_data = []
			trainset_targets = []
			testset_targets = []
			for data in train_loader:
				if train_set.stereo:
					trainset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
				else:
					trainset_data.append(data[0])
				trainset_targets.append(data[1])
				print('trainset iteration', len(trainset_data))
			
			for data in test_loader:
				if test_set.stereo:
					testset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
				else:
					testset_data.append(data[0])
				testset_targets.append(data[1])
				print('testset iteration', len(testset_data))

			
			trainset_data = torch.cat(trainset_data,0)
			testset_data = torch.cat(testset_data,0)
			trainset_targets = torch.cat(trainset_targets,0)
			testset_targets = torch.cat(testset_targets,0)
			
			print('training set shape:', trainset_data.shape)
			no_of_training_samples = trainset_data.shape[0]
			no_of_testing_samples = testset.shape[0]

			logit = linear_model.LogisticRegression(max_iter=10000)		
			logit.fit(trainset_data.reshape(no_of_training_samples,-1), trainset_targets)
			logreg_acc = metrics.accuracy_score(testset_targets, logit.predict(testset_data.reshape(no_of_testing_samples,-1)))
			
			logit = linear_model.SGDClassifier(max_iter=10000)
			logit.fit(trainset_data.reshape(no_of_training_samples,-1), trainset_targets)
			sgd_acc = metrics.accuracy_score(testset_targets, logit.predict(testset_data.reshape(no_of_testing_samples,-1)))
			print('*****')
			print('Dataset: {}, stereo:{}'.format(ds, stereo_boolean))
			print('Accuracy (Logistic Regression): %.4f'%logreg_acc)
			print('Accuracy (SGD): %.4f'%sgd_acc)
			print('*****')
