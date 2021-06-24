import numpy as np
import torch
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn import preprocessing, decomposition, naive_bayes, linear_model, metrics

from dataset_handler import StereoImageFolder

if __name__ == "__main__":
	#Grayscale needs to drop when osycb is loaded

	batch_size = 100
	dataset_list = ['osmnist2c','osmnist2r','osfmnist2c','osfmnist2r','osycb']
	for ds in dataset_list:
		if 'osycb' in ds:
			tfs = transforms.Compose([
				transforms.ToTensor(),
			])
		else:
			tfs = transforms.Compose([
				transforms.Grayscale(),
				transforms.ToTensor(),
			])
		train_set = StereoImageFolder(
			#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
			#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=True,
			stereo=True,
			transform=tfs
			)
		
		train_loader = DataLoader(
		dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
		
		test_set = StereoImageFolder(
			#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
			#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=False,
			stereo=True,
			transform=tfs
		)
		
		
		
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
		
		# accumulate the dataset into memory
		trainset_data = []
		testset_data = []
		trainset_targets = []
		testset_targets = []
		for n, data in enumerate(train_loader):
			if train_set.stereo:
				trainset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
			else:
				trainset_data.append(data[0])
			trainset_targets.append(data[1])
			print(80*' ', end='\r')
			print('trainset sample', n*batch_size, end='\r')

		for n, data in enumerate(test_loader):
			if test_set.stereo:
				testset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
			else:
				testset_data.append(data[0])
			testset_targets.append(data[1])
			print(80*' ', end='\r')
			print('testset sample', n*batch_size, end='\r')


		
		trainset_data = torch.cat(trainset_data,0)
		testset_data = torch.cat(testset_data,0)
		trainset_targets = torch.cat(trainset_targets,0)
		testset_targets = torch.cat(testset_targets,0)
		
		no_of_training_samples = trainset_data.shape[0]
		no_of_testing_samples = testset_data.shape[0]


		# stereo data
		print('training set shape:', trainset_data.shape)
		
		logit = linear_model.LogisticRegression(max_iter=10000)		
		logit.fit(trainset_data.reshape(no_of_training_samples,-1), trainset_targets)
		logreg_acc = metrics.accuracy_score(testset_targets, logit.predict(testset_data.reshape(no_of_testing_samples,-1)))
		
		logit = linear_model.SGDClassifier(max_iter=10000)
		logit.fit(trainset_data.reshape(no_of_training_samples,-1), trainset_targets)
		sgd_acc = metrics.accuracy_score(testset_targets, logit.predict(testset_data.reshape(no_of_testing_samples,-1)))
		
		print('*****')
		print('Dataset: {}, stereo'.format(ds))
		print('Accuracy (Logistic Regression): %.4f'%logreg_acc)
		print('Accuracy (SGD): %.4f'%sgd_acc)
		print('*****')
		
		# mono data
		trainset_data, testset_data = trainset_data[:,0:1,:,:], testset_data[:,0:1,:,:]
		print('training set shape:', trainset_data.shape)
		
		logit = linear_model.LogisticRegression(max_iter=10000)		
		logit.fit(trainset_data.reshape(no_of_training_samples,-1), trainset_targets)
		logreg_acc = metrics.accuracy_score(testset_targets, logit.predict(testset_data.reshape(no_of_testing_samples,-1)))
		
		logit = linear_model.SGDClassifier(max_iter=10000)
		logit.fit(trainset_data.reshape(no_of_training_samples,-1), trainset_targets)
		sgd_acc = metrics.accuracy_score(testset_targets, logit.predict(testset_data.reshape(no_of_testing_samples,-1)))
		
		print('*****')
		print('Dataset: {}, mono'.format(ds))
		print('Accuracy (Logistic Regression): %.4f'%logreg_acc)
		print('Accuracy (SGD): %.4f'%sgd_acc)
		print('*****')
