import numpy as np
import torch
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn import preprocessing, decomposition, naive_bayes, linear_model, metrics

from dataset_handler import StereoImageFolder

if __name__ == "__main__":
	#Grayscale needs to drop when osycb is loaded

	batch_size = 1000
	stereoboolean = True
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
			root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			#root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=True,
			stereo=stereoboolean,
			transform=tfs
			)
		
		#train_set = Subset(train_set, np.arange(0,10000))

		train_loader = DataLoader(
		dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
		
		
		# accumulate the dataset into memory
		trainset_data = []
		trainset_targets = []
		for n, data in enumerate(train_loader):
			if stereoboolean:
				trainset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
			else:
				trainset_data.append(data[0])
			trainset_targets.append(data[1])
			print(80*' ', end='\r')
			print('trainset sample', n*batch_size, end='\r')
		
		trainset_data = torch.cat(trainset_data,0)
		trainset_targets = torch.cat(trainset_targets,0)


		no_of_training_samples = trainset_data.shape[0]
		trainset_reshaped_mono = trainset_data[:,0:1,:,:].reshape(no_of_training_samples,-1)
		
		
		
		# Data fit
		# -----
		
		# stereo data fit
		trainset_data = trainset_data.reshape(no_of_training_samples,-1)
		print('training set shape:', trainset_data.shape)
	
	
		logit_logreg_stereo = linear_model.LogisticRegression(max_iter=10000)		
		logit_logreg_stereo.fit(trainset_data, trainset_targets)
		
		logit_sgd_stereo = linear_model.SGDClassifier(max_iter=10000)
		logit_sgd_stereo.fit(trainset_data, trainset_targets)
		
		
		# mono data fit
		trainset_data = trainset_reshaped_mono
		print('training set shape:', trainset_data.shape)


		logit_logreg_mono = linear_model.LogisticRegression(max_iter=10000)		
		logit_logreg_mono.fit(trainset_data, trainset_targets)
		
		logit_sgd_mono = linear_model.SGDClassifier(max_iter=10000)
		logit_sgd_mono.fit(trainset_data, trainset_targets)
		
		
		# release memory
		del trainset_data
		del trainset_targets
		
		# Accuracy
		# -----
		
		test_set = StereoImageFolder(
			#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
			root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			#root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=False,
			stereo=stereoboolean,
			transform=tfs
		)
		
		
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
		
		testset_data = []
		testset_targets = []
		for n, data in enumerate(test_loader):
			if stereoboolean:
				testset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
			else:
				testset_data.append(data[0])
			testset_targets.append(data[1])
			print(80*' ', end='\r')
			print('testset sample', n*batch_size, end='\r')

		testset_data = torch.cat(testset_data,0)
		testset_targets = torch.cat(testset_targets,0)
		

		
		# mono data extraction
		no_of_testing_samples = testset_data.shape[0]
		testset_reshaped_mono = testset_data[:,0:1,:,:].reshape(no_of_testing_samples,-1)
		
		testset_data = testset_data.reshape(no_of_testing_samples,-1)

		logreg_acc_stereo = metrics.accuracy_score(testset_targets, logit_logreg_stereo.predict(testset_data))
		
		sgd_acc_stereo = metrics.accuracy_score(testset_targets, logit_sgd_stereo.predict(testset_data ))
		
		print('*****')
		print('Dataset: {}, stereo'.format(ds))
		print('Accuracy (Logistic Regression): %.4f'%logreg_acc_stereo)
		print('Accuracy (SGD): %.4f'%sgd_acc_stereo)
		print('*****')
		
		
		# mono data fit
		testset_data = testset_reshaped_mono

		logreg_acc_mono = metrics.accuracy_score(testset_targets, logit_logreg_mono.predict(testset_data))
		
		sgd_acc_mono = metrics.accuracy_score(testset_targets, logit_sgd_mono.predict(testset_data ))

		
		print('*****')
		print('Dataset: {}, mono'.format(ds))
		print('Accuracy (Logistic Regression): %.4f'%logreg_acc_mono)
		print('Accuracy (SGD): %.4f'%sgd_acc_mono)
		print('*****')
