import numpy as np
import torch
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn import preprocessing, decomposition, naive_bayes, linear_model, metrics

from dataset_handler import StereoImageFolder, StereoImageFolderLMDB

import progressbar

if __name__ == "__main__":
	#Grayscale needs to drop when osycb is loaded

	batch_size = 500
	epochs = 100
	dataset_list = [
		('osmnist2c', False),('osmnist2r', False),('osfmnist2r', False),('osfmnist2c', False),('osycb', False),
		('osmnist2c', True),('osmnist2r', True),('osfmnist2r', True),('osfmnist2c', True),('osycb', True)
		]
	
	def evaluate(loader):
		accuracies = []
		for n, data in enumerate(loader):
			if stereoboolean:
				test_data = torch.cat([data[0][0], data[0][1]], axis=1)
			else:
				test_data = data[0]
			test_targets = data[1]
			test_data = test_data.reshape(batch_size,-1)
			acc = metrics.accuracy_score(test_targets, logit_sgd.predict(test_data))
			#print('partial', acc)
			accuracies.append(acc)
		accuracies = np.array(accuracies).mean()
		return accuracies
	
	
	for ds, stereoboolean in dataset_list:
		if 'osycb' in ds:
			tfs = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.,0.,0.), (1.,1.,1.))
			])
		else:
			tfs = transforms.Compose([
				transforms.Grayscale(),
				transforms.ToTensor(),
				transforms.Normalize((0.,), (1.,))	
			])
		
		
		train_set = StereoImageFolder(
			#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
			#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=True,
			stereo=stereoboolean,
			transform=tfs
			)
		
		#train_set = Subset(train_set, np.arange(0,5000))
		
		train_loader = DataLoader(
		dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)
		
		
		test_set = StereoImageFolder(
			#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
			#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
			root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
			train=False,
			stereo=stereoboolean,
			transform=tfs
		)
		
		#test_set = Subset(test_set, np.arange(0,1000))
		
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
		
		
		logit_sgd = linear_model.SGDClassifier(max_iter=10000)
		final_acc = 0
		for e in progressbar.progressbar(range(epochs)):
			for n, data in progressbar.progressbar(enumerate(train_loader), max_value=len(train_set)//batch_size-1):
			#for n, data in enumerate(train_loader):
				if stereoboolean:
					train_data = torch.cat([data[0][0], data[0][1]], axis=1)
				else:
					train_data = data[0]
				train_targets = data[1]
				train_data = train_data.reshape(batch_size,-1)
				logit_sgd.partial_fit(train_data, train_targets, classes=np.arange(10))
			if (e+1)%5 == 0:
				acc = evaluate(test_loader)
				print('***********')
				print('Dataset: {}, stereo: {}'.format(ds, stereoboolean))
				print('Epoch: {} - accuracy (SGD): {}'.format(e, acc))
				print('***********')
				with open('results.txt', 'a') as f:
					f.write('***********\n')
					f.write('Dataset: {}, stereo: {}\n'.format(ds, stereoboolean))
					f.write('Epoch: {} - accuracy (SGD): {}\n'.format(e, acc))
				
				final_acc = max(final_acc, acc)
		
		# print('***********')
		# print('final accuracy (SGD): {}'.format(final_acc))
		# print('***********')
		with open('results.txt', 'a') as f:
			f.write('***********\n')
			f.write('{}, stereo: {}, accuracy (SGD): {}\n'.format(ds, stereoboolean, final_acc))
			f.write('***********\n')


		
		
		# 
		# #full fit
		# # -----		
		# train_set = StereoImageFolder(
		# 	#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
		# 	#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
		# 	#root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
		# 	# root_dir='/Users/markus/mountpoint/{}/'.format(ds),
		# 	root_dir = '/Volumes/Dragonfly/oscar/{}/'.format(ds),
		# 	train=True,
		# 	stereo=stereoboolean,
		# 	transform=tfs
		# 	)
	# 
		# #train_set = Subset(train_set, np.arange(0,6000))
# 
		# train_loader = DataLoader(
		# dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
		# 
		# 
		# 
		# # accumulate the dataset into memory
		# # this takes to much memory
		# 
		# trainset_data = []
		# trainset_targets = []
		# for n, data in enumerate(train_loader):
		# 	if stereoboolean:
		# 		trainset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
		# 	else:
		# 		trainset_data.append(data[0])
		# 	trainset_targets.append(data[1])
		# 	print(80*' ', end='\r')
		# 	print('trainset sample', n*batch_size, end='\r')
		# 
		# trainset_data = torch.cat(trainset_data,0).numpy()
		# trainset_targets = torch.cat(trainset_targets,0).numpy()
# 
# 
		# no_of_training_samples = trainset_data.shape[0]
		# trainset_reshaped_mono = trainset_data[:,0:1,:,:].reshape(no_of_training_samples,-1)
		# 
		# 
		# 
		# # Data fit
		# # -----
		# 
		# # stereo data fit
		# trainset_data = trainset_data.reshape(no_of_training_samples,-1)
		# print('training set shape:', trainset_data.shape)
	# 
	# 
		# logit_logreg_stereo = linear_model.LogisticRegression(max_iter=10000)		
		# logit_logreg_stereo.fit(trainset_data, trainset_targets)
		# 
		# logit_sgd_stereo = linear_model.SGDClassifier(max_iter=10000)
		# logit_sgd_stereo.fit(trainset_data, trainset_targets)
		# 
		# 
		# # mono data fit
		# trainset_data = trainset_reshaped_mono
		# print('training set shape:', trainset_data.shape)
# 
# 
		# logit_logreg_mono = linear_model.LogisticRegression(max_iter=10000)		
		# logit_logreg_mono.fit(trainset_data, trainset_targets)
		# 
		# logit_sgd_mono = linear_model.SGDClassifier(max_iter=10000)
		# logit_sgd_mono.fit(trainset_data, trainset_targets)
		# 
		# 
		# # release memory
		# del trainset_data
		# del trainset_targets
		# 
		# # Accuracy
		# # -----
		# 
# 
		# test_set = StereoImageFolder(
		# 	#root_dir='/Users/markus/Research/Code/titan/datasets/osmnist2_0occ/',
		# 	#root_dir='/Users/markus/Research/Code/titan/datasets/{}_reduced/'.format(ds),
		# 	#root_dir='/home/aecgroup/aecdata/Textures/occluded/datasets/{}/'.format(ds),
		# 	# root_dir='/Users/markus/mountpoint/{}/'.format(ds),
		# 	root_dir = '/Volumes/Dragonfly/oscar/{}/'.format(ds),
		# 	train=False,
		# 	stereo=stereoboolean,
		# 	transform=tfs
		# )
		# 
		# #test_set = Subset(test_set, np.arange(0,1000))
# 
		# 
		# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
		# 
		# testset_data = []
		# testset_targets = []
		# for n, data in enumerate(test_loader):
		# 	if stereoboolean:
		# 		testset_data.append(torch.cat([data[0][0], data[0][1]], axis=1))
		# 	else:
		# 		testset_data.append(data[0])
		# 	testset_targets.append(data[1])
		# 	print(80*' ', end='\r')
		# 	print('testset sample', n*batch_size, end='\r')
# 
		# testset_data = torch.cat(testset_data,0).numpy()
		# testset_targets = torch.cat(testset_targets,0).numpy()
		# 
# 
		# 
		# # mono data extraction
		# no_of_testing_samples = testset_data.shape[0]
		# testset_reshaped_mono = testset_data[:,0:1,:,:].reshape(no_of_testing_samples,-1)
		# 
		# testset_data = testset_data.reshape(no_of_testing_samples,-1)
# 
		# logreg_acc_stereo = metrics.accuracy_score(testset_targets, logit_logreg_stereo.predict(testset_data))
		# 
		# sgd_acc_stereo = metrics.accuracy_score(testset_targets, logit_sgd_stereo.predict(testset_data ))
		# 
		# print('*****')
		# print('Dataset: {}, stereo'.format(ds))
		# print('Accuracy (Logistic Regression): %.4f'%logreg_acc_stereo)
		# print('Accuracy (SGD): %.4f'%sgd_acc_stereo)
		# print('*****')
		# 
		# 
		# # mono data fit
		# testset_data = testset_reshaped_mono
# 
		# logreg_acc_mono = metrics.accuracy_score(testset_targets, logit_logreg_mono.predict(testset_data))
		# 
		# sgd_acc_mono = metrics.accuracy_score(testset_targets, logit_sgd_mono.predict(testset_data ))
# 
		# 
		# print('*****')
		# print('Dataset: {}, mono'.format(ds))
		# print('Accuracy (Logistic Regression): %.4f'%logreg_acc_mono)
		# print('Accuracy (SGD): %.4f'%sgd_acc_mono)
		# print('*****')
