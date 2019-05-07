import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Dataset():
	def __init__(self, train_dir, basic_types = None, shuffle = True):
		self.train_dir = train_dir
		self.basic_types = basic_types
		self.shuffle = shuffle

	def get_loader(self, sz, bs, get_size = False, data_transform = None, num_workers = 1, audio_sample_num = None):
		if(self.basic_types is None):
			if(data_transform == None):
				data_transform = transforms.Compose([
					transforms.Resize(sz),
					transforms.CenterCrop(sz),
					transforms.ToTensor(),
					transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
				])

			train_dataset = datasets.ImageFolder(self.train_dir, data_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			train_dataset_size = len(train_dataset)
			size = train_dataset_size
			
			returns = (train_loader)
			if(get_size):
				returns = returns + (size,)

		elif(self.basic_types == 'MNIST'):
			data_transform = transforms.Compose([
				transforms.Resize(sz),
				transforms.CenterCrop(sz),
				transforms.ToTensor(),
				transforms.Normalize([0.5], [0.5])
			])

			train_dataset = datasets.MNIST(self.train_dir, train = True, download = True, transform = data_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			train_dataset_size = len(train_dataset)
			size = train_dataset_size
			
			returns = (train_loader)
			if(get_size):
				returns = returns + (size,)

		elif(self.basic_types == 'CIFAR10'):
			data_transform = transforms.Compose([
				transforms.Resize(sz),
				transforms.CenterCrop(sz),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])

			train_dataset = datasets.CIFAR10(self.train_dir, train = True, download = True, transform = data_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			train_dataset_size = len(train_dataset)
			size = train_dataset_size
			
			returns = (train_loader)
			if(get_size):
				returns = returns + (size,)

		elif(self.basic_types == 'Audio'):
			train_dataset = Audio_Dataset(self.train_dir, data_transform, audio_sample_num)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			returns = (train_loader)

		return returns

class Audio_Dataset():
	def __init__(self, input_dir, input_transform, num_samples):
		self.input_dir = input_dir
		self.input_transform = input_transform
		self.num_samples = num_samples

		self.audio_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.npy')):
				self.audio_name_list.append(file)

	def __len__(self):
		return len(self.audio_name_list)

	def __getitem__(self, idx):
		input_audio = np.load(os.path.join(self.input_dir, self.audio_name_list[idx]))
		point = random.randint(0, input_audio.shape[0] - self.num_samples)
		input_audio = input_audio[point:point+self.num_samples] / 32768.0
		input_audio = torch.from_numpy(input_audio)
		input_audio = input_audio.view(1, -1).float()

		if(self.input_transform is not None):
			input_audio = self.input_transform(input_audio)

		return (input_audio, 0)
