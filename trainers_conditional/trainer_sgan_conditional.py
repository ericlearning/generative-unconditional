import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, save_fig, save, get_sample_images_list

class Trainer_SGAN_C():
	def __init__(self, netD, netG, n_classes, device, train_dl, lr_D = 0.0002, lr_G = 0.0002, loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots'):
		self.netD = netD
		self.netG = netG
		self.n_classes = n_classes
		self.train_dl = train_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (beta1, 0.999))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (beta1, 0.999))

		self.real_label = 1
		self.fake_label = 0
		self.nz = self.netG.nz

		self.fixed_noise = generate_noise(self.n_classes, self.nz, self.device)
		self.fixed_one_hot_labels = torch.diagflat(torch.ones(self.n_classes)).to(self.device)
		self.loss_interval = loss_interval
		self.image_interval = image_interval
		self.snapshot_interval = snapshot_interval

		self.errD_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		self.save_snapshot_dir = save_snapshot_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)
		if(not os.path.exists(self.save_snapshot_dir)):
			os.makedirs(self.save_snapshot_dir)

	def train(self, num_epoch):
		criterion = nn.BCELoss()
		for epoch in range(num_epoch):
			for i, data in enumerate(tqdm(self.train_dl)):
				# (1) : minimize 0.5 * mean((D(x, y) - 1)^2) + 0.5 * mean((D(G(z, y), y) - 0)^2)
				self.netD.zero_grad()
				real_images = data[0].to(self.device)
				real_class = data[1].to(self.device)

				bs = real_images.size(0)
				# real labels (bs)
				real_label = torch.full((bs, ), self.real_label, device = self.device)
				# fake labels (bs)
				fake_label = torch.full((bs, ), self.fake_label, device = self.device)

				# one hot labels (bs, n_classes)
				one_hot_labels = torch.FloatTensor(bs, self.n_classes).to(self.device)
				one_hot_labels.zero_()
				one_hot_labels.scatter_(1, real_class.view(bs, 1), 1.0)
				
				# noise (bs, nz, 1, 1), fake images (bs, nc, 64, 64)
				noise = generate_noise(bs, self.nz, self.device)

				fake_class = torch.randint(0, self.n_classes, size = (bs, 1)).view(bs, 1).to(self.device)
				# one hot labels (bs, n_classes)
				one_hot_labels_fake = torch.FloatTensor(bs, self.n_classes).to(self.device)
				one_hot_labels_fake.zero_()
				one_hot_labels_fake.scatter_(1, fake_class.view(bs, 1).long(), 1.0)

				fake_images = self.netG(noise, one_hot_labels_fake)

				# calculate the discriminator results for both real & fake
				c_xr = self.netD(real_images, one_hot_labels)				# (bs, 1, 1, 1)
				c_xr = c_xr.view(-1)						# (bs)
				c_xf = self.netD(fake_images.detach(), one_hot_labels_fake)		# (bs, 1, 1, 1)
				c_xf = c_xf.view(-1)						# (bs)
				# calculate the discriminator loss
				errD = criterion(c_xr, real_label) + criterion(c_xf, fake_label)
				errD.backward()
				# update D using the gradients calculated previously
				self.optimizerD.step()

				# (2) : minimize 0.5 * mean((D(G(z)) - 1)^2)
				self.netG.zero_grad()
				if(self.resample):
					noise = generate_noise(bs, self.nz, self.device)
					one_hot_labels_fake = torch.FloatTensor(bs, self.n_classes).to(self.device)
					one_hot_labels_fake.zero_()
					one_hot_labels_fake.scatter_(1, fake_class.view(bs, 1).long(), 1.0)
					fake_images = self.netG(noise, one_hot_labels_fake)
				# we updated the discriminator once, therefore recalculate c_xf
				c_xf = self.netD(fake_images, one_hot_labels_fake)				# (bs, 1, 1, 1)
				c_xf = c_xf.view(-1)						# (bs)
				# calculate the Generator loss
				errG = criterion(c_xf, real_label)			# 0.5 * mean((D(G(z)) - 1)^2)
				errG.backward()
				#update G using the gradients calculated previously
				self.optimizerG.step()

				self.errD_records.append(float(errD))
				self.errG_records.append(float(errG))

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))
				
				if(i % self.image_interval == 0):
					sample_images_list = get_sample_images_list('Conditional', (self.fixed_noise, self.fixed_one_hot_labels, self.n_classes, self.netG))
					plot_fig = plot_multiple_images(sample_images_list, self.n_classes, 1)
					cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
					self.save_cnt += 1
					save_fig(cur_file_name, plot_fig)
					plot_fig.clf()

				if(self.snapshot_interval is not None):
					if(i % self.snapshot_interval == 0):
						save(os.path.join(self.save_snapshot_dir, 'Epoch' + str(epoch) + '_' + str(i) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)