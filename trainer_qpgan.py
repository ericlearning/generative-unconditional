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

class Trainer_QPGAN():
	def __init__(self, netD, netG, device, train_dl, lr_D = 0.0002, lr_G = 0.0002, norm_type = 'L1', loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots', resample = False):
		self.netD = netD
		self.netG = netG
		self.train_dl = train_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.norm_type = norm_type
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (0, 0.9))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0, 0.9))

		self.real_label = 1
		self.fake_label = 0
		self.nz = self.netG.nz

		self.fixed_noise = generate_noise(16, self.nz, self.device)
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
		for epoch in range(num_epoch):
			if(self.resample):
				train_dl_iter = iter(self.train_dl)
			for i, data in enumerate(tqdm(self.train_dl)):
				# (1) : minimizes mean(max(0, 1-D(x))) + mean(max(0, 1+D(G(z))))
				self.netD.zero_grad()
				real_images = data[0].to(self.device)
				bs = real_images.size(0)
				# noise (bs, nz, 1, 1), fake images (bs, cn, 64, 64)
				noise = generate_noise(bs, self.nz, self.device)
				fake_images = self.netG(noise)
				# calculate the discriminator results for both real & fake
				c_xr = self.netD(real_images)				# (bs, 1, 1, 1)
				c_xr = c_xr.view(-1)						# (bs)
				c_xf = self.netD(fake_images.detach())		# (bs, 1, 1, 1)
				c_xf = c_xf.view(-1)						# (bs)
				# calculate the Discriminator loss
				if(self.norm_type == 'L1'):
					denominator = (real_images - fake_images).abs().mean() * 10 * 2
				if(self.norm_type == 'L2'):
					denominator = (real_images - fake_images).mean().sqrt() * 10 * 2

				errD_1 = torch.mean(c_xr) - torch.mean(c_xf)
				errD_2 = (errD_1 ** 2) / denominator
				errD = errD_2 - errD_1
				errD.backward()
				# update D using the gradients calculated previously
				self.optimizerD.step()


				# (2) : minimizes -mean(D(G(z)))
				self.netG.zero_grad()
				if(self.resample):
					real_images = next(train_dl_iter)[0].to(self.device)
					noise = generate_noise(bs, self.nz, self.device)
					fake_images = self.netG(noise)
				# we updated the discriminator once, therefore recalculate c_xr, c_xf
				c_xr = self.netD(real_images)				# (bs, 1, 1, 1)
				c_xr = c_xr.view(-1)						# (bs)
				c_xf = self.netD(fake_images)		# (bs, 1, 1, 1)
				c_xf = c_xf.view(-1)						# (bs)
				# calculate the Generator loss
				errG = torch.mean(c_xr) - torch.mean(c_xf)
				errG.backward()
				# update G using the gradients calculated previously
				self.optimizerG.step()


				self.errD_records.append(float(errD))
				self.errG_records.append(float(errG))

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))

				if(i % self.image_interval == 0):
					sample_images_list = get_sample_images_list('Unsupervised', (self.fixed_noise, self.netG))
					plot_fig = plot_multiple_images(sample_images_list, 4, 4)
					cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
					self.save_cnt += 1
					save_fig(cur_file_name, plot_fig)
					plot_fig.clf()

				if(self.snapshot_interval is not None):
					if(i % self.snapshot_interval == 0):
						save(os.path.join(self.save_snapshot_dir, 'Epoch' + str(epoch) + '_' + str(i) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)


