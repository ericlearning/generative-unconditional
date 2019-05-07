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
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, plot_multiple_spectrograms, save_fig, save, get_sample_images_list

class Trainer_SGAN():
	def __init__(self, netD, netG, device, train_dl, lr_D = 0.0002, lr_G = 0.0002, beta1 = 0.5, loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots', resample = False):
		self.netD = netD
		self.netG = netG
		self.train_dl = train_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample
		self.special = None

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (beta1, 0.999))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (beta1, 0.999))

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
		criterion = nn.BCELoss()
		for epoch in range(num_epoch):
			for i, data in enumerate(tqdm(self.train_dl)):
				# (1) : maximize log(D(x)) + log(1 - D(G(z)))
				# 		also means minimize (-log(D(x))) + (-log(1 - D(G(z))))
				self.netD.zero_grad()

				# first, calculate -log(D(x)) and its gradients using real images
				# real images (bs, nc, 64, 64)
				real_images = data[0].to(self.device)
				bs = real_images.size(0)
				# real labels (bs)
				label = torch.full((bs, ), self.real_label, device = self.device)
				output = self.netD(real_images)				# (bs, 1, 1, 1)
				output = output.view(-1)				# (bs)
				# BCELoss of output(bs), and real label(bs)
				errD_real = criterion(output, label)	# -log(D(x))
				# calculate the gradients
				errD_real.backward()

				# second, calculate -log(1 - D(G(z))) and its gradients using fake images
				# noise (bs, nz, 1, 1), fake images (bs, nc, 64, 64)
				noise = generate_noise(bs, self.nz, self.device)
				fake_images = self.netG(noise)
				# fake labels (bs)
				label.fill_(self.fake_label)
				output = self.netD(fake_images.detach())		# (bs, 1, 1, 1)
				output = output.view(-1)				# (bs)
				# BCELoss of output(bs), and fake labels(bs)
				errD_fake = criterion(output, label)	# -log(1 - D(G(z)))
				# calculate the gradients
				errD_fake.backward()

				# calculate the final loss value, (-log(D(x))) + (-log(1 - D(G(z))))
				errD = errD_real + errD_fake
				# update D using the gradients calculated previously
				self.optimizerD.step()

				# (2) : maximize log(D(G(z)))
				#		also means minimize -log(D(G(z)))
				self.netG.zero_grad()
				if(self.resample):
					noise = generate_noise(bs, self.nz, self.device)
					fake_images = self.netG(noise)

				# first, calculate -log(D(G(z))) and its gradients using fake images
				# real labels (bs)
				label.fill_(self.real_label)
				output = self.netD(fake_images)				# (bs, 1, 1, 1)
				output = output.view(-1)				# (bs)
				# BCELoss of output(bs), and real labels(bs)
				errG = criterion(output, label)			# -log(D(G(z)))
				#calculate the gradients
				errG.backward()

				#update G using the gradients calculated previously
				self.optimizerG.step()

				self.errD_records.append(float(errD))
				self.errG_records.append(float(errG))

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))

				if(i % self.image_interval == 0):
					if(self.special == None):
						sample_images_list = get_sample_images_list('Unsupervised', (self.fixed_noise, self.netG))
						plot_fig = plot_multiple_images(sample_images_list, 4, 4)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						save_fig(cur_file_name, plot_fig)
						plot_fig.clf()

					elif(self.special == 'Wave'):
						sample_audios_list = get_sample_images_list('Unsupervised_Audio', (self.fixed_noise, self.netG))
						plot_fig = plot_multiple_spectrograms(sample_audios_list, 4, 4, freq = 16000)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						save_fig(cur_file_name, plot_fig)
						plot_fig.clf()

				if(self.snapshot_interval is not None):
					if(i % self.snapshot_interval == 0):
						save(os.path.join(self.save_snapshot_dir, 'Epoch' + str(epoch) + '_' + str(i) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)



