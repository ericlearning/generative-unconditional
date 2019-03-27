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
from itertools import chain

class Trainer_LSGAN_Cyclegan():
	def __init__(self, netD_A, netD_B, netG_A2B, netG_B2A, device, train_dl, val_dl, lr_D = 0.0002, lr_G = 0.0002, loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots', resample = None):
		self.netD_A = netD_A
		self.netD_B = netD_B
		self.netG_A2B = netG_A2B
		self.netG_B2A = netG_B2A
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample

		self.optimizerD_A = optim.Adam(self.netD_A.parameters(), lr = self.lr_D, betas = (0.5, 0.999))
		self.optimizerD_B = optim.Adam(self.netD_B.parameters(), lr = self.lr_D, betas = (0.5, 0.999))
		self.optimizerG = optim.Adam(chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr = self.lr_G, betas = (0.5, 0.999))

		self.real_label = 1
		self.fake_label = 0

		self.loss_interval = loss_interval
		self.image_interval = image_interval
		self.snapshot_interval = snapshot_interval

		self.errD_A_records = []
		self.errD_B_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		self.save_snapshot_dir = save_snapshot_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)
		if(not os.path.exists(self.save_snapshot_dir)):
			os.makedirs(self.save_snapshot_dir)

	def train(self, num_epoch):
		l1 = nn.L1Loss()
		for epoch in range(num_epoch):
			if(self.resample):
				train_dl_iter = iter(self.train_dl)
			for i, (a, b) in enumerate(tqdm(self.train_dl)):
				a = a.to(self.device)
				b = b.to(self.device)

				# calculate the generator results
				fake_a = self.netG_B2A(b)
				fake_b = self.netG_A2B(a)

				self.optimizerD_A.zero_grad()
				# calculate the discriminator results
				c_a_fake_a = self.netD_A(fake_a.detach())
				c_a_real_a = self.netD_A(a)
				# calculate the generator loss
				c_a_fake_a_loss = torch.mean((c_a_fake_a - torch.zeros(c_a_fake_a.size()).to(self.device)) ** 2)
				c_a_real_a_loss = torch.mean((c_a_real_a - torch.ones(c_a_real_a.size()).to(self.device)) ** 2)
				c_a_loss = (c_a_fake_a_loss + c_a_real_a_loss)
				c_a_loss = c_a_loss / 2.0		# (discriminator updates slower)
				c_a_loss.backward()
				# update G using the gradients calculated previously
				self.optimizerD_A.step()


				self.optimizerD_B.zero_grad()
				# calculate the discriminator results
				c_b_fake_b = self.netD_B(fake_b.detach())
				c_b_real_b = self.netD_B(b)
				# calculate the generator loss
				c_b_fake_b_loss = torch.mean((c_b_fake_b - torch.zeros(c_b_fake_b.size()).to(self.device)) ** 2)
				c_b_real_b_loss = torch.mean((c_b_real_b - torch.ones(c_b_real_b.size()).to(self.device)) ** 2)
				c_b_loss = (c_b_fake_b_loss + c_b_real_b_loss)
				c_b_loss = c_b_loss / 2.0		# (discriminator updates slower)
				c_b_loss.backward()
				# update G using the gradients calculated previously
				self.optimizerD_B.step()


				self.optimizerG.zero_grad()
				if(self.resample):
					a, b = next(train_dl_iter)
					a = a.to(self.device)
					b = b.to(self.device)
					fake_a = self.netG_B2A(b)
					fake_b = self.netG_A2B(a)

				cycle_a = self.netG_B2A(fake_b)
				cycle_b = self.netG_A2B(fake_a)
				identity_a = self.netG_B2A(a)
				identity_b = self.netG_A2B(b)
				
				# calculate the discriminator results for both real & fake
				c_a_fake_a = self.netD_A(fake_a)
				c_b_fake_b = self.netD_B(fake_b)

				# calculate the generator loss
				c_a_loss = torch.mean((c_a_fake_a - torch.ones(c_a_fake_a.size()).to(self.device)) ** 2)
				c_b_loss = torch.mean((c_b_fake_b - torch.ones(c_b_fake_b.size()).to(self.device)) ** 2)
				cycle_a_loss = l1(cycle_a, a)
				cycle_b_loss = l1(cycle_b, b)
				identity_a_loss = l1(identity_a, a)
				identity_b_loss = l1(identity_b, b)

				errG = c_a_loss + c_b_loss + (cycle_a_loss + cycle_b_loss) * 10.0 + (identity_a_loss + identity_b_loss) * 5.0
				errG.backward()
				# update G using the gradients calculated previously
				self.optimizerG.step()

				self.errD_A_records.append(float(c_a_loss))
				self.errD_B_records.append(float(c_b_loss))
				self.errG_records.append(float(errG))

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD_A : %.4f, errD_B : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, c_a_loss, c_b_loss, errG))

				if(i % self.image_interval == 0):
					sample_images_list = get_sample_images_list('Cyclegan', (self.val_dl,self.netG_A2B, self.netG_B2A, self.device))
					plot_fig = plot_multiple_images(sample_images_list, 3, 6)
					cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
					self.save_cnt += 1
					save_fig(cur_file_name, plot_fig)
					plot_fig.clf()

				if(self.snapshot_interval is not None):
					if(i % self.snapshot_interval == 0):
						save(os.path.join(self.save_snapshot_dir, 'Epoch' + str(epoch) + '_' + str(i) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)
						
