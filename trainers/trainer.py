import os, cv2
import copy
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import *
from losses.losses import *

class Trainer():
	def __init__(self, loss_type, netD, netG, device, train_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'):
		self.loss_type, self.device = loss_type, device
		self.require_type = get_require_type(self.loss_type)
		self.loss = get_gan_loss(self.device, self.loss_type)

		self.netD = netD
		self.netG = netG
		self.train_dl = train_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample
		self.weight_clip = weight_clip
		self.use_gradient_penalty = use_gradient_penalty
		self.special = None

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (0, 0.9))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0, 0.9))

		self.real_label = 1
		self.fake_label = 0
		self.nz = self.netG.nz

		self.fixed_noise = generate_noise(49, self.nz, self.device)
		self.loss_interval = loss_interval
		self.image_interval = image_interval

		self.errD_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)

	def gradient_penalty(self, real_image, fake_image):
		bs = real_image.size(0)
		alpha = torch.FloatTensor(bs, 1, 1, 1).uniform_(0, 1).expand(real_image.size()).to(self.device)
		interpolation = alpha * real_image + (1 - alpha) * fake_image

		c_xi = self.netD(interpolation)
		gradients = autograd.grad(c_xi, interpolation, torch.ones(c_xi.size()).to(self.device),
								  create_graph = True, retain_graph = True, only_inputs = True)[0]
		gradients = gradients.view(bs, -1)
		penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
		return penalty

	def train(self, num_epoch):
		for epoch in range(num_epoch):
			for i, data in enumerate(tqdm(self.train_dl)):
				self.netD.zero_grad()
				real_images = data[0].to(self.device)
				bs = real_images.size(0)

				noise = generate_noise(bs, self.nz, self.device)
				fake_images = self.netG(noise)

				c_xr = self.netD(real_images)
				c_xr = c_xr.view(-1)
				c_xf = self.netD(fake_images.detach())
				c_xf = c_xf.view(-1)

				if(self.require_type == 0 or self.require_type == 1):
					errD = self.loss.d_loss(c_xr, c_xf)
				elif(self.require_type == 2):
					errD = self.loss.d_loss(c_xr, c_xf, real_images, fake_images)
				
				if(self.use_gradient_penalty != False):
					errD += self.use_gradient_penalty * self.gradient_penalty(real_images, fake_images)

				errD.backward()
				self.optimizerD.step()

				if(self.weight_clip != None):
					for param in self.netD.parameters():
						param.data.clamp_(-self.weight_clip, self.weight_clip)

			
				self.netG.zero_grad()
				if(self.resample):
					noise = generate_noise(bs, self.nz, self.device)
					fake_images = self.netG(noise)

				if(self.require_type == 0):
					c_xf = self.netD(fake_images)
					c_xf = c_xf.view(-1)
					errG = self.loss.g_loss(c_xf)
				if(self.require_type == 1 or self.require_type == 2):
					c_xr = self.netD(real_images)
					c_xr = c_xr.view(-1)
					c_xf = self.netD(fake_images)
					c_xf = c_xf.view(-1)
					errG = self.loss.g_loss(c_xr, c_xf)
					
				errG.backward()
				self.optimizerG.step()

				self.errD_records.append(float(errD))
				self.errG_records.append(float(errG))

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))

				if(i % self.image_interval == 0):
					if(self.special == None):
						sample_images_list = get_sample_images_list('Unsupervised', (self.fixed_noise, self.netG))
						plot_img = get_display_samples(sample_images_list, 7, 7)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						cv2.imwrite(cur_file_name, plot_img)

					elif(self.special == 'Wave'):
						sample_audios_list = get_sample_images_list('Unsupervised_Audio', (self.fixed_noise, self.netG))
						plot_fig = plot_multiple_spectrograms(sample_audios_list, 7, 7, freq = 16000)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						save_fig(cur_file_name, plot_fig)
						plot_fig.clf()
						