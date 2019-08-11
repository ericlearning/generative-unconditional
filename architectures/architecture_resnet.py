import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class UpSample(nn.Module):
	def __init__(self):
		super(UpSample, self).__init__()
		self.scale_factor = 2

	def forward(self, x):
		return F.interpolate(x, None, self.scale_factor, 'bilinear', align_corners = True)

class DownSample(nn.Module):
	def __init__(self):
		super(DownSample, self).__init__()

	def forward(self, x):
		return F.avg_pool2d(x, 2)

class SelfAttention(nn.Module):
	def __init__(self, ni):
		super(SelfAttention, self).__init__()
		self.ni = ni
		self.f = nn.Conv2d(self.ni, self.ni//8, 1, 1, 0)
		self.g = nn.Conv2d(self.ni, self.ni//8, 1, 1, 0)
		self.h = nn.Conv2d(self.ni, self.ni, 1, 1, 0)
		self.softmax = nn.Softmax(dim = -1)
		self.alpha = nn.Parameter(torch.tensor(0.0))

	def forward(self, x):
		# x : (bs, ni, sz, sz)
		f_out = self.f(x)
		# (bs, ni // 8, sz, sz)
		f_out = f_out.view(f_out.size(0), self.ni//8, -1)
		# (bs, ni // 8, sz * sz)
		f_out = f_out.permute(0, 2, 1)
		# (bs, sz * sz, ni // 8)

		# x : (bs, ni, sz, sz)
		g_out = self.g(x)
		# (bs, ni // 8, sz, sz)
		g_out = g_out.view(g_out.size(0), self.ni//8, -1)
		# (bs, ni // 8, sz * sz)

		# x : (bs, ni, sz, sz)
		h_out = self.h(x)
		# (bs, ni, sz, sz)
		h_out = h_out.view(h_out.size(0), self.ni, -1)
		# (bs, ni, sz * sz)

		f_g_mult = torch.bmm(f_out, g_out)
		# (bs, sz * sz, sz * sz)
		f_g_mult = self.softmax(f_g_mult)
		# (bs, sz * sz, sz * sz)
		f_g_h_mult = torch.bmm(h_out, f_g_mult)
		# (bs, ni, sz * sz)
		f_g_h_mult = f_g_h_mult.view(*x.shape)
		# (bs, ni, sz, sz)

		out = self.alpha * f_g_h_mult + x
		# (bs, ni, sz, sz)

		return out

class Generative_ResBlock(nn.Module):
	def __init__(self, ic, oc, upsample = True, use_sn = False):
		super(Generative_ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.conv3 = nn.Conv2d(ic, oc, 1, 1, 0)

		if(use_sn == False):
			model_list = [nn.BatchNorm2d(ic), nn.ReLU(inplace = True), self.conv1]
		elif(use_sn == True):
			model_list = [nn.BatchNorm2d(ic), nn.ReLU(inplace = True), SpectralNorm(self.conv1)]
		if(upsample == True):
			model_list.append(UpSample())
		model_list.extend([
			nn.BatchNorm2d(oc),
			nn.ReLU(inplace = True),
		])
		if(use_sn == False):
			model_list.append(self.conv2)
		elif(use_sn == True):
			model_list.append(SpectralNorm(self.conv2))
		self.model = nn.Sequential(*model_list)

		bypass_list = [self.conv3]
		if(upsample == True):
			bypass_list.append(UpSample())
		self.bypass = nn.Sequential(*bypass_list)

	def forward(self, x):
		out = self.model(x) + self.bypass(x)
		return out

class Discriminative_ResBlock_First(nn.Module):
	def __init__(self, ic, oc, downsample = True, use_sn = True):
		super(Discriminative_ResBlock_First, self).__init__()
		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.conv3 = nn.Conv2d(ic, oc, 1, 1, 0)

		model_list = [
			SpectralNorm(self.conv1),
			nn.ReLU(inplace = True),
			SpectralNorm(self.conv2)
		]
		if(downsample == True):
			model_list.append(nn.AvgPool2d(2))
		self.model = nn.Sequential(*model_list)

		bypass_list = [SpectralNorm(self.conv3)]
		if(downsample == True):
			bypass_list.append(nn.AvgPool2d(2))
		self.bypass = nn.Sequential(*bypass_list)

	def forward(self, x):
		out = self.model(x) + self.bypass(x)
		return out

class Discriminative_ResBlock(nn.Module):
	def __init__(self, ic, oc, downsample = True, use_sn = True):
		super(Discriminative_ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.conv3 = nn.Conv2d(ic, oc, 1, 1, 0)
		if(use_sn):
			self.conv1 = SpectralNorm(self.conv1)
			self.conv2 = SpectralNorm(self.conv2)
			self.conv3 = SpectralNorm(self.conv3)

		model_list = [
			nn.ReLU(inplace = True),
			self.conv1,
			nn.ReLU(inplace = True),
			self.conv2
		]

		if(downsample == True):
			model_list.append(nn.AvgPool2d(2))
		self.model = nn.Sequential(*model_list)

		bypass_list = [SpectralNorm(self.conv3)]
		if(downsample == True):
			bypass_list.append(nn.AvgPool2d(2))
		self.bypass = nn.Sequential(*bypass_list)

	def forward(self, x):
		out = self.model(x) + self.bypass(x)
		return out

class ResNet_D(nn.Module):
	def __init__(self, sz, nc, ndf, use_sigmoid = True, use_sn = True, use_self_attention = None):
		super(ResNet_D, self).__init__()
		self.sz = sz
		self.nc = nc
		self.ndf = ndf
		self.use_sn = use_sn
		self.use_self_attention = use_self_attention

		cur_ndf = self.ndf

		self.blocks = [Discriminative_ResBlock_First(self.nc, cur_ndf, True, self.use_sn)]
		for i in range(int(math.log2(self.sz)) - 3):
			if(cur_ndf == self.use_self_attention):
				self.blocks.append(SelfAttention(cur_ndf))
			self.blocks.append(Discriminative_ResBlock(cur_ndf, cur_ndf*2, True, self.use_sn))
			cur_ndf = cur_ndf * 2
		self.blocks.append(Discriminative_ResBlock(cur_ndf, cur_ndf, False, self.use_sn))
		self.blocks = nn.Sequential(*self.blocks)

		self.relu = nn.ReLU(inplace = True)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.dense = nn.Linear(cur_ndf, 1)
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid

		nn.init.xavier_uniform(self.dense.weight.data)
		nn.init.xavier_uniform(self.conv.weight.data)

	def forward(self, x):
		out = self.blocks(x)
		out = self.relu(out)
		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = self.dense(out)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		return out

class ResNet_G(nn.Module):
	def __init__(self, sz, nz, nc, ngf, use_sn = False, use_self_attention = None):
		super(ResNet_G, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.use_sn = use_sn
		self.use_self_attention = use_self_attention

		cur_ngf = self.ngf*self.sz//8
		self.dense = nn.Linear(self.nz, 4*4*cur_ngf)

		self.blocks = [Generative_ResBlock(cur_ngf, cur_ngf, True, self.use_sn)]
		for i in range(int(math.log2(self.sz)) - 3):
			if(cur_ngf == self.use_self_attention):
				self.blocks.append(SelfAttention(cur_ngf))
			self.blocks.append(Generative_ResBlock(cur_ngf, cur_ngf // 2, True, self.use_sn))
			cur_ngf = cur_ngf // 2
		self.blocks = nn.Sequential(*self.blocks)

		self.bn = nn.BatchNorm2d(cur_ngf)
		self.relu = nn.ReLU(inplace = True)
		self.conv = nn.Conv2d(cur_ngf, self.nc, 1, 1, 0)
		self.tanh = nn.Tanh()

		nn.init.xavier_uniform(self.dense.weight.data)
		nn.init.xavier_uniform(self.conv.weight.data)

	def forward(self, x):
		out = x.view(x.size(0), -1)
		out = self.dense(out)
		out = out.view(out.size(0), -1, 4, 4)
		out = self.blocks(out)
		out = self.conv(self.relu(self.bn(out)))
		out = self.tanh(out)
		return out

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, x):
		out = x.view(*self.shape)
		return out
