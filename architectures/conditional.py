import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, use_bn = True, norm_type = 'batchnorm'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		if(pad == None):
			pad = ks // 2 // stride
		self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = False)
		if(self.use_bn == True):
			if(norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)
		self.relu = nn.LeakyReLU(0.2, inplace = True)

	def forward(self, x):
		out = self.conv(x)
		if(self.use_bn == True):
			out = self.bn(out)
		out = self.relu(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, norm_type = 'batchnorm'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		if(pad == None):
			pad = ks // 2 // stride
		self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)
		if(self.use_bn == True):
			if(norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		out = self.deconv(x)
		if(self.use_bn == True):
			out = self.bn(out)
		out = self.relu(out)
		return out

class ConditionalBatchNorm(nn.Module):
	def __init__(self, n_classes, nc):
		super(ConditionalBatchNorm, self).__init__()
		self.nc = nc
		self.n_classes = n_classes
		self.bn = nn.BatchNorm2d(nc, affine = False)
		self.embed = nn.Embedding(n_classes, nc*2)
		self.embed.weights.data[:, :nc].normal_(0.0, 0.02)
		self.embed.weights.data[:, nc:].zero_()

	def forward(self, x, y):
		# x : (bs, nc, sz, sz)
		# y : (bs, n_classes)
		out_x = self.bn(x)
		out_y = self.embed(y)
		# out_y : (bs, nc * 2)
		gamma, beta = torch.chunk(out_y, 2, 1)
		gamma = gamma.view(-1, self.nc, 1, 1)
		beta = beta.view(-1, self.nc, 1, 1)
		# gamma : (bs, nc, 1, 1)
		# beta : (bs, nc, 1, 1)
		# out_x : (bs, nc, sz, sz)
		out = gamma * out_x + beta
		# out : (bs, nc, sz, sz)
		return out

# Conditional DCGAN Architectures

class Conditional_DCGAN_D(nn.Module):
	def __init__(self, sz, nc, n_classes, ndf = 128, use_sigmoid = True):
		super(Conditional_DCGAN_D, self).__init__()
		assert sz > 4, "Image size should be bigger than 4"
		assert sz & (sz-1) == 0, "Image size should be a power of 2"
		self.sz = sz
		self.nc = nc
		self.ndf = ndf
		self.n_classes = n_classes

		cur_ndf = ndf
		layers = []
		self.first_conv = ConvBlock(self.nc, self.ndf, 4, 2, use_bn = False)
		for i in range(int(math.log2(self.sz)) - 3):
			if(i == 0):
				layers.append(ConvBlock(cur_ndf+self.n_classes, cur_ndf * 2, 4, 2))
			else:
				layers.append(ConvBlock(cur_ndf, cur_ndf * 2, 4, 2))
			cur_ndf *= 2
		layers.append(nn.Conv2d(cur_ndf, 1, 4, 1, 0, bias = False))

		self.main = nn.Sequential(*layers)
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, y):
		# x : (bs, nc, sz, sz)
		# y : (bs, n_classes)
		y = y.view(-1, self.n_classes, 1, 1)
		out = self.first_conv(x)
		out = torch.cat([out, y.repeat(1, 1, x.shape[2]//2, x.shape[3]//2)], dim = 1)
		out = self.main(out)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		# out : (bs, 1, 1, 1)
		return out

class Conditional_DCGAN_G(nn.Module):
	def __init__(self, sz, nz, nc, n_classes, ngf = 128):
		super(Conditional_DCGAN_G, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.n_classes = n_classes

		cur_ngf = ngf * self.sz // 8
		layers = [DeConvBlock(self.nz + self.n_classes, cur_ngf, 4, 1, 0)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(DeConvBlock(cur_ngf, cur_ngf // 2, 4, 2))
			cur_ngf = cur_ngf // 2
		layers.extend([nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias = False), nn.Tanh()])

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, y):
		# x : (bs, nz, 1, 1)
		# y : (bs, n_classes)
		y = y.view(-1, self.n_classes, 1, 1)
		out = self.main(torch.cat([x, y], dim = 1))
		# out : (bs, nc, sz, sz)
		return out
