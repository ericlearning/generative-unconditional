import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class Nothing(nn.Module):
	def __init__(self):
		super(Nothing, self).__init__()
		
	def forward(self, x):
		return x

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)

def get_activation(activation_type):
	if(activation_type == 'relu'):
		return nn.ReLU(inplace = True)
	elif(activation_type == 'leakyrelu'):
		return nn.LeakyReLU(0.2, inplace = True)
	elif(activation_type == 'elu'):
		return nn.ELU(inplace = True)
	elif(activation_type == 'selu'):
		return nn.SELU(inplace = True)
	elif(activation_type == 'prelu'):
		return nn.PReLU()
	elif(activation_type == 'tanh'):
		return nn.Tanh()
	elif(activation_type == None):
		return Nothing()

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, use_bn = True, use_sn = False, use_pixelshuffle = False, norm_type = 'batchnorm', activation_type = 'leakyrelu', pad_type = 'Zero'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.use_pixelshuffle = use_pixelshuffle
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad == None):
			pad = ks // 2 // stride

		ni_ = ni
		if(use_pixelshuffle):
			self.pixelshuffle = nn.PixelShuffle(2)
			ni_ = ni // 4
		
		if(pad_type == 'Zero'):
			self.conv = nn.Conv2d(ni_, no, ks, stride, pad, bias = False)
		else:
			self.conv = nn.Sequential(*[
				nn.ReflectionPad2d(pad),
				nn.Conv2d(ni_, no, ks, stride, 0, bias = False)
			])

		if(self.use_bn):
			self.bn = get_norm(norm_type, no)
		if(self.use_sn):
			self.conv = SpectralNorm(self.conv)

		self.act = get_activation(activation_type)

	def forward(self, x):
		out = x
		if(self.use_pixelshuffle):
			out = self.pixelshuffle(out)
		out = self.conv(out)
		if(self.use_bn):
			out = self.bn(out)
		out = self.act(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu', pad_type = 'Zero'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad is None):
			pad = ks // 2 // stride

		if(pad_type == 'Zero'):
			self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)
		else:
			self.deconv = nn.Sequential(*[
				nn.ReflectionPad2d(pad),
				nn.ConvTranspose2d(ni, no, ks, stride, 0, output_padding = output_pad, bias = False)
			])

		if(self.use_bn):
			self.bn = get_norm(norm_type, no)
		if(self.use_sn):
			self.deconv = SpectralNorm(self.deconv)

		self.act = get_activation(activation_type)

	def forward(self, x):
		out = self.deconv(x)
		if(self.use_bn):
			out = self.bn(out)
		out = self.act(out)
		return out

class UpSample(nn.Module):
	def __init__(self):
		super(UpSample, self).__init__()
		self.scale_factor = 2

	def forward(self, x):
		return F.interpolate(x, None, self.scale_factor, 'bilinear', align_corners = True)

class DCGAN_D(nn.Module):
	def __init__(self, sz, nc, ndf = 64, use_sigmoid = True, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_D, self).__init__()
		assert sz > 4, "Image size should be bigger than 4"
		assert sz & (sz-1) == 0, "Image size should be a power of 2"
		self.sz = sz
		self.nc = nc
		self.ndf = ndf
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		cur_ndf = ndf
		layers = [ConvBlock(self.nc, self.ndf, 4, 2, 1, use_bn = False, use_sn = self.use_sn, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(ConvBlock(cur_ndf, cur_ndf * 2, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = self.norm_type, activation_type = activation_type))
			cur_ndf *= 2
		layers.append(ConvBlock(cur_ndf, 1, 4, 1, 0, use_bn = False, use_sn = self.use_sn, activation_type = None))

		self.main = nn.Sequential(*layers)
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		out = self.main(x)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		return out

class DCGAN_G(nn.Module):
	def __init__(self, sz, nz, nc, ngf = 64, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_G, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [DeConvBlock(self.nz, cur_ngf, 4, 1, 0, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = self.norm_type, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(DeConvBlock(cur_ngf, cur_ngf // 2, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = self.norm_type, activation_type = activation_type))
			cur_ngf = cur_ngf // 2
		layers.append(DeConvBlock(self.ngf, self.nc, 4, 2, 1, use_bn = False, use_sn = self.use_sn, activation_type = 'tanh'))

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
					
	def forward(self, x):
		out = self.main(x)
		return out

class DCGAN_G_ResizedConv(nn.Module):
	def __init__(self, sz, nz, nc, ngf = 64, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_G_ResizedConv, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [ConvBlock(self.nz, cur_ngf, 4, 1, 3, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = activation_type), UpSample()]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.extend([ConvBlock(cur_ngf, cur_ngf // 2, 3, 1, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = activation_type), UpSample()])
			cur_ngf = cur_ngf // 2
		layers.append(ConvBlock(self.ngf, self.nc, 3, 1, 1, use_bn = False, use_sn = self.use_sn, activation_type = 'tanh'))

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		out = self.main(x)
		return out

class DCGAN_G_PixelShuffle(nn.Module):
	def __init__(self, sz, nz, nc, ngf = 64, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_G_PixelShuffle, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [ConvBlock(self.nz, cur_ngf, 4, 1, 3, use_bn = self.use_bn, use_sn = self.use_sn, use_pixelshuffle = False, norm_type = norm_type, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.extend([ConvBlock(cur_ngf, cur_ngf // 2, 3, 1, 1, use_bn = self.use_bn, use_sn = self.use_sn, use_pixelshuffle = True, norm_type = norm_type, activation_type = activation_type)])
			cur_ngf = cur_ngf // 2
		layers.append(ConvBlock(self.ngf, self.nc, 3, 1, 1, use_bn = False, use_sn = self.use_sn, use_pixelshuffle = True, activation_type = 'tanh'))

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		out = self.main(x)
		return out

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, x):
		out = x.view(*self.shape)
		return out