import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, use_bn = True, use_pixelshuffle = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_pixelshuffle = use_pixelshuffle
		self.norm_type = norm_type

		if(pad == None):
			pad = ks // 2 // stride

		if(use_pixelshuffle):
			self.conv = nn.Conv2d(ni, no * 2 * 2, ks, stride, pad, bias = False)
			self.pixelshuffle = nn.PixelShuffle(2)
		else:
			self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = False)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)
			elif(self.norm_type == 'spectralnorm'):
				self.conv = SpectralNorm(self.conv)


		if(activation_type == 'relu'):
			self.act = nn.ReLU(inplace = True)
		elif(activation_type == 'leakyrelu'):
			self.act = nn.LeakyReLU(0.2, inplace = True)
		elif(activation_type == 'elu'):
			self.act = nn.ELU(inplace = True)
		elif(activation_type == 'selu'):
			self.act = nn.SELU(inplace = True)

	def forward(self, x):
		out = self.conv(x)
		if(self.use_pixelshuffle == True):
			out = self.pixelshuffle(out)
		if(self.use_bn == True and self.norm_type != 'spectralnorm'):
			out = self.bn(out)
		out = self.act(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.norm_type = norm_type

		if(pad is None):
			pad = ks // 2 // stride

		self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)

		if(self.use_bn == True):
			if(self.norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(self.norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)
			elif(self.norm_type == 'spectralnorm'):
				self.deconv = SpectralNorm(self.deconv)

		if(activation_type == 'relu'):
			self.act = nn.ReLU(inplace = True)
		elif(activation_type == 'leakyrelu'):
			self.act = nn.LeakyReLU(0.2, inplace = True)
		elif(activation_type == 'elu'):
			self.act = nn.ELU(inplace = True)
		elif(activation_type == 'selu'):
			self.act = nn.SELU(inplace = True)

	def forward(self, x):
		out = self.deconv(x)
		if(self.use_bn == True and self.norm_type != 'spectralnorm'):
			out = self.bn(out)
		out = self.act(out)
		return out

class UpSample(nn.Module):
	def __init__(self):
		super(UpSample, self).__init__()
		self.scale_factor = 2

	def forward(self, x):
		return F.interpolate(x, None, self.scale_factor, 'bilinear', align_corners = True)

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

# DCGAN Architectures

class DCGAN_D(nn.Module):
	def __init__(self, sz, nc, ndf = 64, use_sigmoid = True, use_bn = True, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_D, self).__init__()
		assert sz > 4, "Image size should be bigger than 4"
		assert sz & (sz-1) == 0, "Image size should be a power of 2"
		self.sz = sz
		self.nc = nc
		self.ndf = ndf
		self.use_bn = use_bn
		self.norm_type = norm_type

		cur_ndf = ndf
		layers = [ConvBlock(self.nc, self.ndf, 4, 2, use_bn = False, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(ConvBlock(cur_ndf, cur_ndf * 2, 4, 2, use_bn = self.use_bn, norm_type = norm_type, activation_type = activation_type))
			cur_ndf *= 2

		if(self.norm_type == 'spectralnorm'):
			layers.append(SpectralNorm(nn.Conv2d(cur_ndf, 1, 4, 1, 0, bias = False)))
		else:
			layers.append(nn.Conv2d(cur_ndf, 1, 4, 1, 0, bias = False))

		self.main = nn.Sequential(*layers)
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		# x : (bs, nc, sz, sz)
		out = self.main(x)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		# out : (bs, 1, 1, 1)
		return out

class DCGAN_G(nn.Module):
	def __init__(self, sz, nz, nc, ngf = 64, use_bn = True, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_G, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [DeConvBlock(self.nz, cur_ngf, 4, 1, 0, use_bn = use_bn, norm_type = norm_type, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(DeConvBlock(cur_ngf, cur_ngf // 2, 4, 2, use_bn = use_bn, norm_type = norm_type, activation_type = activation_type))
			cur_ngf = cur_ngf // 2

		if(self.norm_type == 'spectralnorm'):
			layers.extend([SpectralNorm(nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias = False)), nn.Tanh()])
		else:
			layers.extend([nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias = False), nn.Tanh()])

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
					
	def forward(self, x):
		# x : (bs, nz, 1, 1)
		out = self.main(x)
		# out : (bs, nc, sz, sz)
		return out

class DCGAN_G_ResizedConv(nn.Module):
	def __init__(self, sz, nz, nc, ngf = 64, use_bn = True, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_G_ResizedConv, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [ConvBlock(self.nz, cur_ngf, 4, 1, 3, use_bn = use_bn, norm_type = norm_type, activation_type = activation_type), UpSample()]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.extend([ConvBlock(cur_ngf, cur_ngf // 2, 3, 1, 1, use_bn = use_bn, norm_type = norm_type, activation_type = activation_type), UpSample()])
			cur_ngf = cur_ngf // 2

		if(self.norm_type == 'spectralnorm'):
			layers.extend([SpectralNorm(nn.Conv2d(self.ngf, self.nc, 3, 1, 1, bias = False)), nn.Tanh()])
		else:
			layers.extend([nn.Conv2d(self.ngf, self.nc, 3, 1, 1, bias = False), nn.Tanh()])

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		# x : (bs, nz, 1, 1)
		out = self.main(x)
		# out : (bs, nc, sz, sz)
		return out

class DCGAN_G_PixelShuffle(nn.Module):
	def __init__(self, sz, nz, nc, ngf = 64, use_bn = True, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(DCGAN_G_PixelShuffle, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [ConvBlock(self.nz, cur_ngf, 4, 1, 3, use_bn = use_bn, use_pixelshuffle = True, norm_type = norm_type, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(ConvBlock(cur_ngf, cur_ngf // 2, 3, 1, 1, use_bn = use_bn, use_pixelshuffle = True, norm_type = norm_type, activation_type = activation_type))
			cur_ngf = cur_ngf // 2
		
		if(self.norm_type == 'spectralnorm'):
			layers.extend([SpectralNorm(nn.Conv2d(self.ngf, self.nc, 3, 1, 1, bias = False)), nn.Tanh()])
		else:
			layers.extend([nn.Conv2d(self.ngf, self.nc, 3, 1, 1, bias = False), nn.Tanh()])

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		# x : (bs, nz, 1, 1)
		out = self.main(x)
		# out : (bs, nc, sz, sz)
		return out


class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, x):
		out = x.view(*self.shape)
		return out
