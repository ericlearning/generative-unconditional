import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, use_bn = True, use_pixelshuffle = False, norm_type = 'batchnorm'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_pixelshuffle = use_pixelshuffle
		if(pad == None):
			pad = ks // 2 // stride
		if(use_pixelshuffle):
			self.conv = nn.Conv2d(ni, no * 2 * 2, ks, stride, pad, bias = False)
			self.pixelshuffle = nn.PixelShuffle(2)
		else:
			self.conv = nn.Conv2d(ni, no, ks, stride, pad, bias = False)

		if(self.use_bn == True):
			if(norm_type == 'batchnorm'):
				self.bn = nn.BatchNorm2d(no)
			elif(norm_type == 'instancenorm'):
				self.bn = nn.InstanceNorm2d(no)
		self.relu = nn.LeakyReLU(0.2, inplace = True)

	def forward(self, x):
		out = self.conv(x)
		if(self.use_pixelshuffle == True):
			out = self.pixelshuffle(out)
		if(self.use_bn == True):
			out = self.bn(out)
		out = self.relu(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, norm_type = 'batchnorm'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		if(pad is None):
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

# Residual Architectures with Spectral Norm and Self Attention

class Generative_ResBlock(nn.Module):
	def __init__(self, ic, oc, upsample = True, use_spectral_norm = False):
		super(Generative_ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.conv3 = nn.Conv2d(ic, oc, 1, 1, 0)
		nn.init.xavier_uniform(self.conv1.weight.data, 1.)
		nn.init.xavier_uniform(self.conv2.weight.data, 1.)
		nn.init.xavier_uniform(self.conv3.weight.data, np.sqrt(2))

		if(use_spectral_norm == False):
			model_list = [nn.BatchNorm2d(ic), nn.ReLU(inplace = True), self.conv1]
		elif(use_spectral_norm == True):
			model_list = [nn.BatchNorm2d(ic), nn.ReLU(inplace = True), SpectralNorm(self.conv1)]
		if(upsample == True):
			model_list.append(UpSample())
		model_list.extend([
			nn.BatchNorm2d(oc),
			nn.ReLU(inplace = True),
		])
		if(use_spectral_norm == False):
			model_list.append(self.conv2)
		elif(use_spectral_norm == True):
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
	def __init__(self, ic, oc, downsample = True):
		super(Discriminative_ResBlock_First, self).__init__()
		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.conv3 = nn.Conv2d(ic, oc, 1, 1, 0)
		nn.init.xavier_uniform(self.conv1.weight.data, 1.)
		nn.init.xavier_uniform(self.conv2.weight.data, 1.)
		nn.init.xavier_uniform(self.conv3.weight.data, np.sqrt(2))

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
	def __init__(self, ic, oc, downsample = True):
		super(Discriminative_ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.conv3 = nn.Conv2d(ic, oc, 1, 1, 0)
		nn.init.xavier_uniform(self.conv1.weight.data, 1.)
		nn.init.xavier_uniform(self.conv2.weight.data, 1.)
		nn.init.xavier_uniform(self.conv3.weight.data, np.sqrt(2))

		model_list = [
			nn.ReLU(inplace = True),
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

class ResNetGan_D(nn.Module):
	def __init__(self, sz, nc, ndf, use_sigmoid = True, self_attention_layer = None):
		super(ResNetGan_D, self).__init__()
		self.sz = sz
		self.nc = nc
		self.ndf = ndf
		self.self_attention_layer = self_attention_layer
		cur_ndf = self.ndf

		self.blocks = [Discriminative_ResBlock_First(self.nc, cur_ndf, True)]
		for i in range(int(math.log2(self.sz)) - 3):
			if(cur_ndf == self_attention_layer):
				self.blocks.append(SelfAttention(cur_ndf))
			self.blocks.append(Discriminative_ResBlock(cur_ndf, cur_ndf*2, True))
			cur_ndf = cur_ndf * 2
		self.blocks.append(Discriminative_ResBlock(cur_ndf, cur_ndf, False))
		self.blocks = nn.Sequential(*self.blocks)

		self.relu = nn.ReLU(inplace = True)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.dense = nn.Linear(cur_ndf, 1)
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid
		
		nn.init.xavier_uniform(self.dense.weight.data, 1.)

	def forward(self, x):
		out = self.blocks(x)
		out = self.relu(out)
		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = self.dense(out)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		return out

class ResNetGan_G(nn.Module):
	def __init__(self, sz, nz, nc, ngf, use_spectral_norm = False, self_attention_layer = None):
		super(ResNetGan_G, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.ngf = ngf
		self.self_attention_layer = self_attention_layer
		cur_ngf = self.ngf*self.sz//8
		self.dense = nn.Linear(self.nz, 4*4*cur_ngf)
		self.use_spectral_norm = use_spectral_norm

		self.blocks = [Generative_ResBlock(cur_ngf, cur_ngf, True, self.use_spectral_norm)]
		for i in range(int(math.log2(self.sz)) - 3):
			if(cur_ngf == self_attention_layer):
				self.blocks.append(SelfAttention(cur_ngf))
			self.blocks.append(Generative_ResBlock(cur_ngf, cur_ngf // 2, True, self.use_spectral_norm))
			cur_ngf = cur_ngf // 2
		self.blocks = nn.Sequential(*self.blocks)

		self.bn = nn.BatchNorm2d(cur_ngf)
		self.relu = nn.ReLU(inplace = True)
		self.conv = nn.Conv2d(cur_ngf, self.nc, 1, 1, 0)
		self.tanh = nn.Tanh()

		nn.init.xavier_uniform(self.dense.weight.data, 1.)
		nn.init.xavier_uniform(self.conv.weight.data, 1.)

	def forward(self, x):
		out = x.view(x.size(0), -1)
		out = self.dense(out)
		out = out.view(out.size(0), -1, 4, 4)
		out = self.blocks(out)
		out = self.conv(self.relu(self.bn(out)))
		out = self.tanh(out)
		return out

# Custom Layers and Models
class Custom128x128_G(nn.Module):
	def __init__(self, nz, oc):
		super(Custom128x128_G, self).__init__()
		self.nz = nz
		self.oc = oc

		self.init_block = nn.Sequential(
			nn.Linear(nz, 64 * 16 * 16),
			Reshape((-1, 64, 16, 16)),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace = True)
		)
		self.blocks = nn.Sequential(
			*([Custom_G_ResBlock(64, 64)] * 16)
		)
		self.blocks2 = nn.Sequential(
			Custom_G_UpBlock(64),
			Custom_G_UpBlock(64),
			Custom_G_UpBlock(64)
		)

		self.bn = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace = True)
		self.conv = nn.Conv2d(64, self.oc, 9, 1, 4, bias = True)
		self.tanh = nn.Tanh()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, z):
		out = self.init_block(z.view(-1, self.nz))
		out = self.relu(self.bn(self.blocks(out))) + out
		out = self.blocks2(out)
		out = self.tanh(self.conv(out))

		return out

class Custom128x128_D(nn.Module):
	def __init__(self, nc, use_sigmoid):
		super(Custom128x128_D, self).__init__()
		self.nc = nc
		self.use_sigmoid = use_sigmoid

		self.conv = nn.Conv2d(nc, 32, 4, 2, 1)
		self.relu = nn.LeakyReLU(0.2, inplace = True)

		self.blocks = nn.Sequential(
			Custom_D_GlobalBlock(32, 64),
			Custom_D_GlobalBlock(64, 128),
			Custom_D_GlobalBlock(128, 256),
			Custom_D_GlobalBlock(256, 512),
			Custom_D_GlobalBlock(512, 1024)
		)
		self.dense = nn.Linear(1024 * 2 * 2, 1)
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		out = self.relu(self.conv(x))
		out = self.blocks(out)
		out = out.view(-1, 1024 * 2 * 2)
		out = self.dense(out)
		out = out.view(-1, 1, 1, 1)
		if(self.use_sigmoid):
			out = self.sigmoid(out)

		return out

class Custom_D_ResBlock(nn.Module):
	def __init__(self, nc, oc):
		super(Custom_D_ResBlock, self).__init__()
		self.nc = nc
		self.oc = oc
		self.conv1 = nn.Conv2d(nc, oc, 3, 1, 1)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
		self.relu = nn.LeakyReLU(0.2, inplace = True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.relu(out)
		out = self.conv2(out)
		out = out + x
		out = self.relu(out)

		return out

class Custom_D_GlobalBlock(nn.Module):
	def __init__(self, nc, oc):
		super(Custom_D_GlobalBlock, self).__init__()
		self.nc = nc
		self.oc = oc
		self.block1 = Custom_D_ResBlock(nc, nc)
		self.block2 = Custom_D_ResBlock(nc, nc)
		self.conv = nn.Conv2d(nc, oc, 4, 2, 1)
		self.relu = nn.LeakyReLU(0.2, inplace = True)

	def forward(self, x):
		out = self.block1(x)
		out = self.block2(out)
		out = self.conv(out)
		out = self.relu(out)

		return out

class Custom_G_ResBlock(nn.Module):
	def __init__(self, nc, oc):
		super(Custom_G_ResBlock, self).__init__()
		self.nc = nc
		self.oc = oc
		self.conv1 = nn.Conv2d(nc, oc, 3, 1, 1, bias = False)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1, bias = False)
		self.bn1 = nn.BatchNorm2d(oc)
		self.bn2 = nn.BatchNorm2d(oc)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = out + x

		return out

class Custom_G_UpBlock(nn.Module):
	def __init__(self, nc):
		super(Custom_G_UpBlock, self).__init__()
		self.nc = nc
		self.conv = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias = False)
		self.pixelshuffle = nn.PixelShuffle(2)
		self.bn = nn.BatchNorm2d(nc)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		# (bs, nc, n, n)
		out = self.conv(x)
		# (bs, nc * 4, n, n)
		out = self.pixelshuffle(out)
		# (bs, nc, n * 2, n * 2)
		out = self.bn(out)
		# (bs, nc, n * 2, n * 2)
		out = self.relu(out)
		# (bs, nc, n * 2, n * 2)

		return out

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, x):
		out = x.view(*self.shape)
		return out

class Wave_D(nn.Module):
	def __init__(self, nz):
		super(Wave_D, self).__init__()
		self.nz = nz
		self.linear = nn.Linear(self.nz, 512)
		#self.conv1 = nn.ConvTransposed1d(512, 256, 4)


class Wave_D(nn.Module):
	def __init__(self, sz):
		super(Wave_D, self).__init__()
		self.sz = sz
		# sz should be divided by 4^5
		self.conv1 = nn.Conv1d(1, 64, 25, 4, 12)
		self.conv2 = nn.Conv1d(64, 128, 25, 4, 12)
		self.conv3 = nn.Conv1d(128, 256, 25, 4, 12)
		self.conv4 = nn.Conv1d(256, 512, 25, 4, 12)
		self.conv5 = nn.Conv1d(512, 512, 25, 4, 12)
		self.reshape = Reshape((-1, sz // (4**5) * 512))
		self.linear = nn.Linear(sz // (4**5) * 512, 1)

	def forward(self, x):
		# (bs, nc, sz)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.reshape(out)
		out = self.linear(out)
		# (bs, 1)
		return out

class Wave_G(nn.Module):
	def __init__(self, nz, sz):
		super(Wave_G, self).__init__()
		self.nz = nz
		self.sz = sz
		# sz should be divided by 4^5
		self.conv5 = nn.ConvTranspose1d(64, 1, 25, 4, 12, output_padding = 3)
		self.conv4 = nn.ConvTranspose1d(128, 64, 25, 4, 12, output_padding = 3)
		self.conv3 = nn.ConvTranspose1d(256, 128, 25, 4, 12, output_padding = 3)
		self.conv2 = nn.ConvTranspose1d(512, 256, 25, 4, 12, output_padding = 3)
		self.conv1 = nn.ConvTranspose1d(512, 512, 25, 4, 12, output_padding = 3)
		self.reshape = Reshape((-1, 512, sz // (4**5)))
		self.linear = nn.Linear(nz, sz // (4**5) * 512)

	def forward(self, x):
		# (bs, nz, 1, 1)
		out = x.view(x.shape[0], x.shape[1])
		out = self.linear(out)
		out = self.reshape(out)
		out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		# (bs, 1, sz)

		return out
