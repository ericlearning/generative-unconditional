import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, x):
		out = x.view(*self.shape)
		return out

class WaveResBlock(nn.Module):
	def __init__(self, ic, oc, last_act = 'leakyrelu', resolution_type = 'upsample'):
		super(WaveResBlock, self).__init__()
		self.last_act = last_act
		if(resolution_type == 'upsample'):
			self.resolution = UpSample1D()
		elif(resolution_type == 'downsample'):
			self.resolution = DownSample1D()
		self.conv1 = nn.Conv1d(ic, ic, 25, 1, 12)
		self.conv2 = nn.Conv1d(ic, ic, 25, 1, 12)
		self.conv3 = nn.Conv1d(ic, oc, 25, 1, 12)
		self.norm1 = nn.InstanceNorm1d(ic)
		self.norm2 = nn.InstanceNorm1d(ic)

		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.relu = nn.LeakyReLU(0.2, inplace = True)

	def forward(self, x):
		out = self.relu(self.norm1(self.conv1(x)))
		out = self.relu(self.norm2(self.conv2(out)))
		out = self.resolution(out + x)
		if(self.last_act == 'sigmoid'):
			out = self.sigmoid(self.conv3(out))
		elif(self.last_act == 'tanh'):
			out = self.tanh(self.conv3(out))
		elif(self.last_act == 'leakyrelu'):
			out = self.relu(self.conv3(out))
		return out

class UpSample1D(nn.Module):
	def __init__(self):
		super(UpSample1D, self).__init__()
		self.scale_factor = 4

	def forward(self, x):
		return F.interpolate(x, None, self.scale_factor, 'linear', align_corners = True)

class DownSample1D(nn.Module):
	def __init__(self):
		super(DownSample1D, self).__init__()
		self.scale_factor = 4

	def forward(self, x):
		return F.avg_pool1d(x, self.scale_factor)

class Wave_G_2(nn.Module):
	def __init__(self, nz, sz):
		super(Wave_G_2, self).__init__()
		self.nz = nz
		self.sz = sz

		self.block1 = WaveResBlock(512, 512)
		self.block2 = WaveResBlock(512, 512)
		self.block3 = WaveResBlock(512, 256)
		self.block4 = WaveResBlock(256, 128)
		self.block5 = WaveResBlock(128, 64)
		self.block6 = WaveResBlock(64, 1, last_act = 'tanh')
		self.reshape = Reshape((-1, 512, sz // (4**6)))
		self.linear = nn.Linear(nz, sz // (4**6) * 512)
		self.act = nn.LeakyReLU(0.2)

		for m in self.modules():
			if(isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()


	def forward(self, x):
		# (bs, nz, 1, 1)
		out = x.view(x.shape[0], x.shape[1])
		out = self.act(self.linear(out))
		out = self.reshape(out)
		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.block4(out)
		out = self.block5(out)
		out = self.block6(out)
		return out

class Wave_D_2(nn.Module):
	def __init__(self, sz, use_sigmoid):
		super(Wave_D_2, self).__init__()
		self.sz = sz
		self.use_sigmoid = use_sigmoid

		self.block1 = WaveResBlock(1, 64, resolution_type = 'downsample')
		self.block2 = WaveResBlock(64, 128, resolution_type = 'downsample')
		self.block3 = WaveResBlock(128, 256, resolution_type = 'downsample')
		self.block4 = WaveResBlock(256, 512, resolution_type = 'downsample')
		self.block5 = WaveResBlock(512, 512, resolution_type = 'downsample')
		self.block6 = WaveResBlock(512, 512, resolution_type = 'downsample')
		self.reshape = Reshape((-1, sz // (4**6) * 512))
		self.linear = nn.Linear(sz // (4**6) * 512, 1)
		self.act = nn.Sigmoid()

		for m in self.modules():
			if(isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()


	def forward(self, x):
		# (bs, nz, 1, 1)
		out = self.block1(x)
		out = self.block2(out)
		out = self.block3(out)
		out = self.block4(out)
		out = self.block5(out)
		out = self.block6(out)
		out = self.reshape(out)
		out = self.linear(out)
		if(self.use_sigmoid):
			out = self.sigmoid(out)
		return out

class Wave_D(nn.Module):
	def __init__(self, sz, use_sigmoid):
		super(Wave_D, self).__init__()
		self.sz = sz
		self.use_sigmoid = use_sigmoid
		# sz should be divided by 4^7
		self.conv1 = nn.Conv1d(1, 64, 25, 4, 12)
		self.conv2 = nn.Conv1d(64, 128, 25, 4, 12)
		self.conv3 = nn.Conv1d(128, 256, 25, 4, 12)
		self.conv4 = nn.Conv1d(256, 512, 25, 4, 12)
		self.conv5 = nn.Conv1d(512, 512, 25, 4, 12)
		self.conv6 = nn.Conv1d(512, 512, 25, 4, 12)
		self.reshape = Reshape((-1, sz // (4**6) * 512))
		self.linear = nn.Linear(sz // (4**6) * 512, 1)
		self.sigmoid = nn.Sigmoid()
		self.act = nn.LeakyReLU(0.2)

		for m in self.modules():
			if(isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		# (bs, nc, sz)
		out = self.act(self.conv1(x))
		out = self.act(self.conv2(out))
		out = self.act(self.conv3(out))
		out = self.act(self.conv4(out))
		out = self.act(self.conv5(out))
		out = self.act(self.conv6(out))
		out = self.reshape(out)
		out = self.linear(out)
		if(self.use_sigmoid):
			out = self.sigmoid(out)
		# (bs, 1)
		return out

class Wave_G(nn.Module):
	def __init__(self, nz, sz):
		super(Wave_G, self).__init__()
		self.nz = nz
		self.sz = sz
		# sz should be divided by 4^5
		self.conv1 = nn.ConvTranspose1d(512, 512, 25, 4, 12, output_padding = 3)
		self.conv2 = nn.ConvTranspose1d(512, 512, 25, 4, 12, output_padding = 3)
		self.conv3 = nn.ConvTranspose1d(512, 256, 25, 4, 12, output_padding = 3)
		self.conv4 = nn.ConvTranspose1d(256, 128, 25, 4, 12, output_padding = 3)
		self.conv5 = nn.ConvTranspose1d(128, 64, 25, 4, 12, output_padding = 3)
		self.conv6 = nn.ConvTranspose1d(64, 1, 25, 4, 12, output_padding = 3)
		self.reshape = Reshape((-1, 512, sz // (4**6)))
		self.linear = nn.Linear(nz, sz // (4**6) * 512)
		self.tanh = nn.Tanh()
		self.act = nn.LeakyReLU(0.2)

		for m in self.modules():
			if(isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		# (bs, nz, 1, 1)
		out = x.view(x.shape[0], x.shape[1])
		out = self.act(self.linear(out))
		out = self.reshape(out)
		out = self.act(self.conv1(out))
		out = self.act(self.conv2(out))
		out = self.act(self.conv3(out))
		out = self.act(self.conv4(out))
		out = self.act(self.conv5(out))
		out = self.conv6(out)
		out = self.tanh(out)
		# (bs, 1, sz)

		return out