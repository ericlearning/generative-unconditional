import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)

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

# U-Net 256x256 Generator
class UNet_G_256x256(nn.Module):
	def __init__(self, ic, oc, use_f = True, norm_type = 'batchnorm'):
		super(UNet_G_256x256, self).__init__()
		self.ic = ic
		self.oc = oc
		self.use_f = use_f
		
		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.relu = nn.ReLU(inplace = True)

		self.enc_conv1 = nn.Conv2d(ic, 64, 4, 2, 1, bias = False)
		self.enc_bn1 = None
		self.enc_conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias = False)
		self.enc_bn2 = get_norm(norm_type, 128)
		self.enc_conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
		self.enc_bn3 = get_norm(norm_type, 256)
		self.enc_conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
		self.enc_bn4 = get_norm(norm_type, 512)
		self.enc_conv5 = nn.Conv2d(512, 512, 4, 2, 1, bias = False)
		self.enc_bn5 = get_norm(norm_type, 512)
		self.enc_conv6 = nn.Conv2d(512, 512, 4, 2, 1, bias = False)
		self.enc_bn6 = get_norm(norm_type, 512)
		self.enc_conv7 = nn.Conv2d(512, 512, 4, 2, 1, bias = False)
		self.enc_bn7 = get_norm(norm_type, 512)
		self.enc_conv8 = nn.Conv2d(512, 512, 4, 2, 1, bias = False)

		self.dec_conv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias = False)
		self.dec_bn1 = get_norm(norm_type, 512)
		self.dec_conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
		self.dec_bn2 = get_norm(norm_type, 512)
		self.dec_conv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
		self.dec_bn3 = get_norm(norm_type, 512)
		self.dec_conv4 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
		self.dec_bn4 = get_norm(norm_type, 512)
		self.dec_conv5 = nn.ConvTranspose2d(1024, 256, 4, 2, 1, bias = False)
		self.dec_bn5 = get_norm(norm_type, 256)
		self.dec_conv6 = nn.ConvTranspose2d(512, 128, 4, 2, 1, bias = False)
		self.dec_bn6 = get_norm(norm_type, 128)
		self.dec_conv7 = nn.ConvTranspose2d(256, 64, 4, 2, 1, bias = False)
		self.dec_bn7 = get_norm(norm_type, 64)
		self.dec_conv8 = nn.ConvTranspose2d(128, oc, 4, 2, 1, bias = False)

		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
	
	def forward(self, x):
		# (bs, nc, 256, 256)
		en1 = self.enc_conv1(x)
		# (bs, 64, 128, 128)
		en2 = self.enc_bn2(self.enc_conv2(self.leaky_relu(en1)))
		# (bs, 128, 64, 64)
		en3 = self.enc_bn3(self.enc_conv3(self.leaky_relu(en2)))
		# (bs, 256, 32, 32)
		en4 = self.enc_bn4(self.enc_conv4(self.leaky_relu(en3)))
		# (bs, 512, 16, 16)
		en5 = self.enc_bn5(self.enc_conv5(self.leaky_relu(en4)))
		# (bs, 512, 8, 8)
		en6 = self.enc_bn6(self.enc_conv6(self.leaky_relu(en5)))
		# (bs, 512, 4, 4)
		en7 = self.enc_bn7(self.enc_conv7(self.leaky_relu(en6)))
		# (bs, 512, 2, 2)
		en8 = self.enc_conv8(self.leaky_relu(en7))
		# (bs, 512, 1, 1)
		if(self.use_f):
			de8 = F.dropout(self.dec_bn1(self.dec_conv1(self.relu(en8))))
			# (bs, 512, 2, 2)
			de7 = F.dropout(self.dec_bn2(self.dec_conv2(self.relu(torch.cat([de8, en7], 1)))))
			# (bs, 512, 4, 4)
			de6 = F.dropout(self.dec_bn3(self.dec_conv3(self.relu(torch.cat([de7, en6], 1)))))
			# (bs, 512, 8, 8)
		else:
			de8 = self.dropout(self.dec_bn1(self.dec_conv1(self.relu(en8))))
			# (bs, 512, 2, 2)
			de7 = self.dropout(self.dec_bn2(self.dec_conv2(self.relu(torch.cat([de8, en7], 1)))))
			# (bs, 512, 4, 4)
			de6 = self.dropout(self.dec_bn3(self.dec_conv3(self.relu(torch.cat([de7, en6], 1)))))
			# (bs, 512, 8, 8)
		de5 = self.dec_bn4(self.dec_conv4(self.relu(torch.cat([de6, en5], 1))))
		# (bs, 512, 16, 16)
		de4 = self.dec_bn5(self.dec_conv5(self.relu(torch.cat([de5, en4], 1))))
		# (bs, 256, 32, 32)
		de3 = self.dec_bn6(self.dec_conv6(self.relu(torch.cat([de4, en3], 1))))
		# (bs, 128, 64, 64)
		de2 = self.dec_bn7(self.dec_conv7(self.relu(torch.cat([de3, en2], 1))))
		# (bs, 64, 128, 128)
		de1 = self.dec_conv8(self.relu(torch.cat([de2, en1], 1)))
		# (bs, 3, 256, 256)
		out = self.tanh(de1)

		del en1, en2, en3, en4, en5, en6, en7, en8, de8, de7, de6, de5, de4, de3, de2, de1

		return out


class PatchGan_D_70x70(nn.Module):
	def __init__(self, ic_1, ic_2, use_sigmoid = True, norm_type = 'batchnorm'):
		super(PatchGan_D_70x70, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.conv1 = nn.Conv2d(self.ic_1 + self.ic_2, 64, 4, 2, 1, bias = False)
		self.bn1 = None
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias = False)
		self.bn2 = get_norm(norm_type, 128)
		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
		self.bn3 = get_norm(norm_type, 256)
		self.conv4 = nn.Conv2d(256, 512, 4, 1, 1, bias = False)
		self.bn4 = get_norm(norm_type, 512)
		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x1, x2):
		out = torch.cat([x1, x2], 1)
		# (bs, ic_1+ic_2, 256, 256)
		out = self.leaky_relu(self.conv1(out))
		# (bs, 64, 128, 128)
		out = self.leaky_relu(self.bn2(self.conv2(out)))
		# (bs, 128, 64, 64)
		out = self.leaky_relu(self.bn3(self.conv3(out)))
		# (bs, 256, 32, 32)
		out = self.leaky_relu(self.bn4(self.conv4(out)))
		# (bs, 512, 31, 31)
		out = self.conv5(out)
		# (bs, 512, 30, 30)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)

		return out


class PatchGan_D_286x286(nn.Module):
	def __init__(self, ic_1, ic_2, use_sigmoid = True, norm_type = 'batchnorm'):
		super(PatchGan_D_286x286, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.conv1 = nn.Conv2d(self.ic_1 + self.ic_2, 64, 4, 2, 1, bias = False)
		self.bn1 = None
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias = False)
		self.bn2 = get_norm(norm_type, 128)
		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
		self.bn3 = get_norm(norm_type, 256)
		self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
		self.bn4 = get_norm(norm_type, 256)
		self.conv5 = nn.Conv2d(512, 512, 4, 2, 1, bias = False)
		self.bn5 = get_norm(norm_type, 256)
		self.conv6 = nn.Conv2d(512, 512, 4, 1, 1, bias = False)
		self.bn6 = get_norm(norm_type, 512)
		self.conv7 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x1, x2):
		out = torch.cat([x1, x2], 1)
		# (bs, ic_1+ic_2, 256, 256)
		out = self.leaky_relu(self.conv1(out))
		# (bs, 64, 128, 128)
		out = self.leaky_relu(self.bn2(self.conv2(out)))
		# (bs, 128, 64, 64)
		out = self.leaky_relu(self.bn3(self.conv3(out)))
		# (bs, 256, 32, 32)
		out = self.leaky_relu(self.bn4(self.conv4(out)))
		# (bs, 256, 16, 16)
		out = self.leaky_relu(self.bn5(self.conv5(out)))
		# (bs, 256, 8, 8)
		out = self.leaky_relu(self.bn6(self.conv6(out)))
		# (bs, 512, 7, 7)
		out = self.conv7(out)
		# (bs, 512, 6, 6)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)

		return out


def receptive_calculator(input_size, ks, stride, pad):
	return int((input_size - ks + 2 * pad) / stride + 1)

def inverse_receptive_calculator(output_size, ks, stride, pad):
	return ((output_size - 1) * stride) + ks

