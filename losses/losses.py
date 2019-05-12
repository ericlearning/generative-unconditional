import torch
import torch.nn as nn
import numpy as np

def get_label(bs):
	label_r = torch.full((bs, ), 1, device = self.device)
	label_f = torch.full((bs, ), 0, device = self.device)
	return label_r, label_f

class SGAN():
	def __init__(self, device):
		self.criterion = nn.BCELoss()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return criterion(c_xr, label_r) + criterion(c_xf, label_f)

	def g_loss(self, c_xf):
		bs = c_xf.shape[0]
		_, label_f = get_label(bs)
		return criterion(c_xf, label_f)

class LSGAN():
	def __init__(self, device):
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return torch.mean((c_xr - label_r) ** 2) + torch.mean((c_xf - label_f) ** 2)

	def g_loss(self, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return torch.mean((c_xf - label_r) ** 2)

class HINGEGAN():
	def __init__(self, device):
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		return torch.mean(torch.nn.ReLU()(1-c_xr)) + torch.mean(torch.nn.ReLU()(1+c_xf))

	def g_loss(self, c_xf):
		return -torch.mean(c_xf)

class WGAN():
	def __init__(self, device):
		self.device = device

	def d_loss(self, c_xr, c_xf):
		return -torch.mean(c_xr) + torch.mean(c_xf)

	def g_loss(self, c_xf):
		return -torch.mean(c_xf)

class RASGAN():
	def __init__(self, device):
		self.device = device
		self.criterion = nn.BCEWithLogitsLoss()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return (self.criterion(c_xr - torch.mean(c_xf), label_r) + self.criterion(c_xf - torch.mean(c_xr), label_f)) / 2.0

	def g_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs)
		return (self.criterion(c_xr - torch.mean(c_xf), label_f) + self.criterion(c_xf - torch.mean(c_xr), label_r)) / 2.0

class RALSGAN():
	def __init__(self, device):
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return (torch.mean((c_xr - torch.mean(c_xf) - label_r)**2) + torch.mean((c_xf - torch.mean(c_xr) + label_r)**2)) / 2.0

	def g_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs)
		return (torch.mean((c_xf - torch.mean(c_xr) - label_r)**2) + torch.mean((c_xr - torch.mean(c_xf) + label_r)**2)) / 2.0

class RAHINGEGAN():
	def __init__(self, device):
		self.device = device

	def d_loss(self, c_xr, c_xf):
		return (torch.mean(torch.nn.ReLU()(1-(c_xr-torch.mean(c_xf)))) + torch.mean(torch.nn.ReLU()(1+(c_xf-torch.mean(c_xr))))) / 2.0

	def g_loss(self, c_xr, c_xf):
		return (torch.mean(torch.nn.ReLU()(1-(c_xf-torch.mean(c_xr)))) + torch.mean(torch.nn.ReLU()(1+(c_xr-torch.mean(c_xf))))) / 2.0

class QPGAN():
	def __init__(self, device, norm_type = 'L1'):
		self.device = device
		self.norm_type = norm_type

	def d_loss(self, c_xr, c_xf, real_images, fake_images):
		if(self.norm_type == 'L1'):
			denominator = (real_images - fake_images).abs().mean() * 10 * 2
		if(self.norm_type == 'L2'):
			denominator = (real_images - fake_images).mean().sqrt() * 10 * 2

		errD_1 = torch.mean(c_xr) - torch.mean(c_xf)
		errD_2 = (errD_1 ** 2) / denominator
		return errD_2 - errD_1

	def g_loss(self, c_xr, c_xf):
		return torch.mean(c_xr) - torch.mean(c_xf)