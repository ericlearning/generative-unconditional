import os
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.architecture_dcgan import DCGAN_D, DCGAN_G
from architectures.architecture_resnet import ResNetGan_D, ResNetGan_G
from architectures.architecture_wavegan import Wave_D, Wave_G
from trainers_advanced.trainer import Trainer
from utils import save, load

dir_name = 'data/celeba'
basic_types = None
# basic_types = 'Audio'

lr_D, lr_G, bs = 0.0002, 0.0002, 128
sz, nc, nz, ngf, ndf = 64, 3, 100, 64, 64
use_sigmoid, spectral_norm, attention_layer = False, True, 256

data = Dataset(dir_name, basic_types)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

netD = DCGAN_D(sz, nc, ndf, use_sigmoid).to(device)
netG = DCGAN_G(sz, nz, nc, ngf).to(device)

loss_type, netD, netG, device, train_dl, lr_D = 0.0002, lr_G = 0.0002, resample = False, weight_clip = None, use_graident_penalty = False, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'

# netD = Wave_D(sample_num, use_sigmoid).to(device)
# netG = Wave_G(nz, sample_num).to(device)

# netD = ResNetGan_D(sz, nc, ndf, use_sigmoid, attention_layer).to(device)
# netG = ResNetGan_G(sz, nz, nc, ngf, spectral_norm, attention_layer).to(device)

trn_dl = data.get_loader(sz, bs)
# trn_dl = data.get_loader(None, bs, audio_sample_num = sample_num)

trainer = Trainer('SGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('LSGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('HINGEGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = 0.01, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = 10, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer = Trainer('RASGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RALSGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RAHINGEGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer = Trainer('QPGAN', netD, netG, device, trn_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_graident_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
torch.save(netG.state_dict(), 'saved/cur_state_G.pth')