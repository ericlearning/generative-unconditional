import os
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.unconditional import DCGAN_D, DCGAN_G
from architectures.unconditional import ResNetGan_D, ResNetGan_G
from trainers.trainer_sgan import Trainer_SGAN
from trainers.trainer_wgan import Trainer_WGAN
from trainers.trainer_wgan_gp import Trainer_WGAN_GP
from trainers.trainer_lsgan import Trainer_LSGAN
from trainers.trainer_rasgan import Trainer_RASGAN
from trainers.trainer_ralsgan import Trainer_RALSGAN
from trainers.trainer_rahingegan import Trainer_RAHINGEGAN
from trainers.trainer_hingegan import Trainer_HINGEGAN
from utils import save, load

dir_name = 'data/celeba'
basic_types = None

lr_D, lr_G, bs = 0.0002, 0.0002, 128
sz, nc, nz, ngf, ndf = 64, 3, 100, 64, 64
use_sigmoid, spectral_norm, attention_layer = False, True, 256

data = Dataset(dir_name, basic_types)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = DCGAN_D(sz, nc, ndf, use_sigmoid).to(device)
netG = DCGAN_G(sz, nz, nc, ngf).to(device)
# netD = ResNetGan_D(sz, nc, ndf, use_sigmoid, attention_layer).to(device)
# netG = ResNetGan_G(sz, nz, nc, ngf, spectral_norm, attention_layer).to(device)

trn_dl = data.get_loader(sz, bs)
trainer = Trainer_RAHINGEGAN(netD, netG, device, trn_dl, lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300)

trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
