import os
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.conditional import Conditional_DCGAN_D, Conditional_DCGAN_G
from trainers_conditional.trainer_sgan_conditional import Trainer_SGAN_C
from trainers_conditional.trainer_lsgan_conditional import Trainer_LSGAN_C
from utils import save, load

dir_name = 'data/mnist'
basic_types = 'MNIST'

lr_D, lr_G, bs = 0.0002, 0.0002, 128
sz, nc, nz, n_classes, ngf, ndf = 64, 3, 100, 10, 64, 64
use_sigmoid, spectral_norm, attention_layer = False, True, 256

data = Dataset(dir_name, basic_types)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = Conditional_DCGAN_D(sz, nc, n_classes, ndf, use_sigmoid).to(device)
netG = Conditional_DCGAN_G(sz, nz, nc, n_classes, ngf).to(device)

trn_dl = data.get_loader(sz, bs)
trainer = Trainer_LSGAN_C(netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300)

trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
