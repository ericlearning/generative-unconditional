import os
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.progressive import PGGAN_D, PGGAN_G
from trainers_progressive.trainer_ralsgan_progressive import Trainer_RALSGAN_Progressive
from trainers_progressive.trainer_rahingegan_progressive import Trainer_RAHINGEGAN_Progressive
from trainers_progressive.trainer_wgan_gp_progressive import Trainer_WGAN_GP_Progressive
from utils import save, load

dir_name = 'data/celeba'
basic_types = None

lr_D, lr_G = 0.001, 0.001
sz, nc, nz = 128, 3, 256
use_sigmoid = False

data = Dataset('data/celeba')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PGGAN_D(sz, nc, use_sigmoid, False, True).to(device)
netG = PGGAN_G(sz, nz, nc, True, True).to(device)

trainer = Trainer_RAHINGEGAN_Progressive(netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300)

trainer.train([4, 8, 8, 8, 8, 8], [0.5, 0.5, 0.5, 0.5, 0.5], [16, 16, 16, 16, 16, 16])
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
