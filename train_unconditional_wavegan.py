import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from dataset import Dataset
from architectures_experimental.unconditional import Wave_D, Wave_G
from trainers.trainer_sgan import Trainer_SGAN
from trainers.trainer_wgan import Trainer_WGAN
from trainers.trainer_wgan_gp import Trainer_WGAN_GP
from trainers.trainer_lsgan import Trainer_LSGAN
from trainers.trainer_rasgan import Trainer_RASGAN
from trainers.trainer_ralsgan import Trainer_RALSGAN
from trainers.trainer_rahingegan import Trainer_RAHINGEGAN
from trainers.trainer_hingegan import Trainer_HINGEGAN
from utils import save, load

dir_name = 'data/audios/full_data'
basic_types = 'Audio'

lr_D, lr_G, bs = 0.0002, 0.0002, 128
nz, use_sigmoid, sample_num = 100, False, 65536

data_transform = None

data = Dataset(dir_name, basic_types)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = Wave_D(sample_num, use_sigmoid).to(device)
netG = Wave_G(nz, sample_num).to(device)

trn_dl = data.get_loader(None, bs, audio_sample_num = sample_num)
trainer = Trainer_RAHINGEGAN(netD, netG, device, trn_dl, lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300)

trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
