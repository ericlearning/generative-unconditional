import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Dataset
from architectures.img2img import ResNet_G_256x256, UNet_G_256x256, UNet_G_512x512
from architectures.img2img import PatchGan_D_70x70
from trainers_pix2pix.trainer_rahingegan_pix2pix import Trainer_RAHINGEGAN_Pix2Pix
from trainers_pix2pix.trainer_lsgan_pix2pix import Trainer_LSGAN_Pix2Pix
from trainers_pix2pix.trainer_sgan_pix2pix import Trainer_Pix2Pix
from utils import save, load

train_dir_name = ['data/file/train/input', 'data/file/train/target']
val_dir_name = ['data/file/val/input', 'data/file/val/target']

lr_D, lr_G, bs = 0.0002, 0.0002, 128
ic, oc, use_sigmoid = 3, 3, False
norm_type = 'batchnorm'

dt = {
	'input' : transforms.Compose([
		transforms.Resize(256),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	]),
	'target' : transforms.Compose([
		transforms.Resize(256),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])
}

train_data = Dataset(train_dir_name, basic_types = 'Pix2Pix', shuffle = True)
val_data = Dataset(val_dir_name, basic_types = 'Pix2Pix', shuffle = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type).to(device)
netG = UNet_G_256x256(ic, oc, True, norm_type).to(device)

trn_dl = train_data.get_loader(256, bs, data_transform = dt)
val_dl = list(val_data.get_loader(256, 3, data_transform = dt))[0]
trainer = Trainer_LSGAN_Pix2Pix(netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300)

trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)