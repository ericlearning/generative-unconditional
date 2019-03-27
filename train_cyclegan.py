import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Dataset
from architectures.img2img import ResNet_G_256x256, UNet_G_256x256, UNet_G_512x512
from architectures.img2img import PatchGan_D_70x70_One_Input
from trainers_cyclegan.trainer_lsgan_cyclegan import Trainer_LSGAN_Cyclegan
from utils import save_extra, load_extra

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

train_data = Dataset(train_dir_name, basic_types = 'CycleGan', shuffle = True)
val_data = Dataset(val_dir_name, basic_types = 'CycleGan', shuffle = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

netD_A = PatchGan_D_70x70_One_Input(ic, use_sigmoid, norm_type).to(device)
netD_B = PatchGan_D_70x70_One_Input(oc, use_sigmoid, norm_type).to(device)
netG_A2B = ResNet_G_256x256(ic, oc, True, norm_type).to(device)
netG_B2A = ResNet_G_256x256(oc, ic, True, norm_type).to(device)

trn_dl = train_data.get_loader(256, bs, data_transform = dt)
val_dl = list(val_data.get_loader(256, 3, data_transform = dt))[0]
trainer = Trainer_LSGAN_Cyclegan(netD_A, netD_B, netG_A2B, netG_B2A, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, loss_interval = 150, image_interval = 300)

trainer.train(5)
save_extra('saved/cur_state.state', netD_A, netD_B, netG_A2B, netG_B2A, trainer.optimizerD_A, trainer.optimizerD_B, trainer.optimizerG)