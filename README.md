# Generative-learning
A python library for generative learning methods with PyTorch.

## List of Trainers Available
This repo has PyTorch implementations of training a Gan.
- Trainers for Unconditional Gans
  - sgan
  - lsgan
  - hingegan
  - wgan
  - wgan-gp
  - qpgan
  - rasgan
  - ralsgan
  - rahingegan
- Trainers for Conditional Gans
  - sgan
  - lsgan
- Trainers for CycleGan
  - lsgan
- Trainers for Pix2Pix
  - sgan
  - lsgan
  - ralsgan
  - rahingegan
- Trainers for Pix2PixHD
  - ralsgan
- Trainers for Progressive Gans
  - rasgan
  - ralsgan
  - rahingegan
  - wgan-gp
  
## List of Architectures Available
- Gans
 - Deconv DCGAN
 - PixelShuffle DCGAN
 - ResizeConv DCGAN
- Conditional Gans
 - Deconv DCGAN (with label)
- Img2Img (Pix2Pix, Pix2PixHD, CycleGan)
 - U-Net (Pix2Pix)
 - ResNetGan (Pix2Pix)
 - PatchGAN (70x70)
 - PatchGAN (286x286)
- Progressive Gan
 - Progressive Architecture
 
Gans, Conditional Gans, Img2Img mostly has options to chose normalization type, such as batchnorm, instancenorm, spectralnorm.

## TODO
- [x] Add Spectral Normalization
- [x] Implement Progressive Gan
- [x] Implement Pix2PixHD
- [ ] Add trained results
- [ ] make the losses into a function and put them in a single file (code cleaning)
- [ ] Implement BicycleGan (multimodal Pix2Pix)
- [ ] Improve README
- [ ] Implement StyleGAN
