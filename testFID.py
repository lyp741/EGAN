 
import argparse
import os
import numpy as np
import math
from sympy import viete
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from DCGAN_model import Generator, Discriminator
import ray
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter

import shutil
import pytorch_fid.fid_score
from inception_score import inception_score

cuda = True if torch.cuda.is_available() else False
# cuda = False
device = torch.device("cuda:0" if cuda else "cpu")

pop_size= 1
fifth = 7
num_workers = pop_size*fifth
classfier = pop_size + 1
PATH = 'Model.pt'

# generator = Generator(opt.img_size, opt.latent_dim, opt.channels)
# sd = torch.load(PATH)['generator']
# generator.load_state_dict(sd)

def test_fid_is(generator):
    generator.eval()
    Tensor = torch.FloatTensor

    if cuda:
        generator.cuda()

    z = Variable(Tensor(np.random.normal(0, 1, (1000, 100)))).to(device)
    # z = Variable(torch.randn(256, opt.latent_dim, 1, 1)).to(device=device)

    gen_imgs = generator(z)
    rgb_imgs = np.concatenate((gen_imgs.cpu().data,)*3, axis=1)
    for i in range(gen_imgs.shape[0]):
        save_image(gen_imgs[i].data, "fake_images_EGAN_softmax/%d.png" % (i+1), nrow=1, normalize=True)

    # dataloader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(
    #         "data/fashion_mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor()]),

    #     ),
    #     # batch_size=opt.batch_size,
    #     # shuffle=True,
    #     # drop_last = True
    # )
    # for idx, (i, c) in enumerate(dataloader):
    #     save_image(i.view(1, opt.img_size, opt.img_size), "fashion_imgs/%d.png" % (idx+1), nrow=1, normalize=True)
    #     if idx == 1000:
    #         break
    is_score = inception_score(rgb_imgs, cuda=True, batch_size=32, resize=True, splits=1)
    print('Inception Score: ', is_score)

    fid = pytorch_fid.fid_score.calculate_fid_given_paths(['mnist_imgs', 'fake_images_EGAN_softmax'], 50, device, 2048, 8)
    print('FID: ', fid)
    generator.train()
    return fid, is_score

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)
    generator = Generator(opt.img_size, opt.latent_dim, opt.channels)
    sd = torch.load(PATH)['generator']
    generator.load_state_dict(sd)
    test_fid_is(generator)