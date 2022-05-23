import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, img_size,latent_dim, channels ):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Base(nn.Module):
    def __init__(self, channels):
        super(Base, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
    
    def forward(self,x):
        x = self.model(x)
        out = x.view(x.shape[0], -1)
        return out
        

class Classifier(nn.Module):
    def __init__(self, base_model,img_size, classfier):
        super(Classifier, self).__init__()
        self.base_model = base_model
        ds_size = img_size // 2 ** 4
        self.fc4 = nn.Sequential(nn.Linear(128 * ds_size ** 2, classfier))
    
    def forward(self,x):
        x = F.leaky_relu(self.base_model(x))
        x = F.softmax(self.fc4(x))
        return x
        

class Discriminator(nn.Module):
    def __init__(self,base_model,img_size):
        super(Discriminator, self).__init__()
        self.base_model = base_model
        ds_size = img_size // 2 ** 4
        self.fc4 = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
    
    def forward(self,x):
        x = F.leaky_relu(self.base_model(x))
        real = F.sigmoid(self.fc4(x))
        return real
        
