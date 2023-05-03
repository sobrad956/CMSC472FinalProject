import torch
import torchvision as tv
import torch.random
import os
import sys
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt 
import matplotlib

# assumes original color temp maxed out at (255,255,255)
# expect Tensor of [..., 3 (r,g,b), H, W]
class RandomSaltPepper(torch.nn.Module):
    def __init__(self, p=0.5, type='salt'):
        super().__init__()
        self.p = p
        self.salt = type in ['salt', 'both']
        self.pepper = type in ['pepper', 'both']
        self.color = []
        if self.salt:
            self.color.append(1)
        if self.pepper:
            self.color.append(0)
    
    def forward(self, imgs):
        if not len(self.color):
            return imgs
        
        singleton = False
        if imgs.dim() == 3:
            imgs = torch.unsqueeze(imgs, 0)
            singleton = True
        
        ni, ch, nx, ny = imgs.shape
    
        if self.salt:
            salt_mask = torch.rand((ni, nx, ny))
            salt_mask = (salt_mask < self.p).long()
            salt_mask = torch.unsqueeze(salt_mask, 1).repeat(1, ch, 1, 1)
            imgs *= 1 - salt_mask # remove things that are salt
            imgs += salt_mask # add salt
        
        if self.pepper:
            pepper_mask = torch.rand((ni, nx, ny))
            pepper_mask = (pepper_mask < self.p).long()
            pepper_mask = torch.unsqueeze(pepper_mask, 1).repeat(1, ch, 1, 1)
            imgs *= 1 - pepper_mask
        
        if singleton:
            imgs = torch.squeeze(imgs, 0)
        return imgs

class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0, var=1):
        super().__init__()
        self.mean = mean
        self.var = var
        self.sigma = var ** 0.5
        
    def forward(self, imgs):
        noise = torch.normal(self.mean, self.sigma, imgs.shape)
        imgs += noise
        imgs = torch.clamp(imgs, 0, 1)
        return imgs

# simple testbench
if __name__ == '__main__':
    root = os.path.expanduser(os.path.join('~', 'data'))
    dataset = Food101(root=root, download=True, transform=ToTensor())
    aug = RandomSaltPepper(p=0.1, type='both')
    aug2 = RandomGaussianNoise(mean=0,var=1./255.)
    oim = torch.unsqueeze(dataset[0][0], 0)
    aim = aug(torch.clone(oim))
    aim2 = aug2(torch.clone(oim))
    plt.figure(0)
    plt.imshow(aim[0].permute(1,2,0))
    plt.figure(1)
    plt.imshow(aim2[0].permute(1,2,0))
    plt.figure(2)
    plt.imshow(oim[0].permute(1,2,0))
    plt.show()
