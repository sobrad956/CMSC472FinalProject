import torch
import torchvision as tv
import torch.random
import os
import sys
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt 
import matplotlib

# Saturation can be done with torchvision.transforms.ColorJitter or torchvision.functional.adjust_saturation

# from https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop/11888449#11888449
kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}

# assumes original color temp maxed out at (255,255,255)
# expect Tensor of [..., 3 (r,g,b), H, W]
class RandomAdjustTemp(torch.nn.Module):
    def __init__(self, color=5500, p=0.5):
        super().__init__()
        self.color = color
        self.p = p
        
        if type(color) is int:
            self.color = [color]
        else:
            self.color = color
        
        if not all([c in kelvin_table for c in self.color]):
            print("Not all color temperatures are valid.")
        
    def forward(self, imgs):
        if torch.rand(1) > self.p:
            return imgs
        
        # could have just used np random but this enforces torch seeded rng
        color = self.color[torch.randint(0, len(self.color), (1,)).item()]
        
        if color in kelvin_table:
            cm = kelvin_table[color]
        else:
            cm = (0,0,0)
            
        # probably slow AF and not pythonic
        for i, img in enumerate(imgs):
            for c, channel in enumerate(img):
                for h, row in enumerate(channel):
                    for w, _ in enumerate(row):
                        imgs[i][c][h][w] *= cm[c] / 255.0
        
        return imgs
        

# Brightness can be done with torchvision.transforms.ColorJitter or torchvision.functional.adjust_brightness

# Sharpness can be done with torchvision.transforms.RandomGaussianBlur

# simple testbench
if __name__ == '__main__':
    root = os.path.expanduser(os.path.join('~', 'data'))
    dataset = CIFAR10(root=root, download=True, transform=ToTensor())
    aug = RandomAdjustTemp(color=9500, p=1)
    oim = torch.unsqueeze(dataset[0][0], 0)
    aim = aug(torch.clone(oim))
    plt.figure(0)
    plt.imshow(aim[0].permute(1,2,0))
    plt.figure(1)
    plt.imshow(oim[0].permute(1,2,0))
    plt.show()