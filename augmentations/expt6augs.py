import torch
import torch.random
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

# assumes original color temp maxed out at (255,255,255)
# expect Tensor of [..., 3 (r,g,b), H, W]
class RandomApplyOne(torch.nn.Module):
    def __init__(self, k=0.5, augs=None):
        super().__init__()
        self.augs = augs
        self.k = k
        
        
    def forward(self, imgs):
        if torch.rand(1) > self.k:
            return imgs

        singleton = False
        if imgs.dim() == 3:
            imgs = torch.unsqueeze(imgs, 0)
            singleton = True
                    
        # probably slow AF and not pythonic
        for i, img in enumerate(imgs):
            if torch.rand(1) <= self.k:
                aug = torch.randint(0,len(self.augs),(1,)).item()
                imgs[i] = self.augs[aug](torch.unsqueeze(img, 0))            

        if singleton:
            imgs = torch.squeeze(imgs, 0)
                          
        return imgs

# simple testbench
if __name__ == '__main__':
    root = os.path.expanduser(os.path.join('~', 'data'))
    dataset = Food101(root=root, download=True, transform=ToTensor())
    augs = [
        transforms.RandomRotation(180), 
        transforms.RandomResizedCrop((512,512),antialias=True),
        transforms.RandomAffine(0, translate=(0.20,0.20)),
        transforms.RandomAffine(0, shear=(20,20))
    ]
    aug = RandomApplyOne(k=1, augs=augs)
    oim = torch.unsqueeze(dataset[0][0], 0)
    plt.figure(0)
    plt.imshow(oim[0].permute(1,2,0))
    for i in range(10):
        aim = aug(torch.clone(oim))
        plt.figure(i + 1)
        plt.imshow(aim[0].permute(1,2,0))
        plt.show()
