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
class RandomRowColDropout(torch.nn.Module):
    def __init__(self, n, m, k=0.5):
        super().__init__()
        self.n = n
        self.m = m
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
            # top-left corner
            mc, mh, mw = img.shape
            
            # could be done with numpy but done for consistency with torch rng
            x = torch.randint(0, mw-self.m, (1,)).item()
            y = torch.randint(0, mh-self.n, (1,)).item()
                        
            imgs[i,:,x:x+self.n,:] = 0
            imgs[i,:,:,y:y+self.m] = 0     
        
        if singleton:
            imgs = torch.squeeze(imgs, 0)
                   
        return imgs

# simple testbench
if __name__ == '__main__':
    root = os.path.expanduser(os.path.join('~', 'data'))
    dataset = Food101(root=root, download=True, transform=ToTensor())
    aug = RandomRowColDropout(n=50, m=50, k=1)
    oim = torch.unsqueeze(dataset[0][0], 0)
    aim = aug(torch.clone(oim))
    plt.figure(0)
    plt.imshow(aim[0].permute(1,2,0))
    plt.figure(1)
    plt.imshow(oim[0].permute(1,2,0))
    plt.show()
