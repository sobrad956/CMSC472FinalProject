import torch
import torch.random
import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

# assumes original color temp maxed out at (255,255,255)
# expect Tensor of [..., 3 (r,g,b), H, W]
class RandomDropout(torch.nn.Module):
    def __init__(self, s=2, i=1, k=0.5):
        super().__init__()
        self.s = s
        self.i = i
        self.k = k
        
    def forward(self, imgs):
        if torch.rand(1) > self.k:
            return imgs
                    
        # probably slow AF and not pythonic
        for i, img in enumerate(imgs):
            # top-left corner
            mc, mh, mw = img.shape
            
            # could be done with numpy but done for consistency with torch rng
            xanch = torch.randint(0, mw-self.s+1, (self.i,))
            yanch = torch.randint(0, mh-self.s+1, (self.i,))

            for x, y in zip(xanch, yanch):
                x = x.item()
                y = y.item()
                imgs[i,:,y:y+self.s,x:x+self.s] = 0
        return imgs

# simple testbench
if __name__ == '__main__':
    root = os.path.expanduser(os.path.join('~', 'data'))
    dataset = CIFAR10(root=root, download=True, transform=ToTensor())
    aug = RandomDropout(k=1, i=5, s=5)
    oim = torch.unsqueeze(dataset[0][0], 0)
    aim = aug(torch.clone(oim))
    plt.figure(0)
    plt.imshow(aim[0].permute(1,2,0))
    plt.figure(1)
    plt.imshow(oim[0].permute(1,2,0))
    plt.show()