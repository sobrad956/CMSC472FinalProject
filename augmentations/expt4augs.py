import torch
import torch.random
import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

# assumes original color temp maxed out at (255,255,255)
# expect Tensor of [..., 3 (r,g,b), H, W]
class RandomRegionSwap(torch.nn.Module):
    def __init__(self, s=2, i=2, k=0.5):
        super().__init__()
        self.s = s
        self.i = i
        self.k = k
        
        
    def forward(self, imgs):
        if torch.rand(1) > self.k or self.i < 2:
            return imgs
                    
        # probably slow AF and not pythonic
        for i, img in enumerate(imgs):
            # top-left corner
            mc, mh, mw = img.shape
            
            # could be done with numpy but done for consistency with torch rng
            xanch = torch.randint(0, mw-self.s+1, (self.i,))
            yanch = torch.randint(0, mh-self.s+1, (self.i,))

            xanch = [x.item() for x in xanch]
            yanch = [y.item() for y in yanch]

            crops = [torch.clone(imgs[i,:,y:y+self.s,x:x+self.s]) for x, y in zip(xanch, yanch)]
            
            # bit of a hack to generate swap locations & prevent a crop from being placed back
            # where it originally was. Could have just done random.choices but this
            # make this more consistent with torch.rand 
            new_inds = []
            for c in range(len(crops)):
                choices = list(range(len(crops)))
                choices.remove(c)
                new_inds.append(choices[torch.randint(0, len(choices), (1,)).item()])
            
            for x, y, c in zip(xanch, yanch, new_inds):
                imgs[i,:,y:y+self.s,x:x+self.s] = crops[c]
                
        return imgs

# simple testbench
if __name__ == '__main__':
    root = os.path.expanduser(os.path.join('~', 'data'))
    dataset = CIFAR10(root=root, download=True, transform=ToTensor())
    aug = RandomRegionSwap(k=1, i=3, s=10)
    oim = torch.unsqueeze(dataset[0][0], 0)
    aim = aug(torch.clone(oim))
    plt.figure(0)
    plt.imshow(aim[0].permute(1,2,0))
    plt.figure(1)
    plt.imshow(oim[0].permute(1,2,0))
    plt.show()