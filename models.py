import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import sys
import math
import argparse
from torchvision.utils import save_image
import os
import shutil
import numpy as np
from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Linear, LogSoftmax
from sklearn.metrics import f1_score 
import matplotlib.pyplot as plt
import pickle

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out
    
class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    # MODELLED AFTER https://github.com/gsp-27/pytorch_Squeezenet
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 101, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        return x

    def fire_layer(inp, s, e):
        f = fire(inp, s, e)
        return f

    def squeezenet(pretrained=False):
        net = SqueezeNet()
        # inp = Variable(torch.randn(64,3,32,32))
        # out = net.forward(inp)
        # print(out.size())
        return net
    
class GoogLeNetV3(Module):
    # MODELLED AFTER https://github.com/Moeo3/GoogLeNet-Inception-V3-pytorch
    def __init__(self, channels_in):
        super(GoogLeNetV3, self).__init__()
        self.in_block = Sequential(
            Conv2d_BN(channels_in, 32, 3, stride=2, padding=1),  # size /= 2
            Conv2d_BN(32, 32, 3, stride=1, padding=1),
            Conv2d_BN(32, 64, 3, stride=1, padding=1),
            MaxPool2d(3, stride=2, padding=1),  # size /= 2
            Conv2d_BN(64, 80, 1, stride=1, padding=0),
            Conv2d_BN(80, 192, 3, stride=1, padding=1),
            MaxPool2d(3, stride=2, padding=1)  # size /= 2
        )  # 192 channels
        self.mix_block = Sequential(
            InceptionA(192, 32),
            InceptionA(256, 64),
            InceptionA(288, 64),
            InceptionB(288),  # size /= 2
            InceptionC(768, 128),
            InceptionC(768, 160),
            InceptionC(768, 160),
            InceptionC(768, 192),
            InceptionD(768),  # size /= 2
            InceptionE(1280),
            InceptionE(2048)
        )  # 2048 channels
        self.out_block = Sequential(
            Conv2d_BN(2048, 1024, 1, stride=1, padding=0),
            AdaptiveAvgPool2d(1)
        )  # 1024 channels
        self.full_connect = Linear(1024, 101)
        self.softmax = LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.in_block(x)
        x = self.mix_block(x)
        x = self.out_block(x)
        x = torch.flatten(x, 1)
        x = self.full_connect(x)
        return self.softmax(x)

class Conv2d_BN(Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride=1, acti=LeakyReLU(0.2, inplace=True)):
        super(Conv2d_BN, self).__init__()
        self.conv2d_bn = Sequential(
            Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
            BatchNorm2d(channels_out),
            acti
        )

    def forward(self, x):
        return self.conv2d_bn(x)

class InceptionA(Module):
    def __init__(self, channels_in, pool_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 64, 1, stride=1, padding=0)  # 64 channels
        self.branch5x5 = Sequential(
            Conv2d_BN(channels_in, 48, 1, stride=1, padding=0),
            Conv2d_BN(48, 64, 5, stride=1, padding=2)
        )  # 64 channels
        self.branch3x3dbl = Sequential(
            Conv2d_BN(channels_in, 64, 1, stride=1, padding=0),
            Conv2d_BN(64, 96, 3, stride=1, padding=1),
            Conv2d_BN(96, 96, 3, stride=1, padding=1)
        )  # 96 channels
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, pool_channels, 1, stride=1, padding=0)
        )  # pool_channels

    def forward(self, x):
        outputs = [self.branch1x1(x), self.branch5x5(x), self.branch3x3dbl(x), self.branch_pool(x)]
        # 64 + 64 + 96 + pool_channels
        return torch.cat(outputs, 1)

class InceptionB(Module):
    def __init__(self, channels_in):
        super(InceptionB, self).__init__()
        self.branch3x3 = Conv2d_BN(channels_in, 384, 3, stride=2, padding=1)  # 384 channels
        self.branch3x3dbl = Sequential(
            Conv2d_BN(channels_in, 64, 1, padding=0),
            Conv2d_BN(64, 96, 3, padding=1),
            Conv2d_BN(96, 96, 3, stride=2, padding=1)
        )  # 96 channels
        self.branch_pool = MaxPool2d(3, stride=2, padding=1)  # channels_in

    def forward(self, x):
        outputs = [self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)]
        # 384 + 96 + channels_in
        return torch.cat(outputs, 1)

class InceptionC(Module):
    def __init__(self, channels_in, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)  # 192 channels
        self.branch7x7 = Sequential(
            Conv2d_BN(channels_in, channels_7x7, 1, stride=1, padding=0),
            Conv2d_BN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(channels_7x7, 192, (7, 1), stride=1, padding=(3, 0))
        )  # 192 channels
        self.branch7x7dbl = Sequential(
            Conv2d_BN(channels_in, channels_7x7, 1, stride=1, padding=0),
            Conv2d_BN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(channels_7x7, 192, (1, 7), stride=1, padding=(0, 3))
        )  # 192 channels
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)
        )  # 192 channels
    
    def forward(self, x):
        outputs = [self.branch1x1(x), self.branch7x7(x), self.branch7x7dbl(x), self.branch_pool(x)]
        # 192 + 192 + 192 + 192 = 768 channels
        return torch.cat(outputs, 1)

class InceptionD(Module):
    def __init__(self, channels_in):
        super(InceptionD, self).__init__()
        self.branch3x3 = Sequential(
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0),
            Conv2d_BN(192, 320, 3, stride=2, padding=1)
        )  # 320 channels
        self.branch7x7x3 = Sequential(
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0),
            Conv2d_BN(192, 192, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(192, 192, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(192, 192, 3, stride=2, padding=1)
        )  # 192 chnnels
        self.branch_pool = MaxPool2d(3, stride=2, padding=1)  # channels_in

    def forward(self, x):
        outputs = [self.branch3x3(x), self.branch7x7x3(x), self.branch_pool(x)]
        # 320 + 192 + channels_in
        return torch.cat(outputs, 1)

class InceptionE(Module):
    def __init__(self, channels_in):
        super(InceptionE, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 320, 1, stride=1, padding=0)  # 320 channels

        self.branch3x3_1 = Conv2d_BN(channels_in, 384, 1, stride=1, padding=0)
        self.branch3x3_2a = Conv2d_BN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch3x3_2b = Conv2d_BN(384, 384, (3, 1), stride=1, padding=(1, 0))
        # 768 channels

        self.branch3x3dbl_1 = Sequential(
            Conv2d_BN(channels_in, 448, 1, stride=1, padding=0),
            Conv2d_BN(448, 384, 3, stride=1, padding=1)
        )
        self.branch3x3dbl_2a = Conv2d_BN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch3x3dbl_2b = Conv2d_BN(384, 384, (3, 1), stride=1, padding=(1, 0))
        # 768 channels
        
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)
        )  # 192 channels
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = torch.cat([self.branch3x3dbl_2a(branch3x3dbl), self.branch3x3dbl_2b(branch3x3dbl)], 1)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        # 320 + 768 + 768 + 192 = 2048 channels
        return torch.cat(outputs, 1)
    
class RunNet():
    def __init__(self, batchSz=64, nEpochs=100, seed=1, model=None, trainLoader=None, valLoader=None, verbose = False, experiment = None):
        self.batchSz = batchSz
        self.nEpochs = nEpochs
        self.seed = seed
        self.cuda = torch.cuda.is_available()
        
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.verbose = verbose
        self.model = model
        
        if experiment != None:
            self.save = 'out/'+str(self.model)+'/'+str(experiment)
        else:
            self.save = 'out/'+str(self.model)
        
        self.lr = 0.001
        self.wd = 5e-4
        self.momentum = 0.9
        self.epoch_55 = True
        
        if self.model == 'SqueezeNet':
            self.net = SqueezeNet()
        elif self.model == 'DenseNet':
            self.net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=101)
        elif self.model == 'InceptionNet':
            self.net = GoogLeNetV3(3)
        else:
            raise Exception("Sorry, that model does not exist")
            
        if self.cuda:
            self.net = net.cuda()
            
        self.trainLosses = []
        self.trainErrors = []
        self.trainF1s = []
        
        self.testLosses = []
        self.testErrors = []
        self.testF1s = []
        
        self.bestError = 200.0
        
    def train(self, epoch, optimizer, trainF):
        self.net.train()
        nProcessed = 0
        nTrain = len(self.trainLoader.dataset)
        epochLoss = []
        epochError = 0
        targets = []
        outputs = []
        for batch_idx, (data, target) in enumerate(self.trainLoader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = self.net(data).squeeze()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            nProcessed += len(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            outputs.append(pred)
            targets.append(target)
            incorrect = pred.ne(target.data).cpu().sum()
            epochLoss.append(loss.item())
            epochError += incorrect
        epochError /= (nProcessed*100.0)
        self.trainErrors.append(epochError)
        
        epochLoss = torch.mean(epochLoss)
        self.trainLosses.append(epochLoss)
        
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        fscore = f1_score(targets, outputs, average="macro")
        self.trainF1s.append(fscore)
        
        if self.verbose:
            print('Train Epoch: {:.2f}, loss: {:.2f}, Error: {:.2f}, F1-Score: {:.2f} '.format(epoch, epochLoss, epochError, fscore))
        trainF.write('Train Epoch: {:.2f}, loss: {:.2f}, Error: {:.2f}, F1-Score: {:.2f} '.format(epoch, epochLoss, epochError, fscore))
        trainF.flush()
            
        
    def val(self, epoch, optimizer, valF):
        self.net.eval()
        val_loss = []
        incorrect = 0
        targets = []
        outputs = []
        for batch_idx, (data, target) in enumerate(self.valLoader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = self.net(data).squeeze()
            loss += F.nll_loss(output, target)
            val_loss.append(loss.item())
            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()
            targets.append(target)
            outputs.append(pred)

        val_loss = torch.mean(val_loss)
        self.testLosses.append(val_loss)
        nTotal = len(self.valLoader.dataset)
        error = (100.0*incorrect)/nTotal
        self.testErrors.append(error)
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        fscore = f1_score(targets, outputs, average="macro")
        self.testF1s.append(fscore)
        
        if self.bestError > error:
            self.bestError = error
            torch.save(self.net.state_dict(), self.save+'/best_model.pth')
        
        if self.verbose:
            print('\nVal set Epoch: {:.2f}, loss: {:.2f}, Error: {:.2f}, F1-Score: {:.2f}'.format(
        epoch, val_loss, error, fscore))
        valF.write('Val set Epoch: {:.2f}, loss: {:.2f}, Error: {:.2f}, F1-Score: {:.2f}'.format(
        epoch, val_loss, error, fscore))
        valF.flush()

    def getschedule(self, epoch):
        p = dict()
        regimes = [[1, 18, 5e-3, 5e-4],
                   [19, 29, 1e-3, 5e-4],
                   [30, 43, 5e-4, 5e-4],
                   [44, 52, 1e-4, 0],
                   [53, 1e8, 1e-5, 0]]
        for i, row in enumerate(regimes):
            if epoch >= row[0] and epoch <= row[1]:
                p['learning_rate'] = row[2]
                p['weight_decay'] = row[3]
        return p
    
    def run(self):
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        if os.path.exists(self.save):
            shutil.rmtree(self.save)
        os.makedirs(self.save, exist_ok=True)
        
        if self.verbose:
            print('  + Number of params: {}'.format(sum([p.data.nelement() for p in self.net.parameters()])))
        
        trainF = open(os.path.join(self.save, 'train.txt'), 'w')
        valF = open(os.path.join(self.save, 'val.txt'), 'w')
        for epoch in range(1, self.nEpochs + 1):
            if self.model == 'SqueezeNet':
                if self.epoch_55:
                    schedule = self.getschedule(epoch)
                    self.lr = schedule['learning_rate']
                    self.wd = schedule['weight_decay']
            if self.model == 'DenseNet':
                if epoch < 150: self.lr = 1e-1
                elif epoch == 150: self.lr = 1e-2
                elif epoch >= 225: self.lr = 1e-3
            if self.model == 'InceptionNet':
                self.momentum = 1.0
            optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
            
            self.train(epoch, optimizer, trainF)
            self.val(epoch, optimizer, valF)
            
            torch.save(self.net, os.path.join(self.save, '/latest.pth'))
        trainF.close()
        valF.close()
        
        plt.plot(self.trainLosses, label='train')
        plt.plot(self.testLosses, label='test')
        plt.legend(loc='best')
        plt.title('Training vs Testing Loss by Epoch')
        plt.ylabel('Epoch #')
        plt.xlabel('Loss')
        plt.savefig(self.save+'/loss.png')
        plt.close()
        
        plt.plot(self.trainErrors, label='train')
        plt.plot(self.testErrors, label='test')
        plt.legend(loc='best')
        plt.title('Training vs Testing Error by Epoch')
        plt.ylabel('Epoch #')
        plt.xlabel('Error %')
        plt.savefig(self.save+'/error.png')
        plt.close()
        
        plt.plot(self.trainF1s, label='train')
        plt.plot(self.testF1s, label='test')
        plt.legend(loc='best')
        plt.title('Training vs Testing F1-Score by Epoch')
        plt.ylabel('Epoch #')
        plt.xlabel('F1-Score')
        plt.savefig(self.save+'/f1.png')
        plt.close()
        
        with open(self.save+'/variables.pkl', 'wb') as file:
            pickle.dump([self.trainLosses, self.testLosses, self.trainErrors, self.testErrors, self.trainF1s, self.testF1s], file)