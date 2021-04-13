import torchvision.models as models
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch
from torch import cat
import sys
sys.path.append('Utils')
import numpy as np


'''
Jigsaw based model pre-trained with imagenet to be used as the base for Rotation SSL task
Based on context-free architecture proposed by Noroozi and Favaro (2016) https://github.com/MehdiNoroozi/JigsawPuzzleSolver
'''
class Model_Jigsaw(nn.Module):
    def __init__(self, num_classes=100):
        super(Model_Jigsaw, self).__init__()
        print('Completing jigsaw SSL pre-training with {}'.format(num_classes))
        base = models.__dict__['resnet34'](pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(512, 1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1', nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(9 * 1024, 4096))
        self.fc7.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7.add_module('drop7', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(4096, num_classes))
#
#
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.base(x[i]).squeeze()
            z = self.fc6(z.view(B, -1))
            z = z.view([B, 1, -1])
            x_list.append(z)

        output = cat(x_list, 1)
        output = self.fc7(output.view(B, -1))
        output = self.classifier(output)

        return output

'''
ResNet based model pre-trained with imagenet to be used as the base for Rotation SSL task
'''
class Model_Rotation(nn.Module):
    def __init__(self, num_classes=4):
        super(Model_Rotation, self).__init__()
        print('Completing rotation SSL pre-training with {}'.format(num_classes))


        base = models.__dict__['resnet34'](pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])

        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.base(x).squeeze()
        output = self.fc1(feat)
        return output



