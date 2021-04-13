import torch
import torch.nn as nn
import torchvision


class Model_imagenet(nn.Module):
    def __init__(self):
        print('using pretrained imagenet model')
        torch.nn.Module.__init__(self)
        base = torchvision.models.resnet34(pretrained=True)

        self.base = nn.Sequential(*list(base.children())[:-1])
        self.emb = nn.Linear(512, 2)

    def forward(self, input):
        feat = self.base(input).squeeze()
        o = self.emb(feat)
        return o


class Model_random(nn.Module):
    def __init__(self):
        print('using random model')
        torch.nn.Module.__init__(self)
        base = torchvision.models.resnet34()

        self.base = nn.Sequential(*list(base.children())[:-1])
        self.emb = nn.Linear(512, 2)

    def forward(self, input):
        feat = self.base(input).squeeze()
        o = self.emb(feat)
        return o


class Model_LoadJigsaw(nn.Module):
    def __init__(self,pretrain_path):
        print('using jigsaw')
        self.pretrain_path = pretrain_path
        self.num_classes=100

        super(Model_LoadJigsaw, self).__init__()
        warmup_dict = torch.load(self.pretrain_path)
        pretrainedm = Model_Jigsaw_Pretrain(num_classes=self.num_classes)

        pretrainedm.load_state_dict(warmup_dict[0])
        pretrainedm.opt = torch.optim.Adam(pretrainedm.parameters())
        pretrainedm.opt.load_state_dict(warmup_dict[1])

        # remove last three fully connected layers
        self.base = nn.Sequential(*list(pretrainedm.children())[:-3])

        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        feat = self.base(x).squeeze()
        output = self.classifier(feat)
        return output



class Model_Jigsaw_Pretrain(nn.Module):
    def __init__(self, num_classes):
        super(Model_Jigsaw_Pretrain, self).__init__()
        print(num_classes)
        base = torchvision.models.__dict__['resnet34']()
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

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.base(x[i]).squeeze()
            z = self.fc6(z.view(B, -1))
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = cat(x_list, 1)
        x = self.fc7(x.view(B, -1))
        x = self.classifier(x)

        return x

class Model_LoadRotation(nn.Module):
    def __init__(self, pretrain_path):
        print('using rotation')
        self.pretrain_path = pretrain_path
        self.num_classes = 4

        super(Model_LoadRotation, self).__init__()
        warmup_dict = torch.load(self.pretrain_path)
        pretrainedm = Model_Rotation_Pretrain(self.num_classes)

        pretrainedm.load_state_dict(warmup_dict[0])
        pretrainedm.opt = torch.optim.Adam(pretrainedm.parameters())
        pretrainedm.opt.load_state_dict(warmup_dict[1])

        # remove last fully connected layer
        self.base = nn.Sequential(*list(pretrainedm.children())[:-1])

        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        feat = self.base(x).squeeze()
        output = self.classifier(feat)

        return output

class Model_Rotation_Pretrain(nn.Module):
    def __init__(self, num_classes):
        super(Model_Rotation_Pretrain, self).__init__()

        base = torchvision.models.__dict__['resnet34'](pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])

        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.base(x).squeeze()
        output = self.fc1(feat)

        return output

