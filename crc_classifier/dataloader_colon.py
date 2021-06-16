'''
This code was created by Eu Wern Teh
'''
import time
import torch
import pickle
import torchvision.transforms as t
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import h5py
import random
from shutil import copyfile
import io
import yaml

train_transform_colon = t.Compose([
    t.Resize((150, 150)),
    t.Pad(19, padding_mode='reflect'),
    t.RandomCrop(150),
    t.RandomHorizontalFlip(0.5),
    t.RandomRotation([0, 360]),
    t.ColorJitter(
        hue=0.4,
        saturation=0.4,
        brightness=0.4,
        contrast=0.4),
    t.ToTensor(),
])
test_transform_colon = t.Compose([
    t.Resize((150, 150)),
    t.ToTensor(),
])


class Colon_dataset(data.Dataset):
    def __init__(self, model, train, seed=0, n_examples=10, dataset_path="/mnt/datasets/colon/colon.h5"):

        self.train = train

        self.dataset_path = dataset_path

        data = h5py.File(self.dataset_path, 'r')
        label = torch.Tensor(data['y']).squeeze()
        data.close()

        self.ys = []
        self.I = []

        index = list(range(len(label)))
        N = 500
        test_ix = index[N * seed: N * seed + N]
        train_ix = []
        test_label = []
        train_label = []
        for ix in index:
            if ix not in test_ix:
                train_ix.append(ix)
                train_label.append(int(label[ix].item()))
            else:
                test_label.append(int(label[ix].item()))

        assert (len(set(train_label)) == 8)
        assert (len(set(test_label)) == 8)
        test_label = torch.Tensor(test_label)
        train_label = torch.Tensor(train_label)

        print(len(test_label), len(train_label))

        if self.train == True:
            self.transform = train_transform_colon

            for uniq_label in set(train_label.tolist()):
                label_pos = (train_label == uniq_label).nonzero()
                random_pos = torch.randperm(len(label_pos))
                ct = 0
                for c_p in random_pos:
                    if ct == n_examples:
                        break
                    self.I.append(train_ix[c_p.item()])
                    self.ys.append(train_label[c_p])
                    ct += 1

        else:
            self.transform = test_transform_colon

            for ix in range(len(test_label)):
                self.I.append(test_ix[ix])
                self.ys.append(test_label[ix])

        pil2tensor = t.ToTensor()
        self.data = h5py.File(self.dataset_path, 'r')

        if model == 'imagenet':
            mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        else:
            if not os.path.exists('data/mean_std_colon.pt'):
                mean_std = {}
                mean_std['mean'] = [0, 0, 0]
                mean_std['std'] = [0, 0, 0]
                labels = torch.Tensor(self.data['y']).squeeze()
                data.close()

                print('Calculating mean and std')
                for ix in tqdm(range(len(labels))):
                    img = pil2tensor(Image.open(io.BytesIO(self.data['x'][ix])))
                    for cix in range(3):
                        mean_std['mean'][cix] += img[cix, :, :].mean()
                        mean_std['std'][cix] += img[cix, :, :].std()

                for cix in range(3):
                    mean_std['mean'][cix] /= len(labels)
                    mean_std['std'][cix] /= len(labels)

                torch.save(mean_std, 'data/mean_std_colon.pt')

            else:
                mean_std = torch.load('data/mean_std_colon.pt')

        self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))
        self.data.close()
        self.data = None

    def __getitem__(self, index):

        if self.data == None:
            self.data = h5py.File(self.dataset_path, 'r')
        curr_index = self.I[index]
        img = Image.open(io.BytesIO(self.data['x'][curr_index]))
        target = self.ys[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ys)

