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

class TextLogger():
    def __init__(self, title, save_path, append=False):
        print(save_path)
        file_state = 'wb'
        if append:
            file_state = 'ab'
        self.file = open(save_path, file_state, 0)
        self.log(title)
    
    def log(self, strdata):
        outstr = strdata + '\n'
        outstr = outstr.encode("utf-8")
        self.file.write(outstr)

    def __del__(self):
        self.file.close()

train_transform = t.Compose([
    t.Resize((96, 96)),
    t.Pad(12, padding_mode='reflect'),
    t.RandomCrop(96),
    t.RandomHorizontalFlip(0.5),
    t.RandomRotation([0, 360]),
    t.ToTensor(),
    ])
test_transform = t.Compose([
    t.Resize((96, 96)),
    t.ToTensor(),
    ])

class ImageDataset_hdf5(data.Dataset):
    def __init__(self, dataset_path, mode, data_percent, init_cond):

        self.mode = mode
        self.data_percent = data_percent
        self.init_cond = init_cond

        target = dataset_path#'/mnt/datasets/pcam/'
        train_x_path = 'camelyonpatch_level_2_split_train_x.h5'
        train_y_path = 'camelyonpatch_level_2_split_train_y.h5'
        valid_x_path = 'camelyonpatch_level_2_split_valid_x.h5'
        valid_y_path = 'camelyonpatch_level_2_split_valid_y.h5'
        test_x_path = 'camelyonpatch_level_2_split_test_x.h5'
        test_y_path = 'camelyonpatch_level_2_split_test_y.h5'


        if "train" in self.mode:
            self.transform = train_transform
            #self.dataset_path = 'train_img_%05d'
            self.h5_file_x = target + train_x_path#'../../dataset/pcam/camelyonpatch_level_2_split_train_x.h5'
            self.h5_file_y = target + train_y_path#'../../dataset/pcam/camelyonpatch_level_2_split_train_y.h5'
        if "valid" in self.mode:
            self.transform = test_transform
            self.h5_file_x = target + valid_x_path#'../../dataset/pcam/camelyonpatch_level_2_split_valid_x.h5'
            self.h5_file_y = target + valid_y_path#'../../dataset/pcam/camelyonpatch_level_2_split_valid_y.h5'

        if "test" in self.mode:
            self.transform = test_transform
            self.h5_file_x = target + test_x_path#'../../dataset/pcam/camelyonpatch_level_2_split_test_x.h5'
            self.h5_file_y = target + test_y_path#'../../dataset/pcam/camelyonpatch_level_2_split_test_y.h5'

        y_f = h5py.File(self.h5_file_y, 'r')
        self.label = torch.Tensor(y_f['y']).squeeze()
        self.random_ixs = list(range(len(self.label)))
        random.shuffle(self.random_ixs)
        y_f.close()

        self.pil2tensor = t.ToTensor()
        self.tensor2pil = t.ToPILImage()
        self.data = h5py.File(self.h5_file_x, 'r')

        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.exists('data/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]
            x_f = h5py.File('../../dataset/pcam/camelyonpatch_level_2_split_train_x.h5')
            y_f = h5py.File('../../dataset/pcam/camelyonpatch_level_2_split_train_y.h5')
            labels = torch.Tensor(y_f['y']).squeeze()
            y_f.close()

            print('Calculating mean and std')
            for ix in tqdm(range(len(labels))):
                np_dat = x_f['x'][ix]
                img = self.pil2tensor(Image.fromarray(np_dat))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(labels)
                mean_std['std'][cix] /= len(labels)

            torch.save(mean_std, 'data/mean_std.pt')

        else:
            mean_std = torch.load('data/mean_std.pt')



        if self.init_cond == 'imagenet':
            self.transform.transforms.append(t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))

        self.data.close()
        self.data = None

    def __getitem__(self, index):
        if self.data == None:
            self.data = h5py.File(self.h5_file_x, 'r')
        img = Image.fromarray(self.data['x'][index])
        target = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return int(self.data_percent/100*len(self.label))


if __name__ == '__main__':
    train_logger = TextLogger('Train loss', 'train_loss.log')
    for ix in range(30):
        print(ix)
        train_logger.log('%s, %s' % (str(torch.rand(1)[0]), str(torch.rand(1)[0])))
        time.sleep(1)


