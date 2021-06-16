
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
from PIL import Image
import math
import matplotlib.pyplot as plt
import Dataset_Combinations
# import cv2

class TextLogger():
    def __init__(self, title, save_path, append=False):
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

test_image_transform = t.Compose([
    t.Resize(600, Image.BILINEAR),
    t.CenterCrop(300),
    t.ToTensor(),
    ])

train_image_transform = t.Compose([
    # t.Resize(96, Image.BILINEAR),
    # t.CenterCrop(90),
    t.Resize(600, Image.BILINEAR),
    t.RandomHorizontalFlip(0.5),
    t.RandomRotation([0, 360]),
    t.RandomCrop(300),
    t.ColorJitter(
        hue= 0.01,
        saturation=0.01,
        brightness=0.01,
        contrast=0.01),
    t.ToTensor(),
    ])

tile_transform = t.Compose([
    t.Resize((100, 100)),
    t.RandomCrop(96),
    # t.RandomCrop(28),
    # t.Resize((32, 32)),
    t.ColorJitter(
        hue= 0.01,
        saturation=0.01,
        brightness=0.01,
        contrast=0.01),
    t.ToTensor(),
    ])




class ImageDataset(data.Dataset):
    def __init__(self, dataset_path, train, is_test, dataset, num_classes):

        self.train = train
        self.is_test = is_test
        self.dataset=dataset
        self.dataset_path = dataset_path
        self.num_classes=num_classes
        target = dataset_path

        #If single dataset will return single, if combo# will return multiple datasets
        self.dataset_list = Dataset_Combinations.dataset_list_convert(self.dataset)
        print(self.dataset_list)
        self.h5_list=[np.zeros(len(self.dataset_list))]

        self.pil2tensor = t.ToTensor()
        self.tensor2pil = t.ToPILImage()


        for dataset_idx in range(len(self.dataset_list)):

            #Specify paths to dataset
            train_path = self.dataset_list[dataset_idx] + '_train.h5'
            valid_path = self.dataset_list[dataset_idx] + '_valid.h5'
            test_path = self.dataset_list[dataset_idx] + '_test.h5'

            #If using combo of multiple datasets
            if len(self.dataset_list) != 1:
                target_mod = target.replace(self.dataset, self.dataset_list[dataset_idx])
            else:
                target_mod = target


            if self.train == True:
                #Training
                self.transform =train_image_transform
                self.h5_file = target_mod + train_path
            else:
                #Validation
                self.transform=train_image_transform
                if self.is_test==False:
                    self.h5_file = target_mod + valid_path
                else:
                    #Testing
                    self.h5_file = target_mod + test_path

            #Save that specific h5 file to index in list
            self.h5_list = (h5py.File(self.h5_file, 'r'))['x']

            # randomly sample 4000 images from entire dataset (4000/#of datasets)
            num_samples =int(4000/len(self.dataset_list))
            if self.train==True:
                print('subsampling {} samples from dataset of length {}'.format(num_samples, len(self.h5_list)))
                random_idx=np.sort(random.sample(range(0,len(self.h5_list)), num_samples))
                self.h5_list = self.h5_list[list(random_idx)]
            else:
                random_idx = np.sort(random.sample(range(0, len(self.h5_list)), int(num_samples/4)))
                self.h5_list = self.h5_list[list(random_idx)]

            if dataset_idx == 0 :
                self.data = self.h5_list
            else:
                self.data = torch.utils.data.ConcatDataset([self.data, self.h5_list])

        self.data_length = len(self.data)
        print('length of data is {}'.format(self.data_length))

        self.random_ixs = list(range(self.data_length))
        random.shuffle(self.random_ixs)

        if not os.path.exists('data/'+dataset):
            os.makedirs('data/'+dataset)

        if not os.path.exists('data/'+dataset+'/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]

            print('Calculating mean and std')
            for ix in tqdm(range(len(self.random_ixs))):
                np_dat = self.data[ix]
                img = self.pil2tensor(Image.fromarray(np_dat))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= self.data_length
                mean_std['std'][cix] /= self.data_length

            torch.save(mean_std, 'data/'+dataset+'/mean_std.pt')

        else:
            mean_std = torch.load('data/'+dataset+'/mean_std.pt')

        normalize = t.Normalize(mean=mean_std['mean'], std=mean_std['std'])

        self.transform.transforms.append(normalize)


    def split_tiles(self, img):
        img = self.transform(img)
        img = self.tensor2pil(img)


        #Load pre-calculated permutations of jigsaw puzzle
        all_perm = np.load('./jigsaw_setup/permutations_%d.npy' % (self.num_classes))

        tiles = [None] * 9
        for n in range(9):

            #Crop each tile
            tile_h, tile_w  = img.size
            left = tile_h/3*(n%3)
            right = tile_h/3*((n%3)+1)
            top = tile_h/3*(math.floor(n/3))
            bottom = tile_h/3*(math.ceil((n+1)/3))

            tile=(img.crop((left,top,right,bottom)))


            #Normalize individual tile
            # norm_tile = Image.fromarray(cv2.normalize(np.uint8(tile), np.uint8(tile), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            norm_tile = tile

            #Apply individual tile transform
            tile = tile_transform(norm_tile)
            tiles[n] = tile

        #Randomly select a permutaiton index to apply
        permute_index = np.random.randint(self.num_classes)
        order = all_perm[permute_index]


        data = [tiles[order[t]] for t in range(9)]

        data = torch.stack(data, 0)
        return data, int(permute_index), img

    def __getitem__(self, index):


        img = Image.fromarray(self.data[self.random_ixs[index]])

        #Run through the jigsaw puzzle
        data, labels, img = self.split_tiles(img)

        img = self.pil2tensor(img)

        return data, labels, img


    def __len__(self):
        return int(self.data_length)


if __name__ == '__main__':
    train_logger = TextLogger('Train loss', 'train_loss.log')
    for ix in range(30):
        # print(ix)
        train_logger.log('%s, %s' % (str(torch.rand(1)[0]), str(torch.rand(1)[0])))
        time.sleep(1)


