import torch
import numpy as np

dataset_org = 'pcam'


def get_combo_dataset(combo_num):
    if combo_num == 1:
        dataset_list = [dataset_org, 'crc'] #Diverse histology images

    if combo_num==2:
        dataset_list = [dataset_org, 'tinyimagenet'] #With Natural images

    if combo_num==3:
        dataset_list = [dataset_org, 'alot', 'crc'] #With all textures

    if combo_num==4:
        dataset_list = ['crc', 'tinyimagenet'] #Two best performing

    if combo_num == 5:
        dataset_list = [dataset_org, 'crc', 'tinyimagenet']  # Two best performing




    # #original plus one other type of image
    # if combo_num==1:
    #     dataset_list = [dataset_org, 'crc'] #Diverse histology images
    #
    # if combo_num==2:
    #     dataset_list = [dataset_org, 'tinyimagenet'] #With Natural images
    #
    # if combo_num==3:
    #     dataset_list = [dataset_org, 'brats', 'minideeplesion'] #with other medical images
    #
    # if combo_num==4:
    #     dataset_list = [dataset_org, 'alot'] #pcam with texture
    #
    # #original plus two other types of image
    # if combo_num==5:
    #     dataset_list = [dataset_org, 'crc', 'brats', 'minideeplesion'] #Histology + other medical
    # if combo_num==6:
    #     dataset_list = [dataset_org, 'crc','alot'] #Histology and texture
    # if combo_num==7:
    #     dataset_list = [dataset_org, 'crc', 'tinyimagenet'] #Hisotlogy + natural images
    #
    # #original plus three other types of image
    # if combo_num==8:
    #     dataset_list = [dataset_org, 'crc', 'brats', 'minideeplesion', 'tinyimagenet'] #Hisoltogy + medical + natural images
    # if combo_num==9:
    #     dataset_list = [dataset_org, 'crc', 'brats', 'minideeplesion', 'alot'] #histology + medical + texture
    #
    # if combo_num==10: #combine all the datasets
    #     dataset_list = [dataset_org, 'crc', 'brats', 'minideeplesion', 'tinyimagenet', 'alot']
    #
    # if combo_num == 11:  # combine all the datasets
    #     dataset_list = ['crc', 'brats']

    return dataset_list

def dataset_list_convert(dataset_name):
    if 'combo' in dataset_name:
        num = str.split(dataset_name, 'combo')
        dataset_list = get_combo_dataset(int(num[1]))

    else:
        dataset_list = [dataset_name]

    # print('{} datasets for pretraining'.format(len(dataset_list)))
    # print('{} datasets for pretraining'.format((dataset_list)))

    return dataset_list

# dataset_list = dataset_list_convert('combo10')
# print('{} datasets for pretraining'.format(len(dataset_list)))
# print('{} datasets for pretraining'.format((dataset_list)))

