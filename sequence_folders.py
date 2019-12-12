import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import pandas as pd


def crawl_folders(root, folders_list):
    '''
    return a list which contains lots of samples : 
    sample = { 'CT_img': CT_img,  'label': []}
    '''
    sample_set = []
    label_table = pd.read_csv(root+'/Lung1.clinical.csv')

    for folder in folders_list:
        folder = Path(root/folder)
        imgs_list = sorted(folder.files('*.jpg'))
        for img in imgs_list:
            sample = { 'CT_img':img , 'label':label_table[label_table['PatientID']== folder] }
            sample_set.append(sample)

    random.shuffle(sample_set)
    return sample_set # 


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.scenes = os.listdir(self.root)
        self.samples = crawl_folders(self.root, self.scenes)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample['CT_img'], sample['label']
        


    def __len__(self):
        return len(self.samples)
