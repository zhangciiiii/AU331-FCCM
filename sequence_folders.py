import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import pandas as pd
import torch


def crawl_folders_train(root, label_root, folders_list):
    '''
    return a list which contains lots of samples : 
    sample = { 'CT_img': CT_img,  'label': []}
    '''
    sample_set = []
    
    label_table = pd.read_csv(label_root+'/Lung1.clinical.csv')
    # data turn the str into num
    label_table.loc[label_table['Overall.Stage']=='I','Overall.Stage'] = 0
    label_table.loc[label_table['Overall.Stage']=='II','Overall.Stage'] = 1
    label_table.loc[label_table['Overall.Stage']=='IIIa','Overall.Stage'] = 2
    label_table.loc[label_table['Overall.Stage']=='IIIb','Overall.Stage'] = 3

    label_table.loc[label_table['Histology']=='large cell','Histology'] = 0
    label_table.loc[label_table['Histology']=='squamous cell carcinoma','Histology'] = 1
    label_table.loc[label_table['Histology']=='adenocarcinoma','Histology'] = 2
    label_table.loc[label_table['Histology']=='nos','Histology'] = 3
    label_table.loc[pd.isnull(label_table['Histology']),'Histology'] = 4 # NA
    

    # one-hot encoding
    dataset = pd.DataFrame()
    dataset['PatientID'] = label_table['PatientID']

    for col in label_table.iloc[:,1:-1].columns :
        for element in (label_table[col].unique()):
            dataset[col+str(element)]=np.zeros_like(label_table[col])
            dataset.loc[label_table[col]==element,col+str(element)]=1
    dataset['Survival.time'] = label_table['Survival.time']
    cols = ['PatientID',
            'Clinical.N.Stage0', 'Clinical.N.Stage1', 'Clinical.N.Stage2','Clinical.N.Stage3', 'Clinical.N.Stage4', 
            'Histology0', 'Histology1', 'Histology2', 'Histology3', 'Histology4', 
            'Overall.Stage0', 'Overall.Stage1', 'Overall.Stage2', 'Overall.Stage3', 
            'clinical.T.Stage1','clinical.T.Stage2', 'clinical.T.Stage3', 'clinical.T.Stage4', 'clinical.T.Stage5',
            'Survival.time']
    # print((cols))

    with open('val_list.txt') as f:
        val_list = f.readlines()

    dataset = dataset.ix[:,cols]
    for folder in folders_list:
        folder = Path(root/folder)
        imgs_list = sorted(folder.files('*.jpg'))
        for img in imgs_list:
            # print(np.array((dataset.loc[dataset['PatientID']== folder.split('/')[-1]])).shape)
            # print(dataset.loc[dataset['PatientID']== folder.split('/')[-1]])
            # print(folder.split('/')[-1] + '/' + img +'\n')
            if (img.split('/')[-2] + '/' + img.split('/')[-1] +'\n') in val_list:
                continue

            tmp = dataset.loc[dataset['PatientID']== folder.split('/')[-1]]
            tmp.pop('PatientID')

            # print(tmp.columns)
            # input()
            # print(folder.split('/'))
            # for x in tmp.columns:
            #     print(tmp[x])
            gt = (np.array(tmp))
            # print(gt)
            sample = { 'CT_img':img , 'label':gt }        

            # input()

            sample_set.append(sample)

    random.shuffle(sample_set)
    return sample_set # 

def load_as_float(path):
    return imread(path).astype(np.float32)

class Generate_train_set(data.Dataset):
    """A sequence data loader where the files are arranged in this way:

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, label_root, transform=None, seed=None, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.label_root = Path(label_root)
        self.scenes = os.listdir(self.root)
        self.samples = crawl_folders_train(self.root, self.label_root, self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.transform is not None:
            imgs = self.transform(load_as_float(sample['CT_img']))
        else:
            imgs = load_as_float(sample['CT_img'])
        return imgs, sample['label']
        


    def __len__(self):
        return len(self.samples)

def crawl_folders_val(root, label_root, folders_list):
    '''
    return a list which contains lots of samples : 
    sample = { 'CT_img': CT_img,  'label': []}
    '''
    sample_set = []
    
    label_table = pd.read_csv(label_root+'/Lung1.clinical.csv')
    # data turn the str into num
    label_table.loc[label_table['Overall.Stage']=='I','Overall.Stage'] = 0
    label_table.loc[label_table['Overall.Stage']=='II','Overall.Stage'] = 1
    label_table.loc[label_table['Overall.Stage']=='IIIa','Overall.Stage'] = 2
    label_table.loc[label_table['Overall.Stage']=='IIIb','Overall.Stage'] = 3

    label_table.loc[label_table['Histology']=='large cell','Histology'] = 0
    label_table.loc[label_table['Histology']=='squamous cell carcinoma','Histology'] = 1
    label_table.loc[label_table['Histology']=='adenocarcinoma','Histology'] = 2
    label_table.loc[label_table['Histology']=='nos','Histology'] = 3
    label_table.loc[pd.isnull(label_table['Histology']),'Histology'] = 4 # NA
    

    # one-hot encoding
    dataset = pd.DataFrame()
    dataset['PatientID'] = label_table['PatientID']

    for col in label_table.iloc[:,1:-1].columns :
        for element in (label_table[col].unique()):
            dataset[col+str(element)]=np.zeros_like(label_table[col])
            dataset.loc[label_table[col]==element,col+str(element)]=1
    dataset['Survival.time'] = label_table['Survival.time']
    cols = ['PatientID',
            'Clinical.N.Stage0', 'Clinical.N.Stage1', 'Clinical.N.Stage2','Clinical.N.Stage3', 'Clinical.N.Stage4', 
            'Histology0', 'Histology1', 'Histology2', 'Histology3', 'Histology4', 
            'Overall.Stage0', 'Overall.Stage1', 'Overall.Stage2', 'Overall.Stage3', 
            'clinical.T.Stage1','clinical.T.Stage2', 'clinical.T.Stage3', 'clinical.T.Stage4', 'clinical.T.Stage5',
            'Survival.time']
    # print((cols))

    with open('val_list.txt') as f:
        val_list = f.readlines()

    dataset = dataset.ix[:,cols]
    for folder in folders_list:
        folder = Path(root/folder)
        imgs_list = sorted(folder.files('*.jpg'))
        for img in imgs_list:
            # print(np.array((dataset.loc[dataset['PatientID']== folder.split('/')[-1]])).shape)
            # print(dataset.loc[dataset['PatientID']== folder.split('/')[-1]])
            # print(folder.split('/')[-1] + '/' + img +'\n')
            if (img.split('/')[-2] + '/' + img.split('/')[-1] +'\n') not in val_list:
                continue

            tmp = dataset.loc[dataset['PatientID']== folder.split('/')[-1]]
            tmp.pop('PatientID')

            # print(tmp.columns)
            # input()
            # print(folder.split('/'))
            # for x in tmp.columns:
            #     print(tmp[x])
            gt = (np.array(tmp))
            # print(gt)
            sample = { 'CT_img':img , 'label':gt }        

            # input()

            sample_set.append(sample)

    random.shuffle(sample_set)
    return sample_set # 

class Generate_val_set(data.Dataset):
    """
    A sequence data loader where the files are arranged in this way:
    transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, label_root, seed=None, ):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.label_root = Path(label_root)
        self.scenes = os.listdir(self.root)
        self.samples = crawl_folders_val(self.root, self.label_root, self.scenes)

    def __getitem__(self, index):
        sample = self.samples[index]

        imgs = load_as_float(sample['CT_img'])
        
        # put it from HWC to CHW format
        # print(im.shape)
        imgs = np.transpose(imgs[..., np.newaxis], (2, 0, 1))
        # handle numpy array
        imgs = torch.from_numpy(imgs).float()/255

        return imgs, sample['label']
        

    def __len__(self):
        return len(self.samples)
