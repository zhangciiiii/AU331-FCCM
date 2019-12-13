import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import pandas as pd

def crawl_folders(root, folders_list):

    with open ('val_list.txt', 'w'):
        pass

    with open ('val_list.txt', 'a') as txt:
        for folder in folders_list:
            folder = Path(root)/folder
            imgs_list = sorted(folder.files('*.jpg')) # all imgs in a folder
            val_list = random.sample(imgs_list, 2)
            for img in val_list:
                val_image = img.split('/')[-2] + '/' + img.split('/')[-1]
                txt.write(val_image + '\n')


    return  

if __name__ == "__main__":
    root = '/home/zhang/documents/data/curriculum/MLproject/pre-process'
    scenes = os.listdir(root)
    crawl_folders(root, scenes)