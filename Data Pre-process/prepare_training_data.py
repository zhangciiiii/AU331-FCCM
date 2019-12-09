from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from PIL import Image
import os
import re

import pydicom
import pydicom.uid
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--with-gt", action='store_true',
                    help="If available (e.g. with KITTI), will store ground truth along with images, for validation")
parser.add_argument("--dump-root", type=str, required=False, help="Where to dump the data")
parser.add_argument("--height", type=int, default=512, help="image height") #from 480 to 384
parser.add_argument("--width", type=int, default=512, help="image width")  #from 640 to 512


args = parser.parse_args()

def main():
    i = 0
    folder_list = os.listdir(args.dataset_dir)
    folder_list = sorted(folder_list, key=embedded_numbers)
    for one_patient in tqdm(folder_list):
        
        Out_dir = Path(args.dump_root+'/'+one_patient)
        Out_dir.makedirs_p()

        for find_study in os.listdir(Path(args.dataset_dir)/one_patient):
            if "StudyID" in find_study or "CT" not in find_study:
                tmp_list = os.listdir(Path(args.dataset_dir)/one_patient/find_study)
                CT_DCM_list = os.listdir(Path(args.dataset_dir)/one_patient/find_study/tmp_list[0])
                CT_DCM_list = sorted(CT_DCM_list, key=embedded_numbers)
                for i,CT_DCM in enumerate(CT_DCM_list):
                    data = pydicom.read_file(Path(args.dataset_dir)/one_patient/find_study/tmp_list[0]/CT_DCM) 
                    data = data.pixel_array
                    scipy.misc.imsave(Out_dir/ str(i).zfill(4)+'.jpg',data)  #

                


        '''
        rgb_images = os.listdir(args.dataset_dir+'/'+one_patient+'/rgb')
        rgb_images = sorted(rgb_images, key=embedded_numbers)

        with open(args.dataset_dir+'/'+one_patient+'/groundtruth.txt') as gt:
            gt_list = gt.readlines()
            # Path(args.dump_root+'/'+one_patient+'/'+'pose.txt').makedirs_p()
            with open ((args.dump_root+'/'+one_patient+'/'+'pose.txt'),'w') as pose:
                for line in gt_list[2:]:
                    pose.write(str(line).lstrip('\n'))
        
        scene_id = 0
        for rgb_image, depth_image in tqdm(zip(rgb_images, depth_images)):

            rgb_image = resize_image(args.dataset_dir+'/'+one_patient+'/rgb/'+rgb_image) # numpy array
            depth_image = resize_image(args.dataset_dir+'/'+one_patient+'/depth/'+depth_image)
            Image.fromarray(rgb_image).convert("RGB").save(Out_dir/str(scene_id).zfill(4)+'.jpg')
            Image.fromarray(depth_image).save(depth_image_dir/str(scene_id).zfill(4)+'.jpg')
            scene_id += 1
        '''
        




def embedded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)           
    pieces[1::2] = map(int, pieces[1::2])    
    return pieces

def resize_image(img_file):
    img = scipy.misc.imread(img_file)
    img = scipy.misc.imresize(img, (args.height, args.width))
    return img

if __name__ == '__main__':
    main()
