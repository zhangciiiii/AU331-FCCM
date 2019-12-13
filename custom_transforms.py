from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize, imrotate

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        if type(images) is not list:
            images = [images]
        for t in self.transforms:
            images = t(images)
        return images


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images


class NormalizeLocally(object):

    def __call__(self, images):
        image_tensor = torch.stack(images)
        assert(image_tensor.size(1)==3)   #3 channel image
        mean = image_tensor.transpose(0,1).contiguous().view(3, -1).mean(1)
        std = image_tensor.transpose(0,1).contiguous().view(3, -1).std(1)

        for tensor in images:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return images


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            # print(im.shape)
            im = np.transpose(im[..., np.newaxis], (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images):

        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]

        else:
            output_images = images

        return output_images

class RandomRotate(object):
    """Randomly rotates images up to 10 degrees and crop them to keep same size as before."""
    def __call__(self, images):
        if np.random.random() > 0.5:
            return images
        else:
            rot = np.random.uniform(0,10)
            rotated_images = [imrotate(im, rot) for im in images]

            return rotated_images




class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""
    def __init__(self, h=0, w=0):
        self.h = h
        self.w = w

    def __call__(self, images):


        in_h, in_w = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.1,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        if self.h and self.w:
            in_h, in_w = self.h, self.w

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]


        return cropped_images

class Scale(object):
    """Scales images to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):

        in_h, in_w = images[0].shape
        scaled_h, scaled_w = self.h , self.w


        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        return scaled_images
