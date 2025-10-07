
import torch
import torchvision.transforms.functional as F
import scipy.ndimage
import random
from PIL import Image
import numpy as np
import cv2
import numbers


class ToTensor(object):
    def __init__(self, train=False):
        self.train = train
    def __call__(self, data):
        image, label = data['image'], data['label']
        if self.train:
            depth = data['depth']
            depth1 = data['depth1']
            return {'image': F.to_tensor(image), 'label': F.to_tensor(label), 'depth': F.to_tensor(depth), 'depth1': F.to_tensor(depth1)}
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}




class Resize(object):

    def __init__(self, size, train=False):
        self.size = size
        self.train = train
    def __call__(self, data):
        image, label = data['image'], data['label']
        if self.train:
            depth = data['depth']
            depth1 = data['depth1']
            return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size),
                    'depth': F.resize(depth, self.size), 'depth1': F.resize(depth1, self.size)}
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size)}





class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, train=False):
        self.p = p
        self.train = train
    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            if self.train:
                depth = data['depth']
                depth1 = data['depth1']
                return {'image': F.hflip(image), 'label': F.hflip(label), 'depth': F.hflip(depth), 'depth1': F.hflip(depth1)}
            return {'image': F.hflip(image), 'label': F.hflip(label)}
        return data


class RandomVerticalFlip(object):
    def __init__(self, p=0.5, train=False):
        self.p = p
        self.train = train
    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            if self.train:
                depth = data['depth']
                depth1 = data['depth1']
                return {'image': F.vflip(image), 'label': F.vflip(label), 'depth': F.vflip(depth), 'depth1': F.vflip(depth1)}
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return data


class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None, train=False):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.train = train
    @staticmethod
    def get_params(degrees):

        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, data):


        image, label = data['image'], data['label']

        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            if self.train:
                depth = data['depth']
                depth1 = data['depth1']
                return {'image': F.rotate(image, angle, self.resample, self.expand, self.center),
                        'label': F.rotate(label, angle, self.resample, self.expand, self.center),
                        'depth': F.rotate(depth, angle, self.resample, self.expand, self.center),
                        'depth1': F.rotate(depth1, angle, self.resample, self.expand, self.center)}

            return {'image': F.rotate(image, angle, self.resample, self.expand, self.center),
                    'label': F.rotate(label, angle, self.resample, self.expand, self.center)}


        return data


class RandomZoom(object):
    def __init__(self, zoom=(0.8, 1.2), train=False):
        self.min, self.max = zoom[0], zoom[1]
        self.train = train
    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)

            zoom = random.uniform(self.min, self.max)
            zoom_image = clipped_zoom(image, zoom)
            zoom_label = clipped_zoom(label, zoom)

            zoom_image = Image.fromarray(zoom_image.astype('uint8'), 'RGB')
            zoom_label = Image.fromarray(zoom_label.astype('uint8'), 'L')
            if self.train:
                depth = data['depth']
                depth = np.array(depth)
                zoom_depth = clipped_zoom(depth, zoom)
                zoom_depth = Image.fromarray(zoom_depth.astype('uint8'), 'RGB')
                depth1 = data['depth1']
                depth1 = np.array(depth1)
                zoom_depth1 = clipped_zoom(depth1, zoom)
                zoom_depth1 = Image.fromarray(zoom_depth1.astype('uint8'), 'L')
                return {'image': zoom_image, 'label': zoom_label, 'depth': zoom_depth, 'depth1': zoom_depth1}
            return {'image': zoom_image, 'label': zoom_label}


        return data


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]


    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)


    if zoom_factor < 1:


        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    elif zoom_factor > 1:

        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        zoom_in = scipy.ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)


        if zoom_in.shape[0] >= h:
            zoom_top = (zoom_in.shape[0] - h) // 2
            sh = h
            out_top = 0
            oh = h
        else:
            zoom_top = 0
            sh = zoom_in.shape[0]
            out_top = (h - zoom_in.shape[0]) // 2
            oh = zoom_in.shape[0]
        if zoom_in.shape[1] >= w:
            zoom_left = (zoom_in.shape[1] - w) // 2
            sw = w
            out_left = 0
            ow = w
        else:
            zoom_left = 0
            sw = zoom_in.shape[1]
            out_left = (w - zoom_in.shape[1]) // 2
            ow = zoom_in.shape[1]

        out = np.zeros_like(img)
        out[out_top:out_top + oh, out_left:out_left + ow] = zoom_in[zoom_top:zoom_top + sh, zoom_left:zoom_left + sw]


    else:
        out = img
    return out






class Normalization(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=False):
        self.mean = mean
        self.std = std
        self.train = train
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        if self.train:
            depth = sample['depth']
            depth = F.normalize(depth, self.mean, self.std)
            return {'image': image, 'label': label, 'depth': depth}
        return {'image': image, 'label': label}
