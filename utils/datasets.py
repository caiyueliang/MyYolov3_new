# encoding:utf-8
import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys
import cv2

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (self.img_shape, self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, root_path, image_file, img_size=416):
        self.root_path = root_path
        self.image_file = image_file
        with open(os.path.join(self.root_path, self.image_file), 'r') as file:
            self.img_files = file.readlines()
        self.img_files = [os.path.join(self.root_path, files.replace('\n', '')) for files in self.img_files]
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt') for path in self.img_files]
        print('img_files len: %d' % len(self.img_files))
        print('label_files len: %d' % len(self.label_files))

        self.img_shape = img_size
        self.max_objects = 50

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))
        # print(img_path)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # show_img = input_img.copy()

        # Resize and normalize
        input_img = resize(input_img, (self.img_shape, self.img_shape, 3), mode='reflect')

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # print('labels', labels)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # print(x1)
            # print(y1)
            # print(x2)
            # print(y2)
            # Calculate ratios from coordinates
            # print('padded_h, padded_w, _', padded_h, padded_w, _)

            labels[:, 1] = ((x1 + x2) / 2) / float(padded_w)               # x轴中心点
            labels[:, 2] = ((y1 + y2) / 2) / float(padded_h)               # y轴中心点
            labels[:, 3] *= float(w) / float(padded_w)                     # w比例
            labels[:, 4] *= float(h) / float(padded_h)                     # h比例
            # print('labels', labels)

            # for i, x in enumerate(x1):
            #     cv2.rectangle(show_img, (int(x), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0))
            # for label in labels:
            #     cv2.rectangle(show_img, (int((label[1] - label[3]/2) * padded_w), int((label[2] - label[4]/2) * padded_h)),
            #                   (int((label[1] + label[3]/2) * padded_w), int((label[2] + label[4]/2) * padded_h)), (0, 255, 0))
            # cv2.imshow('image', show_img)
            # cv2.waitKey(0)

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        # print(img_path)
        # print(input_img)
        # print(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
