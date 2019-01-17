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

    # def __getitem__(self, index):
    #     img_path = self.files[index % len(self.files)]
    #     # Extract image
    #     img = np.array(Image.open(img_path))
    #     h, w, _ = img.shape
    #     dim_diff = np.abs(h - w)
    #     # Upper (left) and lower (right) padding
    #     pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    #     # Determine padding
    #     pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    #     # Add padding
    #     input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
    #     # Resize and normalize
    #     input_img = resize(input_img, (self.img_shape, self.img_shape, 3), mode='reflect')
    #     # Channels-first
    #     input_img = np.transpose(input_img, (2, 0, 1))
    #     # As pytorch tensor
    #     input_img = torch.from_numpy(input_img).float()
    #
    #     return img_path, input_img

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        # Resize and normalize
        input_img = resize(img, (self.img_shape, self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, root_path, image_file, train=False, img_size=416):
        self.root_path = root_path
        self.image_file = image_file
        with open(os.path.join(self.root_path, self.image_file), 'r') as file:
            self.img_files = file.readlines()
        self.img_files = [os.path.join(self.root_path, files.replace('\n', '')) for files in self.img_files]
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt') for path in self.img_files]
        print('img_files len: %d' % len(self.img_files))
        print('label_files len: %d' % len(self.label_files))

        self.img_shape = img_size
        self.train = train
        self.max_objects = 50                       # 每张图片最多支持多少个标签

    # def __getitem__(self, index):
    #
    #     # ---------
    #     #  Image
    #     # ---------
    #
    #     img_path = self.img_files[index % len(self.img_files)].rstrip()
    #     img = np.array(Image.open(img_path))
    #     # print(img_path)
    #
    #     # Handles images with less than three channels
    #     while len(img.shape) != 3:
    #         index += 1
    #         img_path = self.img_files[index % len(self.img_files)].rstrip()
    #         img = np.array(Image.open(img_path))
    #
    #     h, w, _ = img.shape
    #     dim_diff = np.abs(h - w)
    #     # Upper (left) and lower (right) padding
    #     pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    #     # Determine padding
    #     pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    #     # Add padding
    #     input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    #     padded_h, padded_w, _ = input_img.shape
    #     # show_img = input_img.copy()
    #
    #     # Resize and normalize
    #     input_img = resize(input_img, (self.img_shape, self.img_shape, 3), mode='reflect')
    #
    #     # Channels-first
    #     input_img = np.transpose(input_img, (2, 0, 1))
    #     # As pytorch tensor
    #     input_img = torch.from_numpy(input_img).float()
    #
    #     # ---------
    #     #  Label
    #     # ---------
    #
    #     label_path = self.label_files[index % len(self.img_files)].rstrip()
    #
    #     labels = None
    #     if os.path.exists(label_path):
    #         labels = np.loadtxt(label_path).reshape(-1, 5)
    #         # print('labels', labels)
    #         # Extract coordinates for unpadded + unscaled image
    #         x1 = w * (labels[:, 1] - labels[:, 3]/2)
    #         y1 = h * (labels[:, 2] - labels[:, 4]/2)
    #         x2 = w * (labels[:, 1] + labels[:, 3]/2)
    #         y2 = h * (labels[:, 2] + labels[:, 4]/2)
    #         # Adjust for added padding
    #         x1 += pad[1][0]
    #         y1 += pad[0][0]
    #         x2 += pad[1][0]
    #         y2 += pad[0][0]
    #         # print(x1)
    #         # print(y1)
    #         # print(x2)
    #         # print(y2)
    #         # Calculate ratios from coordinates
    #         # print('padded_h, padded_w, _', padded_h, padded_w, _)
    #         labels[:, 1] = ((x1 + x2) / 2) / float(padded_w)                # 第1列表示：x轴中心点（比例） # 第0列表示：类别
    #         labels[:, 2] = ((y1 + y2) / 2) / float(padded_h)                # 第2列表示：y轴中心点（比例）
    #         labels[:, 3] *= float(w) / float(padded_w)                      # 第3列表示：w（比例）
    #         labels[:, 4] *= float(h) / float(padded_h)                      # 第4列表示：h（比例）
    #         # print('labels', labels)
    #
    #         # for i, x in enumerate(x1):
    #         #     cv2.rectangle(show_img, (int(x), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0))
    #         # for label in labels:
    #         #     cv2.rectangle(show_img, (int((label[1] - label[3]/2) * padded_w), int((label[2] - label[4]/2) * padded_h)),
    #         #                   (int((label[1] + label[3]/2) * padded_w), int((label[2] + label[4]/2) * padded_h)), (0, 255, 0))
    #         # cv2.imshow('image', show_img)
    #         # cv2.waitKey(0)
    #
    #     # Fill matrix
    #     filled_labels = np.zeros((self.max_objects, 5))
    #     if labels is not None:
    #         filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
    #     filled_labels = torch.from_numpy(filled_labels)
    #
    #     # print(img_path)
    #     # print(input_img)
    #     # print(filled_labels)
    #
    #     # filled_labels size (50, 5);每一行表示一个标签（最多50个），分别表示：类别，x轴中心点，y轴中心点，w，h
    #     return img_path, input_img, filled_labels

    def __getitem__(self, index):
        # ==============================================================================================
        #  Label
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

        # ==============================================================================================
        #  Image
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        # 图片增广
        if self.train:
            img, labels = self.random_crop(img, labels)             # 随机裁剪
            img = self.random_bright(img)                           # 随机调亮

        # Resize and normalize
        input_img = resize(img, (self.img_shape, self.img_shape, 3), mode='reflect')

        show_img = input_img.copy()
        for label in labels:
            cv2.rectangle(show_img, (int((label[1] - label[3]/2) * self.img_shape), int((label[2] - label[4]/2) * self.img_shape)),
                          (int((label[1] + label[3]/2) * self.img_shape), int((label[2] + label[4]/2) * self.img_shape)), (0, 255, 0))
        cv2.imshow('image', show_img)
        cv2.waitKey(0)

        input_img = np.transpose(input_img, (2, 0, 1))          # Channels-first
        input_img = torch.from_numpy(input_img).float()         # As pytorch tensor

        # ==============================================================================================
        # Fill matrix
        # filled_labels size (50, 5);每一行表示一个标签（最多50个），分别表示：类别，x轴中心点，y轴中心点，w，h
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

    # 随机裁剪
    def random_crop(self, img, labels):
        # print('random_crop', labels)
        # mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
        # random.randrange(int(0.3*short_size), short_size)
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow('img_cv', img_cv)

        imh, imw, _ = img_cv.shape
        # short_size = min(imw, imh)
        # print(imh, imw, short_size)

        left = min(labels[0], labels[4])
        top = min(labels[1], labels[3])
        min_left = min(left, top)

        right = min(imw - labels[2], imw - labels[6])
        bottom = min(imh - labels[5], imh - labels[7])
        min_right = min(right, bottom)

        # print('left, top, right, bottom', left, top, right, bottom)
        # print('min_left, min_right', min_left, min_right)

        x1 = 0
        y1 = 0
        x2 = imw
        y2 = imh

        if random.random() < 0.5:
            rate = random.random()
            crop = int(min_left * rate)
            x1 = crop
            labels[0] = labels[0] - crop
            labels[2] = labels[2] - crop
            labels[4] = labels[4] - crop
            labels[6] = labels[6] - crop

            y1 = crop
            labels[1] = labels[1] - crop
            labels[3] = labels[3] - crop
            labels[5] = labels[5] - crop
            labels[7] = labels[7] - crop

        if random.random() < 0.5:
            rate = random.random()
            crop = int(min_right * rate)
            x2 = imw - crop
            y2 = imh - crop

        # print('x1, y1, x2, y2', x1, y1, x2, y2)
        img_cv = img_cv[y1:y2, x1:x2]
        # cv2.imshow('crop_img_cv', img_cv)
        # cv2.waitKey(0)

        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img, labels

    # # 随机裁剪
    # def random_crop(self, im, boxes, labels):
    #     # print('random_crop', boxes, labels)
    #
    #     imh, imw, _ = im.shape
    #     short_size = min(imw, imh)
    #     # print(imh, imw, short_size)
    #     while True:
    #         mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
    #         if mode is None:
    #             boxes_uniform = boxes / torch.Tensor([imw, imh, imw, imh]).expand_as(boxes)
    #             boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
    #             mask = (boxwh[:, 0] > self.small_threshold) & (boxwh[:, 1] > self.small_threshold)
    #             if not mask.any():
    #                 print('default image have none box bigger than small_threshold')
    #                 im, boxes, labels = self.random_getim()
    #                 imh, imw, _ = im.shape
    #                 short_size = min(imw, imh)
    #                 continue
    #             selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
    #             selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
    #             return im, selected_boxes, selected_labels
    #
    #         for _ in range(10):
    #             w = random.randrange(int(0.3 * short_size), short_size)
    #             h = w
    #
    #             x = random.randrange(imw - w)
    #             y = random.randrange(imh - h)
    #             roi = torch.Tensor([[x, y, x + w, y + h]])
    #
    #             center = (boxes[:, :2] + boxes[:, 2:]) / 2
    #             roi2 = roi.expand(len(center), 4)
    #             mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])
    #             mask = mask[:, 0] & mask[:, 1]
    #             if not mask.any():
    #                 continue
    #
    #             selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
    #             img = im[y:y + h, x:x + w, :]
    #             selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
    #             selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
    #             selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
    #             selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)
    #             # print('croped')
    #
    #             boxes_uniform = selected_boxes / torch.Tensor([w, h, w, h]).expand_as(selected_boxes)
    #             boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
    #             mask = (boxwh[:, 0] > self.small_threshold) & (boxwh[:, 1] > self.small_threshold)
    #             if not mask.any():
    #                 print('crop image have none box bigger than small_threshold')
    #                 im, boxes, labels = self.random_getim()
    #                 imh, imw, _ = im.shape
    #                 short_size = min(imw, imh)
    #                 continue
    #             selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
    #             selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
    #             return img, selected_boxes_selected, selected_labels

    # 随机调亮
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


if __name__ == "__main__":
    torch.utils.data.DataLoader(ListDataset("../Data/yolo/yolo_data_new/car_detect_train", "image_path.txt", train=True),
                                shuffle=False, batch_size=4, num_workers=0)