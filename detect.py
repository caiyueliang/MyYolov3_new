from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
# =============================================================================================================================
# parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
# parser.add_argument('--image_folder', type=str, default='../Data/yolo/yolo_data_new/detect_test', help='path to dataset')
# parser.add_argument('--config_path', type=str, default='config/yolov3-tiny.cfg', help='path to model config file')
# parser.add_argument('--weights_path', type=str, default='weights/yolov3-tiny.weights', help='path to weights file')
# parser.add_argument('--class_num', type=int, default=80, help='class_num')
# parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')

# =============================================================================================================================
parser.add_argument('--image_folder', type=str, default='../Data/yolo/yolo_data_new/detect_test', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/lpr_yolov3-tiny.cfg', help='model config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/lpr_yolo_tiny_best.weights', help='weights file')
parser.add_argument('--class_num', type=int, default=2, help='class_num')
parser.add_argument('--class_path', type=str, default='../Data/yolo/yolo_data_new/lpr.names', help='path to class label file')

parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.2, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

try:
    os.makedirs('output')
except:
    qwe = None

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval()                                                    # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path)                        # Extracts class labels from file
model.load_state_dict(torch.load(opt.weights_path))
# model = torch.load(opt.weights_path)

# print('[print_net] ...')
# params = model.state_dict()
# print(params['module_list.0.conv_0.weight'])

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs_paths = []                                                 # Stores image paths
img_detections = []                                             # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        print ('detections size', detections.size())
        detections = non_max_suppression(detections, opt.class_num, opt.conf_thres, opt.nms_thres)
        if detections[0] is not None:
            print('detections', detections[0].size(), detections)
        else:
            print('detections', detections)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs_paths.extend(img_paths)
    img_detections.extend(detections)

print('images len: %d' % len(imgs_paths))

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print ('\nSaving images:')
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs_paths, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    img = resize(img, (416, 416, 3), mode='reflect')
    print('img shape', img.shape)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # # The amount of padding that was added
    # pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    # pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # # Image height and width after padding is removed
    # unpad_h = opt.img_size - pad_y
    # unpad_w = opt.img_size - pad_x
    # print('unpad_h', unpad_h)
    # print('unpad_w', unpad_w)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # print(x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls_conf.item(), cls_pred.item())
            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
            x1 = x1 if x1 >= 0 else 0
            y1 = y1 if y1 >= 0 else 0

            # Rescale coordinates to original dimensions
            box_h = (y2 - y1)
            box_w = (x2 - x1)
            # y1 = (y1 - pad_y // 2)
            # x1 = (x1 - pad_x // 2)
            # box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            # box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            # y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            # x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()


# for img_i, (path, detections) in enumerate(zip(imgs_paths, img_detections)):
#
#     print ("(%d) Image: '%s'" % (img_i, path))
#
#     # Create plot
#     img = np.array(Image.open(path))
#     print('img shape', img.shape)
#     plt.figure()
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#
#     # The amount of padding that was added
#     pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
#     pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
#     # Image height and width after padding is removed
#     unpad_h = opt.img_size - pad_y
#     unpad_w = opt.img_size - pad_x
#     print('unpad_h', unpad_h)
#     print('unpad_w', unpad_w)
#
#     # Draw bounding boxes and labels of detections
#     if detections is not None:
#         unique_labels = detections[:, -1].cpu().unique()
#         n_cls_preds = len(unique_labels)
#         bbox_colors = random.sample(colors, n_cls_preds)
#         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
#             print(x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls_conf.item(), cls_pred.item())
#             print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
#
#             # Rescale coordinates to original dimensions
#             box_h = ((y2 - y1) / unpad_h) * img.shape[0]
#             box_w = ((x2 - x1) / unpad_w) * img.shape[1]
#             y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
#             x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
#
#             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
#             # Create a Rectangle patch
#             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
#                                     edgecolor=color,
#                                     facecolor='none')
#             # Add the bbox to the plot
#             ax.add_patch(bbox)
#             # Add label
#             plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
#                     bbox={'color': color, 'pad': 0})
#
#     # Save generated image with detections
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(NullLocator())
#     plt.gca().yaxis.set_major_locator(NullLocator())
#     plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
#     plt.close()
