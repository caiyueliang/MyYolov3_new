# encoding: utf-8
from __future__ import division

import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyper_params = module_defs.pop(0)                               # 定义一个变量hyper_params来存储网络的信息
    prev_filters = 3
    # output_filters = [int(hyper_params["channels"])]              # 输入图片的通道数
    output_filters = []                                             # 输入图片的通道数
    module_list = nn.ModuleList()                                   # 返回值nn.ModuleList。这个类相当于一个包含nn.Module对象的普通列表
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    # in_channels=output_filters[-1],
                    in_channels=prev_filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            # print('layers', layers)
            # print('before layers', [output_filters[layer_i] for layer_i in layers])
            filters = sum([output_filters[layer_i] for layer_i in layers])
            # print('filters', filters)
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        # [yolo]
        # mask = 6, 7, 8
        # anchors = 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
        # classes = 80
        # num = 9
        # jitter = .3
        # ignore_thresh = .7
        # truth_thresh = 1
        # random = 1
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]             # anchors: [(116, 90), (156, 198), (373, 326)]
            num_classes = int(module_def["classes"])                # 类别个数
            img_height = int(hyper_params["height"])                # 输入图片的高度
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)                                 # 模块保存
        prev_filters = filters
        output_filters.append(filters)                              # 记录每层的输出维度,构建下一层时也会作为输入

    # print(output_filters)
    return hyper_params, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors                                      # anchors: [(116, 90), (156, 198), (373, 326)]
        self.num_anchors = len(anchors)                             # num_anchors: 3
        self.num_classes = num_classes                              # 类别个数
        self.bbox_attrs = 5 + num_classes                           # bbox的属性： 中心坐标，尺寸，目标分数（5个）和C个类
        self.image_dim = img_dim                                    # 输入图片的高度
        self.ignore_thres = 0.9                                     # iou的忽略阈值 0.5 TODO iou相关
        self.iou_match_thres = 0.9                                  # 真实框和预测框匹配上的阈值
        self.conf_thres = 0.95                                      # 置信度阈值
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)               # Coordinate loss   中心坐标损失函数
        self.bce_loss = nn.BCELoss(size_average=True)               # Confidence loss   置信度损失函数
        self.ce_loss = nn.CrossEntropyLoss()                        # Class loss        交叉熵

    def forward(self, x, targets=None):
        # targets: size (50, 5);每一行表示一个标签（最多50个），分别表示：类别，x轴中心点，y轴中心点，w，h
        # print('targets', targets)

        nA = self.num_anchors           # 锚框个数：3
        nB = x.size(0)                  # batch size大小：16
        nG = x.size(2)                  # 矩阵大小：13
        stride = self.image_dim / nG
        # print(x.size())
        # print(nA)
        # print(nB)
        # print(nG)
        # print(stride)

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # 输入x形状如(-1, 21, 13, 13)
        # 重新变换成矩阵维度为(nB, nA, self.bbox_attrs, nG, nG)，如(-1, 3, 7, 13, 13)
        # 再调整成(nB, nA, nG, nG, self.bbox_attrs)，如(-1, 3, 13, 13, 7)
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])           # Center x      size: (-1, 3, 13, 13)       x轴中心点
        y = torch.sigmoid(prediction[..., 1])           # Center y      size: (-1, 3, 13, 13)       y轴中心点
        w = prediction[..., 2]                          # Width         size: (-1, 3, 13, 13)       w
        h = prediction[..., 3]                          # Height        size: (-1, 3, 13, 13)       h
        pred_conf = torch.sigmoid(prediction[..., 4])   # Conf          size: (-1, 3, 13, 13)       置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])   # Cls pred.     size: (-1, 3, 13, 13, 2)    类别预测

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])   # 锚框，每层3个
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))               # (1, 3, 1, 1): [[[[2.5312]],[[4.2188]],[10.7500]]]]
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))               # (1, 3, 1, 1): [[[[2.5625]],[[5.2812]],[9.9688]]]]
        # print('grid_x', grid_x.size(), grid_x)                            # (1, 1, 13, 13) 值为：[[[0到12], [0到12]]]
        # print('scaled_anchors', scaled_anchors.size(), scaled_anchors)    # (3, 2) 值：[[2.5312, 2.5625],[4.2188, 5.2812],[10.7500, 9.9688]]
        # print('anchor_w', anchor_w.size(), anchor_w)
        # print('anchor_h', anchor_h.size(), anchor_h)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)                 # (-1, 3, 13, 13, 4)
        pred_boxes[..., 0] = x.data + grid_x                                # x轴中心点:关于grid_x（锚框中心点）的偏移，偏移值x.data
        pred_boxes[..., 1] = y.data + grid_y                                # y轴中心点:关于grid_y（锚框中心点）的偏移，偏移值y.data
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w                   # e的w次幂乘以基础锚框的宽
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h                   # e的h次幂乘以基础锚框的高

        # Training
        if targets is not None:
            # x训练代码
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            # nGT:真实标签个数; nCorrect:预测正确的个数
            # mask全为0,预测对的目标为1，用来预测目标
            # conf_mask全为1,预测对的目标为1？，以及与锚框IOU重叠度小的值为0？预测背景？
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,       # (-1, 3, 13, 13, 4)
                pred_conf=pred_conf.cpu().data,         # (-1, 3, 13, 13)
                pred_cls=pred_cls.cpu().data,           # (-1, 3, 13, 13, 2)
                target=targets.cpu().data,              # (-1, 50, 5)           最多50个框：类别,x,y,w,h
                anchors=scaled_anchors.cpu().data,      # (3, 2)                每层3各锚框：w,h
                num_anchors=nA,                         # 3
                num_classes=self.num_classes,           # 2
                grid_size=nG,                           # 13
                ignore_thres=self.ignore_thres,         # iou的忽略阈值
                iou_match_thres=self.iou_match_thres,   # 真实框和预测框匹配上的阈值 0.8
                conf_thres=self.conf_thres,             # 置信度阈值              0.9
                img_dim=self.image_dim,                 # 416
            )

            # print('nGT', nGT)                                               # 一个batch 中，实际标签个数       18
            # print('nCorrect', nCorrect)                                     # 预测正确的个数                  17

            nProposals = int((pred_conf > self.conf_thres).sum().item())    # 建议区域个数:置信度大于阈值的预测框的个数
            # print('nProposals', nProposals)

            recall = float(nCorrect / nGT) if nGT else 1                    # 召回率：预测正确的个数nCorrect除以实际标签个数nGT
            precision = float(nCorrect / nProposals) if nProposals else 0   # 精确率：预测正确的个数nCorrect除以建议区域个数nProposals

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask                           # mask全为0,预测对的目标为1，用来预测目标
            conf_mask_false = conf_mask - mask              # conf_mask全为1,以及与锚框IOU重叠度小的值为0，预测背景

            # Mask outputs to ignore non-existing objects
            # x, y, w, h 用MSELoss, conf 用BCELoss, 类别用CrossEntropyLoss
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + \
                        self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))

            # 总损失是上面各自的损失的累加
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, (
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyper_params, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
                # print(module_def["type"], x.size())

            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                # print(module_def["type"], x.size())

            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
                # print(module_def["type"], x.size())

            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    # x, *losses = module[0](x, targets)                # TODO
                    x, losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
                # print(module_def["type"], x.size())
            layer_outputs.append(x)

        self.losses["recall"] /= 3                                      # 有3个层，所以除以3
        self.losses["precision"] /= 3                                   # 有3个层，所以除以3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):
        print('save_weights: %s' % path)
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


# ================================================================================
# 测试前向传播：创建虚拟输入的函数，我们将把这个输入传递给我们的网络。
def get_test_input(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (416, 416))               # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))     # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0        # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()           # Convert to float
    img_ = Variable(img_)                           # Convert to Variable
    return img_


# 拿一张图片进行测试：该张量的形状为1 x 10647 x 85.第一个维度是批量大小，因为我们使用了单个图像，所以它的大小仅为1。
# 对于批次中的每个图像，我们都有一个10647 x 85的表格。该表格中的每一行代表一个边界框。（4个bbox属性，1个目标分数和80个类别分数）
def my_test(cfg_path, weights_path, image_path):
    model = Darknet(cfg_path)
    model.load_weights(weights_path)
    inp = get_test_input(image_path)
    print('[use GPU] %s' % torch.cuda.is_available())
    if torch.cuda.is_available():
        inp = inp.cuda()
        model = model.cuda()

    start = time.time()
    for i in range(100):
        pred = model(inp)

        # print(pred.data)
        # print(pred.shape)
        # for i in range(len(pred)):
        #     np.savetxt('output_' + str(i) + '.txt', pred[i])

        # output = write_results(pred, 0.8, 80)                       # 筛选网络输出结果，获取可信度高的方框
        # 每个检测有8个属性：即检测的图像在所属批次中的索引，4个角坐标，目标分数，最大置信度类别的分数以及该类别的索引。
        # print(output.data)

    end = time.time()
    print('time: %f' % ((end - start) / 100))


# ================================================================================
if __name__ == '__main__':
    cfg_path = 'config/yolov3.cfg'
    weights_path = 'weights/yolov3.weights'
    # cfg_path = 'config/yolov3-tiny.cfg'
    # weights_path = 'weights/yolov3-tiny.weights'

    # cfg_path = 'config/lpr_yolov3-tiny.cfg'

    test_image_path = 'data/samples/dog.jpg'
    my_test(cfg_path, weights_path, test_image_path)
