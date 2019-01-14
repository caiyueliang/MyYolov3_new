# encoding:utf-8
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def parse_argvs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--decay_epoch", type=int, default=60, help="decay_epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="lr")
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    # parser.add_argument("--model_config_path", type=str, default="config/lpr_yolov3.cfg", help="model_config_path")
    parser.add_argument("--model_config_path", type=str, default="config/lpr_yolov3-tiny.cfg", help="model_config_path")

    parser.add_argument("--checkpoint_name", type=str, default="checkpoints/lpr_yolo_tiny.weights", help="")
    # parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--train_path", type=str, default="../Data/yolo/yolo_data_new/car_detect_train", help="")
    parser.add_argument("--test_path", type=str, default="../Data/yolo/yolo_data_new/car_detect_test", help="")
    parser.add_argument("--image_file", type=str, default="image_path.txt", help="image_file")

    # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="../Data/yolo/yolo_data_new/lpr.names", help="")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    # parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="checkpoint_dir")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    parser.add_argument("--re_train", type=bool, default=False, help="re_train")
    parser.add_argument("--best_loss", type=float, default=10.0, help="best_loss")
    parser.add_argument("--detail_log", type=bool, default=False, help="detail_log")

    opt = parser.parse_args()
    print(opt)
    return opt


class ModuleTrain:
    def __init__(self, opt):
        self.opt = opt
        self.best_loss = self.opt.best_loss
        self.re_train = self.opt.re_train
        self.cuda = torch.cuda.is_available() and opt.use_cuda
        self.decay_epoch = self.opt.decay_epoch
        self.learning_rate = self.opt.lr

        try:
            os.makedirs("output")
        except:
            qwe = None
        try:
            os.makedirs("checkpoints")
        except:
            qwe = None

        self.classes = load_classes(opt.class_path)
        print('classes: %s' % self.classes)

        # Get data configuration
        self.train_path = self.opt.train_path
        self.test_path = self.opt.test_path
        self.image_file = self.opt.image_file

        # Get hyper parameters
        self.hyper_params = parse_model_config(opt.model_config_path)[0]
        self.momentum = float(self.hyper_params["momentum"])
        self.decay = float(self.hyper_params["decay"])
        self.burn_in = int(self.hyper_params["burn_in"])

        # Initiate model
        self.model = Darknet(opt.model_config_path)
        # model.load_weights(opt.weights_path)
        # 加载模型
        if os.path.exists(self.opt.checkpoint_name) and not self.re_train:
            print('[load model] ... %s' % self.opt.checkpoint_name)
            # self.model.load_weights(self.opt.checkpoint_name)
            self.load(self.opt.checkpoint_name)

        self.model.apply(weights_init_normal)

        if self.cuda:
            self.model = self.model.cuda()

        self.tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Get dataloader
        self.train_loader = torch.utils.data.DataLoader(ListDataset(self.train_path, self.image_file), shuffle=True,
                                                        batch_size=self.opt.batch_size, num_workers=self.opt.n_cpu)
        self.test_loader = torch.utils.data.DataLoader(ListDataset(self.test_path, self.image_file), shuffle=False,
                                                       batch_size=self.opt.batch_size, num_workers=self.opt.n_cpu)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)
        # self.model.save(name)

    def train(self):
        for epoch in range(self.opt.epochs):
            print("\n=========================================================")
            self.model.train()

            x_loss = 0.0
            y_loss = 0.0
            w_loss = 0.0
            h_loss = 0.0
            conf_loss = 0.0
            cls_loss = 0.0
            avg_recall = 0.0
            avg_precision = 0.0

            train_loss = 0.0

            if epoch >= self.decay_epoch and epoch % self.decay_epoch == 0:
                self.learning_rate *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            for batch_i, (_, imgs, targets) in enumerate(self.train_loader):
                imgs = Variable(imgs.type(self.tensor))
                targets = Variable(targets.type(self.tensor), requires_grad=False)

                self.optimizer.zero_grad()
                loss = self.model(imgs, targets)
                loss.backward()
                self.optimizer.step()

                if self.opt.detail_log:
                    print("[Epoch %d/%d, Batch %d/%d] [Losses: x %.5f, y %.5f, w %.5f, h %.5f, conf %.5f, cls %.5f,"
                          " total %.5f, recall: %.5f, precision: %.5f]" % (
                                epoch,
                                self.opt.epochs,
                                batch_i,
                                len(self.train_loader),
                                self.model.losses["x"],
                                self.model.losses["y"],
                                self.model.losses["w"],
                                self.model.losses["h"],
                                self.model.losses["conf"],
                                self.model.losses["cls"],
                                loss.item(),
                                self.model.losses["recall"],
                                self.model.losses["precision"],
                                )
                          )

                x_loss += self.model.losses["x"]
                y_loss += self.model.losses["y"]
                w_loss += self.model.losses["w"]
                h_loss += self.model.losses["h"]
                conf_loss += self.model.losses["conf"]
                cls_loss += self.model.losses["cls"]
                avg_recall += self.model.losses["recall"]
                avg_precision += self.model.losses["precision"]

                train_loss += loss.item()
                self.model.seen += imgs.size(0)

            train_loss /= len(self.train_loader)
            x_loss /= len(self.train_loader)
            y_loss /= len(self.train_loader)
            w_loss /= len(self.train_loader)
            h_loss /= len(self.train_loader)
            conf_loss /= len(self.train_loader)
            cls_loss /= len(self.train_loader)
            avg_recall /= len(self.train_loader)
            avg_precision /= len(self.train_loader)
            print ('Epoch [%d/%d] loss: %.5f lr: %.6f [x %.5f, y %.5f, w %.5f, h %.5f, conf %.5f, cls %.5f recall: %.5f, precision: %.5f]' %
                   (epoch + 1,
                    self.opt.epochs,
                    train_loss,
                    self.learning_rate,
                    x_loss,
                    y_loss,
                    w_loss,
                    h_loss,
                    conf_loss,
                    cls_loss,
                    avg_recall,
                    avg_precision
                    )
                   )

            test_loss = self.test()
            if self.best_loss > test_loss:
                self.best_loss = test_loss
                str_list = self.opt.checkpoint_name.split('.')
                best_model_file = ""
                for str_index in range(len(str_list)):
                    best_model_file = best_model_file + str_list[str_index]
                    if str_index == (len(str_list) - 2):
                        best_model_file += '_best'
                    if str_index != (len(str_list) - 1):
                        best_model_file += '.'
                # self.model.save_weights(best_model_file)      # 保存最好的模型
                self.save(best_model_file)                      # 保存最好的模型

        # self.model.save_weights(self.opt.checkpoint_name)     # 保存模型
        self.save(self.opt.checkpoint_name)                     # 保存模型

    def test(self):
        self.model.eval()
        test_loss = 0.0

        x_loss = 0.0
        y_loss = 0.0
        w_loss = 0.0
        h_loss = 0.0
        conf_loss = 0.0
        cls_loss = 0.0
        avg_recall = 0.0
        avg_precision = 0.0

        time_start = time.time()
        # 测试集
        for batch_i, (_, imgs, targets) in enumerate(self.test_loader):
            imgs = Variable(imgs.type(self.tensor))
            targets = Variable(targets.type(self.tensor), requires_grad=False)

            loss = self.model(imgs, targets)
            test_loss += loss.item()

            if self.opt.detail_log:
                print("[Batch %d/%d] [Losses: x %.5f, y %.5f, w %.5f, h %.5f, conf %.5f, cls %.5f,"
                      " total %.5f, recall: %.5f, precision: %.5f]" % (
                            batch_i,
                            len(self.train_loader),
                            self.model.losses["x"],
                            self.model.losses["y"],
                            self.model.losses["w"],
                            self.model.losses["h"],
                            self.model.losses["conf"],
                            self.model.losses["cls"],
                            loss.item(),
                            self.model.losses["recall"],
                            self.model.losses["precision"],
                            )
                      )

            x_loss += self.model.losses["x"]
            y_loss += self.model.losses["y"]
            w_loss += self.model.losses["w"]
            h_loss += self.model.losses["h"]
            conf_loss += self.model.losses["conf"]
            cls_loss += self.model.losses["cls"]
            avg_recall += self.model.losses["recall"]
            avg_precision += self.model.losses["precision"]

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))

        x_loss /= len(self.test_loader)
        y_loss /= len(self.test_loader)
        w_loss /= len(self.test_loader)
        h_loss /= len(self.test_loader)
        conf_loss /= len(self.test_loader)
        cls_loss /= len(self.test_loader)
        avg_recall /= len(self.test_loader)
        avg_precision /= len(self.test_loader)

        avg_loss = test_loss / len(self.test_loader)
        print('[Test] loss: %.5f time: %.5f [x %.5f, y %.5f, w %.5f, h %.5f, conf %.5f, cls %.5f,'
              ' recall: %.5f, precision: %.5f]' % (
                avg_loss,
                time_avg,
                x_loss,
                y_loss,
                w_loss,
                h_loss,
                conf_loss,
                cls_loss,
                avg_recall,
                avg_precision)
              )
        return avg_loss


if __name__ == '__main__':
    args = parse_argvs()

    model_train = ModuleTrain(opt=args)
    model_train.train()
