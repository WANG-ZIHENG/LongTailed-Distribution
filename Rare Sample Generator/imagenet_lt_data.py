from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import json
import numpy as np


class ImageNet_LT(data.Dataset):
    """Dataset class for the Imagenet-LT dataset."""

    def __init__(self, data_dir, cls_num, transform, mode):
        """Initialize and preprocess the Imagenet-LT dataset."""
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.train_txt = '%s/train.csv' % data_dir
        self.val_txt = '%s/val.csv' % data_dir
        self.test_txt = '%s/test.csv' % data_dir
        self.cls_num_list_train = [0] * cls_num
        self.cls_num_list_val = [0] * cls_num
        self.cls_num_list_test = [0] * cls_num
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == "val":
            self.num_images = len(self.val_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        train_file = open(self.train_txt, "r")
        val_file = open(self.val_txt, "r")
        test_file = open(self.test_txt, "r")

        for elem in train_file.readlines()[1:]:
            filename = elem.split(',')[0]
            if filename.startswith('data/'):
                filename = filename[5:]
            filename = '%s/%s' % (self.data_dir, filename)
            label = int(elem.split(',')[1])
            self.train_dataset.append([filename, label])
            self.cls_num_list_train[label] += 1

        for elem in val_file.readlines()[1:]:
            filename = elem.split(',')[0]
            if filename.startswith('data/'):
                filename = filename[5:]
            filename = '%s/%s' % (self.data_dir, filename)
            label = int(elem.split(',')[1])
            self.val_dataset.append([filename, label])
            self.cls_num_list_val[label] += 1

        for elem in test_file.readlines()[1:]:
            filename = elem.split(',')[0]
            if filename.startswith('data/'):
                filename = filename[5:]
            filename = '%s/%s' % (self.data_dir, filename)
            label = int(elem.split(',')[1])
            self.test_dataset.append([filename, label])
            self.cls_num_list_test[label] += 1

    def __getitem__(self, index):
        """Return one image and its corresponding label."""
        if self.mode == "train":
            dataset = self.train_dataset
        elif self.mode == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        filename, label = dataset[index]
        image = Image.open(filename).convert('RGB')

        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

    def get_cls_num_list(self):
        if self.mode == "train":
            print(self.cls_num_list_train)
            return self.cls_num_list_train
        elif self.mode == "val":
            return self.cls_num_list_val
        else:
            return self.cls_num_list_test
