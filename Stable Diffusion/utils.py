import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from transformers import BertTokenizer
from Config import Config



class My_Dataset(Dataset):
    def __init__(self, path, config, iftrain, iftransformer=0):  #### read dataset
        self.config = config
        # training model, load data and labels
        self.iftrain = iftrain
        df = pd.read_csv(path)
        self.img_path = df['img'].to_list()  # [img]

        if iftransformer == 1:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),  # using ImageNet norms
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),  # using ImageNet norms
                                     (0.229, 0.224, 0.225))])

        # training model, load data and labels
        if self.iftrain == 1:
            self.labels = df['label'].to_list()  # [label]

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = self.transform(img)

        if self.iftrain == 1:
            label = int(self.labels[idx])
            label = torch.tensor(label, dtype=torch.long)
            return img.to(self.config.device), label.to(self.config.device)

        else:
            return img.to(self.config.device)

    def __len__(self):
        return len(self.img_path)  # total length of data


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    config = Config()
    train_data = My_Dataset(config.valcsv, config, 1)
    train_iter = DataLoader(train_data, batch_size=32)
    n = 0
    for a, b in train_iter:
        n = n + 1

        print(n, a.shape)
        # print(y)
        print('************')
