# coding: UTF-8
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import sys
import torch
import numpy as np
from models import Mynet,SupConLoss
from tensorboardX import SummaryWriter
from utils import My_Dataset,get_time_dif
from models import *
from Config import Config
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

with open('data/cls_label.json','r')as f:
    cls_dict=json.load(f)

cls_dict=[(int(x[0]),x[1]) for x in cls_dict.items()]

print(cls_dict)
head=[x[0] for x in cls_dict if x[1]>100]
middle=[x[0] for x in cls_dict if 100>=x[1]>=20]
tail=[x[0] for x in cls_dict if x[1]<20]
print('Head',head)
print('Medium',middle)
print('Tail',tail)
config = Config()
test_data = My_Dataset(config.testcsv, config, 1,0)
test_iter = DataLoader(test_data, batch_size=64, shuffle=False)
# train
model = Mynet(config)

model = model.to(config.device)
# test
model.load_state_dict(torch.load(config.save_bestloss))#
model.eval()

alllabel=np.array([], dtype=int)#all labels of test set
allpre=np.array([], dtype=int)#prediction of test
with torch.no_grad():
    for texts, labels in tqdm(test_iter):


        fea, outputs = model(texts)
        labels = labels.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()  ###predicaiton
        alllabel = np.append(alllabel, labels)
        allpre = np.append(allpre , predic)
alllabel = alllabel.tolist()
allpre = allpre.tolist()
print(alllabel)

resdf=pd.DataFrame()
resdf['label']=alllabel
resdf['pre']=allpre

headacc=[]
middleacc=[]
tailacc=[]
print('Overall accuracy:',accuracy_score(alllabel,allpre))

for x in range(100):
    d=resdf[resdf['label']==x]
    acc=accuracy_score(d['label'].values,d['pre'].values)
    if x in head:
        headacc.append(acc)
    elif x in middle:
        middleacc.append(acc)
    elif x in tail:
        tailacc.append(acc)
print('Head accuracy：',np.mean(headacc))
print('Medium accuracy：',np.mean(middleacc))
print('Tail accuracy：',np.mean(tailacc))


