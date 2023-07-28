import pandas as pd
import json
from PIL import Image
import numpy as np

dforg=pd.read_csv('data/train.csv')#raw long tail data
dfgen=pd.read_csv('data/gentrain.csv',header=None)
dfgen.columns=['img','label']
print(dfgen)
label_s={}
for x in range(100):
    l=dforg[dforg['label']==x]
    label_s[x]=len(l)
print(label_s)
with open('data/cls_label.json','w')as f:
    json.dump(label_s,f)

dfnewgen=[]#The generated image is partially damaged and pure black, we will remove it
for x in range(len(dfgen)):
    m,n=dfgen.iloc[x,:]
    mm=m
    m=Image.open(m)
    m=np.mean(m)
    if m!=0:
        dfnewgen.append((mm,n))
dfnewgen=pd.DataFrame(dfnewgen,columns=['img','label'])
print(dfnewgen)
dfall=pd.concat([dforg,dfnewgen],axis=0)#Long tail data and merged data are combined into new training data
print(dfall)
#
dfall.to_csv('data/alltrain.csv',index=None)