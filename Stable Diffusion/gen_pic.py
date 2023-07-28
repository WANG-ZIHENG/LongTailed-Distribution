# house finch
import json
import os.path
import random

from tqdm import tqdm
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import csv
import cv2

#git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
#https://huggingface.co/runwayml/stable-diffusion-v1-5
# Load the image generation pipeline
model_id =pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, "stable-diffusion-v1-5"
 torch_dtype=torch.float16,use_auth_token='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').to(
    "cuda")
# call sd to generate
def gen_pic(img,text,res,label):


    # read initial image
    init_image = Image.open(img).convert("RGB")
    init_image = init_image.resize((100, 100))

    # reasoning
    prompt = text
    generator = torch.Generator(device="cuda").manual_seed(random.randint(0,9999))

    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.8,
        guidance_scale=7.5,
        generator=generator,

    ).images[0]
    image.save(res)
    with open('data/gentrain.csv','a',encoding='utf-8',newline='')as f:
        writer=csv.writer(f)
        writer.writerow((res,label))

with open('data/classname.txt','r')as f:
    text_label=f.read()
text_label=json.loads(text_label)
text_label=[[x[0],x[1].replace('_',' ')] for x in text_label.values()]
text_label=dict(text_label)
print(text_label)#semantic label

df=pd.read_csv('data/train.csv')
dforg=df.copy()
df['dir']=df['img'].apply(lambda x:x.split('/')[-1][:9])
df['text']=df['dir'].apply(lambda x:text_label[x])
print(df)

gendir='data/genimages/'#The storage path of the generated image
if not os.path.exists(gendir):
    os.makedirs(gendir)

n=0
for x in tqdm(range(100)):
    d=df[df['label']==x]#Take out the training data for each class
    gnum=480-len(d)#Number of data to be generated
    i=0
    while 1:
        for y in tqdm(range(len(d)),'Created within a category...'):
            img,_,_,text=d.iloc[y,:]
            print(img,text)
            gen_pic(img,text,gendir+str(n)+'.jpg',x)
            i=i+1
            n=n+1
            if i>=gnum:
                break
        if i >= gnum:
            break
