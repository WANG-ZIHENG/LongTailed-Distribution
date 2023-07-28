from models import *
from Config import Config
from torch.utils.data import DataLoader
from utils import My_Dataset,get_time_dif
from models import *
from torchvision import transforms as transforms
from PIL import Image
import json
from matplotlib import pyplot as plt

import os; os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

config=Config()


img=Image.open(r'data\images\n0153282900000005.jpg')
#Two lines of code for displaying pictures, commented out for batch prediction
plt.imshow(img)
plt.show()

with open(config.index_label,'r')as f:
    index_label=json.load(f)
print(index_label)
#
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize((0.485, 0.456, 0.406),  # using ImageNet norms
                                 (0.229, 0.224, 0.225))])

img=transform(img).unsqueeze(0)#Because of a single image, a batch dimension needs to be added
print(img.shape)
img=img.to(config.device)
model = Mynet(config)

model = model.to(config.device)
model.load_state_dict(torch.load(config.save_path))#Load the trained model
model.eval()


with torch.no_grad():
    fea, outputs = model(img)
    predic = torch.max(outputs.data, 1)[1].cpu().numpy()  ###prediction
    predict=predic[0]#one image one prediction
    print('class:',predict)
    print('label:',index_label[str(predict)])