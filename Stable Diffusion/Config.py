import os.path
import torch

#asl 28 signn24 src26
#Determine the specific structure of the model by modifying self.bert_name and self.resnet_name
class Config(object):
    def __init__(self):

        self.data='data/'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.3
        self.require_improvement = 20000  #If the effect has not improved after more than 2000 batches, the training will end early
        self.num_epochs = 100   # epoch数
        self.batch_size=120
        self.resnet_learning_rate = 5e-4 #The learning rate of resnet is preferably slightly higher than that of bert
        self.other_learning_rate = 5e-4#Learning rates for other layers


        #['resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']
        self.resnet_name='resnet50'#kinds of resnet
        self.resnet_fc=1024#resnet fully connected output dimension
        self.usesloss=True#Whether to use contrastive learning
        self.num_classes=100

        if not os.path.exists('model'):
            os.makedirs('model')

        self.traincsv=self.data+'alltrain.csv'
        self.testcsv=self.data+'test.csv'
        self.valcsv=self.data+'val.csv'
        self.index_label=self.data+'index.json'

        self.save_path = 'model/'+self.resnet_name+'.ckpt'#The path to save the model
        self.save_bestacc= 'model/'+'bestacc.ckpt'#The path to save the model
        self.save_bestloss= 'model/'+'bestloss.ckpt'#The path to save the model
        self.log_dir= './log/'+self.resnet_name#Path to tensorboard log



