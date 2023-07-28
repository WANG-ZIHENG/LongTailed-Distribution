import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2., size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)+1e-8

        log_p = probs.log()+1e-8

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss+1e-8

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        input:
             features: features of the input samples, of size [batch_size, hidden_dim].
             labels: The ground truth label of each sample, the size is [batch_size].
             mask: The mask used for comparative learning, the size is [batch_size, batch_size], if samples i and j belong to the same label, then mask_{i,j}=1
         output:
             loss value
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # About the labels parameter
        if labels is not None and mask is not None:  #Labels and mask cannot define values at the same time, because if there is a label, then the mask needs to be obtained according to the Label
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # If there are no labels and no mask, it is unsupervised learning. The mask is a matrix with a diagonal of 1, indicating that (i,i) belong to the same class
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # If labels are given, the mask is obtained according to the label. When the labels of the two samples i and j are equal, mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        eg: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # When the labels of two samples i and j are equal，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the point product similarity between two samples
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits is the final similarity obtained by subtracting the maximum value of each row from anchor_dot_contrast
        eg: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # create mask
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask+1e-6
        negatives_mask = 1. - mask
        '''
        But for the calculation of Loss, the (i,i) position represents the similarity of the sample itself, 
        which is useless for Loss, so it needs to be masked
        # The ind position of the ind row is filled with 0
        get  logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # In addition to yourself, the number of positive samples  [2 0 2 2]
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)+1e-6

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] /( num_positives_per_row[
                        num_positives_per_row > 0]+1e-6)

        '''
        Calculate the log-likelihood of the positive sample average
        Considering that a category may have only one sample, there is no positive sample. For example, the second category of our labels labels[1,2,1,1]
        So here only calculate the number of positive samples > 0   
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss+1e-8


class Mynet(nn.Module):
    def __init__(self,config):
        super(Mynet, self).__init__()
        self.config=config

        #official method
        resnet_name=self.config.resnet_name# choose a kind of resnet
        if resnet_name=='resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet_name=='resnet34':
            self.resnet = models.resnet34(pretrained=True)
        elif resnet_name=='resnet50':
            self.resnet = models.resnext50_32x4d(pretrained=True)

        elif resnet_name=='resnet101':
            self.resnet = models.resnet101(pretrained=True)
        elif resnet_name=='resnet152':
            self.resnet = models.resnet152(pretrained=True)


        # modify the fully connected layer output
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs,self.config.resnet_fc )


        self.fc_1 = nn.Linear(self.config.resnet_fc, self.config.num_classes)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx):

        img=inx

        # attention_mask=mask
        img=self.resnet(img)

        fea=self.drop(img)
        logits = self.fc_1(img)
        # logits=self.softmax(logits)

        return fea,logits# fea is features needed to be compared,img is features of image，fea is all feature
