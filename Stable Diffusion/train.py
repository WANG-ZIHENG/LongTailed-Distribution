# coding: UTF-8
import numpy as np
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
warnings.filterwarnings("ignore")




def train(config, model, train_iter, dev_iter, test_iter):
    writer = SummaryWriter(log_dir=config.log_dir)
    start_time = time.time()
    model.load_state_dict(torch.load(config.save_path))

    model.train()


    optimizer = torch.optim.Adam(model.parameters() , lr=config.other_learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10, gamma=0.9, last_epoch=-1)

    total_batch = 0  # record how many batches
    dev_best_loss = float('inf')
    dev_best_acc = 0
    last_improve = 0  # record how many batches last time loss down on val_set
    flag = False  # record whether none improvement for a long time

    for epoch in range(config.num_epochs):
        loss_list=[]#loss of batch
        acc_list=[]
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            fea,outputs = model(trains)
            optimizer.zero_grad()

            if config.usesloss:
                bloss = F.cross_entropy(outputs, labels)
                sloss=SupConLoss()
                sloss=sloss(fea,labels=labels)
                floss=FocalLoss(100)
                floss=floss(outputs, labels)

                loss=(bloss+floss+sloss)/3
            else:
                loss = F.cross_entropy(outputs, labels)


            loss.backward()
            optimizer.step()

            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predic)
            writer.add_scalar('train/loss_iter', loss.item(),total_batch)
            writer.add_scalar('train/acc_iter',train_acc,total_batch)
            msg1 = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}'
            if total_batch%20==0:
                print(msg1.format(total_batch, loss.item(), train_acc))
            loss_list.append(loss.item())
            acc_list.append(train_acc)


            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # The verification set loss exceeds 2000 batches and does not drop, so the training ends
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


        dev_acc, dev_loss = evaluate(config, model, dev_iter)#model.eval()
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_bestloss)
            improve = '*'
            last_improve = total_batch
        else:
            improve = ''
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            torch.save(model.state_dict(), config.save_bestacc)
        torch.save(model.state_dict(), config.save_path)
        time_dif = get_time_dif(start_time)
        epoch_loss=np.mean(loss_list)
        epoch_acc=np.mean(acc_list)
        msg2 = 'EPOCH: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
        print(msg2.format(epoch+1,epoch_loss, epoch_acc, dev_loss, dev_acc, time_dif, improve))
        writer.add_scalar('train/loss_epoch',epoch_loss, epoch)
        writer.add_scalar('train/acc_epoch', epoch_acc, epoch)
        writer.add_scalar('val/loss_epoch', dev_loss, epoch)
        writer.add_scalar('val/acc_epoch', dev_acc, epoch)

        model.train()
        scheduler.step()
        print('epoch: ', epoch, 'lr: ', scheduler.get_last_lr())

    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...") #Precision and recall and harmonic mean
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:

            fea,outputs = model(texts)
            if config.usesloss:
                bloss = F.cross_entropy(outputs, labels)
                sloss=SupConLoss()
                sloss=sloss(fea,labels=labels)
                floss=FocalLoss(100)
                floss=floss(outputs, labels)

                loss=(bloss+floss+sloss)/3
            else:
                loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



if __name__ == '__main__':

    config = Config()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # Guarantee the same result every time

    print("Loading data...")


    train_data=My_Dataset(config.traincsv,config,1,1)
    dev_data = My_Dataset(config.valcsv,config,1,0)
    test_data = My_Dataset(config.testcsv,config,1,0)


    train_iter=DataLoader(train_data, batch_size=config.batch_size,shuffle=True)   ##training iterator
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size,shuffle=True)      ###validation iterator
    test_iter = DataLoader(test_data, batch_size=config.batch_size,shuffle=True)   ###test iterator
    # train
    mynet =Mynet(config)

    mynet= mynet.to(config.device)
    print(mynet.parameters)

    #After training, you can comment out the train function and only run the test to evaluate the model performance
    #test(config, mynet, test_iter)
    train(config, mynet, train_iter, dev_iter, test_iter)
    test(config, mynet, test_iter)

#tensorboard --logdir=log/resnet50 --port=6006