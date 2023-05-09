import numpy as np
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
#import torchvision
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import  matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
from model.Lenet5 import blnet
#from Lenet5 import Lenet
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
from sklearn import preprocessing
from scipy.signal import savgol_filter
from model.Lenet5 import D1
from model.Lenet5 import MyDataset


from model.Lenet5 import VGG16
import random
from model.Lenet5 import D2
from model.Lenet5 import BPNet
from model.Lenet5 import ALEXNet
from model.Lenet5 import Lenet
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn import svm
import datetime
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def train_one_epoch(model,optimizer,dataloader,device,epoch):
    model.train()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    train_sum_acc = 0.0
    sum_loss = 0.0  # 初始化损失度为0
    correct = 0.0  # 初始化，正确为0
    total = 0.0  # 初始化总数
    optimizer.zero_grad()
    dataloader = tqdm(dataloader)
    for i, data in enumerate(dataloader):
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
        #output,out_features = model(inputs)  # cnn output
        output = model(inputs)
        loss = loss_function(output, labels).to(device)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        sum_loss += loss.item()  # 每次loss相加，item 为loss转换为float
        _, predicted = torch.max(output.data,
                                 1)  # _ , predicte  d这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 max返回两个，第一个，每行最大的概率，第二个，最大概率的索引
        total += labels.size(0)  # 计算总的数据
        correct += (predicted == labels).cpu().sum().data.numpy()  # 计算相等的数据
        train_sum_acc += correct
        sum_loss += loss.cpu().detach().numpy()
        sum_loss = round(sum_loss, 6)
        dataloader.desc = ("train:epoch = {:} Loss = {:.4f}  Acc= {:.4f}".format((epoch + 1), (loss.item()),
                                                                                   (correct / total)))  # 训练次数，总损失，精确度
        optimizer.step()
        optimizer.zero_grad()
    #return loss.item(),  (correct / total),out_features,labels
    return loss.item(),  (correct / total)
@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total1 = 0
    correct1 = 0.0  # 准确度初始化0
    dataloader = tqdm(dataloader)
    for i, data in enumerate(dataloader):
        inputs1, labels1 = data  # 输入和标签都等于data
        inputs1 = Variable(inputs1).type(torch.FloatTensor).to(device)  # batch x
        labels1 = Variable(labels1).type(torch.LongTensor).to(device)  # batch y
        # inputs, labels = images.to(device), labels.to(device)  # 使用GPU
        #outputs1,out_features = model(inputs1)
        outputs1 = model(inputs1)
        loss1 = loss_function(outputs1, labels1).to(device)  # cross entropy loss
        _, predicted1 = torch.max(outputs1.data,
                                  1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
        total1 += labels1.size(0)  # 计算总的数据
        correct1 += (predicted1 == labels1).cpu().sum().data.numpy()  # 正确数量
        acc = correct1 / total1
        dataloader.desc = ("valid:epoch = {:} Loss = {:.4f}  Acc= {:.4f}".format((epoch + 1), (loss1.item()),
                                                                    (acc)))  # 训练次数，总损失，精确度
    return loss1.item(), correct1 / total1