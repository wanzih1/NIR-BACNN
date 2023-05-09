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
from Lenet5 import blnet
#from Lenet5 import Lenet
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
from sklearn import preprocessing
from scipy.signal import savgol_filter
from Lenet5 import D1
from model.Lenet5 import MyDataset
from model.blnetnew import blnew
from model.blnetnew import blnew_se
from model.Lenet5 import ALEXNet_SE
from model.densenet import densenet121
#from model.d_densenet import densenet121_d
from model.d_densenet import DenseNet
from model.d_densenetse import DenseNetse
from utils.train_val_utilssvm import train_one_epoch
from utils.train_val_utilssvm import evaluate
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import joblib
from sklearn.metrics import recall_score,accuracy_score
from model.Lenet5 import VGG16
import random
from model.Lenet5 import D2
from model.Lenet5 import BPNet
from model.Lenet5 import ALEXNet
from model.Lenet5 import cnn1
from model.Lenet5 import Lenet
from model.resnet import resnet18
from sklearn.decomposition import PCA
from model.densenet_SE import densenet121_SE
from model.densenet_Nonlocal import densenet121_Non
import datetime
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#path = 'nirmyall2.CSV'
path = 'nirmyall-enhensenoise.CSV'
data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
# input ???,1,2074.
print(torch.cuda.is_available())
print("数据测试，直接数据导入")
data_x = data[1:, :-1]
data_y = data[1:, -1]
x_data = np.array(data_x)
y_data = np.array(data_y)
data_x=savgol_filter(x_data,40, 1) # window size 51, polynomial order 3+
#x = np.zeros((906,6950))
# for i in range(0,906,1):
#     for j in range(0,6950,1):
#         data_x[i][j] = data_x[i][j] + random.gauss(0,0.03)
df = pd.DataFrame(data_x)
df.to_csv('PBMC_GS.csv',index= False, header= False)