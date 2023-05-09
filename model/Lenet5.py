import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec, target = self.specs[index], self.labels[index]
        return spec, target

    def __len__(self):
        return len(self.specs)

def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  #绘制贡献率
  tot = sum(eig_val)
  var_exp = [(i / tot) for i in sorted(eig_val, reverse=True)]# 按照降序排列特征值，并计算贡献率
  cum_var_exp = np.cumsum(var_exp)  # 累计贡献度
  plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center', label='individual var')  # 绘制柱状图，
  plt.step(range(1, len(var_exp) + 1), cum_var_exp, where='mid', label='cumulative var')  # 绘制阶梯图
  plt.ylabel('variance rtion')  # 纵坐标
  plt.xlabel('principal components')  # 横坐标
  plt.legend(loc='best')  # 图例位置，右下角
  plt.show()
  return data

def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di

def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        assert channel > reduction, "Make sure your input channel bigger than reduction which equals to {}".format(reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class blnet(nn.Module):
    def __init__(self):
        super(blnet, self).__init__()
        self.conv7_1 = nn.Sequential(
            nn.Conv1d(1,16,7,1,1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv7_2 = nn.Sequential(

            nn.Conv1d(16,16,7,1,1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #SELayer(16)

        )
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(16,64,3,1,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #SELayer(64)
        )
        self.conv1_2 = nn.Sequential(

            nn.Conv1d(64,256,3,1,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SELayer(256)
        )

        self.conv2_1 = nn.Sequential(

            nn.Conv1d(16, 64, 5, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #SELayer(64)
        )
        self.conv2_2 = nn.Sequential(

            nn.Conv1d(64, 256, 5, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SELayer(256)
        )

        self.fc = nn.Sequential(
            #nn.Linear(6400,6946),
            nn.Linear(512, 4096),
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 6)
        )

    def forward(self, x):
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        x1 = x
        x2 = x
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = F.adaptive_max_pool1d(x1, output_size=1)

        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = F.adaptive_max_pool1d(x2, output_size=1)

        out = torch.cat((x1, x2), dim=1)
        x3 = out.view(out.size(0), -1)
        x = self.fc(x3)
        return x



class BPNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(BPNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.Dropout(0.6))
        #self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_1, out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        #x = self.layer2(x)
        x = self.layer3(x)


        return x
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Sequential(

            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPool1d(2, 2)
        )
        # self.conv1_2 = nn.Sequential(
        #
        #     nn.Conv1d(64, 64, 3, 1, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2,2)
        # )
        self.conv2_1 = nn.Sequential(

            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        # self.conv2_2 = nn.Sequential(
        #
        #     nn.Conv1d(128, 128, 3, 1, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2, 2)
        # )
        self.conv3_1 = nn.Sequential(

            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.MaxPool1d(2, 2)
        )

        self.conv3_2 = nn.Sequential(

            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )

        self.conv4_1 = nn.Sequential(

            nn.Conv1d(128, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.MaxPool1d(2, 2)
        )

        self.conv4_2 = nn.Sequential(

            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.conv5_1 = nn.Sequential(

            nn.Conv1d(256,256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.MaxPool1d(2, 2)
        )

        self.conv5_2 = nn.Sequential(

            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(1, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 6),

        )
    def forward(self, x):
        x = self.conv1_1(x)
        #x = self.conv1_2(x)
        x = self.conv2_1(x)
        #x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = F.adaptive_max_pool1d(x, output_size=1)
        x1 = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ALEXNet(nn.Module):
    def __init__(self):
        super(ALEXNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 96, 11, 4, 1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            SELayer(96)
            #nn.MaxPool1d(3,2)
        )
        # self.conv1_2 = nn.Sequential(
        #
        #     nn.Conv1d(64, 64, 3, 1, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2,2)
        # )
        self.conv2 = nn.Sequential(

            nn.Conv1d(96, 256, 5, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SELayer(256)
            #nn.MaxPool1d(3,2)
        )
        # self.conv2_2 = nn.Sequential(
        #
        #     nn.Conv1d(128, 128, 3, 1, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2, 2)
        # )
        self.conv3 = nn.Sequential(

            nn.Conv1d(256, 384, 3, 1, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            SELayer(384)
        )

        self.conv4 = nn.Sequential(

            nn.Conv1d(384, 384, 3, 1, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            SELayer(384)
        )

        self.conv5 = nn.Sequential(

            nn.Conv1d(384, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SELayer(256)
            #nn.MaxPool1d(3,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 6),

        )
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv1_2(x)
        x = self.conv2(x)
        #x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, output_size=1)
        x1 = x.view(x.size(0), -1)
        x = self.fc(x1)
        return x

class ALEXNet_SE(nn.Module):
    def __init__(self):
        super(ALEXNet_SE, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv1d(1, 96, 11, 4, 1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            #nn.MaxPool1d(3,2)
        )
        # self.conv1_2 = nn.Sequential(
        #
        #     nn.Conv1d(64, 64, 3, 1, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2,2)
        # )
        self.conv2 = nn.Sequential(

            nn.Conv1d(96, 256, 5, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.MaxPool1d(3,2)
        )
        # self.conv2_2 = nn.Sequential(
        #
        #     nn.Conv1d(128, 128, 3, 1, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2, 2)
        # )
        self.conv3 = nn.Sequential(

            nn.Conv1d(256, 384, 3, 1, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(

            nn.Conv1d(384, 384, 3, 1, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(

            nn.Conv1d(384, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SELayer(256),
            #nn.MaxPool1d(3,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 6),

        )
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv1_2(x)
        x = self.conv2(x)
        #x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, output_size=1)
        x1 = x.view(x.size(0), -1)
        x = self.fc(x1)
        return x

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv1d(1, 6, 5, 1, 1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            #nn.AvgPool1d(2,2)
        )
        # self.conv1_2 = nn.Sequential(
        #
        #     nn.Conv1d(64, 64, 3, 1, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     #nn.AvgPool1d(2,2)
        # )
        self.conv2 = nn.Sequential(

            nn.Conv1d(6, 16, 5, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.AvgPool1d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16, 1024),
            nn.Linear(1024, 84),
            nn.Linear(84, 6),

        )
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv1_2(x)
        x = self.conv2(x)
        #x = self.conv2_2(x)
        x = F.adaptive_max_pool1d(x, output_size=1)
        x1 = x.view(x.size(0), -1)
        x = self.fc(x1)
        return x
def main():
    #net = Lenet5()
    net = blnet()
    #生成数据
    # tmp = torch.randn(1200, 1, 125)
    # out = net(tmp)
    # print(out[1][:])

if __name__ == "__main__":
    main()