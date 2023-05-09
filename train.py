import numpy as np
import torch

from torch.utils.data import Dataset
import  matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
from model.Lenet5 import blnet
from scipy.signal import savgol_filter
from model.Lenet5 import MyDataset
from utils.train_val_utils import train_one_epoch
from utils.train_val_utils import evaluate
from model.Lenet5 import VGG16
from model.Lenet5 import D2
from model.Lenet5 import BPNet
from model.Lenet5 import ALEXNet
from model.Lenet5 import Lenet
from sklearn.decomposition import PCA
import datetime
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di

def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di

def main():
    test_ratio = 0.3
    EPOCH = 200  # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 800

    path = 'nirmyall.CSV'
    #path = 'nirmyall.CSV'
    train_result_path = 'result/train_result.csv'
    test_result_path = 'result/test_result.csv'

    data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

    # input ???,1,2074
    print(torch.cuda.is_available())
    print("数据测试，直接数据导入")
    data_x = data[1:, :-1]
    data_y = data[1:, -1]
    data_x=savgol_filter(data_x, 11, 2) # window size 51, polynomial order 3+
    #data_x = D1(data_x)
    pca = PCA(n_components=800)
    data_x = pca.fit_transform(data_x)
    x_data = np.array(data_x)


    y_data = np.array(data_y)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio,random_state=3)
    print(X_train.shape)
    print(X_test.shape)
    ##均一化处理
    X_train_Nom = scale(X_train)
    X_test_Nom  = scale(X_test)
    X_train_Nom = X_train_Nom[:, np.newaxis, :]
    X_test_Nom = X_test_Nom[:, np.newaxis, :]
    data_train = MyDataset(X_train_Nom,y_train)

    train_loader = torch.utils.data.DataLoader(data_train,batch_size=BATCH_SIZE,shuffle=True)
    ##使用loader加载测试数据
    data_test = MyDataset(X_test_Nom,y_test)
    test_loader = torch.utils.data.DataLoader(data_test,batch_size=BATCH_SIZE,shuffle=False)
    device = torch.device('cuda')

    model = blnet().to(device)
    #model = densenet121_d().to(device)
    #model = densenet121().to(device)
    #model = resnet18().to(device)
    #model = densenet121_SE().to(device)
    #model = densenet121_Non().to(device)
    #model = VGG16().to(device)
    #model = ALEXNet().to(device)
    #model = densenet121().to(device)
    #model = Lenet().to(device)
    print('*************************************')
    #model = BPNet(in_dim=6950, n_hidden_1=4096, n_hidden_2=1024,out_dim=6).to(device)
    #model = VGGsigmoid().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    # lf = lambda x: ((1 + math.cos(x * math.pi / EPOCH)) / 2) * (1 - 0.01) + 0.01
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3)
    #optimizer = optim.Adagrad(model.parameters(), lr=1e-3)
    print(model)

    log_dir = 'result/model-result.pth'

    with open(train_result_path, "w") as f1:
        f1.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
        f1.write('\n')
        with open(test_result_path, "w") as f2:
            f2.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
            f2.write('\n')
            starttime = datetime.datetime.now()
            for epoch in range(EPOCH):
                train_loss, train_acc = train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    device=device,
                    epoch=epoch
                )
                # lr = lf(epoch)
                # scheduler.step()
                f1.write("{:},{:.4f},{:.4f}".format((epoch + 1), (train_loss), (train_acc)))  # 写入数据
                f1.write('\n')
                f1.flush()
                val_loss, val_acc = evaluate(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    epoch=epoch
                )
                f2.write("{:},{:.4f},{:.4f}".format((epoch + 1), (val_loss), (val_acc)))  # 写入数据
                f2.write('\n')
                f2.flush()
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, log_dir)
            endtime = datetime.datetime.now()
            print((endtime-starttime).seconds)

        # 将每次测试结果实时写入acc.txt文件中





if __name__ == '__main__':
    main()
