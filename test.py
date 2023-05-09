import numpy as np
import torch
import matplotlib.ticker as ticker
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

import  matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils, plot_model
from sklearn import preprocessing
from scipy.signal import savgol_filter
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.signal import savgol_filter
from model.Lenet5 import MyDataset
from model.blnetnew import blnew
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scipy import interp
import time
from model.Lenet5 import ALEXNet
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from itertools import cycle
from sklearn.decomposition import PCA
from model.d_densenet import DenseNet
from model.d_densenetse import DenseNetse
from sklearn.preprocessing import label_binarize
from model.densenet import densenet121
from model.Lenet5 import cnn1
from model.Lenet5 import Lenet
from model.Lenet5 import blnet
from model.resnet import resnet18

config = {
            "font.family": 'serif',
            "font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }


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
test_ratio = 0.3
EPOCH = 200 # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 800

path = 'nirmyal.CSV'
train_result_path = 'train_result.csv'
test_result_path = 'test_result.csv'

data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

# input ???,1,2074
print(torch.cuda.is_available())
print("数据测试，直接数据导入")
data_x = data[1:, :-1]
data_y = data[1:, -1]

data_x=savgol_filter(data_x, 11, 2) # window size 51, polynomial order 3

pca = PCA(n_components=800)
data_x = pca.fit_transform(data_x)

x_lenth = len(data_x[1, :])
print(data_y)
print(x_lenth)

x_data = np.array(data_x)

y_data = np.array(data_y)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio, random_state=3)

X_test_Nom = scale(X_test)

X_test_Nom = X_test_Nom[:, np.newaxis, :]


##使用loader加载测试数据
data_test = MyDataset(X_test_Nom, y_test)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)
print(test_loader)

device = torch.device('cuda')
#model = Lenet5()
model = blnet()

#model = VGGsigmoid()
#model = Lenet()
#model = ALEXNet()
#model = VGG16()
#model = BPNet(in_dim=800,n_hidden_1=4096,n_hidden_2=1024,out_dim=6)
# criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)
#log_dir = 'cifar/结果/ddensenet/nose-sg/model-result.pth'
log_dir = 'cifar/model-result.pth'
#log_dir = 'cifar/结果/resnet18/model-result.pth'
    #log_dir = 'result/bp/model-result.pth'
checkpoint = torch.load(log_dir)
model.load_state_dict(checkpoint['model1'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']
with torch.no_grad():  # 无梯度
    # sum_acc = 0.0
    # for index in range(10):
    correct = 0.0  # 准确度初始化0
    total = 0.0  # 总量为0
    for data in test_loader:
        model.eval()  # 不训练
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor)  # batch x
        labels = Variable(labels).type(torch.LongTensor)  # batch y
        #inputs, labels = images.to(device), labels.to(device)  # 使用GPU
        outputs, out_features = model(inputs) # 输出等于进入网络后的输入
        _, predicted = torch.max(outputs.data,
                                 1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
        total += labels.size(0)  # 计算总的数据
        correct += (predicted == labels).sum().cpu()  # 正确数量
    acc = 100. * correct / total
    print("Acc= {:.4f}".format((acc)))  # 训练次数，总损失，精确度



    f1 = f1_score(y_test, predicted, average='micro')
    p = precision_score(y_test, predicted, average='micro')
    r = recall_score(y_test, predicted, average='micro')
    print('f1')
    print(f1)
    print('p')
    print(p)
    print('R')
    print(r)
    a = classification_report(y_test, predicted, labels=[0, 1, 2,3,4,5],digits=4)

    print(a)

    # 一般字体统一用一个字典控制
    font = {'family': 'Times New Roman"',
            #'style': 'italic',  # 斜体，正常条件下注释就行
            'weight': 'normal',
            'color': 'black',
            'size': 12
            }

    #print("Acc= {:.4f}".format(acc))  # 训练次数，总损失，精确度
    # print("预测值")
    # print(predicted)
    # print("真实值")
    # print(y_test)
    print('测试集混淆矩阵为：\n', confusion_matrix(y_test, predicted))
    print('平均分类准确率为：\n', accuracy_score(y_test, predicted))
    # plt.scatter(range(len(y_test)), y_test, color='red', label='True')
    # plt.scatter(range(len(predicted)), predicted, color='blue', label='Predicted')
    plt.scatter(range(len(y_test)), y_test, color='red')
    plt.scatter(range(len(predicted)), predicted, color='blue')
    plt.xlabel('Number', fontdict=font)
    plt.ylabel('Class', fontdict=font)
    plt.legend()
    plt.savefig("./result.tiff")
    plt.show()
#****************************************************************************

    confusion = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(5, 5))  # 设置图片大小

    # 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar()  # 右边的colorbar

    # 2.设置坐标轴显示列表
    indices = range(len(confusion))
    classes = ['A', 'B', 'C', 'D', 'E', 'F']
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, classes, rotation=45)  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes)

    # 3.设置全局字体
    # 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
    # ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 4.设置坐标轴标题、字体
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion matrix')

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=12)  # 可设置标题大小、字体

    # 5.显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for i in range(len(confusion)):  # 第几行
        for j in range(len(confusion[i])):  # 第几列
            plt.text(j, i, format(confusion[i][j], fmt),
                     fontsize=16,  # 矩阵字体大小
                     horizontalalignment="center",  # 水平居中。
                     verticalalignment="center",  # 垂直居中。
                     color="white" if confusion[i, j] > thresh else "black")

    # 6.保存图片
    plt.savefig("./confusion_matrix.tiff")
    # 7.显示
    plt.show()
#*******************************************************************


    nb_classes = 6
    # Binarize the output
    Y_valid = label_binarize(y_test, classes=[i for i in range(nb_classes)])
    Y_pred = label_binarize(predicted, classes=[i for i in range(nb_classes)])

    # micro：多分类　　
    # weighted：不均衡数量的类来说，计算二分类metrics的平均
    # macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
    precision = precision_score(Y_valid, Y_pred, average='micro')
    recall = recall_score(Y_valid, Y_pred, average='micro')
    f1_score = f1_score(Y_valid, Y_pred, average='micro')
    accuracy_score = accuracy_score(Y_valid, Y_pred)
    print("Precision_score:", precision)
    print("Recall_score:", recall)
    print("F1_score:", f1_score)
    print("Accuracy_score:", accuracy_score)

    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 3
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4, )

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','grey','olive','red','g'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=4 )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate', fontdict= font)
    # plt.ylabel('True Positive Rate', fontdict= font)
    plt.xlabel('False Positive Rate', fontsize=15,fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=15,fontweight='bold')
    plt.yticks(fontproperties='Times New Roman', size=12, weight='bold')  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=12, weight='bold')

    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(12))

    plt.savefig('ROC.tiff',format='tiff')
    plt.show()