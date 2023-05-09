#from sklearn.model_selection import cross_val_score, train_test_split, KFold
#from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import codecs
import csv
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import urllib.request
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
#import seaborn as sns
from matplotlib import pyplot as plt
#url = 'https://raw.githubusercontent.com/onecoinbuybus/Database_chemoinformatics/master/shootout_2012_full_scale.csv'
#urllib.request.urlretrieve(url, 'shootout_2012_full_scale.csv')
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scipy import interp
import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import confusion_matrix
#from keras.models import load_model
from itertools import cycle
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import random

#多元散射校正
def msc(X):
    # +++++ 输入：X = m × p 矩阵，m个样本,p个特征。X 应为 ndarray 数据类型）
    # +++++ 输出：X_msc = m × p
    me = np.mean(X, axis=0)
    [m, p] = np.shape(X)
    X_msc = np.zeros((m, p))

    for i in range(m):
        poly = np.polyfit(me, X[i], 1)  # 每个样本做一次一元线性回归
        j = 0
        for j in range(p):
         X_msc[i, j] = (X[i, j] - poly[1]) / poly[0]

    return X_msc
#标准化
def normalize(X):
    mean = np.mean(X)  # 均值
    std = np.std(X)  # 默认计算每一列的标准差
    X = (X - mean) / std
    return X

def MSC(data_x):
    ## 计算平均光谱做为标准光谱
    mean = np.mean(data_x, axis=0)

    n, p = data_x.shape
    msc_x = np.ones((n, p))

    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_
        b = lin.intercept_
        msc_x[i, :] = (y - b) / k
    return msc_x

#x = pca(x,105)
#一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di

# 二阶导数
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di



housing_data_train =  pd.read_csv('nirmyall.CSV')
housing_data_train = housing_data_train.values

X_data = housing_data_train[1:, :-1]
y_data = housing_data_train[1:, -1]
# for i in range(0, 906, 1):
#     for j in range(0, 6950, 1):
#         X_data[i][j] =  X_data[i][j] + random.gauss(0, 0.02)
X_data=savgol_filter(X_data, 11, 2) # window size 51, polynomial order 3
pca = PCA(n_components=800)
X_data = pca.fit_transform(X_data)
# print(pca.explained_variance_ratio_)
# data_x_1 = X_data[:, 124]
# X_data = np.column_stack([X_data, data_x_1])
# X_data = D1(X_data)
#X_data=savgol_filter(X_data, 11, 2) # window size 51, polynomial order 3
#X_data = normalize(X_data)
#X_data = pca(X_data,128)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=3)

# 训练模型

#model = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=3))
model = OneVsOneClassifier(svm.SVC(kernel='rbf', probability=True, random_state=10))
print("[INFO] Successfully initialize a new model !")
print("[INFO] Training the model…… ")
clt = model.fit(X_train, y_train)
print("[INFO] Model training completed !")
# 保存训练好的模型，下次使用时直接加载就可以了
joblib.dump(clt, "cifar/model")
print("[INFO] Model has been saved !")

y_test_pred = clt.predict(X_test)
ov_acc =accuracy_score(y_test_pred, y_test)
print("overall accuracy: %f" % (ov_acc))
print("===========================================")
acc_for_each_class = precision_score(y_test, y_test_pred, average=None)
print("acc_for_each_class:\n", acc_for_each_class)
print("===========================================")
avg_acc = np.mean(acc_for_each_class)
print("average accuracy:%f" % (avg_acc))
print('real')
print(y_test)
print('pred')
print(y_test_pred)

f1 = f1_score(y_test, y_test_pred, average='macro')
p = precision_score(y_test, y_test_pred, average='macro')
r = recall_score(y_test, y_test_pred, average='macro')
print('f1')
print(f1)
print('p')
print(p)
print('R')
print(r)

a = classification_report(y_test, y_test_pred, labels=[0, 1, 2, 3, 4, 5], digits=4)

print(a)
plt.figure()
plt.scatter(range(len(y_test)),y_test, color='red', label='real rank')
plt.scatter(range(len(y_test_pred)),y_test_pred, color='blue', label='svm predicted')
plt.legend()
plt.show()
print('测试集混淆矩阵为：\n', confusion_matrix(y_test, y_test_pred))
print('平均分类准确率为：\n', accuracy_score(y_test, y_test_pred))

nb_classes = 6
# Binarize the output
Y_valid = label_binarize(y_test, classes=[i for i in range(nb_classes)])
Y_pred = label_binarize(y_test_pred, classes=[i for i in range(nb_classes)])


# micro：多分类　　
# weighted：不均衡数量的类来说，计算二分类metrics的平均
# macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
precision = precision_score(Y_valid, Y_pred, average='micro')
recall = recall_score(Y_valid, Y_pred, average='micro')
f1_score = f1_score(Y_valid, Y_pred, average='micro')
accuracy_score = accuracy_score(Y_valid, Y_pred)
print("Precision_score:",precision)
print("Recall_score:",recall)
print("F1_score:",f1_score)
print("Accuracy_score:",accuracy_score)


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
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','grey','olive','red','g'])
for i, color in zip(range(nb_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
plt.savefig('svmROC预处理.svg',format='svg')
plt.show()



