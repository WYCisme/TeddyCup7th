# -*- coding:utf-8 -*-
'''
通过SVM预测
'''
#兼容python2和python3的print函数
from __future__ import print_function

import csv
from time import time
import logging     #打印日志
import matplotlib.pyplot as plt    #这个库具有绘图功能
 #交叉验证模块
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC



print(__doc__)

csv_file = csv.reader(open('patienLable.csv'))
plable = []
for item in csv_file:
    plable.append(item)
    # print(item)
#打印程序进展信息
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')#打印过程日志

lfw_people = fetch_lfw_people(data_home='./imgData', min_faces_per_person=10, resize=0.4)#下载名人库的数据集
n_samples,h,w = lfw_people.images.shape  #返回数据集的特征值

x = lfw_people.data  #获取数据集特征向量的矩阵
n_features = x.shape[1]  #获取数据集特征向量的维度

y = lfw_people.target  #获取目标标记
target_names = lfw_people.target_names  #获取目标标记的类别值
n_classes = target_names.shape[0]  #返回数据集中有多少类，有多少个人

print('Total dataSet size:')
print("n_samples:%d"%n_samples)
print("n_futures:%d"%n_features)
print("n_classes:%d"%n_classes)

#将数据集拆分为训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

n_components = 150  #组成元素的数量

print("Extracting the top %d eigenfaces from %d faces"%(n_components,x_train.shape[0]))
#每一步的时间
t0 = time()
#使用pca对数据集进行降维处理
# pca = RandomizedPCA(n_components=n_components,whiten=True).fit(x_train)
pca = PCA(n_components=n_components,whiten=True).fit(x_train)
print("done in %0.3fs"%(time()-t0))

#提取人脸图片中的特征值
eigenfaces = pca.components_.reshape((n_components,h,w))
print("projecting the input data on the eigenfaces orthonormal basis ")
t0 = time()
x_train_pca = pca.transform(x_train) #将特征向量进行降维操作
x_test_pca = pca.transform(x_test) #将测试集数据集降维
print("done in %0.3fs"%(time()-t0))

print("Fitting the classifier to the trainning set")
t0 = time()
#c为权重，对错误进行惩罚，根据降维之后的数据结合分类器进行分类
#gamma为核函数的不同表现，表示有多少特征能够被表示，表示比例
param_grid = {'C':[1e3,5e5,1e4,5e4,1e5],
              'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1],}

#建立分类器模型，找出表现最好的核函数
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
#训练模型
clf = clf.fit(x_train_pca,y_train)  #获取使边际最大的超平面

print("done in %0.3fs"%(time()-t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)  #打印分类器的信息


print("Predicting people's names on the test set")
t0 = time()
#进行预测
y_pred = clf.predict(x_test_pca)
print("done in %0.3fs"%(time()-t0))

#将标签值的真实值与预测值之间的比较情况
print(classification_report(y_test,y_pred,target_names=target_names))
#将结果整合在矩阵中
print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))

#将结果可视化
def plot_gallery(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.90,hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ',1)[-1]
    return "predicted: %s\nture:       %s"%(pred_name,true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
            for i in range(y_pred.shape[0])]


plot_gallery(x_test,prediction_titles,h,w)

eigenface_titles = ["eigenface  %d"% i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenface_titles,h,w)

plt.show()
