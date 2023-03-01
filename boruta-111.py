'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-14 22:07:10
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-04 20:59:45
FilePath: \feature-12-1\boruta-111.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import numpy as np
from torch.backends import cudnn
from collections import defaultdict
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

from sklearn.feature_selection import RFE
from sklearn.svm import SVR


parser = argparse.ArgumentParser(description="BORUTA!!!")
parser.add_argument('--data-path', type=str, default=r'C:\Users\26944\Desktop\feature-12-1\feature.mat')
args = parser.parse_args()
args.seed = 1111

## 固定随机数种子
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    
## download
path = args.data_path
mat_dataset = scio.loadmat(path)
feature_dataset = mat_dataset['feature']

X = feature_dataset[:, 0: 18]
# X_mean = np.average(X, axis = 0)
# X_std = np.std(X, axis=0)
# X = (X - X_mean) / X_std
y = feature_dataset[:, 18]

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=5, step=1)
# selector = selector.fit(X, y)

# # 哪些特征入选最后特征，true表示入选
# print(selector.support_)

# # 每个特征的得分排名，特征得分越低（1最好），表示特征越好
# print(selector.ranking_)

# #  挑选了几个特征
# print(selector.n_features_)


feat_selector.fit(X, y)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_
important_list = np.where(feat_selector.ranking_ == 1)
print(important_list)

X_filtered = feat_selector.transform(X)

print('1111')