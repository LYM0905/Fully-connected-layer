'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-01 21:17:23
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-04 21:28:45
FilePath: \feature-1Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
2-1\feature.py
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

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
# from boruta import BorutaPy

def main():
    args = parser.parse_args()
    
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
    
    
    # ### BORUTA select feature
    # X = feature_dataset[:, 0: 18]
    # y = feature_dataset[:, 19]

    # rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    # # define Boruta feature selection method
    # feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # feat_selector.fit(X, y)

    # # check selected features - first 5 features are selected
    # feat_selector.support_

    # # check ranking of features
    # feat_selector.ranking_

    # X_filtered = feat_selector.transform(X)
    
    
    
    ## 分割训练集和测试集
    num_data = feature_dataset.shape[0]
    print('数据数量:', num_data)
    

    estimator = PCA(n_components=6)
    feature_dataset_PCA = estimator.fit_transform(feature_dataset[:, list(range(18))])
    feature_dataset_PCA = np.concatenate((feature_dataset_PCA, feature_dataset[:, -4:]), axis=1)
    
    index = np.arange(num_data)
    train_index = np.random.choice(index, size=int(num_data * args.train_set_rate), replace=False)
    train_dataset = feature_dataset[train_index, :]
    train_dataset_PCA = feature_dataset_PCA[train_index, :]
    import_feature_PCA = [4, 5, 6]
    # import_feature1 = list(range(18))
    # # import_feature1 = [10, 17]
    # import_feature2 = list(range(18))
    # import_feature3 = list(range(18))
    import_feature1 = [ 0,  1,  2,  4,  8, 10, 14]
    # import_feature1 = [10, 17]
    import_feature2 = [3,  5,  6,  7,  9, 11, 12, 13, 14, 17]
    import_feature3 = [3,  5,  6,  7,  9, 11, 12, 13, 14, 15, 17]
    if args.use_important == True:
        train_feature1 = train_dataset[:, import_feature1].astype(np.float32)
        # train_feature1 = X_filtered[train_index, :]
        train_feature2 = train_dataset[:, import_feature2].astype(np.float32)
        train_feature3 = train_dataset[:, import_feature3].astype(np.float32)
        # train_feature1 = train_dataset_PCA[:, import_feature_PCA].astype(np.float32)
        # # train_feature1 = X_filtered[train_index, :]
        # train_feature2 = train_dataset_PCA[:, import_feature_PCA].astype(np.float32)
        # train_feature3 = train_dataset_PCA[:, import_feature_PCA].astype(np.float32)
    else:
        train_feature = train_dataset[:, :18].astype(np.float32)

    train_target = train_dataset[:, -1].astype(np.int64)
    train_target1 = train_dataset[:, -4].astype(np.int64)
    train_target2 = train_dataset[:, -3].astype(np.int64)
    train_target3 = train_dataset[:, -2].astype(np.int64)

    test_index = np.delete(index, train_index)
    test_dataset = feature_dataset[test_index, :]
    test_dataset_PCA = feature_dataset_PCA[test_index, :]

    if args.use_important == True:
        # test_feature1 = test_dataset_PCA[:, import_feature_PCA].astype(np.float32)
        # # test_feature1 = X_filtered[:, test_index]
        # test_feature2 = test_dataset_PCA[:, import_feature_PCA].astype(np.float32)
        # test_feature3 = test_dataset_PCA[:, import_feature_PCA].astype(np.float32)
        test_feature1 = test_dataset[:, import_feature1].astype(np.float32)
        # test_feature1 = X_filtered[:, test_index]
        test_feature2 = test_dataset[:, import_feature2].astype(np.float32)
        test_feature3 = test_dataset[:, import_feature3].astype(np.float32)
    else:
        test_feature = test_dataset[:, :18].astype(np.float32)
    # test_feature = test_dataset[:, :18].astype(np.float32)
    test_target = test_dataset[:, -1].astype(np.int64)
    test_target1 = test_dataset[:, -4].astype(np.int64)
    test_target2 = test_dataset[:, -3].astype(np.int64)
    test_target3 = test_dataset[:, -2].astype(np.int64)
    
    if args.use_PCA == True:
        estimator = PCA(n_components=6)
        train_feature1 = estimator.fit_transform(train_feature1)
        test_feature1 = estimator.fit_transform(test_feature1)
        train_feature1 = estimator.fit_transform(train_feature1)
        test_feature1 = estimator.fit_transform(test_feature1)
        train_feature1 = estimator.fit_transform(train_feature1)
        test_feature1 = estimator.fit_transform(test_feature1)
        # print('降维后的特征shape:', train_feature3.shape, test_feature3.shape)
    
    if args.use_important == False:
        args.in_features = train_feature.shape[1]
    else:
        num_feature1 = train_feature1.shape[1]
        num_feature2 = train_feature2.shape[1]
        num_feature3 = train_feature3.shape[1]
        
    
    ## model
    # drop_out_rate = args.drop_out_rate
    # in_features = args.in_features
    # out_features = args.num_cls
    
    
    class Block(nn.Module):
        def  __init__(self, hidden_features=10):
            super(Block, self).__init__()
            self.hidden_features = hidden_features
            self.hidden_layer = nn.Linear(hidden_features, hidden_features)
            self.bn = nn.BatchNorm1d(hidden_features)
            self.dropout = nn.Dropout(p=args.drop_out_rate)
            
        def forward(self, x):
            x = self.hidden_layer(x)
            # x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            return x
        

    
    class FC(nn.Module):
        def __init__(self, depth, in_features, hidden_features=10, out_features=2, drop_out_rate=0):
            super(FC, self).__init__()
            self.depth = depth
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.out_features = out_features
            self.drop_out_rate = drop_out_rate
            self.head = nn.Linear(self.in_features, self.hidden_features)
            self.classifier = nn.Linear(self.hidden_features, self.out_features)
            self.body = nn.ModuleList([
                Block(hidden_features=self.hidden_features) for i in range(self.depth)
            ])
            
            self.bn_in = nn.BatchNorm1d(self.in_features)
            self.bn = nn.BatchNorm1d(self.hidden_features)
            self.bn_out = nn.BatchNorm1d(self.out_features)
            
            
            self.dropout = nn.Dropout(p=args.drop_out_rate)
            
            
        def forward(self, x):
            if len(x.shape) == 1:
                x = x.unsqueeze(1)
                
            # x_original_plt = np.array(x[:, 0].cpu())
            # plt.hist(x_original_plt, bins=20, normed = True, color=sns.desaturate("indianred", .8), alpha=.4)
            
            x = self.bn_in(x)
            # x_bn_plt = np.array(x[:, 0].detach().cpu())
            
            
            # plt.hist([x_original_plt, x_bn_plt], bins=50, label=['a', 'b'])
            # plt.show()
            
            # x = F.normalize(x, p=2, dim=0)
            x = self.head(x)
            x = F.relu(x)
            x = self.dropout(x)
            # x = self.bn(x)
            x = F.relu(x)
            for blk in self.body:
                x = blk(x)
                # x = self.bn(x)
            x = self.classifier(x)
            x = F.relu(x)
            x = self.bn_out(x)

            return x
        
    if args.use_important == False:
        num_cls = 8
    else:
        num_cls = 2
        
    if args.use_important == False:
    
        FC_model = FC(depth=args.depth, drop_out_rate=args.drop_out_rate, 
                    hidden_features=args.hidden_features,
                    in_features = args.in_features, out_features = num_cls)
        FC_model
        params = [{"params": [value]} for value in FC_model.parameters()]
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=args.lr1, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=args.lr1, momentum=0.9, weight_decay=args.weight_decay)
            
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
    else:
        FC_model1 = FC(depth=args.depth, drop_out_rate=args.drop_out_rate, 
                    hidden_features=args.hidden_features,
                    in_features = num_feature1, out_features = num_cls)
        FC_model2 = FC(depth=args.depth, drop_out_rate=args.drop_out_rate, 
                    hidden_features=args.hidden_features,
                    in_features = num_feature2, out_features = num_cls)
        FC_model3 = FC(depth=args.depth, drop_out_rate=args.drop_out_rate, 
                    hidden_features=args.hidden_features,
                    in_features = num_feature3, out_features = num_cls)
        FC_model1.cuda()
        FC_model2.cuda()
        FC_model3.cuda()
        
        params1 = [{"params": [value]} for value in FC_model1.parameters()]
        params2 = [{"params": [value]} for value in FC_model2.parameters()]
        params3 = [{"params": [value]} for value in FC_model3.parameters()]
            
        if args.optimizer == 'Adam':
            optimizer1 = torch.optim.Adam(params1, lr=args.lr1, weight_decay=args.weight_decay)
            optimizer2 = torch.optim.Adam(params2, lr=args.lr2, weight_decay=args.weight_decay)
            optimizer3 = torch.optim.Adam(params3, lr=args.lr3, weight_decay=args.weight_decay)
        else:
            optimizer1 = torch.optim.SGD(params1, lr=args.lr1, momentum=0.9, weight_decay=args.weight_decay)
            optimizer2 = torch.optim.SGD(params2, lr=args.lr2, momentum=0.9, weight_decay=args.weight_decay)
            optimizer3 = torch.optim.SGD(params3, lr=args.lr3, momentum=0.9, weight_decay=args.weight_decay)
            
        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)
        lr_scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=args.step_size, gamma=args.gamma)
            
    
    # loss_func = torch.nn.CrossEntropyLoss()
    Loss_dict = defaultdict()
    Loss_dict['epoch'] = list()
    Loss_dict['loss1_train'] = list()
    Loss_dict['loss2_train'] = list()
    Loss_dict['loss3_train'] = list()
    Loss_dict['loss1_test'] = list()
    Loss_dict['loss2_test'] = list()
    Loss_dict['loss3_test'] = list()

        
    if args.use_important == False:
            
        for i in range(args.num_epoch):
            
            FC_model.cuda()
            FC_model.train()
            train_feature = torch.tensor(train_feature).cuda()
            train_target = torch.tensor(train_target).cuda()
            train_target_one_hot = torch.zeros(train_target.shape[0], num_cls).cuda().scatter_(1, train_target.unsqueeze(1), 1)
            output = FC_model(train_feature)
            if args.loss_type == 'CEloss':
                loss = - torch.mean(F.log_softmax(output) * train_target_one_hot)
            if i > 10:
                Loss_dict['epoch'].append(i + 1)
                Loss_dict['loss_train'].append(loss.item())
            
            predict = torch.argmax(F.softmax(output, dim=1), dim=1)          
            acc = torch.sum(train_target == predict) / train_target.shape[0]      
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            FC_model.eval()
            test_feature= torch.tensor(test_feature).cuda()           
            test_target = torch.tensor(test_target).cuda()
            output_test = FC_model(test_feature)
            predict_test = torch.argmax(F.softmax(output_test, dim=1), dim=1)
            acc_test = torch.sum(test_target == predict_test) / test_target.shape[0]
            
            print('TRAIN-SET /// epoch:{:3d} /// LOSS:{:.4f} /// Acc:{:3.1%} /// \n'
                    .format(i, loss.item(), acc.item()))
            print('TEST-SET /// epoch:{:3d} ///  TEST_Acc:{:3.1%} ///\n'
                    .format(i,  acc_test.item()))
            
        
    else:
        
        for i in range(args.num_epoch):
        
            FC_model1.train()
            FC_model2.train()
            FC_model3.train()
            
            ## 生成one-hot-vector
            train_feature1 = torch.tensor(train_feature1).cuda()
            train_feature2 = torch.tensor(train_feature2).cuda()
            train_feature3 = torch.tensor(train_feature3).cuda()
            train_target1 = torch.tensor(train_target1).cuda()
            train_target2 = torch.tensor(train_target2).cuda()
            train_target3 = torch.tensor(train_target3).cuda()
            train_target_one_hot1 = torch.zeros(train_target1.shape[0], num_cls).cuda().scatter_(1, train_target1.unsqueeze(1), 1)
            train_target_one_hot2 = torch.zeros(train_target2.shape[0], num_cls).cuda().scatter_(1, train_target2.unsqueeze(1), 1)
            train_target_one_hot3 = torch.zeros(train_target3.shape[0], num_cls).cuda().scatter_(1, train_target3.unsqueeze(1), 1)
            
            output1 = FC_model1(train_feature1)
            output2 = FC_model2(train_feature2)
            output3 = FC_model3(train_feature3)
            
            if i == 100:
                print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii--------\n")
            
            if args.loss_type == 'CEloss':
                loss1 = - torch.mean(F.log_softmax(output1) * train_target_one_hot1)
                loss2 = - torch.mean(F.log_softmax(output2) * train_target_one_hot2)
                loss3 = - torch.mean(F.log_softmax(output3) * train_target_one_hot3)
            elif args.loss_type == 'hingeloss':
                margin = args.margin
                train_target_one_hot1 = train_target_one_hot1 * 2 - 1
                train_target_one_hot2 = train_target_one_hot2 * 2 - 1
                train_target_one_hot3 = train_target_one_hot3 * 2 - 1
                
                y_yhat1 = train_target_one_hot1 * output1
                loss1 = torch.mean(torch.max(torch.zeros_like(y_yhat1[:, 0]), margin - (y_yhat1[:, 0] + y_yhat1[:, 1])))

                y_yhat2 = train_target_one_hot2 * output2
                loss2 = torch.mean(torch.max(torch.zeros_like(y_yhat2[:, 0]), margin - (y_yhat2[:, 0] + y_yhat2[:, 1])))
                
                y_yhat3 = train_target_one_hot3 * output3
                loss3 = torch.mean(torch.max(torch.zeros_like(y_yhat3[:, 0]), margin - (y_yhat3[:, 0] + y_yhat3[:, 1])))
            
            if i > 5:
                Loss_dict['epoch'].append(i + 1)
                Loss_dict['loss1_train'].append(loss1.item())
                Loss_dict['loss2_train'].append(loss2.item())
                Loss_dict['loss3_train'].append(loss3.item())
            
            predict1 = torch.argmax(F.softmax(output1, dim=1), dim=1)
            predict2 = torch.argmax(F.softmax(output2, dim=1), dim=1)
            predict3 = torch.argmax(F.softmax(output3, dim=1), dim=1)
            
            acc1 = torch.sum(train_target1 == predict1) / train_target1.shape[0] 
            acc2 = torch.sum(train_target2 == predict2) / train_target2.shape[0]
            acc3 = torch.sum(train_target3 == predict3) / train_target3.shape[0]       
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss1.backward()
            loss2.backward()
            loss3.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            lr_scheduler1.step()
            lr_scheduler2.step()
            lr_scheduler3.step()
            
            ## evaluate
            FC_model1.eval()
            FC_model2.eval()
            FC_model3.eval()
            
            test_feature1= torch.tensor(test_feature1).cuda()
            test_feature2= torch.tensor(test_feature2).cuda()
            test_feature3= torch.tensor(test_feature3).cuda()
            
            test_target1 = torch.tensor(test_target1).cuda()
            output_test1 = FC_model1(test_feature1)
            predict_test1 = torch.argmax(F.softmax(output_test1, dim=1), dim=1)
            acc_test1 = torch.sum(test_target1 == predict_test1) / test_target1.shape[0]
            
            test_target2 = torch.tensor(test_target2).cuda()
            output_test2 = FC_model2(test_feature2)
            predict_test2 = torch.argmax(F.softmax(output_test2, dim=1), dim=1)
            acc_test2 = torch.sum(test_target2 == predict_test2) / test_target2.shape[0]
            
            test_target3 = torch.tensor(test_target3).cuda()
            output_test3 = FC_model3(test_feature3)
            predict_test3 = torch.argmax(F.softmax(output_test3, dim=1), dim=1)
            acc_test3 = torch.sum(test_target3 == predict_test3) / test_target3.shape[0]
            
            test_target_one_hot1 = torch.zeros(test_target1.shape[0], num_cls).cuda().scatter_(1, test_target1.unsqueeze(1), 1)
            test_target_one_hot2 = torch.zeros(test_target2.shape[0], num_cls).cuda().scatter_(1, test_target2.unsqueeze(1), 1)
            test_target_one_hot3 = torch.zeros(test_target3.shape[0], num_cls).cuda().scatter_(1, test_target3.unsqueeze(1), 1)
            
            if args.loss_type == 'CEloss':
                loss1_test = - torch.mean(F.log_softmax(output_test1) * test_target_one_hot1)
                loss2_test = - torch.mean(F.log_softmax(output_test2) * test_target_one_hot2)
                loss3_test = - torch.mean(F.log_softmax(output_test3) * test_target_one_hot3)
            elif args.loss_type == 'hingeloss':
                margin = args.margin
                test_target_one_hot1 = test_target_one_hot1 * 2 - 1
                test_target_one_hot2 = test_target_one_hot2 * 2 - 1
                test_target_one_hot3 = test_target_one_hot3 * 2 - 1
                
                y_yhat1 = test_target_one_hot1 * output_test1
                loss1_test = torch.mean(torch.max(torch.zeros_like(y_yhat1[:, 0]), margin - (y_yhat1[:, 0] + y_yhat1[:, 1])))

                y_yhat2 = test_target_one_hot2 * output_test2
                loss2_test = torch.mean(torch.max(torch.zeros_like(y_yhat2[:, 0]), margin - (y_yhat2[:, 0] + y_yhat2[:, 1])))
                
                y_yhat3 = test_target_one_hot3 * output_test3
                loss3_test = torch.mean(torch.max(torch.zeros_like(y_yhat3[:, 0]), margin - (y_yhat3[:, 0] + y_yhat3[:, 1])))
            
            print('TRAIN-SET /// epoch:{:3d} /// LOSS1:{:.4f} /// Acc1:{:3.1%} /// LOSS2:{:.4f} /// Acc2:{:3.1%} /// LOSS3:{:.4f} /// Acc3:{:3.1%} /// \n'
                    .format(i, loss1.item(), acc1.item(), loss2.item(), acc2.item(), loss3.item(), acc3.item()))
            print('TEST-SET /// epoch:{:3d} ///  TEST_Acc1:{:3.1%} /// TEST_Acc2:{:3.1%} /// TEST_Acc3:{:3.1%} ///\n'
                    .format(i,  acc_test1.item(), acc_test2.item(), acc_test3.item()))
            
            if i>5:
                Loss_dict['loss1_test'].append(loss1_test.item())
                Loss_dict['loss2_test'].append(loss2_test.item())
                Loss_dict['loss3_test'].append(loss3_test.item())
            
    # plt.figure(figsize=(8, 5),dpi=(80))
    # x = Loss_dict['epoch']
    # y1_train = Loss_dict['loss1_train'] 
    # y1_test = Loss_dict['loss1_test']
    # plt.title('错误一分类器的Loss Function收敛曲线')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.xlabel('EPOCH')  # x轴标题
    # plt.ylabel('Loss Function')  # y轴标题
    # plt.plot(x, y1_train)
    # plt.plot(x, y1_test)
    # plt.legend(['train', 'test'])
    # plt.show()
    
    # plt.figure(figsize=(8, 5),dpi=(80))
    # x = Loss_dict['epoch']
    # y2_train = Loss_dict['loss2_train'] 
    # y2_test = Loss_dict['loss2_test']
    # plt.title('错误二分类器的Loss Function收敛曲线')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.xlabel('EPOCH')  # x轴标题
    # plt.ylabel('Loss Function')  # y轴标题
    # plt.plot(x, y2_train)
    # plt.plot(x, y2_test)
    # plt.legend(['train', 'test'])
    # plt.show()
       
    # plt.figure(figsize=(8, 5),dpi=(80))
    # x = Loss_dict['epoch']
    # y3_train = Loss_dict['loss3_train'] 
    # y3_test = Loss_dict['loss3_test']
    # plt.title('错误三分类器的Loss Function收敛曲线')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.xlabel('EPOCH')  # x轴标题
    # plt.ylabel('Loss Function')  # y轴标题
    # plt.plot(x, y3_train)
    # plt.plot(x, y3_test)
    # plt.legend(['train', 'test'])
    # plt.show()

    


if __name__ == '__main__':
    ## defualt
    parser = argparse.ArgumentParser(description="feature classifier")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--depth', type=int, default=2, help="depth of neural network")
    parser.add_argument('--data-path', type=str, default=r'C:\Users\26944\Desktop\feature-12-1\feature.mat')
    parser.add_argument('--train-set-rate', type=float, default=0.6, help='训练集占比')
    ## input layer 2 * hidden layer output layer
    ## model
    parser.add_argument('--drop-out-rate', type=float, default=0.3) ## 0.3
    # parser.add_argument('--in-features', type=float, default=5)
    parser.add_argument('--hidden_features', type=float, default=2048)
    parser.add_argument('--all-feature', type=float, default=False)
    parser.add_argument('--num-epoch', type=float, default=200)
    
    ## optimizer
    parser.add_argument('--optimizer', type=str, default='Adam') ## SGD
    parser.add_argument('--lr1', type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--lr2', type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--lr3', type=float, default=0.005,
                        help="learning rate") ## 0.005
    
    parser.add_argument('--weight-decay', type=float, default=1e-4) ## 1e-4
    parser.add_argument('--step-size', type=int, default=300) ## 200
    parser.add_argument('--gamma', default=0.1) ## 0.3
    ### 错误三预测正确率100% lr=0.05 gamma=0.3
    ### 错误一训练正确率100，预测正确率98.4 lr=0.005 gamma=0.3 step size=100
    parser.add_argument('--use_PCA', default=False)
    parser.add_argument('--use-important', default=True)
    
    parser.add_argument('--loss-type', default='CEloss')
    parser.add_argument('--margin', default=1, type=float)
    main()
    
    ### 设置：num_epoch = 2000, weight_decay = 1e-4 lr = 0.005 gamma=0.5 margin=1
    ## Learning rate=0.005,Weight decay=1e-4,Strep-size=200, Gamma=0.1,Drop-out-rate=0.3,Proportion of training set=0.6,Number of inputlayer=1,
    ## Number of hidden layer=2 Number of outputlayer=1,Number of hidden layer nodes=2048,
    ## Iteration times=200,Optimizer：Adam,Loss-type：CEloss