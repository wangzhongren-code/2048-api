import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy

test1 = xlrd.open_workbook('/home/dl/机器学习大作业/test.xls')
test2 = xlrd.open_workbook('/home/dl/机器学习大作业/test2.xls')
test3 = xlrd.open_workbook('/home/dl/机器学习大作业/test3.xls')
rows = 0
cols = 0
for i in range(0,26):
    sht1 = test1.sheets()[i]
    rows1 = sht1.nrows
    rows += sht1.nrows
for i in range(0,8):
    sht1 = test2.sheets()[i]
    rows1 = sht1.nrows
    rows += sht1.nrows
for i in range(0,13):
    sht1 = test3.sheets()[i]
    rows1 = sht1.nrows
    rows += sht1.nrows    
    
x_data = np.zeros((rows,16))
y_data = np.zeros((rows,1))

tmprows = 0
for i in range(0,26):
    sht1 = test1.sheets()[i]
    rows1 = sht1.nrows
    for k in range(int(rows1)):
        if sht1.cell_value(k,16)!=0:
            y_data[k+tmprows][0] = (sht1.cell_value(k,16))
        else:
            y_data[k+tmprows][0] = 0
        
        for j in range (16):
            if sht1.cell_value(k,j)!=0:
                x_data[k+tmprows][j] = np.log2(sht1.cell_value(k,j) )
            else:
                 x_data[k+tmprows][j] = 0
    tmprows+= rows1
    
for i in range(0,8):
    sht2 = test2.sheets()[i]
    rows1 = sht2.nrows
    for k in range(int(rows1)):
        if sht2.cell_value(k,16)!=0:
            y_data[k+tmprows][0] = (sht2.cell_value(k,16))
        else:
            y_data[k+tmprows][0] = 0
        
        for j in range (16):
            if sht2.cell_value(k,j)!=0:
                x_data[k+tmprows][j] = np.log2(sht2.cell_value(k,j) )
            else:
                 x_data[k+tmprows][j] = 0
    tmprows+= rows1    
    
for i in range(0,13):
    sht3 = test3.sheets()[i]
    rows1 = sht3.nrows
    for k in range(int(rows1)):
        if sht3.cell_value(k,16)!=0:
            y_data[k+tmprows][0] = (sht3.cell_value(k,16))
        else:
            y_data[k+tmprows][0] = 0
        
        for j in range (16):
            if sht3.cell_value(k,j)!=0:
                x_data[k+tmprows][j] = np.log2(sht3.cell_value(k,j) )
            else:
                 x_data[k+tmprows][j] = 0
    tmprows+= rows1     
    
print(rows)

print((x_data))
#print(x_data.shape())

test1 = xlrd.open_workbook('/home/dl/机器学习大作业/test.xls')
rows = 0
cols = 0
for i in range(26,27):
    sht1 = test1.sheets()[i]
    rows1 = sht1.nrows
    rows += sht1.nrows
    
test_x_data = np.zeros((rows,16))
test_y_data = np.zeros((rows,1))

tmprows = 0
for i in range(26,27):
    sht1 = test1.sheets()[i]
    rows1 = sht1.nrows
    for k in range(int(rows1)):
        if sht1.cell_value(k,16)!=0:
            test_y_data[k+tmprows][0] = (sht1.cell_value(k,16))
        else:
            test_y_data[k+tmprows][0] = 0
        
        for j in range (16):
            if sht1.cell_value(k,j)!=0:
                test_x_data[k+tmprows][j] = np.log2(sht1.cell_value(k,j) )
            else:
                 test_x_data[k+tmprows][j] = 0
    tmprows+= rows1
    

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 2,1,1), nn.ReLU())
        
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Linear(12,4)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Linear(12,4)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            #nn.Linear(16,4)
        )        
        self.fc5 = nn.Sequential(
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Linear(16,4)
        )  
        self.fc6 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16,4)
        ) 
        
        self.dropout = nn.Dropout(0.08)      #使用dropout来防止过拟合
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        x = self.dropout(x)    
        
        x = self.fc6(x)
        x = self.dropout(x)
        x = F.softmax(x)
        return x        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()# 损失函数使用交叉熵
optimizer = optim.Adam(model.parameters(),lr = 0.001)# 优化函数使用Adam算法

NUM_EPOCHS = 230
BATCH_SIZE = 128
features = torch.tensor(x_data, dtype=torch.float)
print(y_data)
labels = torch.tensor(y_data, dtype=torch.float)

dataset = data.TensorDataset(features, labels)

data_iter = data.DataLoader(dataset, BATCH_SIZE, shuffle = True)

test_labels = torch.tensor(test_y_data, dtype=torch.float)
test_features = torch.tensor(test_x_data, dtype=torch.float)
test_dataset = data.TensorDataset(test_features, test_labels)
test_data_iter = data.DataLoader(test_dataset, BATCH_SIZE, shuffle = True)


for epoch in range(NUM_EPOCHS):
    print("epoch:"+' ' +str(epoch))

    model.train()    #训练模式
    sum_loss = 0.0
    correct = 0
    total = 0
    for i,data1 in enumerate(data_iter):
        inputs, labels = data1
        
        inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
        inputs = inputs.reshape((inputs.shape[0],1,4,4))
        
        optimizer.zero_grad()  #将梯度归零
        outputs = model(inputs)  #前向运算
        loss = criterion(outputs, labels.squeeze().long())  #损失函数
        loss.backward()  #反向传播
        optimizer.step()  #做一步参数更新
        
        _,pred = torch.max(outputs, 1)
        num_correct = (pred == labels).sum()
        
        m = pred.reshape(1,inputs.shape[0])==labels.reshape(1,inputs.shape[0])
        correct += m[0].sum().item()
        total += inputs.shape[0]
    print("Train accuracy: ",(correct /total))
    model.eval()  #测试模式
    correct = 0
    total = 0
    for data_test in test_data_iter:
        inputs, labels = data_test
        inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
        inputs = inputs.reshape((inputs.shape[0],1,4,4))
        
        
        testout = model(inputs)
        testloss = criterion(testout,labels.squeeze().long())
        _,pred = torch.max(testout, 1)
        
        
        num_correct = (pred == labels).sum()
        m = pred.reshape(1,inputs.shape[0])==labels.reshape(1,inputs.shape[0])
        correct += m[0].sum().item()
        total += inputs.shape[0]

    print("Test accuracy: ",(correct /total))  #输出结果          
        
        
torch.save(model,'/home/dl/机器学习大作业/model.xls')
torch.save(model.state_dict(),'/home/dl/机器学习大作业/model2.xls')

