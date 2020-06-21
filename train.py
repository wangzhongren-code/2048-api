import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from game2048.agents import YourOwnAgent as TestAgent
import numpy as np
import pandas as pd
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent,YourOwnAgent,SimpleNet
display1 = Display()
display2 = IPythonDisplay()

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(12, 128, 3,1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(12, 128, 2,1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(12, 128, 4,1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(12, 128, (4,1), 1), nn.ReLU())
        
        self.conv5 = nn.Sequential(nn.Conv2d(12, 128, (1,4), 1), nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(2*2*128+9*128+1*128+8*128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )    

        self.dropout = nn.Dropout(0.23)      #使用dropout来防止过拟合
    
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        x3 = x3.view(x3.size()[0], -1)
        x4 = x4.view(x4.size()[0], -1)
        x5 = x5.view(x5.size()[0], -1)
        x = torch.cat((x1,x2,x3,x4,x5),1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = F.softmax(x)
        return x            
    
def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=None, **kwargs)
    agent.play(verbose=False)
    return game.score 

GAME_SIZE = 4
SCORE_TO_WIN = 2048
N_TESTS = 10



#input the data 
df=pd.read_csv('lastdata.xls',header=None,error_bad_lines=False)

rows = 0;
for row in df.iterrows():
    rows = rows+1
    if (rows%500000==0):
      print(rows)
print (rows)


tmpdata =  np.zeros((rows,17))
x_data = np.zeros((rows,16))
y_data = np.zeros((rows,1))
i = 0
for row in df.iterrows():
    tmpdata[i] = row[1][0:17]
    i = i+1
    if (i%500000==0):
      print(i)
      
for j in range(rows):  
  for k in range (16):
      if int(tmpdata[j][k])!=0:
          x_data[j][k] = np.log2(int(tmpdata[j,k]) )
      else:
          x_data[j][k] = 0
  if int(tmpdata[j][16])!=0:
        y_data[j][0] = (int(tmpdata[j,16]))
  else:
      y_data[j][0] = 0
  if (j%500000==0):
    print(j)
            
            
tdata = np.zeros((rows,12,4,4),dtype=bool)
tmp = x_data[0]
print(tmp[0])
print(tmp)
for j in range(rows):
  tmp = x_data[j]
  for i in range(16):
      row1 = i/4
      col1 = i%4
      tdata[j,int(tmp[i]),int(row1),int(col1)]= 1
  if (j%500000==0):
    print(j)
    

#train the model
device = torch.device('cuda')
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()# 损失函数使用交叉熵
optimizer = optim.Adam(model.parameters(),lr = 0.000230)# 优化函数使用Adam算法
NUM_EPOCHS = 80
BATCH_SIZE = 512
features = torch.tensor(tdata, dtype=torch.float32)
print(tdata.shape)
print(y_data.shape)
labels = torch.tensor(y_data, dtype=torch.float32)
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, BATCH_SIZE, shuffle = True)
#print (test_features)


for epoch in range(NUM_EPOCHS):
    print("epoch:"+' ' +str(epoch))
    model.train()   
    sum_loss = 0.0
    correct = 0
    total = 0
    for i,data1 in enumerate(data_iter):
        inputs, labels = data1
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()  #将梯度归零
        outputs = model(inputs)  #前向运算
    
        loss = criterion(outputs, labels.squeeze().long())  #损失函数
        sum_loss += loss
        loss.backward()  #反向传播
        optimizer.step()  #做一步参数更新
        _,pred = torch.max(outputs, 1)
        num_correct = (pred == labels).sum()
        m = pred.reshape(1,inputs.shape[0])==labels.reshape(1,inputs.shape[0])

        correct += m[0].sum().item()
        total += inputs.shape[0]
        #print(correct,total)

    print("sum_loss:",(sum_loss))
    print("mean_loss:",(sum_loss/total)*BATCH_SIZE)
    print("Train accuracy: ",(correct /total))
    torch.save(model,'model.xls')  #the model we use
    torch.save(model.state_dict(),'model2.xls')
    

   
    scores = []
    for j in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,AgentClass=TestAgent)
        scores.append(score)
        #print(score)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
