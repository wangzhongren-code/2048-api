import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas
from train import SimpleNet

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False,currentsht=70,url='/home/dl/机器学习大作业/test.xls'):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            
            direction = self.step()
            self.game.move(direction)
            
            #print(self.game.board)
            
            tmp1 = self.game.board.reshape(1,16)
            #print(tmp1)
            #test1 = xlrd.open_workbook(url)
            #test2 = copy(test1)
            #sht = test2.get_sheet((currentsht))
            df = pandas.DataFrame(list(tmp1))
            df.to_csv('/home/dl/机器学习大作业/test4.xls',index=0,mode='a')
            #for i in range(0,16):
              #  sht.write(n_iter,i,tmp1[0,i])
            
            #sht.write(n_iter,16,direction)
            n_iter += 1
            #test2.save(url)
            
            
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                  ["left", "down", "right", "up"][direction]))
                
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        
        from .expectimax import board_to_move
        self.search_func = board_to_move
    
    def pri(self):
        print(self.game.board)
        
    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class YourOwnAgent(Agent):
    def step(self):
        #model = SimpleNet()
        model = torch.load('/home/dl/机器学习大作业/model.xls')
        model.load_state_dict(torch.load('/home/dl/机器学习大作业/model2.xls'))
        model.eval()
        #print(model.state_dict())
        tmpboard = self.game.board.reshape(1,16)

        tmpboard = torch.tensor(tmpboard, dtype=torch.float)
        for i in range (16):
            if tmpboard[0,i]!=0:
                tmpboard[0,i] = np.log2(tmpboard[0,i])
            else:
                tmpboard[0,i] = 0
        print(tmpboard)
        tmpboard = tmpboard.reshape((1,1,4,4))
        output = model(tmpboard)
        tmp = 0

        for i in range(0,4):
            if output[0,i].item()>output[0,tmp].item():
                tmp = i
        
        direction = tmp
        return direction
    
    
    
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
        
        
