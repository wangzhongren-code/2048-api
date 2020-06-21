import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False,currentsht=70):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            
            direction = self.step()
            self.game.move(direction)
    
            n_iter += 1

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
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
    def step(self):
        model = torch.load('model.xls')
        model.eval()
        tmpboard = self.game.board.reshape(1,16)
        #get the board 
        tmpboard = torch.tensor(tmpboard, dtype=torch.float)
        for i in range (16):
            if tmpboard[0,i]!=0:
                tmpboard[0,i] = np.log2(tmpboard[0,i])
            else:
                tmpboard[0,i] = 0

        
        tdata = np.zeros((12,4,4))

        for i in range(16):
            row1 = i/4
            col1 = i%4
            tdata[int(tmpboard[0,i]),int(row1),int(col1)]= 1
            
        tdata = tdata.reshape((1,12,4,4))

        tdata = torch.from_numpy(tdata)
        tdata = torch.tensor(tdata, dtype=torch.float32)
        tdata = tdata.cuda()
        output = model(tdata)
        tmp = 0
        #get the direction
        for i in range(0,4):
            if output[0,i].item()>output[0,tmp].item():
                tmp = i
        
        direction = tmp
        return direction
    
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
