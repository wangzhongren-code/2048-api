#this is for generating the data we need
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent,YourOwnAgent,SimpleNet
import numpy as np
import pandas as pd

display1 = Display()
display2 = IPythonDisplay()

%%time

for i in range(0,3000):
    print(i)
    game = Game(4, score_to_win=2048, random=False)
    #display2.display(game)
    agent = ExpectiMaxAgent(game, display=display2)
    max_iter=np.inf
    n_iter = 0
    
    while (n_iter < max_iter) and (not game.end):
        tmp1 = game.board.reshape(1,16)
        direction = np.array(agent.step())
        tmp3 = direction.reshape(1,1)

        a = np.hstack((tmp1,tmp3))
        
        
        n_iter += 1
        agent.game.move(direction)
        df1 = pd.DataFrame((a))
        df1.to_csv('lastdata.xls',index=0,mode='a',header=0)
