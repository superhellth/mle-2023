import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.logger.info("Random model Setup.")
    np.random.seed()

def act(agent,game_state : dict):
    return np.random.choice(ACTIONS,p=[0.2,0.2,0.2,0.2,0.05,0.15])
for i in range(4):
    agent = 0
    game_state = {}
    x = act(agent,game_state)
    print(x)
