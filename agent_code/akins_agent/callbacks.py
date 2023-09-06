import os
import pickle
import random

import numpy as np


ACTIONS = ['UP','RIGHT', 'DOWN','LEFT'] #['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    print("Setting up.")
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    #round,step,field,self,others,bombs,coins,user_input,explosion_map,round,step,field,self,others,bombs,coins,user_input,explosion_map

    #Filter valid actions
    #Check if there is a wall/crate, you cannot walk into the wall/crate
    field = game_state['field']
    x = game_state['self'][3][0] #player_x_coordinate
    y = game_state['self'][3][1] #player_y_coordinate
    possible_actions = []
    invalid_actions = []
    if field[x-1][y] == -1:
        invalid_actions.append("LEFT")
    if field[x+1][y] == -1:
        invalid_actions.append("RIGHT")
    if field[x][y-1] == -1:
        invalid_actions.append("UP")
    if field[x][y+1] == -1:
        invalid_actions.append("DOWN")
    for actions in ACTIONS:
        if actions not in invalid_actions:
            possible_actions.append(actions)
    for coin in game_state['coins']:
        field[coin[0]][coin[1]] = 2

    #rufe auf mit Argument ((x,y,[])
    def breitenSuche_Coin(coordinates): #currentPath = ['LEFT', 'DOWN', 'LEFT']
        newCoordinates = []
        for coordinate in coordinates:
            #Terminieren wenn Coin gefunden
            if coordinate[0]-1 >= 0 and field[coordinate[0]-1, coordinate[1]] == 2:
                    return coordinate[2]+["l"]
            elif coordinate[0]+1 < 17 and field[coordinate[0]+1, coordinate[1]] == 2:
                    return coordinate[2]+["r"]
            elif coordinate[1]-1 >= 0 and field[coordinate[0], coordinate[1]-1] == 2:
                    return coordinate[2]+["u"]
            elif coordinate[1]+1 < 17 and field[coordinate[0], coordinate[1]+1] == 2:
                    return coordinate[2]+["d"]
            #Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen)
            if coordinate[0] - 1 >= 0:  #Prüfe ob wir den Rand noch nicht erreicht haben
                if field[coordinate[0] - 1][coordinate[1]] != -1 and coordinate[2][-1] != "r": #Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                    newCoordinates.append((coordinate[0] - 1, coordinate[1],coordinate[2]+["l"])) # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
            if coordinate[0] + 1 < 17:
                if field[coordinate[0] + 1][coordinate[1]] != -1 and coordinate[2][-1] != "l":
                    newCoordinates.append((coordinate[0] + 1, coordinate[1],coordinate[2]+["r"]))
            if coordinate[1] - 1 >= 0:
                if field[coordinate[0]][coordinate[1] - 1] != -1 and coordinate[2][-1] != "d":
                    newCoordinates.append((coordinate[0], coordinate[1] - 1,coordinate[2]+["u"]))
            if coordinate[1] + 1 < 17:
                if field[coordinate[0]][coordinate[1] + 1] != -1 and coordinate[2][-1] != "u":
                    newCoordinates.append((coordinate[0], coordinate[1] + 1,coordinate[2]+["d"]))
        if len(newCoordinates)>0:
            return breitenSuche_Coin(newCoordinates)
        else:
            return "Keine Lösung"
    print("\n\n",breitenSuche_Coin([(x,y,["i"])]))

    #Introduce some randomness in training mode with small probabiliy 10%
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(possible_actions) #second argument p=[.2, .2, .2, .2, .1, .1]

    self.logger.debug("Querying model for action.")
    return "WAIT"
    #return np.random.choice(possible_actions)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
