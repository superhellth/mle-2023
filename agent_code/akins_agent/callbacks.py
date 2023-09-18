import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    # round,step,field,self,others,bombs,coins,user_input,explosion_map,round,step,field,self,others,bombs,coins,user_input,explosion_map

    # Filter valid actions
    # Check if there is a wall/crate, you cannot walk into the wall/crate
    field = game_state['field']
    x = game_state['self'][3][0]  # player_x_coordinate
    y = game_state['self'][3][1]  # player_y_coordinate
    possible_actions = []
    invalid_actions = []
    if field[x - 1][y] == -1:
        invalid_actions.append("LEFT")
    if field[x + 1][y] == -1:
        invalid_actions.append("RIGHT")
    if field[x][y - 1] == -1:
        invalid_actions.append("UP")
    if field[x][y + 1] == -1:
        invalid_actions.append("DOWN")
    for actions in ACTIONS:
        if actions not in invalid_actions:
            possible_actions.append(actions)
    for coin in game_state['coins']:
        field[coin[0]][coin[1]] = 2

    def kuerzesterWegZumTile(x, y, field, tile_value):
        '''This function searches the closest path to a coin given the agents coordinates (x,y) and the field of the current gamestate and the value of the tile that should be found.
        For coins it is the value 2.'''
        visitedCoordinates = []
        field_length = len(field[0])  # We can deduct the length from variable field, in ths case 17
        print(field_length)

        # rufe auf mit Argument ((x,y,[],[]) X,Y Koordinaten [] aktuell mitgeführter Weg [] schon besuche Koordinaten
        def breitenSuche(coordinates):  # currentPath = ['LEFT', 'DOWN', 'LEFT']
            newCoordinates = []  # Neue Knoten mit der letzten Schrittrichtung mitgeführt
            for coordinate in coordinates:
                x = coordinate[0]
                y = coordinate[1]
                # Terminieren wenn Coin gefunden
                if x - 1 >= 0 and field[x - 1][y] == tile_value:
                    return coordinate[2] + ["LEFT"]
                elif x + 1 < field_length and field[x + 1][y] == tile_value:
                    return coordinate[2] + ["RIGHT"]
                elif y - 1 >= 0 and field[x][y - 1] == tile_value:
                    return coordinate[2] + ["UP"]
                elif y + 1 < field_length and field[x][y + 1] == tile_value:
                    return coordinate[2] + ["DOWN"]
                # Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen
                if x - 1 >= 0:  # Prüfe ob wir den Rand noch nicht erreicht haben
                    if field[x - 1][y] != -1 and field[x - 1][y] != 1 and (x - 1,
                                                                           y) not in visitedCoordinates:  # Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                        newCoordinates.append((x - 1, y, coordinate[2] + [
                            "LEFT"]))  # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
                        visitedCoordinates.append((x - 1, y))
                if x + 1 < field_length:
                    if field[x + 1][y] != -1 and field[x + 1][y] != 1 and (x + 1, y) not in visitedCoordinates:
                        newCoordinates.append((x + 1, y, coordinate[2] + ["RIGHT"]))
                        visitedCoordinates.append((x + 1, y))
                if y - 1 >= 0:
                    if field[x][y - 1] != -1 and field[x][y - 1] != 1 and (x, y - 1) not in visitedCoordinates:
                        newCoordinates.append((x, y - 1, coordinate[2] + ["UP"]))
                        visitedCoordinates.append((x, y - 1))
                if y + 1 < field_length:
                    if field[x][y + 1] != -1 and field[x][y + 1] != 1 and (x, y + 1) not in visitedCoordinates:
                        newCoordinates.append((x, y + 1, coordinate[2] + ["DOWN"]))
                        visitedCoordinates.append((x, y + 1))
            if len(newCoordinates) > 0:
                return breitenSuche(newCoordinates)
            else:
                return False  # "Keine Lösung"

        return breitenSuche([(x, y, [])])

    # print(kuerzesterWegZumTile(x,y,game_state['field'],2))

    def cropSevenTiles(x, y, field):
        '''
        This function crops the 17x17 field into a 7x7 with the player centered , i.e. the surrounding matrix.
        This will preprocessed further before used as a state in Q-Learning.
        '''
        x += 2
        y += 2  # Since we increase the matrix for easy cropping we need to adjust the new coordinates
        field = [[-1] * 17] * 2 + field + [[-1] * 17] * 2
        for i in range(len(field)):
            field[i] = [-1, -1] + field[i] + [-1, -1]
        sevenTiles = []
        for i in range(-3, 4):
            sevenTiles.append(field[x + i][
                              y - 3:y + 4])  # Von absolut bis weniger als bedeutet das ":" wortwörtlich mit Bezug auf Index.
            # Beispiel:  arr = [0,1,2,3,4,5,6,7] arr[0:6] liefert nur 0 bis 5
        return sevenTiles

    crop = cropSevenTiles(x, y, field.tolist())

    # print(crop)
    def checkLineOfSight(tile_x, tile_y, x, y, field):
        '''
        This function checks if a given tile in the cropped view has any path towards the agent by using BFS.
        The first two arguments are the tile we want to check its path towards the player at coordinates x,y.
        The last argument is the cropped view on the field.
        '''
        field[tile_x][tile_y] = 100  # Marks the tile as a "find me" tile
        if not kuerzesterWegZumTile(x, y, field, 100):
            return False
        else:
            return True

    print(checkLineOfSight(0, 0, 3, 3,crop))  #schaut beispielsweise, ob im Field eine freie Sicht gibt von Spielerkoordinate und dem Tile drei über ihn.

    # Introduce some randomness in training mode with small probabiliy 10%
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(possible_actions)  # second argument p=[.2, .2, .2, .2, .1, .1]

    self.logger.debug("Querying model for action.")
    return "WAIT"
    # return np.random.choice(possible_actions)


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
