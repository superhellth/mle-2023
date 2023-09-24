import os
import pickle
import random
import json

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN','LEFT','WAIT','BOMB']  # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
purelyRandom = ['UP', 'RIGHT', 'DOWN','LEFT','WAIT']  # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
    print("Setting up Q-Table.")


    training_folder_name = "Training"
    qtable_file_name = "qtable.pkl"
    file_path = os.path.join(training_folder_name, qtable_file_name)
    if os.path.exists(file_path): #Model from scratch if no model exists or train mode.
        with open(file_path, "rb") as file:
           self.qtable = pickle.load(file) # Load from model, in act() we just check if (state,*) is available and look for the according action in the table.
    else:
        self.qtable = {} #If empty we need to train from scratch


    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.step_counter = 0



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
    bomb_boolean = game_state['self'][2] #if bomb is ticking or not
    explosion_map = game_state['explosion_map']

    def explosionToField(explosion_map, field):
        explosion_map_array = np.array(explosion_map)
        field_array = np.array(field)

        # Only update the dangerous bomb zone (where explosion_map is 1)
        field_array[explosion_map_array == 1] += 8

        # Update the original field with the modified values
        field[:] = field_array
    explosionToField(explosion_map,field)
    def coinsToField(coins,field):
        for coin in coins:
            field[coin[0]][coin[1]] = 2
    coinsToField(game_state['coins'],field)
    def opponentToField(others,field):
        for opponent in others:
            (opponent_x,opponent_y) = opponent[3]
            field[opponent_x][opponent_y]=3
    opponentToField(game_state['others'],field)
    def bombToField(bombs, field):  # Explosion Map missing
        for bomb in bombs:
            (bomb_x, bomb_y) = bomb[0]
            timer = bomb[1]
            for i in range(-3, 4):
                if bomb_x + i < 17 and bomb_x + i >= 0:
                    field[bomb_x + i][bomb_y] = 4 + timer
            for j in range(-3, 4):
                if bomb_y + j < 17 and bomb_y + j >= 0:
                    field[bomb_x][bomb_y + j] = 4 + timer
    bombToField(game_state['bombs'],field)
    def kuerzesterWegZumTile(x, y, field, tile_value):
        '''This function searches the closest path to a coin given the agents coordinates (x,y) and the field of the current gamestate and the value of the tile that should be found.
        For coins it is the value 2.'''
        visitedCoordinates = []
        field_length = len(field[0])  # We can deduct the length from variable field, in ths case 17
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
    def naherTile(x, y, field, tile_value):
        '''This function returns the coordinates of closest Tile with the value tile_value'''
        visitedCoordinates = []
        field_length = len(field[0])  # We can deduct the length from variable field, in ths case 17
        # rufe auf mit Argument ((x,y,[],[]) X,Y Koordinaten [] aktuell mitgeführter Weg [] schon besuche Koordinaten
        def breitenSuche(coordinates):  # currentPath = ['LEFT', 'DOWN', 'LEFT']
            newCoordinates = []  # Neue Knoten mit der letzten Schrittrichtung mitgeführt
            for coordinate in coordinates:
                x = coordinate[0]
                y = coordinate[1]
                # Terminieren wenn Coin gefunden
                if x - 1 >= 0 and field[x - 1][y] == tile_value:
                    return (x-1,y)
                elif x + 1 < field_length and field[x + 1][y] == tile_value:
                    return (x+1,y)
                elif y - 1 >= 0 and field[x][y - 1] == tile_value:
                    return (x,y-1)
                elif y + 1 < field_length and field[x][y + 1] == tile_value:
                    return (x,y+1)
                # Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen
                if x - 1 >= 0:  # Prüfe ob wir den Rand noch nicht erreicht haben
                    if field[x - 1][y] != -1 and field[x - 1][y] != 1 and (x - 1,y) not in visitedCoordinates:  # Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                        newCoordinates.append((x - 1, y))  # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
                        visitedCoordinates.append((x - 1, y))
                if x + 1 < field_length:
                    if field[x + 1][y] != -1 and field[x + 1][y] != 1 and (x + 1, y) not in visitedCoordinates:
                        newCoordinates.append((x + 1, y))
                        visitedCoordinates.append((x + 1, y))
                if y - 1 >= 0:
                    if field[x][y - 1] != -1 and field[x][y - 1] != 1 and (x, y - 1) not in visitedCoordinates:
                        newCoordinates.append((x, y - 1))
                        visitedCoordinates.append((x, y - 1))
                if y + 1 < field_length:
                    if field[x][y + 1] != -1 and field[x][y + 1] != 1 and (x, y + 1) not in visitedCoordinates:
                        newCoordinates.append((x, y + 1))
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
        old_Tile_value = field[tile_x][tile_y]
        field[tile_x][tile_y] = 100  # Marks the tile as a "find me" tile
        if not kuerzesterWegZumTile(x, y, field, 100):
            field[tile_x][tile_y] = old_Tile_value
            return False
        else:
            field[tile_x][tile_y] = old_Tile_value
            return True

    def reduceInformation(crop):
        '''
        This function takes the cropped matrix and reduces unrelevant information within this crop.
        We specifically set all Tiles to the value 0 that are not accessible by the centered player.
        The purpose of this function is to reduce the amount of possible states the agent has to train for.
        '''
        crop_length = len(crop[0])
        for tile_x in range(crop_length): #Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
            for tile_y in range(crop_length):
                if not checkLineOfSight(tile_x,tile_y,3,3,crop):
                    crop[tile_x][tile_y] = 0
    reduceInformation(crop)
    def keepOneCoin(crop):
        crop_length = len(crop[0])
        closest_coin = naherTile(3, 3, crop, 2)
        for tile_x in range(crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
            for tile_y in range(crop_length):
                if crop[tile_x][tile_y]==2:
                    crop[tile_x][tile_y] = 0
        if closest_coin:
            (closest_coin_x,closest_coin_y)=closest_coin
            crop[closest_coin_x][closest_coin_y] = 2

    keepOneCoin(crop)
    def keepOneEnemy(crop):
        crop_length = len(crop[0])
        closest_coin = naherTile(3, 3, crop, 3)
        for tile_x in range(crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
            for tile_y in range(crop_length):
                if crop[tile_x][tile_y] == 3:
                    crop[tile_x][tile_y] = 0
        if closest_coin:
            (closest_coin_x, closest_coin_y) = closest_coin
            crop[closest_coin_x][closest_coin_y] = 3
    keepOneEnemy(crop)

    def coinDirection(x, y, field, crop):  # gives additional hint of the direction towards the next coin outside (even trough crates) if there is no coin in crop
        if naherTile(3, 3, crop, 2):
            return (0, 0)
        else:
            coin_coord = naherTile(x, y, field, 2)
            if coin_coord:
                (x_coin, y_coin) = coin_coord
                normalize_y = abs(y - y_coin)
                normalize_x = abs(x - x_coin)
                if normalize_x == 0:
                    normalize_x = 1
                if normalize_y == 0:
                    normalize_y = 1
                return ((x - x_coin) / normalize_x, (y - y_coin) / normalize_y)  # normalisieren!!! weniger Q states
            return (0, 0)

    def coinExistsOutside(x,y,field,crop):
        if naherTile(3, 3, crop, 2):
            return 0 #Tell the agent to focus on crop view
        else:
            if naherTile(x, y, field, 2):
                return 1 #Tell agent there exists a coin outside
            return 0 #Tell agent there is no coin outside
    direction_advice1 = coinDirection(x,y,field,crop)

    # 0 0
    # 1 1 links oben
    # -1 1 rechts oben
    # 1 -1 links unten
    # -1 -1 rechts unten

    #for element in crop:
    #   print(element)
    '''TODOS:
    WITHIN THE GAME_STATE:
    MOVE ALL PLAYERS INTO THE GAME_STATE FIELD check
    MOVE ALL BOMBS INTO THE GAME_STATE FIELD check
    
    WITHIN THE CROP:
    LEAVE ONLY THE CLOSEST COIN ON MAP check
    LEAVE ONLY ONE ENEMY IN THE CROP check
    
    Whats missing?:
    Move explosion counter into subfield
    '''
    # Introduce some randomness in training mode with small probabiliy 10%


    self.logger.debug("Querying model for action.")


    def filterInvalidActions(x, y, bomb_boolean, actions): #You cannot walk into opponents or walls or crates
        '''This function filters invalid moves. Make sure this is executed after bomb,player information is inserted into the field.
        And make sure that field is not touched anymore (e.g. do not remove player information again for optimization reasons). You can modify crop as you wish.'''
        invalid_actions = []
        possible_actions = []
        if field[x - 1][y] in [-1,1,3]:
            invalid_actions.append("LEFT")
        if field[x + 1][y] in [-1,1,3]:
            invalid_actions.append("RIGHT")
        if field[x][y - 1] in [-1,1,3]:
            invalid_actions.append("UP")
        if field[x][y + 1] in [-1,1,3]:
            invalid_actions.append("DOWN")
        if not bomb_boolean:
            invalid_actions.append("BOMB")
        for action in actions:
            if action not in invalid_actions:
                possible_actions.append(action)
        return possible_actions
    possible_actions = filterInvalidActions(x, y, bomb_boolean, ACTIONS)





    best_action = None
    if not self.train:
        random_prob = .05
        if random.random() < random_prob:#if self.train and random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            best_action =  np.random.choice(purelyRandom)  # second argument p=[.2, .2, .2, .2, .1, .1]


    #state =  (tuple(tuple(row) for row in crop),direction_advice1) #We gotta make state hashable i.e by turning it into a tuple to use it as key in dictionary
    if best_action == None:
        state = (tuple(tuple(row) for row in crop),direction_advice1)
        best_qvalue = float('-inf')



        final_action = None
        if not self.train:
            for action in possible_actions:
                qvalue = self.qtable.get((state,action),float('-inf'))
                if qvalue > best_qvalue:
                    best_action = action
                    best_qvalue = qvalue

        if self.train:
            random_prob = .8
            if random.random() < random_prob:  # if self.train and random.random() < random_prob:
                for action in possible_actions:
                    qvalue = self.qtable.get((state,action),float('-inf'))
                    if qvalue > best_qvalue:
                        best_action = action
                        best_qvalue = qvalue
        """for action in possible_actions:
            qvalue = self.qtable.get((state,action),float('-inf'))
            if qvalue > best_qvalue:
                best_action = action
                best_qvalue = qvalue"""

        if best_qvalue == float('-inf'): #Sadly no entry found in qtable
            best_action = random.choice(possible_actions)
    return best_action

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
