from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import math
import events as e
from .callbacks import state_to_features

import json
import os

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

training_folder_name = "Training"
qtable_file_name = "qtable.pkl"
btable_file_name = "btable.pkl"


def setup_directory():
    if not os.path.exists(training_folder_name):
        os.makedirs(training_folder_name)


def setup_qtable(self):
    file_path = os.path.join(training_folder_name, qtable_file_name)
    file_path_btable = os.path.join(training_folder_name, btable_file_name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as file:
            self.qtable = {}
    else:
        with open(file_path, "rb") as file:
           self.qtable = pickle.load(file) # Load pretrained model, continue from here
    if not os.path.exists(file_path_btable):
        with open(file_path_btable, "wb") as file:
            self.btable = {}

    else:
        with open(file_path_btable, "rb") as file:
           self.btable = pickle.load(file) # Load pretrained btable



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    setup_directory()
    setup_qtable(self)  # here we will write our dictionary later
    # self.qtable = {}
    self.action_history = []
    self.round_counter = 0 #To track the rounds and periodically store the qtable into the file
    self.period = 200 #
    #self.visited = []
    self.history_size = 4
    self.history = []
    self.crop_size = 5#odd numbers only!


def explosionToField(explosion_map, field):
    for tile_x in range(len(explosion_map[0])):
        for tile_y in range(len(explosion_map[0])):
            if explosion_map[tile_x][tile_y] == 1:
                    field[tile_x][tile_y] = 4
def coinsToField(coins, field):
    for coin in coins:
        field[coin[0]][coin[1]] = 2

def opponentToField(others, field):
    for opponent in others:
        (opponent_x, opponent_y) = opponent[3]
        field[opponent_x][opponent_y] = 3


'''def bombToField(bombs, field):  # Explosion Map missing
    for bomb in bombs:
        (bomb_x, bomb_y) = bomb[0]
        field[bomb_x][bomb_y] = 4'''
def bombToField(bombs, field):  # Explosion Map missing
    for bomb in bombs:
        (bomb_x, bomb_y) = bomb[0]
        timer = bomb[1]
        for i in range(-3, 4):
            if bomb_x + i < 17 and bomb_x + i >= 0:
                field[bomb_x + i][bomb_y] = 4
        for j in range(-3, 4):
            if bomb_y + j < 17 and bomb_y + j >= 0:
                field[bomb_x][bomb_y + j] = 4


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
                if field[x-1][y] not in [-1, 1, 4] and (x - 1,
                                                          y) not in visitedCoordinates:  # Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                    newCoordinates.append((x - 1, y, coordinate[2] + [
                        "LEFT"]))  # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
                    visitedCoordinates.append((x - 1, y))
            if x + 1 < field_length:
                if field[x+1][y] not in [-1, 1, 4] and (x + 1, y) not in visitedCoordinates:
                    newCoordinates.append((x + 1, y, coordinate[2] + ["RIGHT"]))
                    visitedCoordinates.append((x + 1, y))
            if y - 1 >= 0:
                if field[x][y - 1] not in [-1, 1, 4] and (x, y - 1) not in visitedCoordinates:
                    newCoordinates.append((x, y - 1, coordinate[2] + ["UP"]))
                    visitedCoordinates.append((x, y - 1))
            if y + 1 < field_length:
                if field[x][y + 1] not in [-1, 1, 4] and (x, y + 1) not in visitedCoordinates:
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
                return (x - 1, y)
            elif x + 1 < field_length and field[x + 1][y] == tile_value:
                return (x + 1, y)
            elif y - 1 >= 0 and field[x][y - 1] == tile_value:
                return (x, y - 1)
            elif y + 1 < field_length and field[x][y + 1] == tile_value:
                return (x, y + 1)
            # Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen
            if x - 1 >= 0:  # Prüfe ob wir den Rand noch nicht erreicht haben
                if field[x - 1][y] not in [-1, 1, 4] and (x - 1,
                                                          y) not in visitedCoordinates:  # Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                    newCoordinates.append(
                        (x - 1, y))  # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
                    visitedCoordinates.append((x - 1, y))
            if x + 1 < field_length:
                if field[x + 1][y] not in [-1, 1, 4] and (x + 1, y) not in visitedCoordinates:
                    newCoordinates.append((x + 1, y))
                    visitedCoordinates.append((x + 1, y))
            if y - 1 >= 0:
                if field[x][y - 1] not in [-1, 1, 4] and (x, y - 1) not in visitedCoordinates:
                    newCoordinates.append((x, y - 1))
                    visitedCoordinates.append((x, y - 1))
            if y + 1 < field_length:
                if field[x][y + 1] not in [-1, 1, 4] and (x, y + 1) not in visitedCoordinates:
                    newCoordinates.append((x, y + 1))
                    visitedCoordinates.append((x, y + 1))
        if len(newCoordinates) > 0:
            return breitenSuche(newCoordinates)
        else:
            return False  # "Keine Lösung"

    return breitenSuche([(x, y, [])])

# print(kuerzesterWegZumTile(x,y,game_state['field'],2))

def cropSevenTiles(x, y, field,crop_size):
    '''
    This function crops the 17x17 field into a 7x7 with the player centered , i.e. the surrounding matrix.
    This will preprocessed further before used as a state in Q-Learning.
    '''
    x += math.floor(crop_size / 2) - 1
    y += math.floor(crop_size / 2) - 1  # Since we increase the matrix for easy cropping we need to adjust the new coordinates
    field = [[-1] * 17] * (math.floor(crop_size/2)-1) + field + [[-1] * 17] * (math.floor(crop_size/2)-1)
    for i in range(len(field)):
        field[i] = [-1]*(math.floor(crop_size/2)-1) + field[i] + [-1]*(math.floor(crop_size/2)-1)
    sevenTiles = []
    for i in range(-math.floor(crop_size/2), math.floor(crop_size/2)+1):
        sevenTiles.append(field[x + i][
                          y - math.floor(crop_size/2):y + math.floor(crop_size/2) + 1])  # Von absolut bis weniger als bedeutet das ":" wortwörtlich mit Bezug auf Index.
        # Beispiel:  arr = [0,1,2,3,4,5,6,7] arr[0:6] liefert nur 0 bis 5
    return sevenTiles
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


def reduceInformation(crop, crop_size):
    '''
    This function takes the cropped matrix and reduces unrelevant information within this crop.
    We specifically set all Tiles to the value 0 that are not accessible by the centered player.
    The purpose of this function is to reduce the amount of possible states the agent has to train for.
    '''
    crop_length = len(crop[0])
    for tile_x in range(
            crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
        for tile_y in range(crop_length):
            if not checkLineOfSight(tile_x, tile_y, math.floor(crop_size / 2), math.floor(crop_size / 2), crop):
                crop[tile_x][tile_y] = 0


def keepOneCoin(crop, crop_size):
    crop_length = len(crop[0])
    closest_coin = naherTile(math.floor(crop_size / 2), math.floor(crop_size / 2), crop, 2)
    for tile_x in range(
            crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
        for tile_y in range(crop_length):
            if crop[tile_x][tile_y] == 2:
                crop[tile_x][tile_y] = 0
    if closest_coin:
        (closest_coin_x, closest_coin_y) = closest_coin
        crop[closest_coin_x][closest_coin_y] = 2


def keepOneEnemy(crop, crop_size):
    crop_length = len(crop[0])
    closest_coin = naherTile(math.floor(crop_size / 2), math.floor(crop_size / 2), crop, 3)
    for tile_x in range(
            crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
        for tile_y in range(crop_length):
            if crop[tile_x][tile_y] == 3:
                crop[tile_x][tile_y] = 0
    if closest_coin:
        (closest_coin_x, closest_coin_y) = closest_coin
        crop[closest_coin_x][closest_coin_y] = 3


def coinDirection(x, y, field, crop,
                  crop_size):  # gives additional hint of the direction towards the next coin outside (even trough crates) if there is no coin in crop
    if naherTile(math.floor(crop_size / 2), math.floor(crop_size / 2), crop, 2):
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
def filterInvalidActions(x, y,field, bomb_possible, actions): #You cannot walk into opponents or walls or crates
    '''This function filters invalid moves. Make sure this is executed after bomb,player information is inserted into the field.
    And make sure that field is not touched anymore (e.g. do not remove player information again for optimization reasons). You can modify crop as you wish.'''
    invalid_actions = []
    possible_actions = []
    if field[x - 1][y] in [-1, 1, 3]:
        invalid_actions.append("LEFT")
    if field[x + 1][y] in [-1, 1, 3]:
        invalid_actions.append("RIGHT")
    if field[x][y - 1] in [-1, 1, 3]:
        invalid_actions.append("UP")
    if field[x][y + 1] in [-1, 1, 3]:
        invalid_actions.append("DOWN")
    if not bomb_possible:
        invalid_actions.append("BOMB")
    for action in actions:
        if action not in invalid_actions:
            possible_actions.append(action)
    return possible_actions
def getCrop(self,game_state):
    field = game_state['field']
    x = game_state['self'][3][0]  # player_x_coordinate
    y = game_state['self'][3][1]  # player_y_coordinate
    explosionToField(game_state['explosion_map'], field)
    coinsToField(game_state['coins'], field)
    opponentToField(game_state['others'], field)
    bombToField(game_state['bombs'], field)
    crop = cropSevenTiles(x, y, field.tolist(),self.crop_size)
    reduceInformation(crop,self.crop_size)
    keepOneCoin(crop,self.crop_size)
    keepOneEnemy(crop,self.crop_size)

    return crop

def rotate_matrix_90_degrees(matrix): #we can use the symmetry of the game to store 4 states at the same time
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    rotated_matrix = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            rotated_matrix[j][rows - 1 - i] = matrix[i][j]
    return rotated_matrix

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    reward = 0

    field = old_game_state['field']
    x = old_game_state['self'][3][0]  # player_x_coordinate
    y = old_game_state['self'][3][1]  # player_y_coordinate

    bomb_active = not old_game_state['self'][2]
    #action = self_action
    action = self_action
    # How do we get our action? By the event...

    crop1 = getCrop(self, old_game_state)
    crop2 = getCrop(self, new_game_state)  # Very inefficient

    '''rotate_direction = [['RIGHT','DOWN','LEFT','UP'],
                        ['DOWN','LEFT','UP','RIGHT'],
                        ['LEFT','UP','RIGHT','DOWN'],
                        ['UP','RIGHT','DOWN','LEFT']]
    rotate_coin_direction_vector = [[(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0)],
                                    [(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1)],
                                    [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)],
                                    [(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1)]]



    crop1_2 = rotate_matrix_90_degrees(crop1) # right becomes down, down becomes left, left becomes up, up becomes right
    crop1_3 = rotate_matrix_90_degrees(crop1_2)  #right becomes left, down becomes up, left becomes right, up becomes down
    crop1_4 = rotate_matrix_90_degrees(crop1_3) #right becomes up, down becomes right, left becomes down, up becomes left


    '''

    direction_advice1 = coinDirection(x, y, old_game_state['field'], crop1,self.crop_size)
    direction_advice2 = coinDirection(x, y, new_game_state['field'], crop2,self.crop_size)

    '''crop_array = [crop1_2, crop1_3, crop1_4]
    new_action_array = [action, action, action]
    new_coin_array = [(0,0), (0,0), (0,0)]
    if action not in ['BOMB', 'WAIT']:  # no matter the orientation, these actions remain same
        for i in range(1, 4):
            new_action_array[i - 1] = rotate_direction[i][rotate_direction[0].index(action)]
            if direction_advice1 != (0,0):
                new_coin_array[i - 1] = rotate_coin_direction_vector[i][rotate_coin_direction_vector[0].index(direction_advice1)]
    '''
    state1 = (tuple(tuple(row) for row in crop1),direction_advice1)  # state1 alte gamestate
    state2 = (tuple(tuple(row) for row in crop2),direction_advice2) # state2 ist der neue gamestate

    #Calculate states with rotated crop and adjusted direction advice
    '''    state_rotate_1 = (tuple(tuple(row) for row in crop_array[0]), new_coin_array[0])
    state_rotate_2 = (tuple(tuple(row) for row in crop_array[1]), new_coin_array[1])
    state_rotate_3 = (tuple(tuple(row) for row in crop_array[2]), new_coin_array[2])'''

    #if (x,y) in self.visited:
    #   reward-=15

    #self.visited.append((x,y))


    if direction_advice1[0] == 1:
        if 'LEFT' == action:
            #print("General move towards coin")
            reward += 35
        else:
            reward -= 10
    if direction_advice1[1] == 1:
        if 'UP'== action:
            #print("General move towards coin")
            reward += 35
        else:
            reward -= 10
    if direction_advice1[0] == -1:
        if 'RIGHT' == action:
            #print("General move towards coin")
            reward += 35
        else:
            reward -= 10
    if direction_advice1[1] == -1:
        if 'DOWN' == action:

            #print("General move towards coin")
            reward += 35
        else:
            reward -= 10

    #Reward for walking outside the bomb efficiently
    crop_self_xy = math.floor(self.crop_size / 2)
    if crop1[crop_self_xy][crop_self_xy] == 4:
        #print("Player is on the bomb! Take escape route now")
        escape_route = kuerzesterWegZumTile(math.floor(self.crop_size/2), math.floor(self.crop_size/2), crop1, 0)
        if escape_route is not False:
            if action == escape_route[0]:
                #print("Nice move! bonus!")
                reward+=100
        else:
            #print("Bad move! Fine!")
            reward-= 50
    bomb_coord = naherTile(math.floor(self.crop_size/2), math.floor(self.crop_size/2), crop1, 4)
    if bomb_coord:
        (bomb_x, bomb_y) = bomb_coord
        if bomb_x == math.floor(self.crop_size/2):
            #print("Axis with bomb same")
            reward -= 50
        if bomb_y == math.floor(self.crop_size/2):
            #print("Axis with bomb same")
            reward -= 50


    '''#Negative reward for walking from safe zone into bomb
    if crop1[crop_self_xy][crop_self_xy] == 0 and crop1[crop_self_xy+1][crop_self_xy] in [4,5] and action == 'RIGHT':
        #print("Brutal mistake")
        reward-=100
    if crop1[crop_self_xy][crop_self_xy] == 0 and crop1[crop_self_xy - 1][crop_self_xy] in [4,5] and action == 'LEFT':
        #print("Brutal mistake")
        reward -= 100
    if crop1[crop_self_xy][crop_self_xy] == 0 and crop1[crop_self_xy][crop_self_xy+1] in [4,5] and action == 'DOWN':
        #print("Brutal mistake")
        reward-=100
    if crop1[crop_self_xy][crop_self_xy] == 0 and crop1[crop_self_xy][crop_self_xy-1] in [4,5] and action == 'UP':
        #print("Brutal mistake")
        reward-=100'''


    '''for i in range(len(self.history) - 1):
        for dir1, dir2 in opposite_directions:
            if self.history[i] == dir1 and self.history[i + 1] == dir2:
                reward -= 10  # Penalize for opposite directions between adjacent moves'''

    if 'INVALID_ACTION' in events:
        reward-=1000 #kann passieren, da bombe -1 tiles uberschreibt
    if 'GOT_KILLED' in events:
        reward-=10000
    '''if 'SURVIVE_ROUND' in events:
        reward+=500 #Keep this as big as the number of steps'''
    if 'CRATE_DESTROYED' in events:
        reward += 200
    if 'COIN_FOUND' not in events:
        reward+=30
    if 'COIN_COLLECTED' in events:
        reward+=100
    if 'COIN_FOUND' in events:
        reward+=50
    if 'WAIT' in events:
        reward+=2
    if 'BOMB' in events: #Hier kann man präziser werden. Droppe nur wenn an KRater drumherum liegt oder Gegner
        reward+=50
    if 'UP' in events:
        reward += 2
    if 'DOWN' in events:
        reward += 2
    if 'RIGHT' in events:
        reward += 2
    if 'LEFT' in events:
        reward += 2
    coin_direction = kuerzesterWegZumTile(math.floor(self.crop_size/2),math.floor(self.crop_size/2), crop1, 2)
    if coin_direction:#in the crop there exists a coin, award him
        if action == coin_direction[0]:
            #print("Shortest move towards coin")
            reward+=40
        else:
            #print("Moves away")
            reward-=40

    # 0 0
    # 1 1 links oben
    # -1 1 rechts oben
    # 1 -1 links unten
    # -1 -1 rechts unten


    '''self.action_history.append(action)
    if len(self.action_history) == 2:
        if self.action_history[0]==self.action_history[1]:
            reward-=100
        else:
            reward+=20
        self.action_history = self.action_history[1:] #Remove oldest action, and now compare the next two new actions'''
    #ungefähre Richtung mitgeben in den state

    x_new = new_game_state['self'][3][0]  # player_x_coordinate
    y_new = new_game_state['self'][3][1]  # player_y_coordinate
    epsilon = 0.99 #0.7
    def maxQ(state):
        best_qvalue = float('-inf')
        possible_actions = filterInvalidActions(x_new,y_new,new_game_state['field'],new_game_state['self'][2],['UP', 'RIGHT', 'DOWN', 'LEFT','WAIT','BOMB'])
        for action in possible_actions:
            qvalue = self.qtable.get((state, action), 0)
            if qvalue == 0: #Initialize with 0 if new state
                self.qtable[(state, action)] = 0
            if qvalue >= best_qvalue:
                best_qvalue = qvalue
        return best_qvalue

    bn = self.btable.get((state1, action), 0)
    q_value_old = self.qtable.get((state1, action), 0)
    alpha = 1/(1+bn)
    self.qtable[(state1,action)] = (1-alpha)*q_value_old+alpha*(reward + epsilon * maxQ(state2)) #
    self.btable[(state1,action)] = bn + 1

    '''self.qtable[(state_rotate_1, new_action_array[0])] = (1 - alpha) * q_value_old + alpha * (reward + epsilon * maxQ(state2))  #
    self.btable[(state_rotate_1, new_action_array[0])] = bn + 1

    self.qtable[(state_rotate_2, new_action_array[1])] = (1 - alpha) * q_value_old + alpha * (reward + epsilon * maxQ(state2))  #
    self.btable[(state_rotate_2, new_action_array[1])] = bn + 1

    self.qtable[(state_rotate_3, new_action_array[2])] = (1 - alpha) * q_value_old + alpha * (reward + epsilon * maxQ(state2))  #
    self.btable[(state_rotate_3, new_action_array[2])] = bn + 1'''





    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    game_events_occurred(self, last_game_state, last_action,last_game_state, events)

    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """


    file_path = os.path.join(training_folder_name, qtable_file_name)
    file_path_b_table = os.path.join(training_folder_name, btable_file_name)
    if self.round_counter % self.period == 0:
        with open(file_path, "wb") as file:
            pickle.dump(self.qtable, file)

        with open(file_path_b_table, "wb") as file:
            pickle.dump(self.btable, file)
    self.round_counter += 1
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
