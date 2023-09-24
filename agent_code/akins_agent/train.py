from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
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
    '''if not os.path.exists(file_path_btable):
        self.btable = {}
        with open(file_path_btable, "wb") as file:
            pass

    else:
        with open(file_path_btable, "rb") as file:
           self.btable = pickle.load(file) # Load pretrained btable'''



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


def explosionToField(explosion_map, field):
    explosion_map_array = np.array(explosion_map)
    field_array = np.array(field)
    field_array[explosion_map_array == 1] += 8
    field[:] = field_array
def coinsToField(coins, field):
    for coin in coins:
        field[coin[0]][coin[1]] = 2

def opponentToField(others, field):
    for opponent in others:
        (opponent_x, opponent_y) = opponent[3]
        field[opponent_x][opponent_y] = 3

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
                return (x - 1, y)
            elif x + 1 < field_length and field[x + 1][y] == tile_value:
                return (x + 1, y)
            elif y - 1 >= 0 and field[x][y - 1] == tile_value:
                return (x, y - 1)
            elif y + 1 < field_length and field[x][y + 1] == tile_value:
                return (x, y + 1)
            # Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen
            if x - 1 >= 0:  # Prüfe ob wir den Rand noch nicht erreicht haben
                if field[x - 1][y] != -1 and field[x - 1][y] != 1 and (x - 1,
                                                                       y) not in visitedCoordinates:  # Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                    newCoordinates.append(
                        (x - 1, y))  # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
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
    for tile_x in range(
            crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
        for tile_y in range(crop_length):
            if not checkLineOfSight(tile_x, tile_y, 3, 3, crop):
                crop[tile_x][tile_y] = 0

def keepOneCoin(crop):
    crop_length = len(crop[0])
    closest_coin = naherTile(3, 3, crop, 2)
    for tile_x in range(
            crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
        for tile_y in range(crop_length):
            if crop[tile_x][tile_y] == 2:
                crop[tile_x][tile_y] = 0
    if closest_coin:
        (closest_coin_x, closest_coin_y) = closest_coin
        crop[closest_coin_x][closest_coin_y] = 2

def keepOneEnemy(crop):
    crop_length = len(crop[0])
    closest_coin = naherTile(3, 3, crop, 3)
    for tile_x in range(
            crop_length):  # Wir schauen alle Tiles an, ob eine Verbindung zum Spieler gibt. Wenn nicht => unnötige Information
        for tile_y in range(crop_length):
            if crop[tile_x][tile_y] == 3:
                crop[tile_x][tile_y] = 0
    if closest_coin:
        (closest_coin_x, closest_coin_y) = closest_coin
        crop[closest_coin_x][closest_coin_y] = 3

def coinDirection(x,y,field,crop): #gives additional hint of the direction towards the next coin outside (even trough crates) if there is no coin in crop
    if naherTile(3,3,crop,2):
        return (0,0)
    else:
        coin_coord = naherTile(x,y,field,2)
        if coin_coord:
            (x_coin,y_coin) = coin_coord
            normalize_y = abs(y-y_coin)
            normalize_x= abs(x-x_coin)
            if normalize_x == 0:
                normalize_x=1
            if normalize_y == 0:
                normalize_y=1
            return ((x-x_coin)/normalize_x,(y-y_coin)/normalize_y) #normalisieren!!! weniger Q states
        return (0,0)


def coinExistsOutside(x, y, field, crop):
    if naherTile(3, 3, crop, 2):
        return 0
    else:
        if naherTile(x, y, field, 2):
            return 1
        return 0

def getCrop(game_state):
    field = game_state['field']
    x = game_state['self'][3][0]  # player_x_coordinate
    y = game_state['self'][3][1]  # player_y_coordinate
    explosionToField(game_state['explosion_map'], field)
    coinsToField(game_state['coins'], field)
    opponentToField(game_state['others'], field)
    bombToField(game_state['bombs'], field)
    crop = cropSevenTiles(x, y, field.tolist())
    keepOneCoin(crop)
    keepOneEnemy(crop)
    reduceInformation(crop)

    return crop

def last_action(self, best_action):
    if len(self.history) < self.history_size:
        self.history.append(best_action)
    else:
        self.history.append(best_action)
        self.history = self.history[1:] #Forget first element

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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = 0

    field = old_game_state['field']
    x = old_game_state['self'][3][0]  # player_x_coordinate
    y = old_game_state['self'][3][1]  # player_y_coordinate

    bomb_active = not old_game_state['self'][2]
    #action = self_action
    action = None
    # How do we get our action? By the event...
    if 'WAIT' in events:
        action = 'WAIT'
    if 'BOMB' in events:
        action = 'BOMB'
    if 'LEFT' in events:
        action = 'LEFT'
    if 'RIGHT' in events:
        action = 'RIGHT'
    if 'UP' in events:
        action = 'UP'
    if 'DOWN' in events:
        action = 'DOWN'

    crop1 = getCrop(old_game_state)
    crop2 = getCrop(new_game_state)  # Very inefficient


    direction_advice1 = coinDirection(x, y, old_game_state['field'], crop1)
    direction_advice2 = coinDirection(x, y, new_game_state['field'], crop2)

    state1 = (tuple(tuple(row) for row in crop1),direction_advice1)  # state1 alte gamestate
    state2 = (tuple(tuple(row) for row in crop2),direction_advice2) # state2 ist der neue gamestate

    #if (x,y) in self.visited:
    #   reward-=15

    #self.visited.append((x,y))


    if direction_advice1[0] == 1:
        if 'LEFT' == action:
            reward += 35
        else:
            reward -= 10
    if direction_advice1[1] == 1:
        if 'UP'== action:
            reward += 35
        else:
            reward -= 10
    if direction_advice1[0] == -1:
        if 'RIGHT' == action:
            reward += 35
        else:
            reward -= 10
    if direction_advice1[1] == -1:
        if 'DOWN' == action:
            reward += 35
        else:
            reward -= 10

    #Reward for walking outside the bomb efficiently
    for i in range(4, 9):
        if crop1[3][3] == i:
            escape_route = kuerzesterWegZumTile(3, 3, crop1, 0)
            if escape_route is not False:
                if action == escape_route[0]:
                    reward+=70
            else:
                escape_route = kuerzesterWegZumTile(3, 3, crop1, 2)
                if escape_route is not False:
                    if action == escape_route[0]:
                        reward += 70
                else:
                    reward-= 50
        bomb_coord = naherTile(3, 3, crop1, i)
        if bomb_coord:

            (bomb_x, bomb_y) = bomb_coord
            if bomb_x == 3 or bomb_y == 3:
                reward -= 50
    if bomb_active:
        if crop1[3][3] == 0:
            reward+=3
    #Negative reward for walking from safe zone into bomb
    if crop1[3][3] == 0 and crop1[3+1][3] in [4,5,6,7,8] and action == 'RIGHT':
        reward-=200
    if crop1[3][3] == 0 and crop1[3 - 1][3] in [4, 5, 6, 7, 8] and action == 'LEFT':
        reward -= 200
    if crop1[3][3] == 0 and crop1[3][3+1] in [4,5,6,7,8] and action == 'DOWN':
        reward-=200
    if crop1[3][3] == 0 and crop1[3][3-1] in [4,5,6,7,8] and action == 'UP':
        reward-=200




    '''for i in range(len(self.history) - 1):
        for dir1, dir2 in opposite_directions:
            if self.history[i] == dir1 and self.history[i + 1] == dir2:
                reward -= 10  # Penalize for opposite directions between adjacent moves'''

    if 'INVALID_ACTION' in events:
        reward-=100
    if 'GOT_KILLED' in events:
        reward-=200
    if 'SURVIVE_ROUND' in events:
        reward+=0
    if 'CRATE_DESTROYED' in events and 'GOT_KILLED' not in events:
        reward += 1000
    if 'COIN_FOUND' not in events:
        reward+=30
    if 'COIN_COLLECTED' in events:
        reward+=150
    if 'COIN_FOUND' in events:
        reward+=10
    if 'WAIT' in events:
        reward+=2
    if 'BOMB' in events: #Hier kann man präziser werden. Droppe nur wenn an KRater drumherum liegt oder Gegner
        reward+=80
    if 'UP' in events:
        reward += 2
    if 'DOWN' in events:
        reward += 2
    if 'RIGHT' in events:
        reward += 2
    if 'LEFT' in events:
        reward += 2
    if bomb_active is False: #only reward coin collection if own bomb is not ticking
        coin_direction = kuerzesterWegZumTile(3, 3, crop1, 2)
        if coin_direction:#in the crop there exists a coin, award him
            if action == coin_direction[0]:
                reward+=30
            else:
                reward-=10


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


    epsilon = 0.6
    def maxQ(state):
        best_qvalue = 0
        for action in ['UP', 'RIGHT', 'DOWN', 'LEFT','WAIT','BOMB']:
            qvalue = self.qtable.get((state, action), 0) #Simulates the "initialize all Q-values with 0
            if qvalue > best_qvalue:
                best_qvalue = qvalue
        return best_qvalue
    self.qtable[(state1,action)] = reward + epsilon * maxQ(state2) #
    #if self.qtable.get((state1, action), 0) < -1000000000:
    #    print("something dangerous")
    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
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

    if self.round_counter % self.period == 0:
        with open(file_path, "wb") as file:
            pickle.dump(self.qtable, file)
    self.round_counter += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
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
