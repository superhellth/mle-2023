import os
import ujson
import json
import random
from .state import GameState
from collections import defaultdict

import numpy as np


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
    self.logger.info("Agent setup")
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self.TRAINING_DATA_DIRECTORY = "./training_data/"
    if self.train or not os.path.isfile(self.TRAINING_DATA_DIRECTORY + "q.json"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        with open(self.TRAINING_DATA_DIRECTORY + "q.json", "rb") as file:
            self.Q = defaultdict(float)
            loaded_dict = ujson.loads(file.read())
            for key in loaded_dict:
                splits = key[1:-1].split(", ")
                hash = int(splits[0])
                action = splits[1][1:-1]
                self.Q[(hash, action)] = loaded_dict[key]
    if not self.train:
        self.hash_to_features = dict()
        self.hash_to_action_values = dict()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    game_state = GameState(game_state)
    # print(f"Game state is surviveable: {game_state.can_agent_survive()}")
    # print(game_state.agent_position)
    # print(game_state.explosion_map)
    # if game_state.to_features() == -1:
    #     print("DEAD")
    # else:
    if self.train and random.random() < self.EPSILON:
        return np.random.choice(game_state.get_possible_moves())
    hashed_gamestate = game_state.to_hashed_features()
    action_values = dict()
    for action in self.ACTIONS:
        if (hashed_gamestate, action) in self.Q:
            action_values[action] = self.Q[(hashed_gamestate, action)]
        else:
            action_values[action] = 0
    # print(action_values)
    possible_moves = game_state.get_possible_moves()
    action_values = {game_state.adjust_movement(action): action_values[action] for action in action_values}
    chosen_action = sorted([(action, action_values[action]) for action in possible_moves], key=lambda x: x[1], reverse=True)[0][0]
    default_valued_actions = [action for action in possible_moves if int(action_values[action]) == 0]
    # only take gamestates as training data, which have been explored a bit
    if len(default_valued_actions) < 3 and not self.train:
        self.hash_to_features[hashed_gamestate] = game_state.to_features()
        self.hash_to_action_values[hashed_gamestate] = {action: action_values[action] for action in self.ACTIONS}
    if max(action_values[action] for action in possible_moves) == 0 and len(default_valued_actions) > 1:
        if not self.train:
            pass
            # print("Choosing random action")
            # print(game_state.to_features())
        # print(action_values)
        return np.random.choice(possible_moves)
    elif not self.train:
        pass
        # print(action_values)
        # print(chosen_action)
        # print(game_state.get_closest_coin_distance())
    # print(chosen_action)

    if game_state.round % 10 == 0 and not self.train:
        with open(self.TRAINING_DATA_DIRECTORY + "hash_to_features.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(self.hash_to_features, cls=NumpyEncoder))
        with open(self.TRAINING_DATA_DIRECTORY + "hash_to_action_values.json", "w", encoding="utf-8") as f:
            f.write(ujson.dumps(self.hash_to_action_values))

    return chosen_action


def cropSevenTiles(game_state):
    '''
    This function crops the 17x17 field into a 7x7 with the player centered , i.e. the surrounding matrix.
    This will preprocessed further before used as a state in Q-Learning.
    '''
    x,y = game_state.get_agent_position()
    field_prepared = game_state.get_field()
    #Calculate positions on the field affected by bombs
    bomb_position = []
    for (bomb,_) in game_state.get_bombs_position():
        bomb_position.append(game_state.get_bomb_explosion_squares(bomb))
    flattened_list = [item for sublist in bomb_position for item in sublist]
    bomb_position_array = np.array(flattened_list)
    bomb_position = np.unique(bomb_position_array,axis=0)

    #Positions affected by bombs marked with -2
    for coordinate in bomb_position:
        x = coordinate[0]
        y = coordinate[1]
        field_prepared[x][y] = -2

    #Positions with coins marked with 4
    coins = game_state.get_coins()
    for coin in coins:
        x = coin[0]
        y = coin[1]
        field_prepared[x][y] = 4

    #Mark enemy agent positions with 3
    other_agents = game_state.get_other_agents_position()
    for other_agent in other_agents:
        x = other_agent[3][0]
        y = other_agent[3][1]
        field_prepared[x][y] = 3

    x = x+2
    y = y+2
    padded_array = np.pad(field_prepared, 2, mode='constant', constant_values=-1)
    croped_array = padded_array[x-3:x+4, y-3:y+4]
    #croped_array = np.transpose(croped_array)

    #5 decodes agents position
    croped_array[3][3]=5
    #Mark all positions unaccessibale for the player with -1 except of crates (1)
    croped_array,other_agent = calculate_accessible_parts(croped_array)
    print(croped_array,other_agent)

    return croped_array,other_agent




def explore_field(field, x, y, explored,other_agent):
    if x < 0 or y < 0 or x >= field.shape[0] or y >= field.shape[1] or field[x, y] in [-1, 1, 3]:
        return
    
    if (x, y) not in explored:
        explored.append((x, y))
        if field[x][y] == 3:
            other_agent.append((x,y))
    
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            explore_field(field, x + dx, y + dy, explored,other_agent)

def calculate_accessible_parts(field):
    # Copy the field to avoid modifying the original array
    result_field = field.copy()
    
    # List to keep track of explored fields and if other agent was found
    explored = []
    other_agent = []
    
    # Perform DFS to explore accessible parts
    explore_field(result_field, 3, 3, explored,other_agent)  # Assuming the starting position is (3, 3)
    
    # Mark unexplored fields as -1
    for i in range(result_field.shape[0]):
        for j in range(result_field.shape[1]):
            if (i, j) not in explored and field[i][j]!=1 and field[i][j]!=3:
                result_field[i, j] = -1
    
    return result_field,other_agent


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)