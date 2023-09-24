import os
import ujson
import json
import random
from agent_code.subfield_agent.state import GameState
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
    if self.train or not os.path.isfile(self.TRAINING_DATA_DIRECTORY + "q_24_09.json"):
        self.logger.info("Setting up model from scratch.")
    else:
        """If not in train mode the agent uses two Q-Tables, the first optimised for dealing with crates and coins
        and navigating throuh the level, the second one for fighting against other agents and surviving their attacks."""
        self.logger.info("Loading model from saved state.")
        with open(self.TRAINING_DATA_DIRECTORY + "q_nico.json", "rb") as file:
            self.Q_COINS_CRATES = defaultdict(float)
            loaded_dict = ujson.loads(file.read())
            for key in loaded_dict:
                splits = key[1:-1].split(", ")
                hash = int(splits[0])
                action = splits[1][1:-1]
                self.Q_COINS_CRATES[(hash, action)] = loaded_dict[key]

        with open(self.TRAINING_DATA_DIRECTORY + "q_24_09.json", "rb") as file:
            self.Q_HITMAN = defaultdict(float)
            loaded_dict = ujson.loads(file.read())
            for key in loaded_dict:
                splits = key[1:-1].split(", ")
                hash = int(splits[0])
                action = splits[1][1:-1]
                self.Q_HITMAN[(hash, action)] = loaded_dict[key]
        
    if not self.train:
        self.hash_to_features_coins_crates = dict()
        self.hash_to_action_values_coins_crates = dict()
        self.hash_to_features_hitman = dict()
        self.hash_to_action_values_hitman = dict()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    game_state = GameState(game_state)

    if self.train and random.random() < self.EPSILON:
        return np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    
    if not self.train and random.random() > 0.8:
        return np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    
    game_state_subfield = game_state.to_features_subfield()

    #Check rather other agent is close, in this case I an 7x7 field around the player
    if isinstance(game_state_subfield, dict):
        try:
            agent_in_subfield = game_state_subfield["other_agent_in_subfield"]
        except KeyError:
            agent_in_subfield = False
    else:
        agent_in_subfield = False

    #If other agent is close switch to combate mode (hitman), else stay in normal mode
    if agent_in_subfield == True:
        hashed_gamestate = game_state.to_hashed_features()
        action_values = dict()
        for action in self.ACTIONS:
            if (hashed_gamestate, action) in self.Q_HITMAN:
                action_values[action] = self.Q_HITMAN[(hashed_gamestate, action)]
            else:
                action_values[action] = 0

        possible_moves = game_state.get_possible_moves()
        action_values = {game_state.adjust_movement(action): action_values[action] for action in action_values}
        chosen_action = sorted([(action, action_values[action]) for action in possible_moves], key=lambda x: x[1], reverse=True)[0][0]
        default_valued_actions = [action for action in possible_moves if int(action_values[action]) == 0]

        # only take gamestates as training data, which have been explored a bit
        if len(default_valued_actions) < 3 and not self.train:
            self.hash_to_features_hitman[hashed_gamestate] = game_state.to_features_subfield()
            self.hash_to_action_values_hitman[hashed_gamestate] = {action: action_values[action] for action in self.ACTIONS}
        if max(action_values[action] for action in possible_moves) == 0 and len(default_valued_actions) > 1:
            if not self.train:
                pass

            return np.random.choice(possible_moves)

        elif not self.train:
            pass

        if game_state.round % 10 == 0 and not self.train:
            with open(self.TRAINING_DATA_DIRECTORY + "hash_to_features_coins_crates.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.hash_to_features_hitman, cls=NumpyEncoder))
            with open(self.TRAINING_DATA_DIRECTORY + "hash_to_action_values_coins_crates.json", "w", encoding="utf-8") as f:
                f.write(ujson.dumps(self.hash_to_action_values_hitman))

        return chosen_action

    else:
        hashed_gamestate = game_state.to_hashed_features()
        action_values = dict()
        for action in self.ACTIONS:
            if (hashed_gamestate, action) in self.Q_COINS_CRATES:
                action_values[action] = self.Q_COINS_CRATES[(hashed_gamestate, action)]
            else:
                action_values[action] = 0

        possible_moves = game_state.get_possible_moves()
        action_values = {game_state.adjust_movement(action): action_values[action] for action in action_values}
        chosen_action = sorted([(action, action_values[action]) for action in possible_moves], key=lambda x: x[1], reverse=True)[0][0]
        default_valued_actions = [action for action in possible_moves if int(action_values[action]) == 0]

        # only take gamestates as training data, which have been explored a bit
        if len(default_valued_actions) < 3 and not self.train:
            self.hash_to_features_coins_crates[hashed_gamestate] = game_state.to_features_subfield()
            self.hash_to_action_values_coins_crates[hashed_gamestate] = {action: action_values[action] for action in self.ACTIONS}
        if max(action_values[action] for action in possible_moves) == 0 and len(default_valued_actions) > 1:
            if not self.train:
                pass
            return np.random.choice(possible_moves)

        elif not self.train:
            pass

        if game_state.round % 10 == 0 and not self.train:
            with open(self.TRAINING_DATA_DIRECTORY + "hash_to_features_hitman.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.hash_to_features_coins_crates, cls=NumpyEncoder))
            with open(self.TRAINING_DATA_DIRECTORY + "hash_to_action_values_hitman.json", "w", encoding="utf-8") as f:
                f.write(ujson.dumps(self.hash_to_action_values_coins_crates))

        return chosen_action


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)