import torch
from .state import GameState

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
    self.logger.info("Loading model from saved state.")
    self.model = torch.load(self.TRAINING_DATA_DIRECTORY + "model.pt")
    self.model.eval()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    game_state = GameState(game_state)
    print(game_state.to_vec(game_state.to_features()))
    hashed_gamestate = game_state.to_hashed_features()
    possible_moves = game_state.get_possible_moves()
    return np.random.choice(self.ACTIONS)