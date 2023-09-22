import torch
from .network import Net
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
    self.model = Net()
    self.model.load_state_dict(torch.load(self.TRAINING_DATA_DIRECTORY + "deep_q.pt"))
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
    x = torch.Tensor(np.array(dict_to_vec(game_state.to_features())))
    # print(x)
    pred = self.model(x)
    actions_in_order = ["LEFT", "RIGHT", "UP", "DOWN", "WAIT", "BOMB"]
    return actions_in_order[torch.argmax(pred)]

def dict_to_vec(feature_dict):
    if feature_dict == -1:
        return np.array([0 for i in range(23)]).flatten()
    # local map is always a feature
    local_map_vec = np.array(feature_dict["local_map"]).flatten()
    # would survive bomb
    if "would_survive_bomb" in feature_dict.keys():
        would_survive_bomb_vec = np.array([feature_dict["would_survive_bomb"]])
    else:
        would_survive_bomb_vec = np.array([True])
    # local prec explosion map
    if "local_prec_explosion_map" in feature_dict.keys():
        local_prec_explosion_map_vec = np.array(feature_dict["local_prec_explosion_map"]).flatten()
    else:
        local_prec_explosion_map_vec = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).flatten()
    # agent to nearest safe square
    if "agent_to_nearest_safe_square" in feature_dict.keys() and not np.array(feature_dict["agent_to_nearest_safe_square"]).shape[0] == 2:
        agent_to_nearest_safe_square_vec = np.array(feature_dict["agent_to_nearest_safe_square"]).flatten()
    else:
        agent_to_nearest_safe_square_vec = np.array([0, 0]).flatten()
    # agent to nearest coin
    if "agent_to_nearest_coin" in feature_dict.keys():
        agent_to_nearest_coin_vec = np.array(feature_dict["agent_to_nearest_coin"]).flatten()
    else:
        agent_to_nearest_coin_vec = np.array([0, 0]).flatten()
    return np.concatenate((local_map_vec, would_survive_bomb_vec, local_prec_explosion_map_vec, agent_to_nearest_safe_square_vec, agent_to_nearest_coin_vec)).flatten()
