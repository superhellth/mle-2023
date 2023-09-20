import hashlib
import numpy as np
import networkx as nx


class GameState:
    """
    This class stores all information of one game state. Provides utility functions and calculates features.
    """

    def __init__(self, game_state: dict, dead=False):
        if dead:
            self.dead = True
        else:
            self.dead = False
            self.round = game_state["round"]
            self.step = game_state["step"]
            self.field = game_state["field"]
            self.agent_name = game_state["self"][0]
            self.agent_score = game_state["self"][1]
            self.agent_bombs_left = game_state["self"][2]
            self.agent_position = game_state["self"][3]
            self.other_agents = game_state["others"]
            self.bombs = game_state["bombs"]
            self.coins = game_state["coins"]
            self.explosion_map = game_state["explosion_map"]

    def get_agent_position(self):
        return self.agent_position
    
    def get_bombs_position(self):
        return self.bombs
    
    def get_coins_position(self):
        return self.coins
    
    def get_field(self):
        return self.field