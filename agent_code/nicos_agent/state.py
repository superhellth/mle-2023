import numpy as np
import hashlib

class GameState:

    def __init__(self, game_state: dict):
        self.round = game_state["round"]
        self.step = game_state["step"]
        self.field = game_state["field"]
        self.flipped_vertical = False
        self.flipped_horizontal = False
        self.flipped_slash = False
        self.flipped_backslash = False
        self.agent_name = game_state["self"][0]
        self.agent_score = game_state["self"][1]
        self.agent_bombs_left = game_state["self"][2]
        self.agent_position = game_state["self"][3]
        self.other_agents = game_state["others"]
        self.bombs = game_state["bombs"]
        self.coins = game_state["coins"]
        self.explosion_map = game_state["explosion_map"]

    def get_possible_moves(self):
        x = self.agent_position[0]
        y = self.agent_position[1]
        possible_moves = ["WAIT"]
        if y % 2 == 1 and x != 15:
            possible_moves.append("RIGHT")
        if y % 2 == 1 and x != 1:
            possible_moves.append("LEFT")
        if x % 2 == 1 and y != 15:
            possible_moves.append("DOWN")
        if x % 2 == 1 and y != 1:
            possible_moves.append("UP")

        return possible_moves

    def adjust_movement(self, original_move):
        straight_axis_untangled = original_move
        if self.flipped_horizontal:
            if original_move == "UP":
                straight_axis_untangled = "DOWN"
            elif original_move == "DOWN":
                straight_axis_untangled = "UP"
        if self.flipped_vertical:
            if original_move == "LEFT":
                straight_axis_untangled = "RIGHT"
            elif original_move == "RIGHT":
                straight_axis_untangled = "LEFT"
        backslash_axis_untangled = straight_axis_untangled
        if self.flipped_backslash:
            if straight_axis_untangled == "RIGHT":
                backslash_axis_untangled = "DOWN"
            elif straight_axis_untangled == "LEFT":
                backslash_axis_untangled = "UP"
            elif straight_axis_untangled == "DOWN":
                backslash_axis_untangled = "RIGHT"
            elif straight_axis_untangled == "UP":
                backslash_axis_untangled = "LEFT"
        return backslash_axis_untangled

    def adjust_position(self, position):
        """Adjust position according to field rotations done.

        Args:
            position (list): List of x and y corrdinate.

        Returns:
            list: Adjusted x and y coordinate.
        """
        if self.flipped_vertical:
            position[0] = 16 - position[0]
        if self.flipped_horizontal:
            position[1] = 16 - position[1]
        if self.flipped_backslash:
            old_x = position[0]
            old_y = position[1]
            position[0] = old_y
            position[1] = old_x
        return position

    def get_distance(self, pos1, pos2):
        """Calculate the distance between two coordinates.

        Args:
            pos1 (list): Position 1.
            pos2 (list): Position 2.

        Returns:
            float: Euclidian distance.
        """
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def to_features(self):
        """Convert game state to more generic feature vector.

        Returns:
            tuple: List of features.
        """
        # round ignored
        # step ignored
        ## field: exploit symmetry, only consider boxes nearby, (cut off outer ring/only boxes)
        # agent_name ignored
        # agent_score ignored
        ## agent_bombs_left
        ## agent_position: exploit symmetry
        ## other_agents: position and bombs_left, exploit_symmetry
        ## bombs: only nearby, position and timer, exploit symmetry
        ## coins: only nearby
        ## explosion_map: exploit symmetry

        feature_field = self.field
        feature_agent_position = np.array([self.agent_position[0], self.agent_position[1]])
        # print(f"Original agent position: {feature_agent_position}")

        ### rotate field
        # vertical axis
        if self.agent_position[0] > 8:
            self.flipped_vertical = True
            feature_field = np.fliplr(feature_field)
            feature_agent_position[0] = 16 - feature_agent_position[0]
            # print(f"Mirrored vertically: {feature_agent_position}")
        # horizontal axis
        if self.agent_position[1] > 8:
            self.flipped_horizontal = True
            feature_field = np.flipud(feature_field)
            feature_agent_position[1] = 16 - feature_agent_position[1]
            # print(f"Mirrored horizontally: {feature_agent_position}")
        # backslash axis
        # if feature_agent_position[0] < feature_agent_position[1]:
        #     self.flipped_backslash = True
        #     feature_field = feature_field.T
        #     # mirror agent position
        #     old_x = feature_agent_position[0]
        #     old_y = feature_agent_position[1]
        #     feature_agent_position[0] = old_y
        #     feature_agent_position[1] = old_x
            # print(f"Mirrored backslash: {feature_agent_position}")
        
        ### rotate coins and filter out closest one
        feature_coins = [np.array(self.adjust_position([coin[0], coin[1]])) for coin in self.coins]
        nearest_coins = sorted(feature_coins, key=lambda x: self.get_distance(x, feature_agent_position))
        # nearest_coins = sorted(self.coins, key=lambda x: self.get_distance(x, self.agent_position))
        if len(nearest_coins) > 0:
            if nearest_coins[0][0] == feature_agent_position[0] and nearest_coins[0][1] == feature_agent_position[1] and len(nearest_coins) > 1:
                nearest_coin = nearest_coins[1]
            else:
                nearest_coin = nearest_coins[0]
        else:
            nearest_coin = np.array([0, 0])

        vector = np.array(nearest_coin) - np.array(feature_agent_position)
        # print(vector)

        return [vector, np.array([feature_agent_position[0] % 2, feature_agent_position[1] % 2])]


    def to_hashed_features(self):
        """Hash this game state.

        Returns:
            int: Hash of game state.
        """
        hash_list = list()
        features = self.to_features()
        for feature in features:
            # print(feature)
            hash_list.append(int(hashlib.md5(str(feature).encode()).hexdigest(), 16))
        final_hash = hash_list[0]
        for a_hash in hash_list[1:]:
            final_hash ^= a_hash
        # print(f"Features: {features} hash to: {final_hash}")
        return final_hash


    def get_potential(self):
        return 20 * self.agent_score - self.get_closest_coin_distance()


    def get_closest_coin_distance(self):
        if len(self.coins) > 0:
            return sorted([self.get_distance(coin, self.agent_position) for coin in self.coins])[0]
        else:
            return 0
