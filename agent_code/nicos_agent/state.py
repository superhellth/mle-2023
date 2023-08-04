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
            closest_bomb = -1
            self.agent_can_escape_bomb = True
            if len(self.bombs) > 0:
                closest_bomb = sorted(self.bombs, key=lambda x: self.get_rockless_distance(
                    x[0], self.agent_position))[0]
                if closest_bomb[0][0] == self.agent_position[0] and closest_bomb[0][1] == self.agent_position[1]:
                    if closest_bomb[1] < 3:
                        self.agent_can_escape_bomb = False
                elif closest_bomb[0][0] == self.agent_position[0] and self.agent_position[0] % 2 == 1:
                    if closest_bomb[1] < 3 - abs(closest_bomb[0][1] - self.agent_position[1]):
                        self.agent_can_escape_bomb = False
                elif closest_bomb[0][1] == self.agent_position[1] and self.agent_position[1] % 2 == 1:
                    if closest_bomb[1] < 3 - abs(closest_bomb[0][0] - self.agent_position[0]):
                        self.agent_can_escape_bomb = False

    def get_possible_moves(self):
        """Get a list of all possible moves in the current game state.

        Returns:
            list: List of all possible moves.
        """
        x = self.agent_position[0]
        y = self.agent_position[1]
        possible_moves = ["WAIT"]
        if self.agent_bombs_left:
            possible_moves.append("BOMB")
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
        """Adjust the target movement according to the rotations done by get_features().

        Args:
            original_move (str): Move in feature space.

        Returns:
            str: Move in real environment.
        """
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

    def get_shortest_path(self, pos1, pos2, ignore_crates=True):
        """Find the shortest path from pos1 to pos2.

        Args:
            pos1 (list or np.array): Starting position.
            pos2 (list or np.array): Target destination.
            ignore_crates (bool, optional): If true crates are treated as walkable spaces. If False crates are treated as rock. Defaults to True.

        Returns:
            list: List of positions to walk to one by one.
        """
        n = len(self.field)
        G = nx.Graph()

        for x in range(n):
            for y in range(n):
                if self.field[x][y] == 0 or (self.field[new_x][new_y] == 1) and ignore_crates:
                    G.add_node((x, y))
                    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < n and 0 <= new_y < n and (self.field[new_x][new_y] == 0 or (self.field[new_x][new_y] == 1) and ignore_crates):
                            G.add_edge((x, y), (new_x, new_y))
        try:
            return nx.shortest_path(G, source=(pos1[0], pos1[1]), target=(pos2[0], pos2[1]))
        except nx.NetworkXNoPath:
            return -1  # no valid path

    def get_shortest_path_length(self, pos1, pos2, ignore_crates=True):
        """Get the length of the shortest path with the possibility of considering crates.

        Args:
            pos1 (list or np.array): Starting position.
            pos2 (list or np.array): Target position.
            ignore_crates (bool, optional): If True pretend crates are walkable. If False navigate around crates. Defaults to True.

        Returns:
            int: Length of shortest path. -1 if not path possible.
        """
        return len(self.get_shortest_path(pos1, pos2, ignore_crates=ignore_crates))

    def get_rockless_distance(self, pos1, pos2):
        """Calculate distance of shortest path from pos1 to pos2. Does not consider crates or bombs.

        Args:
            pos1 (list or np.array): Starting position.
            pos2 (list or np.array): Target position.

        Returns:
            float: Shortest path distance.
        """
        if pos1[0] == pos2[0] and pos1[1] != pos2[1] and pos1[0] % 2 == 0:
            dx = 2
        else:
            dx = abs(pos1[0] - pos2[0])
        if pos1[1] == pos2[1] and pos1[0] != pos2[0] and pos1[1] % 2 == 0:
            dy = 2
        else:
            dy = abs(pos1[1] - pos2[1])
        # + 0.01 * (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))
        return dx + dy

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
        # field: exploit symmetry, only consider boxes nearby, (cut off outer ring/only boxes)
        # other_agents: position and bombs_left, exploit_symmetry
        # bombs: only nearby, position and timer, exploit symmetry
        # explosion_map: exploit symmetry

        feature_field = self.field
        feature_explosion_map = self.explosion_map
        feature_agent_position = np.array(
            [self.agent_position[0], self.agent_position[1]])

        # rotate field
        # vertical axis
        if self.agent_position[0] > 8:
            self.flipped_vertical = True
            feature_field = np.fliplr(feature_field)
            feature_explosion_map = np.fliplr(feature_explosion_map)
            feature_agent_position[0] = 16 - feature_agent_position[0]
        # horizontal axis
        if self.agent_position[1] > 8:
            self.flipped_horizontal = True
            feature_field = np.flipud(feature_field)
            feature_explosion_map = np.flipud(feature_explosion_map)
            feature_agent_position[1] = 16 - feature_agent_position[1]
        # backslash axis
        # if feature_agent_position[0] < feature_agent_position[1]:
        #     self.flipped_backslash = True
        #     feature_field = feature_field.T
        #     # mirror agent position
        #     old_x = feature_agent_position[0]
        #     old_y = feature_agent_position[1]
        #     feature_agent_position[0] = old_y
        #     feature_agent_position[1] = old_x

        # rotate coins and bombs and filter out closest one
        feature_coins = [np.array(self.adjust_position(
            [coin[0], coin[1]])) for coin in self.coins]
        nearest_coins = sorted(feature_coins, key=lambda x: self.get_rockless_distance(
            x, feature_agent_position))
        feature_bombs = [[np.array(self.adjust_position(
            [bomb[0][0], bomb[0][1]])), bomb[1]] for bomb in self.bombs]
        nearest_bombs = sorted(feature_bombs, key=lambda x: self.get_rockless_distance(
            x[0], feature_agent_position))
        if len(nearest_coins) > 0:
            if nearest_coins[0][0] == feature_agent_position[0] and nearest_coins[0][1] == feature_agent_position[1] and len(nearest_coins) > 1:
                nearest_coin = nearest_coins[1]
            else:
                nearest_coin = nearest_coins[0]
        else:
            nearest_coin = np.array([0, 0])
        if len(nearest_bombs) > 0:
            nearest_bomb = nearest_bombs[0]
        else:
            nearest_bomb = -1

        agent_to_nearest_coin = np.array(
            nearest_coin) - np.array(feature_agent_position)
        if nearest_bomb == -1:
            agent_to_nearest_bomb = -1
            nearest_bomb_timer = -1
        else:
            agent_to_nearest_bomb = nearest_bomb[0] - \
                np.array(feature_agent_position)
            nearest_bomb_timer = nearest_bomb[1]
        agent_can_move_left = self.field[self.agent_position[0] -
                                         1][self.agent_position[1]] == 0
        agent_can_move_right = self.field[self.agent_position[0] +
                                          1][self.agent_position[1]] == 0
        agent_can_move_up = self.field[self.agent_position[0]
                                       ][self.agent_position[1] - 1] == 0
        agent_can_move_down = self.field[self.agent_position[0]
                                         ][self.agent_position[1] + 1] == 0
        local_explosion_map = np.array([[0, feature_explosion_map[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                                        [feature_explosion_map[feature_agent_position[0] - 1][feature_agent_position[1]], feature_explosion_map[feature_agent_position[0]]
                                            [feature_agent_position[1]], feature_explosion_map[feature_agent_position[0] + 1][feature_agent_position[1]]],
                                        [0, feature_explosion_map[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])

        feature_list = [agent_can_move_up, agent_can_move_down, agent_can_move_left, agent_can_move_right]
        if np.max(local_explosion_map) != 0:
            feature_list.append(local_explosion_map)
        if self.position_is_close_to_danger(self.agent_position):
            feature_list.append(agent_to_nearest_bomb)
            feature_list.append(nearest_bomb_timer)
        elif np.max(local_explosion_map) == 0:
            feature_list.append(agent_to_nearest_coin)
        return feature_list

    def position_is_in_danger(self, position):
        """Check if the given position is inside the range of a bomb explosion.

        Args:
            position (list or np.array): Position to check.

        Returns:
            bool: Whether or not position is in bomb range.
        """
        if len(self.bombs) > 0:
            for bomb in self.bombs:
                if bomb[0][0] == position[0] and bomb[0][1] == position[1]:
                    return True
                elif bomb[0][0] == position[0] and position[0] % 2 == 1:
                    if abs(bomb[0][0] - position[0]) <= 3:
                        return True
                elif bomb[0][1] == position[1] and position[1] % 2 == 1:
                    if abs(bomb[0][1] - position[1]) <= 3:
                        return True
        return False

    def position_is_close_to_danger(self, position):
        """This checks if any of the squares around the player is in range of a bomb explosion.

        Args:
            position (list or np.array): Position to check.

        Returns:
            bool: Whether or not position is close to danger.
        """
        nearby_positons = [[position[0], position[1] + 1], [position[0], position[1] - 1],
                           [position[0] - 1, position[1]], [position[0] + 1, position[1]], position]
        for position in nearby_positons:
            if self.position_is_in_danger(position):
                return True
        return False

    def can_agent_escape_bomb(self):
        """Find out whether or not the agent can escape the explosion of the closest bomb.

        Returns:
            bool: True if can survive, False if can't survive.
        """
        return self.agent_can_escape_bomb

    def to_hashed_features(self):
        """Hash this game state.

        Returns:
            int: Hash of game state.
        """
        if self.dead:
            return 0
        hash_list = list()
        features = self.to_features()
        for feature in features:
            hash_list.append(
                int(hashlib.md5(str(feature).encode()).hexdigest(), 16))
        final_hash = hash_list[0]
        for a_hash in hash_list[1:]:
            final_hash ^= a_hash
        return final_hash

    def get_potential(self):
        """Calculate the potential of this game state.

        Returns:
            float: Potential of game state.
        """
        if self.dead:
            return -30
        closest_coin_distance = self.get_closest_coin_distance()
        if closest_coin_distance == 0:
            add_one = 1
        else:
            add_one = 0
        if self.can_agent_escape_bomb():
            if self.position_is_in_danger(self.agent_position):
                danger_penalty = 2
            else:
                danger_penalty = 0
            return 20 * (self.agent_score + add_one) - self.get_closest_coin_distance(k=add_one) - danger_penalty
        else:
            return -30

    def get_closest_coin_distance(self, k=0):
        """Get the distnce to the kth nearest coin to the agent.

        Args:
            k (int, optional): kth nearest coin to calculate distance from. Defaults to 0.

        Returns:
            float: Distance from agent to closest coin.
        """
        if len(self.coins) > k:
            return sorted([self.get_rockless_distance(coin, self.agent_position) for coin in self.coins])[k]
        else:
            return 0
