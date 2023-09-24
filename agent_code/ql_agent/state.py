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
            self.prec_explosion_map = [list(x) for x in self.explosion_map]
            # precalculate explosion from bombs about to explode
            for bomb in self.bombs:
                if bomb[1] == 0:
                    for tile in self.get_bomb_explosion_squares(bomb[0]):
                        self.prec_explosion_map[tile[0]][tile[1]] = 1
            # can the agent survive with perfect bomb escape?
            self.is_surviveable = self.is_position_survivable(
                self.agent_position, self.bombs)
            # could the agent survive if it would place a bomb rn?
            self.would_be_survivable =  self.is_position_survivable(self.agent_position, [
                ((self.agent_position[0], self.agent_position[1]), 3)], hypothetical=True)
            # will the agent die if it does not move?
            self.is_agent_in_danger = self.is_position_in_danger(self.agent_position)
            # can the agent die in the next time step by making a move in the wrong direction?
            self.is_agent_close_to_danger = self.is_position_close_to_danger(self.agent_position)

    def get_possible_moves(self):
        """Get a list of all possible moves in the current game state.

        Returns:
            list: List of all possible moves.
        """
        x = self.agent_position[0]
        y = self.agent_position[1]
        other_agent_positions = [agent[3] for agent in self.other_agents]
        possible_moves = ["WAIT"]
        if self.agent_bombs_left:
            possible_moves.append("BOMB")
        if self.field[x + 1][y] == 0 and not (x + 1, y) in other_agent_positions:
            possible_moves.append("RIGHT")
        if self.field[x - 1][y] == 0 and not (x - 1, y) in other_agent_positions:
            possible_moves.append("LEFT")
        if self.field[x][y + 1] == 0 and not (x, y + 1) in other_agent_positions:
            possible_moves.append("DOWN")
        if self.field[x][y - 1] == 0 and not (x, y - 1) in other_agent_positions:
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
            position[0], position[1] = position[1], position[0]
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
        return dx + dy

    def to_features(self):
        """Convert game state to more generic feature vector.

        Returns:
            tuple: List of features.
        """
        feature_field = self.field
        feature_explosion_map = self.explosion_map
        feature_prec_explosion_map = self.prec_explosion_map
        feature_agent_position = np.array(
            [self.agent_position[0], self.agent_position[1]])

        ### rotate field
        # vertical axis
        if self.agent_position[0] > 8:
            self.flipped_vertical = True
            feature_field = np.flipud(feature_field)
            feature_explosion_map = np.flipud(feature_explosion_map)
            feature_prec_explosion_map = np.flipud(feature_prec_explosion_map)
            feature_agent_position[0] = 16 - feature_agent_position[0]
        # horizontal axis
        if self.agent_position[1] > 8:
            self.flipped_horizontal = True
            feature_field = np.fliplr(feature_field)
            feature_explosion_map = np.fliplr(feature_explosion_map)
            feature_prec_explosion_map = np.fliplr(feature_prec_explosion_map)
            feature_agent_position[1] = 16 - feature_agent_position[1]
        # backslash axis
        # if feature_agent_position[0] < feature_agent_position[1]:
        #     self.flipped_backslash = True
        #     feature_field = feature_field.T
        #     feature_explosion_map = feature_explosion_map.T
        #     # mirror agent position
        #     feature_agent_position[0], feature_agent_position[1] = feature_agent_position[1], feature_agent_position[0]

        # rotate coins and filter out closest one
        feature_coins = [np.array(self.adjust_position(
            [coin[0], coin[1]])) for coin in self.coins]
        nearest_coins = sorted(feature_coins, key=lambda x: self.get_rockless_distance(
            x, feature_agent_position))
        if len(nearest_coins) > 0:
            if nearest_coins[0][0] == feature_agent_position[0] and nearest_coins[0][1] == feature_agent_position[1] and len(nearest_coins) > 1:
                nearest_coin = nearest_coins[1]
            else:
                nearest_coin = nearest_coins[0]
        else:
            nearest_coin = np.array([0, 0])
        agent_to_nearest_coin = np.array(
            nearest_coin) - np.array(feature_agent_position)

        # crop explosion map to local env
        local_prec_explosion_map = np.array([[0, feature_prec_explosion_map[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                                        [feature_prec_explosion_map[feature_agent_position[0] - 1][feature_agent_position[1]], 0,
                                         feature_prec_explosion_map[feature_agent_position[0] + 1][feature_agent_position[1]]],
                                        [0, feature_prec_explosion_map[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])
        # crop field to local env
        local_map_1 = np.array([[0, feature_field[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                              [feature_field[feature_agent_position[0] - 1][feature_agent_position[1]], 0,
                               feature_field[feature_agent_position[0] + 1][feature_agent_position[1]]],
                              [0, feature_field[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])

        # if the game state is not surviveable, the features do not matter
        if self.dead or not self.can_agent_survive():
            return -1
        feature_dict = {}

        # always tell the agent his surroundings
        feature_dict["local_map"] = local_map_1

        # if the agent can place a bomb, pass whether the bomb would be surviveable or not
        if self.agent_bombs_left:
            feature_dict["would_survive_bomb"] = self.would_be_survivable

        # if there are active explosions around the agent, pass the local explosion map
        if np.max(local_prec_explosion_map) != 0:
            feature_dict["local_prec_explosion_map"] = local_prec_explosion_map

        # if the agent is in or close to danger, pass a vector to the nearest safe square
        if self.is_agent_close_to_danger or self.is_agent_in_danger:
            nearest_safe_square = self.get_closest_safe_square()
            if nearest_safe_square == -1:
                feature_dict["agent_to_nearest_safe_square"] = -1
            else:
                agent_to_nearest_safe_square = np.array(self.adjust_position(nearest_safe_square) - np.array(feature_agent_position))
                feature_dict["agent_to_nearest_safe_square"] = agent_to_nearest_safe_square

        # if the agent is not in or close to danger and has no explosions around
        elif np.max(local_prec_explosion_map) == 0:
            # if there are coins on the map, pass a vector to the nearest coin
            if len(self.coins) > 0:
                feature_dict["agent_to_nearest_coin"] = agent_to_nearest_coin

        return feature_dict

    def get_closest_safe_square(self):
        """Find the closest reachable square that is outside the radius of all bombs.

        Returns:
            np.array: Coordinates of the closest safe square. -1 if none found.
        """
        n_bombs = len(self.bombs)
        if n_bombs == 0:
            return [0, 0]
        safe_reachable_per_bomb = []
        for bomb in self.bombs:
            safe_reachable_squares = self.get_safe_reachable_squares(self.agent_position, bomb)
            safe_reachable_squares = [square for square in safe_reachable_squares if self.prec_explosion_map[square[0]][square[1]] == 0]
            safe_reachable_per_bomb.append(safe_reachable_squares)
        safe_from_all_bombs = set(safe_reachable_per_bomb[0])
        for i in range(1, n_bombs):
            safe_from_all_bombs &= set(safe_reachable_per_bomb[i])
        safe_from_all_bombs = list(safe_from_all_bombs)
        if len(safe_from_all_bombs) == 0:
            return -1
        return sorted([[pos[0], pos[1]] for pos in safe_from_all_bombs], key=lambda x: self.get_rockless_distance([x[0], x[1]], self.agent_position))[0]

    def can_agent_survive(self):
        """Check if the current game state is surviveable.

        Returns:
            bool: Surviveability.
        """
        if self.dead:
            return False
        return self.is_surviveable

    def is_position_survivable(self, agent_position, bombs, hypothetical=False):
        """Check if the given position is survivable.

        Args:
            agent_position (list or np.array): Position of agent.
            bombs (list or np.array): List of bombs.

        Returns:
            bool: Whether or not position is survivable.
        """
        if self.explosion_map[agent_position[0]][agent_position[1]] == 1:
            return False
        if self.is_position_in_danger(agent_position) or hypothetical:
            for bomb in bombs:
                if not self.is_bomb_escapable(agent_position, bomb):
                    return False

        return True

    def is_bomb_escapable(self, agent_position, bomb):
        """Check if the given bomb is escapeable. Taking into account crates and rock.

        Args:
            agent_position (list or np.array): Current position of agent.
            bomb (list): Bomb to consider.

        Returns:
            bool: Whether or not bomb is escapeable.
        """
        if self.is_bomb_danger_to_position(agent_position, bomb):
            return len(self.get_safe_reachable_squares(agent_position, bomb)) > 0
        return True

    def get_safe_reachable_squares(self, agent_position, bomb):
        """Find all squares reachable in the bomb timer time from the given position that lay outside
        the explosion of the given bomb.

        Args:
            agent_position (list or np.array): Position of the agent.
            bomb (bomb tuple): Position and timer of the given bomb.

        Returns:
            list: List of reachable squares.
        """
        reachable_locations = {(agent_position[0], agent_position[1])}
        for i in range(bomb[1] + 1):
            reachable_next_step = set()
            for location in reachable_locations:
                possibly_reachable = [[location[0], location[1] + 1], [location[0], location[1] - 1], [
                    location[0] - 1, location[1]], [location[0] + 1, location[1]]]
                reachable_next_step |= {
                    (l[0], l[1]) for l in possibly_reachable if self.field[l[0]][l[1]] == 0}
            reachable_locations |= reachable_next_step
        return [pos for pos in reachable_locations if [pos[0], pos[1]] not in self.get_bomb_explosion_squares(bomb[0])]

    def get_bomb_explosion_squares(self, bomb_position):
        """Get a list of all position the bomb explosion is going to be on.

        Args:
            bomb_position (list or np.array): Position of bomb.

        Returns:
            list: List of affected position.
        """
        explosion = list()
        # bomb on open file
        if bomb_position[1] % 2 == 1:
            explosion += [[x, bomb_position[1]]
                          for x in range(max(1, bomb_position[0] - 3), min(15, bomb_position[0] + 3) + 1)]
        # bomb on open rank
        if bomb_position[0] % 2 == 1:
            explosion += [[bomb_position[0], y]
                          for y in range(max(1, bomb_position[1] - 3), min(15, bomb_position[1] + 3) + 1)]
        return explosion

    def is_bomb_danger_to_position(self, position, bomb):
        """Check if the given bomb is a danger to the agent. So check if agent
        is in range of bomb explosion.

        Args:
            position (list or np.array): Possible agent position.
            bomb (list or np.array): Bomb to check.

        Returns:
            bool: Whether or not agent would die if bomb would explode as is.
        """
        # agent on same square as bomb
        if bomb[0][0] == position[0] and bomb[0][1] == position[1]:
            return True
        # agent on same x as bomb
        elif bomb[0][0] == position[0] and position[0] % 2 == 1:
            if abs(bomb[0][0] - position[0]) <= 3:
                return True
        # agent on same y as bomb
        elif bomb[0][1] == position[1] and position[1] % 2 == 1:
            if abs(bomb[0][1] - position[1]) <= 3:
                return True

    def is_position_in_danger(self, position):
        """Check if the given position is inside the range of a bomb explosion.

        Args:
            position (list or np.array): Position to check.

        Returns:
            bool: Whether or not position is in bomb range.
        """
        if len(self.bombs) > 0:
            for bomb in self.bombs:
                if self.is_bomb_danger_to_position(position, bomb):
                    return True
        return False

    def is_position_close_to_danger(self, position):
        """This checks if any of the squares around the player is in range of a bomb explosion.

        Args:
            position (list or np.array): Position to check.

        Returns:
            bool: Whether or not position is close to danger.
        """
        nearby_positons = [[position[0], position[1] + 1], [position[0], position[1] - 1],
                           [position[0] - 1, position[1]], [position[0] + 1, position[1]], position]
        for position in nearby_positons:
            if self.is_position_in_danger(position):
                return True
        return False

    def to_hashed_features(self):
        """Hash this game state.

        Returns:
            int: Hash of game state.
        """
        hash_list = list()
        features = self.to_features()
        if features == -1:
            return 0
        for feature in features:
            hash_list.append(
                int(hashlib.md5((feature + str(features[feature])).encode()).hexdigest(), 16))
        final_hash = hash_list[0]
        for a_hash in hash_list[1:]:
            final_hash ^= a_hash
        return final_hash

    def get_potential(self, print_components=False):
        """Calculate the potential of this game state.

        Returns:
            float: Potential of game state.
        """
        if self.dead:
            return -1000000
        closest_coin_distance = self.get_closest_coin_distance()
        if closest_coin_distance == 0:
            add_one = 1
        else:
            add_one = 0
        if self.can_agent_survive():
            # discourage going near bombs
            if self.is_agent_in_danger:
                danger_penalty = 1
            else:
                danger_penalty = 0
            # encourage blowing up crates
            crate_potential = 0
            for bomb in self.bombs:
                affected_tiles = self.get_bomb_explosion_squares(bomb[0])
                crate_potential += len(
                    [pos for pos in affected_tiles if self.field[pos[0]][pos[1]] == 1])
                # potential weights
                # coin distance: 1
                # danger penalty: 5 (coin distance + 1)
                # crate potential: 6 (danger penalty + 1)
                # coin count: 19 (crate potential * 3 + 1)
                # agent score: 30 ((15 + 15) * coin distance)
                # distance to closest bomb
            n_crates = len([tile_value for x in self.field for tile_value in x if tile_value == 1])
            if print_components:
                print(f"Agent score: {self.agent_score}")
                print(f"Closest coin distance: {self.get_closest_coin_distance()}")
                print(f"Closest bomb distance: {self.get_closest_bomb_distance()}")
                print(f"Danger penalty: {danger_penalty}")
                print(f"Number of crates: {n_crates}")
                print(f"Crate potential: {crate_potential}")
            return 30 * (self.agent_score + add_one) - self.get_closest_coin_distance(k=add_one) + self.get_closest_bomb_distance() - 5 * danger_penalty - 9 * n_crates + 6 * crate_potential
        else:
            return -1000000

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
            return 10

    def get_closest_bomb_distance(self, k=0):
        """Get the distnce to the kth nearest bomb to the agent.

        Args:
            k (int, optional): kth nearest bomb to calculate distance from. Defaults to 0.

        Returns:
            float: Distance from agent to closest coin.
        """
        if len(self.bombs) > k:
            return sorted([self.get_rockless_distance(bomb[0], self.agent_position) for bomb in self.bombs])[k]
        else:
            return 4

    def get_all_attributes(self):
            return {
                "dead": self.dead,
                "round": self.round,
                "step": self.step,
                "field": self.field,
                "flipped_vertical": self.flipped_vertical,
                "flipped_horizontal": self.flipped_horizontal,
                "flipped_slash": self.flipped_slash,
                "flipped_backslash": self.flipped_backslash,
                "agent_name": self.agent_name,
                "agent_score": self.agent_score,
                "agent_bombs_left": self.agent_bombs_left,
                "agent_position": self.agent_position,
                "other_agents": self.other_agents,
                "bombs": self.bombs,
                "coins": self.coins,
                "explosion_map": self.explosion_map
            }

    def get_agent_position(self):
        return self.agent_position
    
    def get_field(self):
        return self.field
    
    def get_coins(self):
        return self.coins
    
    def get_bombs_position(self):
        return self.bombs
    
    def get_explosion_map(self):
        return self.explosion_map
    
    def get_other_agents_position(self):
        return self.other_agents