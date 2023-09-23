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
            # print(self.bombs)
            # print(self.explosion_map)
            # print(self.prec_explosion_map)
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
            #Calculate subfield around agent
            self.subfield = self.cropSevenTiles()[0]
            #Holds position of other agent(s) in subfield in case they are reachable, else []
            self.other_agent_in_subfield = self.cropSevenTiles()[1]

    def get_possible_moves(self):
        """Get a list of all possible moves in the current game state.

        Returns:
            list: List of all possible moves.
        """
        x = self.agent_position[0]
        y = self.agent_position[1]
        other_agent_positions = [agent[3] for agent in self.other_agents]
        possible_moves = ["WAIT"]
        # new_bombs = [([bomb[0][0], bomb[0][1]], bomb[1] - 1) for bomb in self.bombs]
        # if self.is_agent_close_to_danger and not self.is_agent_in_danger or np.max(self.explosion_map) == 1 and self.explosion_map[x][y] == 0:
        #     return ["WAIT"]
        if self.agent_bombs_left: # and self.would_be_survivable and len(self.coins) == 0:
            possible_moves.append("BOMB")
        # if self.is_position_survivable([x, y], new_bombs, hypothetical=True):
        #    possible_moves.append("WAIT")
        if self.field[x + 1][y] == 0 and not (x + 1, y) in other_agent_positions: # and self.explosion_map[x + 1][y] == 0 and self.is_position_survivable([x + 1, y], new_bombs, hypothetical=True):
            possible_moves.append("RIGHT")
        if self.field[x - 1][y] == 0 and not (x - 1, y) in other_agent_positions: # and self.explosion_map[x - 1][y] == 0 and self.is_position_survivable([x - 1, y], new_bombs, hypothetical=True):
            possible_moves.append("LEFT")
        if self.field[x][y + 1] == 0 and not (x, y + 1) in other_agent_positions: # and self.explosion_map[x][y + 1] == 0 and self.is_position_survivable([x, y + 1], new_bombs, hypothetical=True):
            possible_moves.append("DOWN")
        if self.field[x][y - 1] == 0 and not (x, y - 1) in other_agent_positions: # and self.explosion_map[x][y - 1] == 0 and self.is_position_survivable([x, y - 1], new_bombs, hypothetical=True):
            possible_moves.append("UP")

        # if len(possible_moves) == 0:
        #     return ["WAIT"] 
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
        # + 0.01 * (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))
        return dx + dy

    def to_features_subfield(self):
        """Convert game state to more generic feature vector.

        Returns:
            tuple: List of features.
        """
        # field: exploit symmetry, only consider boxes nearby, (cut off outer ring/only boxes)
        # other_agents: position and bombs_left, exploit_symmetry
        # bombs: only nearby, position and timer, exploit symmetry
        # explosion_map: exploit symmetry
        # path to safety feature

        feature_field = self.field
        feature_subfield = self.subfield
        feature_other_agent_in_subfield = self.other_agent_in_subfield
        feature_explosion_map = self.explosion_map
        feature_prec_explosion_map = self.prec_explosion_map
        feature_agent_position = np.array(
            [self.agent_position[0], self.agent_position[1]])
        feature_bombs = [[np.array(self.adjust_position(
            [bomb[0][0], bomb[0][1]])), bomb[1]] for bomb in self.bombs]
        nearest_bombs = sorted(feature_bombs, key=lambda x: self.get_rockless_distance(
            x[0], feature_agent_position))

        # rotate field
        # vertical axis
        if self.agent_position[0] > 8:
            self.flipped_vertical = True
            feature_field = np.flipud(feature_field)
            feature_explosion_map = np.flipud(feature_explosion_map)
            feature_prec_explosion_map = np.flipud(feature_prec_explosion_map)
            feature_agent_position[0] = 16 - feature_agent_position[0]
            if feature_other_agent_in_subfield != []:
                feature_other_agent_in_subfield[0] = 6 - feature_other_agent_in_subfield[0]
            feature_subfield = np.flipud(feature_subfield)
        # horizontal axis
        if self.agent_position[1] > 8:
            self.flipped_horizontal = True
            feature_field = np.fliplr(feature_field)
            feature_explosion_map = np.fliplr(feature_explosion_map)
            feature_prec_explosion_map = np.fliplr(feature_prec_explosion_map)
            feature_agent_position[1] = 16 - feature_agent_position[1]
            if feature_other_agent_in_subfield != []:
                feature_other_agent_in_subfield[1] = 6 - feature_other_agent_in_subfield[1]
            feature_subfield = np.fliplr(feature_subfield)
        # backslash axis
        # if feature_agent_position[0] < feature_agent_position[1]:
        #     self.flipped_backslash = True
        #     feature_field = feature_field.T
        #     feature_explosion_map = feature_explosion_map.T
        #     # mirror agent position
        #     feature_agent_position[0], feature_agent_position[1] = feature_agent_position[1], feature_agent_position[0]

        # if the game state is not surviveable, the features do not matter
        if len(nearest_bombs) > 0:
            nearest_bomb = nearest_bombs[0]
        else:
            nearest_bomb = -1
        if self.dead or not self.can_agent_survive():
            return -1
        feature_dict = {}

        closest_agent = ()
        distance_to_enemy = 34 #Maximum number
        for agent in self.other_agents:
            if self.get_rockless_distance(self.agent_position,agent[3]) < distance_to_enemy:
                closest_agent = agent
                distance_to_enemy = self.get_rockless_distance(self.agent_position,agent[3])
        feature_dict["distance_to_next_enemy"] = distance_to_enemy

        #Encourage to walk more to the centre
        steps_to_wall = [self.calculate_step_to_next_wall(self.agent_position)]
        actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        actions_away_from_wall = [action for action in actions if action not in steps_to_wall]
        feature_dict["actions_away_from_wall"] = actions_away_from_wall

        if nearest_bomb == -1:
            agent_to_nearest_bomb = -1
            nearest_bomb_timer = -1
        else:
            agent_to_nearest_bomb = nearest_bomb[0] - \
                np.array(feature_agent_position)
            nearest_bomb_timer = nearest_bomb[1]
        agent_can_move_left = feature_field[feature_agent_position[0] -
                                            1][feature_agent_position[1]] == 0
        agent_can_move_right = feature_field[feature_agent_position[0] +
                                             1][feature_agent_position[1]] == 0
        agent_can_move_up = feature_field[feature_agent_position[0]
                                          ][feature_agent_position[1] - 1] == 0
        agent_can_move_down = feature_field[feature_agent_position[0]
                                            ][feature_agent_position[1] + 1] == 0
        local_prec_explosion_map = np.array([[0, feature_prec_explosion_map[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                                        [feature_prec_explosion_map[feature_agent_position[0] - 1][feature_agent_position[1]], 0,
                                         feature_prec_explosion_map[feature_agent_position[0] + 1][feature_agent_position[1]]],
                                        [0, feature_prec_explosion_map[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])
        local_map_1 = np.array([[0, feature_field[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                              [feature_field[feature_agent_position[0] - 1][feature_agent_position[1]], 0,
                               feature_field[feature_agent_position[0] + 1][feature_agent_position[1]]],
                              [0, feature_field[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])


        # print(self.field)
        feature_dict["local_map"] = local_map_1
        # feature_dict["can_move_up"] = agent_can_move_up
        # feature_dict["can_move_down"] = agent_can_move_down
        # feature_dict["can_move_left"] = agent_can_move_left
        # feature_dict["can_move_right"] = agent_can_move_right

        # if the agent can place a bomb, pass whether the bomb would be surviveable or not
        if self.agent_bombs_left:
            feature_dict["would_survive_bomb"] = self.would_be_survivable

        # if there are active explosions around the agent, pass the local explosion map
        if np.max(local_prec_explosion_map) != 0:
            # print("Local explosion map:")
            # print(local_prec_explosion_map)
            feature_dict["local_prec_explosion_map"] = local_prec_explosion_map

        # if the agent is in or close to danger, pass a vector to the nearest safe square
        if self.is_agent_close_to_danger or self.is_agent_in_danger:
            # print("IN DANGER")
            # print(self.can_agent_survive())
            # feature_dict["agent_to_nearest_bomb"] = agent_to_nearest_bomb
            # print("Danger")
            # print(nearest_bomb)
            # print(feature_agent_position)
            # feature_dict["nearest_bomb_timer"] = nearest_bomb_timer
            # if self.is_position_in_danger(self.agent_position):
            nearest_safe_square = self.get_closest_safe_square()
            if nearest_safe_square == -1:
                feature_dict["agent_to_nearest_safe_square"] = -1
            else:
                agent_to_nearest_safe_square = np.array(self.adjust_position(nearest_safe_square) - np.array(feature_agent_position))
                feature_dict["agent_to_nearest_safe_square"] = agent_to_nearest_safe_square

        if feature_other_agent_in_subfield != []:
            feature_dict["other_agent_in_subfield"] = True
            min_distance = float("inf")
            closest_agent = None
            for agent in feature_other_agent_in_subfield:
                distance = self.get_rockless_distance(agent,feature_agent_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_agent = agent
            feature_dict["closest_agent"] = [closest_agent,distance]
            affected_by_own_bomb = self.get_bomb_explosion_squares(feature_agent_position)
            if closest_agent != None:
                if closest_agent in affected_by_own_bomb and self.is_position_survivable(self.get_agent_position(),[self.get_agent_position()])==True:
                    feature_dict["closest_agent_is_in_danger"] = True
                    if self.is_position_survivable(closest_agent,[feature_agent_position])==False:
                        feature_dict["closest_agent_cant_survive"] = True
                else:
                    feature_dict["closest_agent_is_in_danger"] = False
                    feature_dict["closest_agent_cant_survive"] = False
        else:
            feature_dict["other_agent_in_subfield"] = False
            feature_dict["closest_agent_is_in_danger"] = False
            feature_dict["closest_agent_cant_survive"] = False
        

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

    def get_potential_subfield(self, print_components=False):
        """Calculate the potential of this game state.

        Returns:
            float: Potential of game state.
        """
        game_state_feature = self.to_features_subfield()
        if self.dead:
            #print("State: Agent dead")
            return -1000000
        
        if self.can_agent_survive():
            #print("State: Agent acts")
            # discourage going near bombs
            if self.is_agent_in_danger:
                danger_penalty = 1
            else:
                danger_penalty = 0
            # encourage blowing up crates
            if game_state_feature["other_agent_in_subfield"] == True:
                closest_agent,distance_to_enemy = game_state_feature["closest_agent"]
            else:
                closest_agent = ()
                distance_to_enemy = 34 #Maximum number
                for agent in self.other_agents:
                    if self.get_rockless_distance(self.agent_position,agent[3]) < distance_to_enemy:
                        closest_agent = agent
                        distance_to_enemy = self.get_rockless_distance(self.agent_position,agent[3])

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

            #Encourage placing bombs near to opponents
            if game_state_feature["closest_agent_is_in_danger"] == True:
                bomb_near_opponent = 1
            else:
                bomb_near_opponent = 0

            #Encourage brining opponenet in unsurvivable positions:
            if game_state_feature["closest_agent_cant_survive"] == True:
                opponent_cant_survive = 1
            else:
                opponent_cant_survive = 0

            if print_components:
                print(f"Agent score: {self.agent_score}")
                #print(f"Closest coin distance: {self.get_closest_coin_distance()}")
                print(f"Closest bomb distance: {self.get_closest_bomb_distance()}")
                print(f"Danger penalty: {danger_penalty}")
                print(f"Distance to enemy: {distance_to_enemy}")
                print(f"Bomb near opponent: {bomb_near_opponent}")
                print(f"Oponenet cant survive: {opponent_cant_survive}")
            #return 30 * (self.agent_score) + self.get_closest_bomb_distance()-15*distance_to_enemy + 3*bomb_near_opponent +4*opponent_cant_survive - 10 * danger_penalty
            return 30 * (self.agent_score) + self.get_closest_bomb_distance()-5*distance_to_enemy + 1*bomb_near_opponent +3*opponent_cant_survive
        else:
            #print("State: Cant surivive")
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
                "explosion_map": self.explosion_map,
                "subfield": self.subfield,
                "other_agent_in_subfield": self.other_agent_in_subfield
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
    
    def cropSevenTiles(self):
        '''
        This function crops the 17x17 field into a 7x7 with the player centered , i.e. the surrounding matrix.
        This will preprocessed further before used as a state in Q-Learning.
        '''
        x,y = self.get_agent_position()
        field_prepared = self.get_field()
        #Calculate positions on the field affected by bombs
        bomb_position = []
        for (bomb,_) in self.get_bombs_position():
            bomb_position.append(self.get_bomb_explosion_squares(bomb))
        flattened_list = [item for sublist in bomb_position for item in sublist]
        bomb_position_array = np.array(flattened_list)
        bomb_position = np.unique(bomb_position_array,axis=0)

        #Positions affected by bombs marked with -2
        for coordinate in bomb_position:
            x = coordinate[0]
            y = coordinate[1]
            field_prepared[x][y] = -2

        #Positions with coins marked with 4
        coins = self.get_coins()
        for coin in coins:
            x = coin[0]
            y = coin[1]
            field_prepared[x][y] = 4

        #Mark enemy agent positions with 3
        other_agents = self.get_other_agents_position()
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
        croped_array,other_agent = self.calculate_accessible_parts(croped_array)

        return croped_array,other_agent




    def explore_field(self,field, x, y, explored,other_agent):
        if x < 0 or y < 0 or x >= field.shape[0] or y >= field.shape[1] or field[x, y] in [-1, 1, 3]:
            return
        
        if (x, y) not in explored:
            explored.append((x, y))
            if field[x][y] == 3:
                other_agent.append((x,y))
        
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                self.explore_field(field, x + dx, y + dy, explored,other_agent)

    def calculate_accessible_parts(self,field):
        # Copy the field to avoid modifying the original array
        result_field = field.copy()
        
        # List to keep track of explored fields and if other agent was found
        explored = []
        other_agent = []
        
        # Perform DFS to explore accessible parts
        self.explore_field(result_field, 3, 3, explored,other_agent)  # Assuming the starting position is (3, 3)
        
        # Mark unexplored fields as -1
        for i in range(result_field.shape[0]):
            for j in range(result_field.shape[1]):
                if (i, j) not in explored and field[i][j]!=1 and field[i][j]!=3:
                    result_field[i, j] = -1
        
        return result_field,other_agent
    
    def to_features(self):
        """Convert game state to more generic feature vector.

        Returns:
            tuple: List of features.
        """
        # field: exploit symmetry, only consider boxes nearby, (cut off outer ring/only boxes)
        # other_agents: position and bombs_left, exploit_symmetry
        # bombs: only nearby, position and timer, exploit symmetry
        # explosion_map: exploit symmetry
        # path to safety feature

        feature_field = self.field
        feature_explosion_map = self.explosion_map
        feature_prec_explosion_map = self.prec_explosion_map
        feature_agent_position = np.array(
            [self.agent_position[0], self.agent_position[1]])

        # rotate field
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
        agent_can_move_left = feature_field[feature_agent_position[0] -
                                            1][feature_agent_position[1]] == 0
        agent_can_move_right = feature_field[feature_agent_position[0] +
                                             1][feature_agent_position[1]] == 0
        agent_can_move_up = feature_field[feature_agent_position[0]
                                          ][feature_agent_position[1] - 1] == 0
        agent_can_move_down = feature_field[feature_agent_position[0]
                                            ][feature_agent_position[1] + 1] == 0
        local_prec_explosion_map = np.array([[0, feature_prec_explosion_map[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                                        [feature_prec_explosion_map[feature_agent_position[0] - 1][feature_agent_position[1]], 0,
                                         feature_prec_explosion_map[feature_agent_position[0] + 1][feature_agent_position[1]]],
                                        [0, feature_prec_explosion_map[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])
        local_map_1 = np.array([[0, feature_field[feature_agent_position[0]][feature_agent_position[1] - 1], 0],
                              [feature_field[feature_agent_position[0] - 1][feature_agent_position[1]], 0,
                               feature_field[feature_agent_position[0] + 1][feature_agent_position[1]]],
                              [0, feature_field[feature_agent_position[0]][feature_agent_position[1] + 1], 0]])

        # if the game state is not surviveable, the features do not matter
        if self.dead or not self.can_agent_survive():
            return -1
        feature_dict = {}

        # print(self.field)
        feature_dict["local_map"] = local_map_1
        # feature_dict["can_move_up"] = agent_can_move_up
        # feature_dict["can_move_down"] = agent_can_move_down
        # feature_dict["can_move_left"] = agent_can_move_left
        # feature_dict["can_move_right"] = agent_can_move_right

        # if the agent can place a bomb, pass whether the bomb would be surviveable or not
        if self.agent_bombs_left:
            feature_dict["would_survive_bomb"] = self.would_be_survivable

        # if there are active explosions around the agent, pass the local explosion map
        if np.max(local_prec_explosion_map) != 0:
            # print("Local explosion map:")
            # print(local_prec_explosion_map)
            feature_dict["local_prec_explosion_map"] = local_prec_explosion_map

        # if the agent is in or close to danger, pass a vector to the nearest safe square
        if self.is_agent_close_to_danger or self.is_agent_in_danger:
            # print("IN DANGER")
            # print(self.can_agent_survive())
            # feature_dict["agent_to_nearest_bomb"] = agent_to_nearest_bomb
            # print("Danger")
            # print(nearest_bomb)
            # print(feature_agent_position)
            # feature_dict["nearest_bomb_timer"] = nearest_bomb_timer
            # if self.is_position_in_danger(self.agent_position):
            nearest_safe_square = self.get_closest_safe_square()
            if nearest_safe_square == -1:
                feature_dict["agent_to_nearest_safe_square"] = -1
            else:
                agent_to_nearest_safe_square = np.array(self.adjust_position(nearest_safe_square) - np.array(feature_agent_position))
                feature_dict["agent_to_nearest_safe_square"] = agent_to_nearest_safe_square
            # print(agent_to_nearest_safe_square)
            # print(feature_bombs)
            # print(feature_agent_position)
            # print(agent_to_nearest_safe_square)
            # print(local_map)

        # if the agent is not in or close to danger and has no explosions around
        elif np.max(local_prec_explosion_map) == 0:
            # print("SAFE")

            # if there are coins on the map, pass a vector to the nearest coin
            if len(self.coins) > 0:
                feature_dict["agent_to_nearest_coin"] = agent_to_nearest_coin

            # if there are not coins on the map, pass the local map
            # else:
            #     feature_dict["local_map"] = local_map
                # if self.agent_bombs_left:
                #     feature_dict["would_bomb_be_surviveable"] = self.would_be_survivable
                # if len(self.bombs) > 0:
                #     feature_dict["agent_to_nearest_bomb"] = agent_to_nearest_bomb
        # print(feature_list)
        return feature_dict
    
    def get_potential(self, print_components=False):
        """Calculate the potential of this game state.
        Returns:
            float: Potential of game state.
        """
        if self.dead:
            #print("Dead")
            return -1000000
        closest_coin_distance = self.get_closest_coin_distance()
        if closest_coin_distance == 0:
            add_one = 1
        else:
            add_one = 0
        if self.can_agent_survive():
            game_state_feature = self.to_features_subfield()
            #print("Survive")
            # discourage going near bombs
            if self.is_agent_in_danger:
                danger_penalty = 1
            else:
                danger_penalty = 0

            # encourage blowing up crates
            if game_state_feature["other_agent_in_subfield"] == True:
                closest_agent,distance_to_enemy = game_state_feature["closest_agent"]
                agent_in_subfield = 1
            else:
                agent_in_subfield = 0
                closest_agent = ()
                distance_to_enemy = 34 #Maximum number
                for agent in self.other_agents:
                    if self.get_rockless_distance(self.agent_position,agent[3]) < distance_to_enemy:
                        closest_agent = agent
                        distance_to_enemy = self.get_rockless_distance(self.agent_position,agent[3])
            
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

            #Encourage placing bombs near to opponents
            if game_state_feature["closest_agent_is_in_danger"] == True:
                bomb_near_opponent = 1
            else:
                bomb_near_opponent = 0

            #Encourage brining opponenet in unsurvivable positions:
            if game_state_feature["closest_agent_cant_survive"] == True:
                opponent_cant_survive = 1
            else:
                opponent_cant_survive = 0

            if print_components:
                print(f"Agent score: {self.agent_score}")
                print(f"Closest coin distance: {self.get_closest_coin_distance()}")
                print(f"Closest bomb distance: {self.get_closest_bomb_distance()}")
                print(f"Danger penalty: {danger_penalty}")
                print(f"Number of crates: {n_crates}")
                print(f"Crate potential: {crate_potential}")
            return 100 * (self.agent_score + add_one) - self.get_closest_coin_distance(k=add_one) + self.get_closest_bomb_distance() - 5 * danger_penalty - 9 * n_crates + 6 * crate_potential-1*distance_to_enemy + 10*bomb_near_opponent +25*opponent_cant_survive+15*agent_in_subfield
        else:
            #print("Cant surivive")
            return -1000000
        
    def calculate_step_to_next_wall(self,agent_position):

        if agent_position[0] < 16-agent_position[0]:
            direction_in_x = "LEFT"
            distance_in_x = agent_position[0]
        else:
            direction_in_x = "RIGHT"
            distance_in_x = 16-agent_position[0]
    
        if agent_position[1] < 16-agent_position[1]:
            direction_in_y = "UP"
            distance_in_y = agent_position[1]
        else:
            direction_in_y = "DOWN"
            distance_in_y = 16-agent_position[1]

        if distance_in_x < distance_in_y:
            return direction_in_x
        else:
            return direction_in_y
