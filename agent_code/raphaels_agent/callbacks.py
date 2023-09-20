import numpy as np
import os
import ujson
from collections import defaultdict
from .state import GameState
from collections import deque


def setup(self):
    np.random.seed()
    self.logger.info("Agent setup")
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self.DIMENSIONS_MAP = (17,17)

    self.TRAINING_DATA_DIRECTORY = "./training_data/"
    print(self.train)

    if self.train or not os.path.isfile(self.TRAINING_DATA_DIRECTORY + "q.json"):
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model from saved state.")
        with open(self.TRAINING_DATA_DIRECTORY + "q.json", "rb") as file:
            self.Q_TABLE = defaultdict(float)
            loaded_dict = ujson.loads(file.read())
            self.Q_TABLE = np.array(loaded_dict)
    self.EPSILON = 0.9



def act(self,game_state : dict):
    game_state = GameState(game_state)
    agent_position = game_state.get_agent_position()
    x=cropSevenTiles(game_state)
    self.logger.info("Random Q Table model Act.")
    """if not self.train and np.random.random(1) > self.EPSILON:
        action = np.random.choice(self.ACTIONS, p=[0.225, 0.225, 0.225, 0.225, 0.05, 0.05])
        print("Random: "+action)
        return action"""
    if self.train and np.random.random(1) > self.EPSILON:
        action = np.random.choice(self.ACTIONS, p=[0.225, 0.225, 0.225, 0.225, 0.05, 0.05])
        print("Random: "+action)
        return action
    else:
        slice = self.Q_TABLE[agent_position[0], agent_position[1], :]
        action_id = np.argmax(slice)
        subfield=get_9x9_submatrix(game_state.get_field(),agent_position)
        #print(game_state.get_field(),game_state.get_coins_position())
        #test = state_to_features(game_state)
        #print(test)
        #print(agent_position)
        action = self.ACTIONS[4]
        if action == "BOMB":
            print("BOMB")
        #print("Not Random: "+action)
        return action
        #return "BOMB"
    
def state_to_features(game_state) -> defaultdict:
    #Load GameState and its important informations about the current state
    field = game_state.get_field()
    agent_position = game_state.get_agent_position()
    subfield=get_9x9_submatrix(game_state.get_field(),agent_position)
    coin_positions = game_state.get_coins_position()

    #Check if coin is subfield around the player
    coins_available = coins_in_subfield(subfield,coin_positions)
    close_coin = False
    if coins_available != []:
         close_coin = True

    
    #Calculate closest coin
    path_to_closest_coin = breitenSuche_Coin(count=0,field=field,coins=coin_positions,coordinates=[((agent_position[0],agent_position[1],["i"]))])
    game_state_dict = defaultdict(int)
    game_state_dict["agent_position"] = agent_position
    game_state_dict["subfield"] = subfield
    game_state_dict["coins_available"] = coins_available
    game_state_dict["close_coin"] = close_coin
    game_state_dict["path_to_closest_coin"] = path_to_closest_coin
    game_state_dict["impossible_moves"] = impossible_moves(field,agent_position)
    game_state_dict["save_moves"] = save_moves(field,agent_position,game_state.get_bombs_position(),game_state_dict["impossible_moves"],['UP', 'RIGHT', 'DOWN', 'LEFT'])
    return game_state_dict


def breitenSuche_Coin(count,field,coins,coordinates): #currentPath = ['LEFT', 'DOWN', 'LEFT']
    #print("call")
    if count > 34 or coins==[]: #Coin is in the opposite corner
        return []
    newCoordinates = []
    for coordinate in coordinates:
        #Terminieren wenn Coin gefunden
        if coordinate[0]-1 >= 0 and any((coordinate[0]-1, coordinate[1]) == c for c in coins):
            return coordinate[2]+["LEFT"]
        elif coordinate[0]+1 < 17 and any((coordinate[0]+1, coordinate[1]) == c for c in coins):
            return coordinate[2]+["RIGHT"]
        elif coordinate[1]-1 >= 0 and any((coordinate[0], coordinate[1]-1) == c for c in coins):
            return coordinate[2]+["UP"]
        elif coordinate[1]+1 < 17 and any((coordinate[0], coordinate[1]+1) == c for c in coins):
            return coordinate[2]+["DOWN"]

        #Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen)
        if coordinate[0] - 1 >= 0:  #Prüfe ob wir den Rand noch nicht erreicht haben
            if field[coordinate[0] - 1][coordinate[1]] != -1 and coordinate[2][-1] != "RIGHT": #Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                newCoordinates.append((coordinate[0] - 1, coordinate[1],coordinate[2]+["LEFT"])) # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
        if coordinate[0] + 1 < 17:
            if field[coordinate[0] + 1][coordinate[1]] != -1 and coordinate[2][-1] != "LEFT":
                newCoordinates.append((coordinate[0] + 1, coordinate[1],coordinate[2]+["RIGHT"]))
        if coordinate[1] - 1 >= 0:
            if field[coordinate[0]][coordinate[1] - 1] != -1 and coordinate[2][-1] != "DOWN":
                newCoordinates.append((coordinate[0], coordinate[1] - 1,coordinate[2]+["UP"]))
        if coordinate[1] + 1 < 17:
            if field[coordinate[0]][coordinate[1] + 1] != -1 and coordinate[2][-1] != "UP":
                newCoordinates.append((coordinate[0], coordinate[1] + 1,coordinate[2]+["DOWN"]))
    if len(newCoordinates)>0:
        return breitenSuche_Coin(count+1,field,coins,newCoordinates)
    else:
        return []


def get_9x9_submatrix(matrix, position):
    x, y = position
    half_size = 4  # Halbe Größe des 9x9 Feldes

    # Begrenze die Koordinaten, um sicherzustellen, dass der Ausschnitt innerhalb der Matrix bleibt
    x = max(half_size, min(matrix.shape[0] - 1 - half_size, x))
    y = max(half_size, min(matrix.shape[1] - 1 - half_size, y))

    # Erzeuge ein Gitter von Koordinaten für die Submatrix
    subgrid = np.indices((9, 9)).T

    # Passe die Koordinaten an, um sie an die Position in der Hauptmatrix anzupassen
    subgrid += [x - half_size, y - half_size]
    submatrix_coordinates = np.array([subgrid[:,:,0], subgrid[:,:,1]])

    rows=submatrix_coordinates[0].shape[0]
    cols=submatrix_coordinates[0].shape[1]

    matrix = np.random.randint(0, 10, size=(rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            matrix[i][j][0]= submatrix_coordinates[0][i][j]
            matrix[i][j][1]= submatrix_coordinates[1][i][j]

    return matrix

def coins_in_subfield(subfield,coin_positions):
    coins_in_subfield = []
    for field in subfield:
        field_tuples = [tuple(item) for item in field]
        common_combinations = set(field_tuples) & set(coin_positions)

        if common_combinations:
             coins_in_subfield.append(common_combinations)
    return coins_in_subfield

def impossible_moves(field,position):
    impossible_moves = []
    if field[position[0]-1,position[1]] == -1:
        impossible_moves.append("LEFT")
    if field[position[0]+1,position[1]] == -1:
        impossible_moves.append("RIGHT")
    if field[position[0],position[1]-1] == -1:
        impossible_moves.append("UP")
    if field[position[0],position[1]+1] == -1:
        impossible_moves.append("DOWN")
    
    return impossible_moves

from collections import deque

def save_moves(field, position, bombs, impossible_moves, actions):
    affected_positions = set()
    possible_moves = [elem for elem in actions if elem not in impossible_moves]
    if bombs == []:
        return possible_moves
    possible_moves_set = set(possible_moves)

    for bomb_position, countdown in bombs:
        x, y = bomb_position
        for i in range(1, 7 + 1):
            if 0 <= x + i < len(field) and 0 <= y < len(field[0]):
                affected_positions.add((x + i, y))  # Right
            if 0 <= x - i < len(field) and 0 <= y < len(field[0]):
                affected_positions.add((x - i, y))  # Left
            if 0 <= x < len(field) and 0 <= y + i < len(field[0]):
                affected_positions.add((x, y + i))  # Down
            if 0 <= x < len(field) and 0 <= y - i < len(field[0]):
                affected_positions.add((x, y - i))  # Up

    affected_bombs = [(bomb_position, countdown) for bomb_position, countdown in bombs if bomb_position in affected_positions]

    if position in affected_positions:
        safe_moves = []
        x, y = position

        if 0 <= x + 1 < len(field) and (x + 1, y) not in affected_positions:
            safe_moves.append('RIGHT')
        if 0 <= x - 1 < len(field) and (x - 1, y) not in affected_positions:
            safe_moves.append('LEFT')
        if 0 <= y + 1 < len(field[0]) and (x, y + 1) not in affected_positions:
            safe_moves.append('DOWN')
        if 0 <= y - 1 < len(field[0]) and (x, y - 1) not in affected_positions:
            safe_moves.append('UP')
        safe_moves_set = set(safe_moves)
        intersection = safe_moves_set.intersection(possible_moves_set)

        safe_moves = list(intersection)
        if safe_moves != []:
            return safe_moves
        else:
            queue = deque([(x, y, 0, [])])
            visited = set()
            if len([countdown for _, countdown in affected_bombs])==0:
                step_limit = 1
            else:
                step_limit = min([countdown for _, countdown in affected_bombs])
            while queue:
                cx, cy, steps, path = queue.popleft()
                if (cx, cy) not in visited and field[cx][cy] != -1:
                    visited.add((cx, cy))
                    if (cx, cy) not in affected_positions:
                        # and steps < min([countdown for _, countdown in affected_bombs])
                        return path

                    if steps < step_limit:
                        for dx, dy, move in [(1, 0, 'RIGHT'), (-1, 0, 'LEFT'), (0, 1, 'DOWN'), (0, -1, 'UP')]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < len(field) and 0 <= ny < len(field[0]):
                                queue.append((nx, ny, steps + 1, path + [move]))

            return ["Position not survivable"]

    return possible_moves

def cropSevenTiles(game_state):
    '''
    This function crops the 17x17 field into a 7x7 with the player centered , i.e. the surrounding matrix.
    This will preprocessed further before used as a state in Q-Learning.
    '''
    x,y = game_state.get_agent_position()
    field = game_state.get_field()

    x = x+2
    y = y+2
    padded_array = np.pad(field, 2, mode='constant', constant_values=-1)
    croped_array = padded_array[x-3:x+4, y-3:y+4]
    croped_array = np.transpose(croped_array)

    #5 decodes agents position
    croped_array[3][3]=5
    print(game_state.get_bombs_position())
    print("############")
    print(game_state.explosion_map())
    print("'''''''''''''''")
    print(game_state.get_coins())
    print("MMMMMMMMMM")

    return padded_array