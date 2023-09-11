import numpy as np
import os
import ujson
from collections import defaultdict
from .state import GameState


def setup(self):
    np.random.seed()
    self.logger.info("Agent setup")
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self.DIMENSIONS_MAP = (17,17)

    self.TRAINING_DATA_DIRECTORY = "./training_data/"

    if self.train or not os.path.isfile(self.TRAINING_DATA_DIRECTORY + "q.json"):
        self.logger.info("Setting up model from scratch.")
        random_value = np.random.rand(1)[0]
        self.Q_TABLE = defaultdict(lambda: random_value)
        self.Q_TABLE = np.random.rand(self.DIMENSIONS_MAP[0],self.DIMENSIONS_MAP[1],len(self.ACTIONS))
        self.Q_TABLE = 2 * self.Q_TABLE - 1

        q_table_as_list = self.Q_TABLE.tolist()
        print(q_table_as_list)

        if not os.path.isfile(self.TRAINING_DATA_DIRECTORY + "q.json"):
            # If the file does not exist, create the directory
            os.makedirs(os.path.dirname(self.TRAINING_DATA_DIRECTORY), exist_ok=True)

            # Create the file and write data to it
            with open(self.TRAINING_DATA_DIRECTORY + "q.json", "w", encoding="utf-8") as f:
                f.write(ujson.dumps(q_table_as_list))

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
    self.logger.info("Random Q Table model Act.")
    if not self.train and np.random.random(1) > self.EPSILON:
        return np.random.choice(self.ACTIONS,p=[0.2,0.2,0.2,0.2,0.05,0.15])
    elif (self.train and np.random.random(1) > self.EPSILON) or agent_position == None:
        return np.random.choice(self.ACTIONS,p=[0.2,0.2,0.2,0.2,0.05,0.15])
    else:
        with open("./training_data/" + "q.json", "r", encoding="utf-8") as f:
            try:
                q_table_as_list = ujson.load(f)
            except ujson.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                q_table_as_list = None
        self.Q_TABLE = np.array(q_table_as_list)
        slice = self.Q_TABLE[agent_position[0], agent_position[1], :]
        action_id = np.argmax(slice)
        subfield=get_9x9_submatrix(game_state.get_field(),agent_position)
        #print(game_state.get_field(),game_state.get_coins_position())
        test = state_to_features(game_state)
        print(test)
        #print(agent_position)
        return self.ACTIONS[action_id]
        #return "BOMB"
    
def state_to_features(game_state) -> defaultdict:
    coins = game_state.get_coins_position()

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

    #print(close_coin,coins_available)

    #Calculate closest coin
    path_to_closest_coin = breitenSuche_Coin(count=0,field=field,coins=coins,coordinates=[((agent_position[0],agent_position[1],["i"]))])
    game_state_dict = defaultdict(int)
    game_state_dict["subfield"] = subfield
    game_state_dict["coins_available"] = coins_available
    game_state_dict["close_coin"] = close_coin
    game_state_dict["path_to_closest_coin"] = path_to_closest_coin
    game_state_dict["impossible_moves"] = impossible_moves(field,agent_position)
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
