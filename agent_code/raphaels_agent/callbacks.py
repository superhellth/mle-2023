import numpy as np
from .state import GameState

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIMENSIONS_MAP = (17,17)
EPSILON = 0.9

def setup(self):
    np.random.seed()
    self.Q_TABLE = np.random.rand(DIMENSIONS_MAP[0],DIMENSIONS_MAP[1],len(ACTIONS))
    self.Q_TABLE = 2 * self.Q_TABLE - 1
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self.DIMENSIONS_MAP = (17,17)
    self.EPSILON = 0.9


def act(self,game_state : dict):
    game_state = GameState(game_state)
    agent_position = game_state.get_agent_position()
    self.logger.info("Random Q Table model Act.")
    if np.random.random(1) > self.EPSILON or agent_position == None:
        return np.random.choice(ACTIONS,p=[0.2,0.2,0.2,0.2,0.05,0.15])
    else:
        slice = self.Q_TABLE[agent_position[0], agent_position[1], :]
        action_id = np.argmax(slice)
        subfield=get_9x9_submatrix(game_state.get_field(),agent_position)
        state_to_features(game_state)
        return self.ACTIONS[action_id]
    
def state_to_features(game_state) -> np.array:
    #Load GameState and its important informations about the current state
    field = game_state.get_field()
    agent_position = game_state.get_agent_position()
    print(agent_position)
    subfield=get_9x9_submatrix(game_state.get_field(),agent_position)
    coin_positions = game_state.get_coins_position()

    #Check if coin is subfield around the player
    coins_available = coins_in_subfield(subfield,coin_positions)
    close_coin = False
    if coins_available != []:
         close_coin = True
    print(close_coin,coins_available)
    #Calculate closest coin
    #path_to_closest_coin = breitenSuche_Coin(field,agent_position)
    #print(path_to_closest_coin)


    


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

def breitenSuche_Coin(field,coordinates): #currentPath = ['LEFT', 'DOWN', 'LEFT']
        newCoordinates = []
        for coordinate in coordinates:
            #Terminieren wenn Coin gefunden
            if coordinate[0]-1 >= 0 and field[coordinate[0]-1, coordinate[1]] == 2:
                    return coordinate[2]+["l"]
            elif coordinate[0]+1 < 17 and field[coordinate[0]+1, coordinate[1]] == 2:
                    return coordinate[2]+["r"]
            elif coordinate[1]-1 >= 0 and field[coordinate[0], coordinate[1]-1] == 2:
                    return coordinate[2]+["u"]
            elif coordinate[1]+1 < 17 and field[coordinate[0], coordinate[1]+1] == 2:
                    return coordinate[2]+["d"]
            #Wir wollen für jeden Knoten/Koordinate den Nachfolger durchsuchen (also die nächste Ebene durchkämmen)
            if coordinate[0] - 1 >= 0:  #Prüfe ob wir den Rand noch nicht erreicht haben
                if field[coordinate[0] - 1][coordinate[1]] != -1 and coordinate[2][-1] != "r": #Prüfe ob es ein legaler Pfad ist und ob wir den schon besucht haben
                    newCoordinates.append((coordinate[0] - 1, coordinate[1],coordinate[2]+["l"])) # Legitimer Nachfolgeknoten von den wir aus weiter expandieren können
            if coordinate[0] + 1 < 17:
                if field[coordinate[0] + 1][coordinate[1]] != -1 and coordinate[2][-1] != "l":
                    newCoordinates.append((coordinate[0] + 1, coordinate[1],coordinate[2]+["r"]))
            if coordinate[1] - 1 >= 0:
                if field[coordinate[0]][coordinate[1] - 1] != -1 and coordinate[2][-1] != "d":
                    newCoordinates.append((coordinate[0], coordinate[1] - 1,coordinate[2]+["u"]))
            if coordinate[1] + 1 < 17:
                if field[coordinate[0]][coordinate[1] + 1] != -1 and coordinate[2][-1] != "u":
                    newCoordinates.append((coordinate[0], coordinate[1] + 1,coordinate[2]+["d"]))
        if len(newCoordinates)>0:
            return breitenSuche_Coin(newCoordinates)
        else:
            return []