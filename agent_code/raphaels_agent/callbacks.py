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
        sub_matrix=get_9x9_submatrix(game_state.get_field(),agent_position)
        print(sub_matrix)
        print("____________")
        print(game_state.get_field())
        print(agent_position)
        print("-------------")
        print("###################")
        print(game_state.get_coins_position())
        return self.ACTIONS[action_id]
    
def state_to_features(game_state: dict) -> np.array:
    pass

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

    combined_matrix = list(zip(submatrix_coordinates[:,:,0].flatten(), submatrix_coordinates[:,:,1].flatten()))

    # Reshape zur gewünschten 9x9 Matrix
    #combined_matrix = np.array(combined_matrix).reshape(9, 9, 2)

    rows=submatrix_coordinates[0].shape[0]
    print(rows)
    cols=submatrix_coordinates[0].shape[1]

    matrix = np.random.randint(0, 10, size=(rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            matrix[i][j][0]= submatrix_coordinates[0][i][j]
            matrix[i][j][1]= submatrix_coordinates[1][i][j]

    return matrix
