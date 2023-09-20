import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import GameState


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        size_input = 100
        size_hidden = 120
        self.fc1 = nn.Linear(size_input, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_hidden)
        self.fc3 = nn.Linear(size_hidden, 6)

    def forward(self, game_state: GameState):
        x = game_state.to_vec() # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)