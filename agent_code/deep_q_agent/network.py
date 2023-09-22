import torch
import torch.nn as nn
import ujson
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # an affine operation: y = Wx + b
        size_input = 23
        size_hidden = 40
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(size_input, size_hidden),
            nn.ReLU(),
            nn.Linear(size_hidden, size_hidden),
            nn.ReLU(),
            nn.Linear(size_hidden, 6)
        )
        # self.fc1 = nn.Linear(size_input, size_hidden)
        # self.fc2 = nn.Linear(size_hidden, size_hidden)
        # self.fc3 = nn.Linear(size_hidden, 6)

    def forward(self, x):
        # x = torch.randn(25, 23)
        # print(x)
        x = torch.nn.functional.normalize(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        # print(logits)
        # print(logits)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return logits


def dict_to_vec(feature_dict):
    if feature_dict == -1:
        return np.array([0 for i in range(23)]).flatten()
    # local map is always a feature
    local_map_vec = np.array(feature_dict["local_map"]).flatten()
    # would survive bomb
    if "would_survive_bomb" in feature_dict.keys():
        would_survive_bomb_vec = np.array([feature_dict["would_survive_bomb"]])
    else:
        would_survive_bomb_vec = np.array([True])
    # local prec explosion map
    if "local_prec_explosion_map" in feature_dict.keys():
        local_prec_explosion_map_vec = np.array(feature_dict["local_prec_explosion_map"]).flatten()
    else:
        local_prec_explosion_map_vec = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).flatten()
    # agent to nearest safe square
    if "agent_to_nearest_safe_square" in feature_dict.keys() and not np.array(feature_dict["agent_to_nearest_safe_square"]).shape[0] == 2:
        agent_to_nearest_safe_square_vec = np.array(feature_dict["agent_to_nearest_safe_square"]).flatten()
    else:
        agent_to_nearest_safe_square_vec = np.array([0, 0]).flatten()
    # agent to nearest coin
    if "agent_to_nearest_coin" in feature_dict.keys():
        agent_to_nearest_coin_vec = np.array(feature_dict["agent_to_nearest_coin"]).flatten()
    else:
        agent_to_nearest_coin_vec = np.array([0, 0]).flatten()
    return np.concatenate((local_map_vec, would_survive_bomb_vec, local_prec_explosion_map_vec, agent_to_nearest_safe_square_vec, agent_to_nearest_coin_vec)).flatten()

def action_values_to_vec(action_values):
    actions = ["LEFT", "RIGHT", "UP", "DOWN", "WAIT", "BOMB"]
    return np.array([action_values[action] for action in actions])

# load data from file
hash_to_features = dict()
hash_to_q = dict()
with open("./training_data/hash_to_features.json", "r") as f:
    loaded_dict = ujson.loads(f.read())
    for hash in loaded_dict.keys():
        hash_to_features[int(hash)] = loaded_dict[hash]
with open("./training_data/hash_to_action_values.json", "r") as f:
    loaded_dict = ujson.loads(f.read())
    for hash in loaded_dict.keys():
        hash_to_q[int(hash)] = loaded_dict[hash]

all_hashes = list(hash_to_features.keys())
for hash_value in hash_to_features.keys():
    q_dict = hash_to_q[hash_value]
    # print([item[1] for item in q_dict.items()])
    if max([abs(item[1]) for item in q_dict.items()]) == 0:
        all_hashes.remove(hash_value)
batch_size = 25
train_data = []
for i in range(len(all_hashes) // batch_size):
    batch_X = []
    batch_y = []
    for j in range(i * 25, (i + 1) * 25):
        batch_X.append(dict_to_vec(hash_to_features[all_hashes[j]]))
        batch_y.append(action_values_to_vec(hash_to_q[all_hashes[j]]))
    # print(np.array(batch_y))
    train_data.append((torch.Tensor(np.array(batch_X)), torch.Tensor(np.array(batch_y))))

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# X = torch.rand(25, 23)
# logits = model(X)
# print(logits)
loss_fn = nn.MSELoss()



def train_loop(train_data, model, loss_fn, optimizer):
    size = len(all_hashes)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(train_data):
        # Compute prediction and loss
        pred = model(X)
        y = torch.nn.functional.normalize(y)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 1000
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.9)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_data, model, loss_fn, optimizer)
    scheduler.step()

torch.save(model.state_dict(), "./training_data/deep_q.pt")