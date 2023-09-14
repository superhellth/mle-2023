from collections import namedtuple, deque

import pickle
from typing import List
import os
from collections import defaultdict
import ujson
import numpy as np

import events as e
from .callbacks import state_to_features
from .state import GameState

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
WALKED_TO_NEXT_COIN = "WALKED_TO_NEXT_COIN"
WALKED_AWAY_FROM_COIN_OR_WAITED =  "WALKED_AWAY_FROM_COIN_OR_WAITED"
FIELD_ALREADY_VISITED = "FIELD_ALREADY_VISITED"
MOVE_NOT_SAVE = "MOVE_NOT_SAVE"
COINS_COLLECTED_HIGHER_THAN_MAX = "COINS_COLLECTED_HIGHER_THAN_MAX"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.logger.info("Starting Training Setup")
    self.TRAINING_DATA_DIRECTORY = "./training_data/"
    if not os.path.exists(self.TRAINING_DATA_DIRECTORY):
        os.mkdir(self.TRAINING_DATA_DIRECTORY)

    self.DIMENSIONS_MAP = (17,17)
    self.EPSILON = 0.9
    self.Q_TABLE = defaultdict(float)
    random_value = np.random.rand(1)[0]
    self.Q_TABLE = defaultdict(lambda: random_value)
    self.Q_TABLE = np.random.rand(self.DIMENSIONS_MAP[0],self.DIMENSIONS_MAP[1],len(self.ACTIONS))
    self.Q_TABLE = 2 * self.Q_TABLE - 1
    self.prev_Q_TABLE = self.Q_TABLE

    q_table_as_list = self.Q_TABLE.tolist()
    #print(q_table_as_list)

    if not os.path.isfile(self.TRAINING_DATA_DIRECTORY + "q.json"):
        # If the file does not exist, create the directory
        os.makedirs(os.path.dirname(self.TRAINING_DATA_DIRECTORY), exist_ok=True)

    # Create the file and write data to it
    with open(self.TRAINING_DATA_DIRECTORY + "q.json", "w", encoding="utf-8") as f:
        f.write(ujson.dumps(q_table_as_list))

    self.LEARNING_RATE = 0.1
    self.DISCOUNT = 0.9
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self.cummulative_reward = 0
    self.already_visited = np.zeros(self.DIMENSIONS_MAP)
    self.coins_collected_in_game=0
    self.max_coins_collected=0



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    """print("Game event occurred")
    print(events)
    print(self_action)"""
    """# Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)


    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))"""
    if "COIN_COLLECTED" in events:
         self.coins_collected_in_game += 1
         
    old_game_state = GameState(old_game_state)
    new_game_state = GameState(new_game_state)

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    old_agent_position = old_features["agent_position"]
    new_agent_position = new_features["agent_position"]

    if self_action == new_features["path_to_closest_coin"][1]:
        events.append(WALKED_TO_NEXT_COIN)
    elif self_action == "BOMB":
        pass
    else:
        events.append(WALKED_AWAY_FROM_COIN_OR_WAITED)
    if self_action in old_features["impossible_moves"]:
        events.append(e.INVALID_ACTION)
    if self.already_visited[new_agent_position] == 1 or self_action == "WAIT":
        print("ALREADY THERE")
        events.append(FIELD_ALREADY_VISITED)
    if not self_action in old_features["save_moves"]:
         events.append(MOVE_NOT_SAVE)

    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    self_action_index = self.ACTIONS.index(self_action)
    current_q_value = self.Q_TABLE[old_agent_position][self_action_index]
    max_future_q_value = max(self.Q_TABLE[new_agent_position][i] for i in range(len(self.ACTIONS)))
    reward = reward_from_events(self, events)
    self.cummulative_reward += reward
    new_q_value = current_q_value + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q_value - current_q_value)
    self.Q_TABLE[old_agent_position][self_action_index] = new_q_value
    self.already_visited[new_agent_position] = 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    
    #print(state_to_features(last_game_state))
    last_game_state = GameState(last_game_state)
    last_action_index = self.ACTIONS.index(last_action)

    if self.coins_collected_in_game >= self.max_coins_collected:
         events.append(COINS_COLLECTED_HIGHER_THAN_MAX)
         self.max_coins_collected = self.coins_collected_in_game

    self.cummulative_reward += reward_from_events(self,events)
    last_agent_position = last_game_state.get_agent_position()
    current_q_value = self.Q_TABLE[last_agent_position][last_action_index]
    self.Q_TABLE[last_agent_position][last_action_index] = (1-self.LEARNING_RATE)*current_q_value+self.LEARNING_RATE*self.cummulative_reward

    # Store the model
    q_table_as_list = self.Q_TABLE.tolist()
    self.already_visited = np.zeros(self.DIMENSIONS_MAP)
    self.coins_collected_in_game=0
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    with open(self.TRAINING_DATA_DIRECTORY + "q.json", "w", encoding="utf-8") as f:
                print("Write")
                f.write(ujson.dumps(q_table_as_list))

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 50,
        WALKED_TO_NEXT_COIN: 30,
        e.INVALID_ACTION: -100,
        e.KILLED_SELF:-10000,
        e.GOT_KILLED:-500,
        e.WAITED:-300,
        e.MOVED_UP:-1,
        e.MOVED_LEFT:-1,
        e.MOVED_RIGHT:-1,
        e.MOVED_DOWN:-1,
        WALKED_AWAY_FROM_COIN_OR_WAITED:-30,
        FIELD_ALREADY_VISITED:-300,
        MOVE_NOT_SAVE:40,
        e.SURVIVED_ROUND:5000,
        COINS_COLLECTED_HIGHER_THAN_MAX:1000
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
