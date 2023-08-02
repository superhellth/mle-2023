import os
import ujson
from typing import List
import numpy as np
from random import shuffle
import random
import events as e
from .state import GameState
from collections import defaultdict

EPSILON = 0.3
GAMMA = 1
LEARNING_RATE = 0.1

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Starting Training Setup")
    if not os.path.exists(self.TRAINING_DATA_DIRECTORY):
        os.mkdir(self.TRAINING_DATA_DIRECTORY)

    self.EPSILON = EPSILON
    self.Q = defaultdict(float)
    self.prev_Q = self.Q
    self.experience_buffer = list()
    self.current_game_list = list()
    self.logger.info("Finished Training Setup")


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
    self.logger.info("Game events occured")
    the_old_game_state = GameState(old_game_state)
    the_old_game_state.to_hashed_features() # ??????????????????????????????????????????????????
    self.current_game_list.append({"old_game_state": the_old_game_state, "new_game_state": GameState(new_game_state), "action": the_old_game_state.adjust_movement(self_action), "events": events})


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
    self.logger.info("Round ended!")
    self.experience_buffer.append(self.current_game_list)
    game_actions_seen = list()
    for step in sample_training_data(self):
        old_game_state = step["old_game_state"]
        new_game_state = step["new_game_state"]
        old_state_feature_hash = old_game_state.to_hashed_features()
        new_state_feature_hash = new_game_state.to_hashed_features()
        action = step["action"]
        if old_state_feature_hash == new_state_feature_hash or (old_state_feature_hash, action) in game_actions_seen:
            continue
        v = max([self.prev_Q[(new_state_feature_hash, a)] for a in self.ACTIONS])
        auxilliary_reward = auxilliary_rewards(old_game_state, new_game_state)
        event_reward = 0 # reward_from_events(self, step["events"])
        total_update = LEARNING_RATE * (auxilliary_reward + event_reward) # + GAMMA * v - self.prev_Q[((old_state_feature_hash, action))]
        self.Q[(old_state_feature_hash, action)] = self.prev_Q[(old_state_feature_hash, action)] + total_update
        game_actions_seen.append((old_state_feature_hash, action))
    self.prev_Q = self.Q
    self.current_game_list = list()
    self.logger.info(f"Size of Q: {len(self.Q.keys())}")
    with open(self.TRAINING_DATA_DIRECTORY + "q.json", "w", encoding="utf-8") as f:
        f.write(ujson.dumps(self.Q))


def sample_training_data(self, size=400):
    num_games = len(self.experience_buffer)
    points_per_game = int(size / num_games)
    random_sample = []
    for game in range(num_games):
        subsample = random.sample(self.experience_buffer[game], points_per_game)
        shuffle(subsample)
        random_sample += subsample
    shuffle(random_sample)
    return random_sample


def auxilliary_rewards(old_game_state: GameState, new_game_state: GameState):
    return GAMMA * new_game_state.get_potential() - old_game_state.get_potential()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
