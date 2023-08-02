import os
import ujson
from typing import List
import numpy as np
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
    self.steps_of_current_game = dict()
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
    identifier = the_old_game_state.to_hashed_features()
    self.steps_of_current_game[(identifier, self_action)] = {"old_game_state": the_old_game_state, "new_game_state": GameState(new_game_state), "action": the_old_game_state.adjust_movement(self_action), "events": events}


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
    for step in self.steps_of_current_game.values():
        old_game_state = step["old_game_state"]
        new_game_state = step["new_game_state"]
        # print("From:")
        old_state_feature_hash = old_game_state.to_hashed_features()
        # print(old_game_state.get_closest_coin_distance())
        # print("To:")
        new_state_feature_hash = new_game_state.to_hashed_features()
        # print(new_game_state.get_closest_coin_distance())
        v = max([self.prev_Q[(new_state_feature_hash, action)] for action in self.ACTIONS])
        action = step["action"]
        auxilliary_reward = auxilliary_rewards(old_game_state, new_game_state)
        event_reward = 0 # reward_from_events(self, step["events"])
        total_update = LEARNING_RATE * (auxilliary_reward + event_reward) # + GAMMA * v - self.prev_Q[((old_state_feature_hash, action))]
        # print(f"Auxillary reward: {auxilliary_reward}")
        # print(f"Gamma part: {GAMMA * v - self.prev_Q[((old_state_feature_hash, action))]}")
        # print(f"Rewarded: {total_update}")
        if old_state_feature_hash == new_state_feature_hash:
            continue
        self.Q[(old_state_feature_hash, action)] = self.prev_Q[(old_state_feature_hash, action)] + total_update
    self.prev_Q = self.Q
    self.steps_of_current_game = dict()
    self.logger.info(f"Size of Q: {len(self.Q.keys())}")
    with open(self.TRAINING_DATA_DIRECTORY + "q.json", "w", encoding="utf-8") as f:
        f.write(ujson.dumps(self.Q))


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
