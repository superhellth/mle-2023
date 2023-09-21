import os
import sys
import ujson
from typing import List
import numpy as np
from random import shuffle
import random
import events as e
import settings
from .state import GameState
from collections import defaultdict
import math

NEW_FIELD_VISITED = "NEW_FIELD_VISITED"
OPPONENT_IN_DANGER_AND_PLACED_BOMB = "OPPONENT_IN_DANGER_AND_PLACED_BOMB"
OPPONENT_CANT_SURVIVE_AND_PLACED_BOMB = "OPPONENT_CANT_SURVIVE_AND_PLACED_BOMB"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Starting Training Setup")
    if not os.path.exists(self.TRAINING_DATA_DIRECTORY):
        os.mkdir(self.TRAINING_DATA_DIRECTORY)

    self.DIMENSIONS_MAP = (17,17)
    self.EPSILON = 1.0
    self.STEP_DISCOUNT = 0.3
    self.LEARNING_RATE = 0.1
    self.Q = defaultdict(float)
    self.prev_Q = self.Q
    self.experience_buffer = list()
    self.current_game_list = list()
    self.current_game_hashes = set()
    self.logger.info("Finished Training Setup")
    self.previous_agent_position = None
    self.already_visited = np.zeros(self.DIMENSIONS_MAP)


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
    the_old_game_state = GameState(old_game_state)
    the_new_game_state = GameState(new_game_state)
    the_old_game_state_feature = the_old_game_state.to_features()
    the_new_game_state_feature = the_old_game_state.to_features()

    old_game_state_hash = the_old_game_state.to_hashed_features()
    new_game_state_hash = the_new_game_state.to_hashed_features()
    self.already_visited[the_old_game_state.get_agent_position()] = 1
    if self.already_visited[the_new_game_state.get_agent_position()] == 0 and self_action != "WAITED":
        #print("ALREADY THERE")
        events.append(NEW_FIELD_VISITED)
        self.already_visited[the_new_game_state.get_agent_position()] == 1
    if type(the_old_game_state_feature) != int and the_old_game_state_feature["closest_agent_is_in_danger"]==True and self_action=="BOMB":
        events.append(OPPONENT_IN_DANGER_AND_PLACED_BOMB)
    if type(the_old_game_state_feature) != int and the_old_game_state_feature["closest_agent_cant_survive"]==True and self_action=="BOMB":
        events.append(OPPONENT_CANT_SURVIVE_AND_PLACED_BOMB)
    if old_game_state_hash == new_game_state_hash:
        events.append("REPEATED GAMESTATE")
    self.current_game_hashes.add(old_game_state_hash) 
    self.current_game_list.append({"old_game_state": the_old_game_state, "new_game_state": the_new_game_state,
                                  "action": the_old_game_state.adjust_movement(self_action), "events": events})


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
    the_last_game_state = GameState(last_game_state)
    self.current_game_hashes.add(the_last_game_state.to_hashed_features())
    if last_game_state["step"] < 390:
        self.current_game_list.append({"old_game_state": the_last_game_state, "new_game_state": GameState(None, dead=True),
                                    "action": the_last_game_state.adjust_movement(last_action), "events": events})
    self.logger.info(f"Round {the_last_game_state.round} ended!")
    self.experience_buffer.append(self.current_game_list)
    random_batch = sample_training_data(self, size=200)
    for i, (game_number, step_number, step) in enumerate(random_batch):
        n = min(4, len(self.experience_buffer[game_number]) - step_number - 1)
        sum = 0
        current_state = step["old_game_state"]
        if not current_state.can_agent_survive():
            continue
        current_hash = current_state.to_hashed_features()
        current_action = step["action"]
        # print("------------")
        # print("Old state:")
        # print(current_state.to_features())
        # print(current_state.agent_position)
        # print("Action:")
        # print(current_action)
        # print("New state:")
        # other_state = self.experience_buffer[game_number][step_number + 1]["old_game_state"]
        # print(other_state.to_features())
        # print(other_state.agent_position)
        killed_by_waiting = False
        for future_step in range(step_number + 1, step_number + n + 1):
            game_state_before = self.experience_buffer[game_number][future_step -
                                                                    1]["old_game_state"]
            game_state_after = self.experience_buffer[game_number][future_step]["old_game_state"]
            print_components = False

            if future_step == step_number + 1 and current_action == "WAIT" and not game_state_after.can_agent_survive():
                killed_by_waiting = True

            # if current_action == "WAIT":
            #     print_components = True

            # gamestate is not surviveable -> don't need to learn
            if not game_state_before.can_agent_survive():
                break
            # nothing happend between gamestates
            if game_state_before.to_hashed_features() == game_state_after.to_hashed_features():
                continue
            # the step is a bullshit move
            if future_step >= step_number + 2 and game_state_before.can_agent_survive() and not game_state_after.can_agent_survive():
                break
            final_state = self.experience_buffer[game_number][step_number +
                                                              n - 1]["new_game_state"]
            # + reward_from_events(self, events)
            reward = auxilliary_rewards(game_state_before, game_state_after, print_components=print_components) # + reward_from_events(self, events)
            
            #Add rewards from events
            #ToDo
            #Falsch, auf Events im Experience Buffer zugreifen
            reward += reward_from_events(self,self.experience_buffer[game_number][future_step]["events"])
            print(self.experience_buffer[game_number][future_step]["events"])
            if n > 1 and final_state.get_potential() > 0:
                v = max([self.prev_Q[(final_state.to_hashed_features(), a)]
                        for a in final_state.get_possible_moves()])
            else:
                v = 0
            # + GAMMA**n * v - self.prev_Q[(current_hash, current_action)]
            weight_of_step = self.STEP_DISCOUNT**(future_step - step_number - 1) * reward # + self.STEP_DISCOUNT**n * v - self.prev_Q[(current_hash, current_action)]
            # if current_action == "WAIT" and current_state.bombs != [] and current_state.bombs[0][1] == 0:
            #     print(" - ")
            #     print(weight_of_step)
            sum += weight_of_step
        old_q = self.Q[(current_hash, current_action)]
        self.Q[(current_hash, current_action)] = self.prev_Q[(
            current_hash, current_action)] + self.LEARNING_RATE * sum
        # if current_action == "WAIT" and sum < 0 and not killed_by_waiting:
        #     print("Current State:")
        #     print(f"Agent pos: {current_state.agent_position}")
        #     print(f"Is agent in danger: {current_state.is_agent_in_danger}")
        #     print(current_state.bombs)
        #     print(current_state.field)
        #     print(f"Rewarded: {sum}")
        #     print(f"Old value of Q: {old_q}")
        #     print(f"New value of Q: {self.Q[(current_hash, current_action)]}")
        #     print("-------------")
        #     print()
        # print(f"Rewarded: {sum}")
    self.prev_Q = self.Q
    self.current_game_list = list()
    self.logger.info(f"Size of Q: {len(self.Q.keys())}")
    if self.EPSILON > 0.03:
        self.EPSILON *= 0.998
    if len(self.experience_buffer) % 30 == 0 and self.LEARNING_RATE > 0.01:
        self.LEARNING_RATE *= 0.95
    if len(self.experience_buffer) % 50 == 0:
        print(f" Size of Q after {the_last_game_state.round} rounds: {len(self.Q)}")
        print(f"Size in memory: {sys.getsizeof(self.Q)}")
        print(f"Size of experience buffer in memory: {sys.getsizeof(self.experience_buffer)}")
        print(f"Epsilon: {self.EPSILON}")
        print(f"Learning rate: {self.LEARNING_RATE}")
        with open(self.TRAINING_DATA_DIRECTORY + "q.json", "w", encoding="utf-8") as f:
            f.write(ujson.dumps(self.Q))
    self.already_visited = np.zeros(self.DIMENSIONS_MAP)

def sample_training_data(self, size=400):
    """Create a random subsample of the experience buffer for training.

    Args:
        size (int, optional): Size of the sample. Defaults to 400.

    Returns:
        list: Each entry is a tuple of (game_number, step_number, step)
    """
    num_games = len(self.experience_buffer)
    points_per_game = int(size / num_games)
    games = np.arange(num_games)
    if points_per_game < 3 or num_games > 100:
        points_per_game = int(size / 50)
        # only consider steps from the last 100 games
        games = np.random.choice(np.arange(num_games - 100, num_games), 50, replace=False)
    random_sample = []
    for game in games:
        game_steps = self.experience_buffer[game]
        points_this_game = min(len(game_steps), points_per_game)
        subsample_indicies = np.random.choice(
            np.arange(0, len(game_steps), 1), points_this_game, replace=False)
        subsample = [game_steps[i] for i in subsample_indicies]
        subsample = [(game, subsample_indicies[i], subsample[i])
                     for i in range(points_this_game)]
        shuffle(subsample)
        random_sample += subsample
    shuffle(random_sample)
    return random_sample


def auxilliary_rewards(old_game_state: GameState, new_game_state: GameState, print_components=False):
    # print("Old game state:")
    pot0 = old_game_state.get_potential(print_components=print_components)
    # print(f"Old game state potential: {pot0}")
    # print()
    # print("New game state:")
    pot1 = new_game_state.get_potential(print_components=print_components)
    # print(f"New game state potential: {pot1}")
    return pot1 - pot0


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.KILLED_SELF: -30,
        NEW_FIELD_VISITED: 0,
        e.KILLED_OPPONENT:19,
        OPPONENT_IN_DANGER_AND_PLACED_BOMB:15,
        OPPONENT_CANT_SURVIVE_AND_PLACED_BOMB:25,
        e.WAITED:-50
    }

    """e.MOVED_UP:-.001,
        e.MOVED_LEFT:-.001,
        e.MOVED_RIGHT:-.001,
        e.MOVED_DOWN:-.001,"""
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum