from dataclasses import dataclass
from functools import wraps
from typing import Union
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class HyperParameter:
    """Class for hyper-parameter for Q-learning.
    
    Args:
    ---
    inverse_temperature: The lower the temperature, the more random the action is chosen.

    """
    inverse_temperature: float = 1.0
    num_episode: int = 100
    learning_rate: float = 0.1
    epsilon: float = 0.5
    gamma: float = 0.9

class Action(Enum):
    up: int = 0
    right: int = 1
    down: int = 2
    left: int = 3

class PolicyGenerator:
    def __init__(self, map_info: np.ndarray, inverse_temperature: float) -> None:
        self.__map_info: np.ndarray = map_info
        self.__pi: np.ndarray = None
        self.__inverse_temperature: float = inverse_temperature

    @property
    def map_info(self) -> np.ndarray:
        return self.__map_info
            
    @property
    def pi(self) -> Union[np.ndarray, None]:
        return self.__pi

    def __check_pi_none(self) -> bool:
        return True if self.__pi is None else False

    def __check_zero_nansum_exist(self, dim: int = 0) -> bool:
        """Check whether the theta's nansum is zero or not.
        The nansum results are derived from either row or col.
        This option is controlled by dim argument.
        If dim is 0, then nansum is calculated for each row of the ndarray shape.
        If dim is 1, then nansum is calculated for each col of the ndarray shape.
        """
        [row, col] = self.__map_info.shape  # Shape of the theta matrix
        if dim == 0:
            for i in range(0, row):
                if np.nansum(self.__map_info[i, :]) == 0:
                    return True
        elif dim == 1:
            for i in range(0, col):
                if np.nansum(self.__map_info[:, i]) == 0:
                    return True
        else:
            raise ValueError(f"The dim value {dim} is not supported.")
        return False

    def __init_action_prob_by_softmax(self) -> np.ndarray:
        """Support softmax-based policy rate for defined actions"""
        [row, col] = self.__map_info.shape  # Shape of the theta matrix
        pi = np.zeros((row, col))
        map_exp_info: np.ndarray = np.exp(self.__inverse_temperature * self.__map_info)
        for r in range(0, row):
            pi[r, :] = map_exp_info[r, :] / np.nansum(map_exp_info[r, :])
        pi = np.nan_to_num(pi)
        return pi

    def __init_action_prob_by_simple(self) -> np.ndarray:
        """Support basic mode to calculate the policy rate for defined actions"""
        [row, col] = self.__map_info.shape  # Shape of the theta matrix
        pi: np.ndarray= np.zeros((row, col))

        if self.__check_zero_nansum_exist(0) is True:
            raise ValueError(f"The certain row nansum is zero. Cannot proceed the zero division.")
        
        for r in range(0, row):
            pi[r, :] = self.__map_info[r, :] / np.nansum(self.__map_info[r, :])
        pi = np.nan_to_num(pi)
        return pi
    
    def pi_init_test(method):
        @wraps(method)
        def _impl(self, *args, **kwargs):
            if self.__check_pi_none():
                method(self, *args, **kwargs)
                return True
            else:
                print(f"The pi has already been allocated.")
                return False
        return _impl
    
    @pi_init_test
    def make_action_prob_by_softmax(self) -> bool:
        self.__pi = self.__init_action_prob_by_softmax()

    @pi_init_test
    def make_action_prob_by_simple(self) -> bool:
        self.__pi = self.__init_action_prob_by_simple()


class Agent:
    def __init__(self, pi: np.ndarray):
        self.__pi = pi
        self.__state = 0 # TODO What does zero means?

    @property
    def state(self) -> int:
        return self.__state
    
    def get_action_and_next_state(self, cur_state: int):
        direction = [Action.up.value, Action.right.value, Action.down.value, Action.left.value]
        next_action = np.random.choice(direction, p=self.__pi[cur_state, :])
        if next_action == Action.up.value:
            return [next_action, cur_state - 3]
        elif next_action == Action.right.value:
            return [next_action, cur_state + 1]
        elif next_action == Action.down.value:
            return [next_action, cur_state + 3]
        elif next_action == Action.left.value:
            return [next_action, cur_state - 1]
        else:
            raise ValueError(f"Unexpected next action {next_action} was occurred.")

    def init_state(self):
        self.__state = 0


class MazeProblemSolver:
    def __init__(self, hyper_parameter: HyperParameter, policy_generator: PolicyGenerator, agent: Agent):
        self.__hyper_parameter = hyper_parameter
        self.__policy_generator = policy_generator
        self.__q_table = None
        self.agent = agent
        self.__init_hyper_parameter()
        self.__init_policy()
        self.__init_q_table()
        self.__init_episode_result()
        
    @property
    def q_table(self):
        return self.__q_table
    
    @property
    def episode_reward(self):
        return self.__episode_reward
    
    @property
    def episode_len(self):
        return self.__episode_len
    
    def __init_hyper_parameter(self):
        self.__num_episode = self.__hyper_parameter.num_episode
        self.__learning_rate = self.__hyper_parameter.learning_rate
        self.__epsilon = self.__hyper_parameter.epsilon
        self.__gamma = self.__hyper_parameter.gamma
    
    def __init_policy(self):
        self.__map_info = self.__policy_generator.map_info
        self.__pi = self.__policy_generator.pi

    def __init_q_table(self):
        [row, col] = self.__map_info.shape
        self.__q_table = np.random.rand(row, col) * self.__learning_rate

    def __init_episode_result(self):
        self.__episode_reward = [0] * self.__num_episode
        self.__episode_len = [0] * self.__num_episode

    def __q_learning(self, state, action, reward, next_state):
        if next_state == 8:
            self.__q_table[state, action] = self.__q_table[state, action] + self.__learning_rate * (reward - self.__q_table[state, action])
        else:
            self.__q_table[state, action] = self.__q_table[state, action] + self.__learning_rate * (reward + self.__gamma * np.nanmax(self.__q_table[next_state, :]) - self.__q_table[state, action])

    def __do_explore_maze(self):
        self.agent.init_state()
        cur_state = self.agent.state
        state_action_history = [[cur_state, np.nan]] # List for recording (state, action) of Agent

        while True:
            [next_action, next_state] = self.agent.get_action_and_next_state(cur_state)
            state_action_history[-1][1] = next_action
            state_action_history.append([next_state, np.nan])
            
            if next_state == 8:
                self.__q_learning(cur_state, next_action, reward=1, next_state=next_state)
                break
            else:
                self.__q_learning(cur_state, next_action, reward=0, next_state=next_state)
                cur_state = next_state

        return state_action_history
    
    def __decrease_epsilon(self):
        """Decrease the epsilon value"""
        self.__epsilon = self.__epsilon / 2

    def __record_episode_result(self, episode: int, history: list):
        self.__episode_len[episode] = len(history) - 2
        state_value = np.nanmax(self.__q_table, axis=1)
        self.__episode_reward[episode] = np.sum(np.abs(state_value))

    def explore_maze(self):
        """Explore the maze to learn"""
        for episode in range(self.__num_episode):
            print(f"Episode # {episode} has been begun.")
            self.__decrease_epsilon()
            collected_state_action_history = self.__do_explore_maze()
            self.__record_episode_result(episode, collected_state_action_history)
            print(f"The number of steps it took to reach the destination point is {self.__episode_len[episode]}")
            print(f"The value of steps for reaching the destination point is {self.__episode_reward[episode]}")
    
    def stat_solver(self):
        print("========== STAT OF MAZE PROBLEM SOLVER ==========")
        for state in range(len(self.__q_table)):
            max_action = np.argmax(self.__q_table[state])
            if max_action == Action.up.value:
                print(f"Up is desirable at the state S{state}")
            elif max_action == Action.right.value:
                print(f"Right is desirable at the state S{state}")
            elif max_action == Action.down.value:
                print(f"Down is desirable at the state S{state}")
            elif max_action == Action.left.value:
                print(f"Left is desirable at the state S{state}")
        print("=================================================")
class MazePlotter:
    def draw_episode_len(self, maze_problem_solver: MazeProblemSolver):
        plt.plot(mps.episode_len)
        plt.title('Episode length over time')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.savefig('maze/maze_episode_length.png', dpi=300)

    def draw_episode_reward(self):
        pass

if __name__ == "__main__":
    # Row: The number of states (S0 ~ S7, except S8 since S8 is our goal)
    # Col: The number of possible moves (Up, Right, Down, Left)
    map_info = np.array([[np.nan, 1, 1, np.nan],
                        [np.nan, 1, np.nan, 1],
                        [np.nan, np.nan, 1, 1],
                        [1, 1, 1, np.nan],
                        [np.nan, np.nan, 1, 1],
                        [1, np.nan, np.nan, np.nan],
                        [1, np.nan, np.nan, np.nan],
                        [1, 1, np.nan, np.nan],
                        ])

    hp = HyperParameter()
    pg = PolicyGenerator(map_info, hp.inverse_temperature)
    pg.make_action_prob_by_softmax()

    agent = Agent(pg.pi)
    mps = MazeProblemSolver(hp, pg, agent)
    mps.explore_maze()
    mps.stat_solver()
    
    # mp = MazePlotter()
    # mp.draw_episode_len(mps)