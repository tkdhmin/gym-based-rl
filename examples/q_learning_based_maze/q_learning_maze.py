from dataclasses import dataclass
from functools import wraps
from typing import Union
from enum import Enum
import numpy as np


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

#TODO: impl next
def Q_learning():
    """Q-learning"""
    pass


class PolicyGenerator:
    def __init__(self, theta: np.ndarray, inverse_temperature: float) -> None:
        self.__theta: np.ndarray = theta
        self.__pi: np.ndarray = None
        self.__inverse_temperature: float = inverse_temperature

    @property
    def theta(self) -> np.ndarray:
        return self.__theta
            
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
        [row, col] = self.__theta.shape  # Shape of the theta matrix
        if dim == 0:
            for i in range(0, row):
                if np.nansum(self.__theta[i, :]) == 0:
                    return True
        elif dim == 1:
            for i in range(0, col):
                if np.nansum(self.__theta[:, i]) == 0:
                    return True
        else:
            raise ValueError(f"The dim value {dim} is not supported.")
        return False

    #TODO: To make a class supporting the various mode to calculate the action probablity.
    def __init_action_prob_by_softmax(self) -> np.ndarray:
        """Support softmax-based policy rate for defined actions"""
        [row, col] = self.__theta.shape  # Shape of the theta matrix
        pi = np.zeros((row, col))
        exp_theta: np.ndarray = np.exp(self.__inverse_temperature * self.__theta)
        for r in range(0, row):
            pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])
        pi = np.nan_to_num(pi)
        return pi

    def __init_action_prob_by_simple(self) -> np.ndarray:
        """Support basic mode to calculate the policy rate for defined actions"""
        [row, col] = self.__theta.shape  # Shape of the theta matrix
        pi: np.ndarray= np.zeros((row, col))

        if self.__check_zero_nansum_exist(0) is True:
            raise ValueError(f"The certain row nansum is zero. Cannot proceed the zero division.")
        
        for r in range(0, row):
            pi[r, :] = self.__theta[r, :] / np.nansum(self.__theta[r, :])
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

# 행동가치 함수 Q의 초기 상태
[a, b] = theta_0.shape  # # 열과 행의 갯수를 변수 a, b에 저장
Q = np.random.rand(a, b) * theta_0 * 0.1
# * theta0 로 요소 단위 곱셈을 수행, Q에서 벽 방향으로 이동하는 행동에는 nan을 부여

# ε-greedy 알고리즘 구현

# Q러닝 알고리즘으로 행동가치 함수 Q를 수정


def Q_learning(s, a, r, s_next, Q, eta, gamma):

    if s_next == 8:  # 목표 지점에 도달한 경우
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next,: ]) - Q[s, a])

    return Q

# Q러닝 알고리즘으로 미로를 빠져나오는 함수, 상태 및 행동 그리고 Q값의 히스토리를 출력한다


def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0  # 시작 지점
    a = a_next = get_action(s, Q, epsilon, pi)  # 첫 번째 행동
    s_a_history = [[0, np.nan]]  # 에이전트의 행동 및 상태의 히스토리를 기록하는 리스트

    while (1):  # 목표 지점에 이를 때까지 반복
        a = a_next  # 행동 결정

        s_a_history[-1][1] = a
        # 현재 상태(마지막으로 인덱스가 -1)을 히스토리에 추가

        s_next = get_s_next(s, a, Q, epsilon, pi)
        # 다음 단계의 상태를 구함

        s_a_history.append([s_next, np.nan])
        # 다음 상태를 히스토리에 추가, 행동은 아직 알 수 없으므로 nan으로 둔다

        # 보상을 부여하고 다음 행동을 계산함
        if s_next == 8:
            r = 1  # 목표 지점에 도달했다면 보상을 부여
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
            # 다음 행동 a_next를 계산

        # 가치함수를 수정
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)

        # 종료 여부 판정
        if s_next == 8:  # 목표 지점에 도달하면 종료
            break
        else:
            s = s_next

    return [s_a_history, Q]

# Q러닝 알고리즘으로 미로 빠져나오기

eta = 0.1  # 학습률
gamma = 0.9  # 시간할인율
epsilon = 0.5  # ε-greedy 알고리즘 epsilon 초깃값
v = np.nanmax(Q, axis=1)  # 각 상태마다 가치의 최댓값을 계산
is_continue = True
episode = 1


class MazeExplorer:
    def __init__(self, hyper_parameter: HyperParameter, policy_generator: PolicyGenerator, agent: Agent):
        self.hyper_parameter = hyper_parameter
        self.policy_generator = policy_generator
        self.agent = agent
        self.__init_hyper_parameter(self.hyper_parameter)
        self.__init_policy(self.policy_generator)
        self.__init_q_table()
        
    @property
    def q_table(self):
        return self.__q_table
    
    def __init_hyper_parameter(self, hp: HyperParameter):
        self.__num_episode = hp.num_episode
        self.__learning_rate = hp.learning_rate
        self.__epsilon = hp.epsilon
        self.__gamma = hp.gamma
    
    def __init_policy(self, pg:PolicyGenerator):
        self.__theta = pg.theta
        self.__pi = pg.pi

    def __init_q_table(self):
        [row, col] = self.__theta.shape
        self.__q_table = np.random.rand(row, col) * self.__learning_rate

    def __q_learning(self, state, action, reward, next_state):
        if next_state == 8:
            self.__q_table[state, action] = self.__q_table[state, action] + self.__learning_rate * (reward - self.__q_table[state, action])
        else:
            self.__q_table[state, action] = self.__q_table[state, action] + self.__learning_rate * (reward + self.__gamma * np.nanmax(self.__q_table[next_state, :]) - self.__q_table[state, action])

    def __do_explore_maze(self):
        self.agent.init_state()
        cur_state = self.agent.state
        state_action_history = [[cur_state, np.nan]] # List for recording (state, action) of Agent
        print(type(state_action_history))

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

    def explore_maze(self):
        """Explore the maze to learn"""
        # V = []  # 에피소드 별로 상태가치를 저장
        # V.append(np.nanmax(Q, axis=1))  # 상태 별로 행동가치의 최댓값을 계산
        for episode in range(self.__num_episode):
            print(f"Episode # {episode} has been begun.")
            self.__decrease_epsilon()
            collected_state_action_history = self.agent.get_action_and_next_state
            # Q러닝으로 미로를 빠져나온 후, 결과로 나온 행동 히스토리와 Q값을 변수에 저장
            
            # # 상태가치의 변화
            # new_v = np.nanmax(Q, axis=1)  # 각 상태마다 행동가치의 최댓값을 계산
            # print(np.sum(np.abs(new_v - v)))  # 상태가치 함수의 변화를 출력
            # v = new_v
            # V.append(v)  # 현재 에피소드가 끝난 시점의 상태가치 함수를 추가
            print(f"The number of steps it took to reach the destination point is {0}")
    
    def init_q_table(self):
        [row, col] = self.policy_generator.theta.shape
        self.__q_table = np.random.rand(row, col) * self.__theta.shape * self.__learning_rate


if __name__ == "__main__":
    # Row: The number of states (S0 ~ S7, except S8 since S8 is our goal)
    # Col: The number of possible moves (Up, Right, Down, Left)
    theta_0 = np.array([[np.nan, 1, 1, np.nan],
                        [np.nan, 1, np.nan, 1],
                        [np.nan, np.nan, 1, 1],
                        [1, 1, 1, np.nan],
                        [np.nan, np.nan, 1, 1],
                        [1, np.nan, np.nan, np.nan],
                        [1, np.nan, np.nan, np.nan],
                        [1, 1, np.nan, np.nan],
                        ])

    hyper_parameter = HyperParameter()
    policy_generator = PolicyGenerator(theta_0, hyper_parameter.inverse_temperature)
    policy_generator.make_action_prob_by_softmax()

    agent = Agent(policy_generator.pi)
    maze_explorer = MazeExplorer(hyper_parameter, policy_generator, agent)
    maze_explorer.explore_maze()