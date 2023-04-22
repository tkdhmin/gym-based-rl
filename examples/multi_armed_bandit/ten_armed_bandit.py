from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, num_k_options: int = 1, epsilons: float = 0.) -> None:
        self.__num_k_options = num_k_options
        self.__epsilons = epsilons
        self.__estimated_q_value = np.zeros(self.__num_k_options)
        self.reset()

    @property
    def epsilons(self) -> float:
        return self.__epsilons

    @property
    def best_option(self) -> int:
        return self.__best_option
    
    def reset(self):
        self.__estimated_q_value = np.zeros(self.__num_k_options)
        self.__avg_reward = 0
        self.__time = 0
        self.__best_option = np.argmax(np.random.randn(self.__num_k_options)) # fixed per iteration
        self.__current_reward = 0
        self.__current_option = None

    def __decide_option(self) -> None:
        """Support epsilon-greedy policy to choose an next option."""
        if np.random.rand() < self.__epsilons: # Exploration
            self.__current_option = np.random.choice(np.arange(self.__num_k_options))
        else: # Exploitation
            self.__current_option = np.argmax(self.__estimated_q_value)

    def __update_avg_reward(self, reward: float) -> None:
        self.__time += 1
        self.__avg_reward += (reward - self.__avg_reward) / self.__time

    def __update_estimated_reward(self, option: int, reward: float) -> None:
        self.__estimated_q_value[option] += (reward - self.__estimated_q_value[option]) * 0.1

    def __update_reward(self) -> None:
        """Update reward based on selected option."""
        self.__current_reward = np.random.randn()
        self.__update_avg_reward(self.__current_reward)
        self.__update_estimated_reward(self.__current_option, self.__current_reward)

    def proceed(self) -> None:
        self.__decide_option()
        self.__update_reward()
    
    def get_latest_option_and_reward(self) -> Tuple[int, float]:
        if self.__current_option is None or self.__current_reward is None:
            raise ValueError(f"Current action and its reward at time {self.__time} is None.")
        return self.__current_option, self.__current_reward

class SlotMachineSimulator(Bandit):
    def __init__(self, num_k_options: int = 1, epsilons: float = 0.) -> None:
        super().__init__(num_k_options, epsilons)
        self.__expected_best_option = None
        self.__expected_rewards = None
    
    @property
    def expected_best_options(self):
        return self.__expected_best_option

    @property
    def expected_rewards(self):
        return self.__expected_rewards
    
    def run(self, num_operations = None, time_steps = None) -> None:
        if num_operations is None or time_steps is None:
            raise ValueError(f"Neither num_operation nor time_step can be None.")
        
        rewards = np.zeros([num_operations, time_steps])
        best_options = np.zeros([num_operations, time_steps])

        for operation in range(num_operations):
            super().reset()
            for time_step in range(time_steps):
                self.proceed()
                option, reward = self.get_latest_option_and_reward()
                rewards[operation, time_step] = reward
                if option == self.best_option:
                    best_options[operation, time_step] = 1
        
        self.__expected_best_option = best_options.mean(axis=0)
        self.__expected_rewards = rewards.mean(axis=0)


num_options = 10
num_operations = 2000
time_steps = 1000
epsilons = [0.0, 0.1, 0.01]
best_options = []
rewards = []

for idx in range(len(epsilons)):
    sms = SlotMachineSimulator(num_options, epsilons[idx])
    sms.run(num_operations, time_steps)
    best_options.append(sms.expected_best_options)
    rewards.append(sms.expected_rewards)
    print(f"[NOTICE] SMS_ver{idx+1} has been done with eps: {epsilons[idx]}")

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.ylim([-1, 1])
for eps, reward in zip(epsilons, rewards):
    plt.plot(reward, label='$\epsilon = %.02f$' % (eps))
plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend()

plt.subplot(2, 1, 2)
plt.ylim([0, 100])
for eps, count in zip(epsilons, best_options):
    plt.plot(count * 100, label='$\epsilon = %.02f$' % (eps))
plt.xlabel('steps')
plt.ylabel('%% optimal action')
plt.legend()

plt.savefig('images/figure_2_2.png')
plt.close()