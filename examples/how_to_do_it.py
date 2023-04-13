import gym
import torch 


env = gym.make('CartPole-v1')
n_state = env.observation_space.shape[0]
print(n_state)
n_action = env.action_space.n
print(n_action) # Two of actions are defined at CartPole class.

### Function for running each episode
def run_episode(env, weight):
    state = env.reset()[0]
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state, weight))
        state, reward, is_done, truncated, _ = env.step(action.item())
        total_reward += reward
    return total_reward

n_episode = 1000
best_total_reward = 0
best_weight = None
total_rewards = []

### Iteration for finding total best reward and action upon state infomation.
for episode in range(n_episode):
    weight = torch.rand(n_state, n_action)
    total_reward = run_episode(env, weight)
    print('Episode {}: {}'.format(episode+1, total_reward))
    if total_reward > best_total_reward:
        best_weight = weight
        best_total_reward = total_reward
    total_rewards.append(total_reward)
print('Average total reward over {} episode: {}'.format(n_episode, sum(total_rewards) / n_episode))

### Evaluation with result
n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
    total_reward = run_episode(env, best_weight)
    print('Episode {}: {}'.format(episode+1, total_reward))
    total_rewards_eval.append(total_reward)