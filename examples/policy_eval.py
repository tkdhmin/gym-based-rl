import torch
import matplotlib.pyplot as plt

# Define the Transistion matrix
t = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                    [[0.7, 0.2, 0.1],
                    [0.1, 0.8, 0.1]],
                    [[0.6, 0.2, 0.2],
                     [0.1, 0.4, 0.5]]]
                     )

# Define the reward function and discount factor, gamma
r = torch.tensor([1.0, 0, -1.0])
gamma = 0.99

# Threshold used to determine when the stop the evaluation process.
threshold = 0.0001

# Define the optimal policy where action a0 is chosen under all circumstances.
policy_optimal = torch.tensor([[1.0, 0.0],
                               [1.0, 0.0],
                               [1.0, 0.0]])

# Evaluation function
def policy_evaluation(policy, trans_matrix, rewards, gamma, threshold):
    """Perform policy evaluation."""

    n_state = policy.shape[0]
    v = torch.zeros(n_state)
    while True:
        v_temp = torch.zeros(n_state)
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                v_temp[state] += action_prob * (rewards[state] + gamma * torch.dot(trans_matrix[state, action], v))
        max_delta = torch.max(torch.abs(v - v_temp))
        v = v_temp.clone()
        if max_delta <= threshold:
            break
    return v

# # Eval history
def policy_eval_history(policy, trans_matrix, rewards, gamma, threshold):
    """Keep track the evaluation history"""
    n_state = policy.shape[0]
    v = torch.zeros(n_state)
    v_history = [v]
    while True:
        v_temp = torch.zeros(n_state)
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                v_temp[state] += action_prob * (rewards[state] + gamma * torch.dot(trans_matrix[state, action], v))
        max_delta = torch.max(torch.abs(v - v_temp))
        v = v_temp.clone()
        v_history.append(v)
        if max_delta <= threshold:
            break
    return v, v_history

v = policy_evaluation(policy_optimal, t, r, gamma, threshold)
print("The value is {}".format(v))

# Experiment with another policy
policy_random = torch.tensor([[0.5, 0.5], 
                              [0.5, 0.5], 
                              [0.5, 0.5]])

v = policy_evaluation(policy_random, t, r, gamma, threshold)
print("The value is {}".format(v))

# Experiment with policy evaluation history
v, v_history = policy_eval_history(policy_optimal, t, r, gamma, threshold)

# Save pyplot result as png
s0, = plt.plot([v_item[0] for v_item in v_history])
s1, = plt.plot([v_item[1] for v_item in v_history])
s2, = plt.plot([v_item[2] for v_item in v_history])
plt.title('Optimal policy with gamma = {}'.format(str(gamma)))
plt.xlabel('Iteration')
plt.ylabel('Policy Values')
plt.legend([s0, s1, s2],
           ["State s0", "State s1", "State s2"],
           loc="upper left")
plt.savefig('policy_history/opt_policy_with_gamma({}).png'.format(str(gamma)), dpi=300)