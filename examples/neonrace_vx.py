import gym
env = gym.make('flashgames.NeonRace-v0', render_mode="human")
env.reset()

for episode in range(1000):
    observation = env.reset() # Init the environment for each episode
    for time in range(10000):
        env.render()
        print(observation)
        action = env.action_space.sample() # Pick an one of the actions from the defined action space.
        observation, reward, terminated_done,_, info = env.step(action) # Store the observation, reward, done, and info vars.
        # The observation means the object that indicates the observation of the environment.
        # The reward means the bonus reward from the previous action.
        # The done ia a boolean-typed variable that indicates whether the episode has been done or not.
        # The info is nothing but debugging and logging info.

        if terminated_done:
            print(f"{time} timesteps taken for the episode.")
            break
