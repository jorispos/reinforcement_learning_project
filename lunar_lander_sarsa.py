import gymnasium as gym
import numpy as np
import collections

def reset_environment(render=False):
    global env, observation, info
    if env is not None:
        env.close()
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2", render_mode=None)
    observation, info = env.reset(seed=42)

def state_extractor(s):
    # Modified state extractor based on the example
    return (min(5, max(-5, int(s[0] / 0.05))),
            min(5, max(-1, int(s[1] / 0.1))),
            min(3, max(-3, int(s[2] / 0.1))),
            min(3, max(-3, int(s[3] / 0.1))),
            min(3, max(-3, int(s[4] / 0.1))),
            min(3, max(-3, int(s[5] / 0.1))),
            int(s[6]),
            int(s[7]))

def lr_scheduler(episode):
    # Learning rate scheduler
    return max(0.3 * np.exp(-episode / 10000), 0.01)

def choose_action(state, Q, episode):
    # Adaptive exploration policy based on the episode number
    threshold = 50
    if episode > 200:
        threshold = 10
    if episode > 2000:
        threshold = 5
    if episode > 5000:
        threshold = 1
    if episode > 7500:
        threshold = 0

    if np.random.randint(0, 100) >= threshold:
        return np.argmax(Q[str(state)])
    else:
        return env.action_space.sample()

# Initialize Q-table and environment
Q = collections.defaultdict(float)
gamma = 0.95  # Discount factor
num_episodes = 10000  # Total episodes for training
render_every = 1000   # Render the environment every 10000 episodes
env = None  # Initialize the environment variable

for episode in range(num_episodes):
    total_reward = 0
    steps = 0
    alpha = lr_scheduler(episode)  # Learning rate

    # Reset environment with or without rendering
    reset_environment(render=(episode % render_every == 0))

    while True:
        state = state_extractor(observation)
        action = choose_action(state, Q, episode)
        new_observation, reward, terminated, truncated, info = env.step(action)
        new_state = state_extractor(new_observation)
        new_action = choose_action(new_state, Q, episode)

        # SARSA update rule
        sa = str(state) + " " + str(action)
        new_sa = str(new_state) + " " + str(new_action)
        Q[sa] += alpha * (reward + gamma * Q[new_sa] - Q[sa])

        total_reward += reward
        steps += 1
        observation = new_observation

        if terminated or truncated:
            break

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}")

# Close the environment after training is done
if env is not None:
    env.close()