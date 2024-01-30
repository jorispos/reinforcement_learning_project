import numpy as np
import keras
from keras.activations import relu, linear
import lunar_lander as lander
from collections import deque
import gym
import random
from keras.utils import to_categorical

# Set hyperparameters
learning_rate = 0.001
epsilon = 1
gamma = .99
batch_size = 64
memory = deque(maxlen=1000000)
min_eps = 0.01

# Define the model architecture
model = keras.Sequential()
model.add(keras.layers.Dense(64, input_dim=8, activation=relu))
model.add(keras.layers.Dense(64, activation=relu))
model.add(keras.layers.Dense(4, activation=linear))
model.compile(loss="mse", optimizer=keras.optimizers.adam(lr=learning_rate))

def replay_experiences():
    if len(memory) >= batch_size:
        # Randomly sample experiences from memory
        sample_indices = np.random.choice(len(memory), batch_size)
        mini_batch = np.array(memory)[sample_indices]

        # Extract states, actions, next states, rewards, and finishes from mini batch
        states, actions, next_states, rewards, finishes = zip(*mini_batch)
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        finishes = np.array(finishes)

        # Reshape states and next states
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Compute Q-values for Q-values to update (target Q-values)
        q_vals_target = model.predict_on_batch(states)

        # Compute max Q-values for next states
        q_vals_next_state = model.predict_on_batch(next_states)
        max_q_values_next_state = np.amax(q_vals_next_state, axis=1)

        # Update target Q-values using Bellman equation
        q_vals_target[np.arange(batch_size), actions] = rewards + gamma * (max_q_values_next_state) * (1 - finishes)

        # Train the model using states and target Q-values
        model.fit(states, q_vals_target, verbose=0)

        # Update epsilon
        global epsilon
        if epsilon > min_eps:
            epsilon *= 0.996


if __name__ == '__main__':
    env = lander.LunarLander()  # Create the LunarLander environment
    num_episodes = 400  # Number of episodes to run
    np.random.seed(0)  # Set the random seed for reproducibility
    scores = []  # List to store the scores of each episode

    for i in range(num_episodes + 1):
        score = 0  # Initialize the score for the current episode
        state = env.reset()  # Reset the environment and get the initial state
        finished = False  # Flag to indicate if the episode is finished

        if i != 0 and i % 50 == 0:
            model.save(".\saved_models\model_" + str(i) + "_episodes.h5")  # Save the model every 50 episodes

        for j in range(3000):
            state = np.reshape(state, (1, 8))  # Reshape the state to match the input shape of the model

            if np.random.random() <= epsilon:
                action = np.random.choice(4)  # Choose a random action with probability epsilon
            else:
                action_values = model.predict(state)  # Get the predicted action values from the model
                action = np.argmax(action_values[0])  # Choose the action with the highest predicted value

            env.render()  # Render the environment
            next_state, reward, finished, metadata = env.step(action)  # Take the chosen action and get the next state, reward, and finished flag
            next_state = np.reshape(next_state, (1, 8))  # Reshape the next state

            memory.append((state, action, next_state, reward, finished))  # Add the experience to the memory
            replay_experiences()  # Replay experiences from memory and update the model

            score += reward  # Update the score
            state = next_state  # Update the current state

            if finished:
                scores.append(score)  # Add the score to the list of scores
                avg_score = np.mean(scores[-100:])  # Calculate the average score over the last 100 episodes
                print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, avg_score))  # Print the episode number, score, and average score
                break
