import gymnasium as gym
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.activations import linear
from collections import deque

# Set hyperparameters
LEARNING_RATE = 0.0001
EPSILON = 1.0
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY = deque(maxlen=32 * 2048)
MIN_EPS = 0.01
UPDATE_EVERY = 4
MAX_EPISODES = 2500


def reset_environment(env, seed=None, render_mode='human'):
    if env is not None:
        env.close()
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    observation, info = env.reset(seed=seed)
    return env, observation


def save_scores_to_csv(scores, episode, file_path="./Data/Linear/"):
    filename = f"{file_path}model_{episode}data.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Episode', 'Score'])  # Writing the headers
        for i, score in enumerate(scores):
            csvwriter.writerow([i, score])  # Writing the score data


def build_model():
    model = Sequential()
    model.add(Dense(units=4, input_dim=8, activation=linear))
    model.compile(loss="mse", optimizer=SGD(learning_rate=LEARNING_RATE))
    return model


def replay_experiences(memory, model):
    if len(memory) < BATCH_SIZE:
        return

    sample_indices = np.random.choice(len(memory), BATCH_SIZE, replace=False)
    mini_batch = [memory[i] for i in sample_indices]

    states = np.array([experience[0][0] for experience in mini_batch])
    actions = np.array([experience[1] for experience in mini_batch])
    next_states = np.array([experience[2][0] for experience in mini_batch])
    rewards = np.array([experience[3] for experience in mini_batch])
    finishes = np.array([experience[4] for experience in mini_batch])

    q_vals_target = model.predict_on_batch(states)
    q_vals_next_state = model.predict_on_batch(next_states)
    max_q_values_next_state = np.amax(q_vals_next_state, axis=1)

    q_vals_target[np.arange(BATCH_SIZE), actions] = rewards + GAMMA * max_q_values_next_state * (1 - finishes)
    model.fit(states, q_vals_target, verbose=0)


def main():
    global EPSILON
    model = build_model()
    np.random.seed(0)

    env = None
    scores = []

    render_every = 100
    for episode in range(1, MAX_EPISODES):
        render_mode = 'human' if episode % render_every == 0 else None
        env, observation = reset_environment(env, seed=0, render_mode=render_mode)
        state = np.reshape(observation, (1, 8))
        score = 0
        step_count = 0

        for t in range(3000):
            if np.random.random() <= EPSILON:
                action = env.action_space.sample()
            else:
                action_values = model.predict(state, verbose=0)
                action = np.argmax(action_values[0])

            observation, reward, terminated, truncated, _ = env.step(action)
            next_state = np.reshape(observation, (1, 8))
            finished = terminated or truncated

            MEMORY.append((state, action, next_state, reward, finished))
            step_count += 1

            if step_count % UPDATE_EVERY == 0:
                replay_experiences(MEMORY, model)

            state = next_state
            score += reward

            if finished:
                break

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode: {episode}, Score: {score}, Average Score: {avg_score}")

        EPSILON = max(MIN_EPS, EPSILON * 0.994)

        if episode % 50 == 0:
            model.save(f"./saved_models/Linear/model_{episode}_episodes.h5")
            save_scores_to_csv(scores, episode)

    if env is not None:
        env.close()


if __name__ == '__main__':
    main()
