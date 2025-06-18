# Name : Nanayakkara Weragoda Vidanalage Chamod Kanishka Chathuranga
# Student No: M23W0628
import numpy as np
import traci  # Make sure SUMO's TraCI module is installed
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import multiprocessing

# Define traffic lights and their phases (you can update the phases as per your network)
traffic_lights = {
    "joinedS_3150402309_662901136_cluster_4793209901_662901137": list(range(11)),
    "cluster_127658071_3185210184_3185210187": list(range(8)),
    "206595627": list(range(8)),
    "joinedS_2066330708_3150421787_3150421800_635117662": list(range(4))
}

# DQN parameters
state_size = 12  # Example state size; adjust based on your environment
action_size = max(len(phases) for phases in traffic_lights.values())
gamma = 0.9  # Discount factor
alpha = 0.01  # Learning rate
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.99
batch_size = 128
num_episodes = 100  # Number of training episodes

# Create individual models and replay memories for each traffic light
models = {}
memories = {}

for tl in traffic_lights.keys():
    models[tl] = Sequential([
        Dense(32, input_dim=state_size, activation='relu'),
        Dense(32, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    models[tl].compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    memories[tl] = deque(maxlen=2000)

def choose_action(state, tl):
    """Epsilon-greedy action selection for a specific traffic light."""
    if np.random.rand() <= epsilon:
        return random.choice(traffic_lights[tl])  # Random valid action
    q_values = models[tl].predict(state, verbose=0)
    valid_actions = traffic_lights[tl]
    return valid_actions[np.argmax([q_values[0][a] for a in valid_actions])]

def replay(tl):
    """Replay experiences and train the model for a specific traffic light."""
    if len(memories[tl]) < batch_size:
        return
    minibatch = random.sample(memories[tl], batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(models[tl].predict(next_state, verbose=0)[0])
        target_f = models[tl].predict(state, verbose=0)
        target_f[0][action] = target
        models[tl].fit(state, target_f, epochs=1, verbose=0)

def preprocess_state(tl):
    """Preprocess the state representation."""
    # Example: Encode traffic light state as one-hot vector
    phase = traci.trafficlight.getPhase(tl)
    state = np.zeros(state_size)
    state[phase] = 1
    return np.reshape(state, [1, state_size])

def train_single_traffic_light(tl):
    """Train the DQN model for a specific traffic light."""
    global epsilon
    traci.start(["sumo", "-c", "test.sumocfg"])  # Your SUMO configuration file

    done = False
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1} for traffic light {tl}")
        done = False
        while not done:
            state = preprocess_state(tl)
            action = choose_action(state, tl)
            if action in traffic_lights[tl]:  # Ensure action is within the valid range
                traci.trafficlight.setPhase(tl, action)
            traci.simulationStep()  # Step forward in the simulation

            reward = -1  # Placeholder: Define a real reward function
            next_state = preprocess_state(tl)
            done = traci.simulation.getMinExpectedNumber() <= 0

            memories[tl].append((state, action, reward, next_state, done))
            replay(tl)

    traci.close()  # Close simulation after each episode
    print(f"Episode {episode + 1} completed for traffic light {tl}")
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def train_dqn():
    """Train the DQN models for each traffic light using multiprocessing."""
    processes = []
    for tl in traffic_lights.keys():
        p = multiprocessing.Process(target=train_single_traffic_light, args=(tl,))
        processes.append(p)
        p.start()

    # Ensure all processes finish before closing
    for p in processes:
        p.join()

def test_dqn():
    """Test the trained DQN models for each traffic light."""
    traci.start(["sumo", "-c", "test.sumocfg"])  # Your SUMO configuration file

    done = False
    while not done:
        for tl in traffic_lights.keys():
            state = preprocess_state(tl)
            action = np.argmax(models[tl].predict(state, verbose=0)[0])
            if action in traffic_lights[tl]:  # Ensure action is within the valid range
                traci.trafficlight.setPhase(tl, action)
            traci.simulationStep()

        done = traci.simulation.getMinExpectedNumber() <= 0

    traci.close()  # Close simulation after testing
    print("Testing completed.")

def main():
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train_dqn()  # Train the DQN models using multiprocessing
    elif mode == "test":
        test_dqn()  # Test the DQN models
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")

if __name__ == "__main__":
    main()
