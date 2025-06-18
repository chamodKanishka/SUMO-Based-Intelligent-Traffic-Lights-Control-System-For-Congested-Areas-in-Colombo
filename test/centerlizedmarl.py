# Name : Nanayakkara Weragoda Vidanalage Chamod Kanishka Chathuranga
# Student No: M23W0628
import numpy as np
import traci  # Make sure SUMO's TraCI module is installed
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from multiprocessing import Process, Manager


# Define traffic lights and their phases (you can update the phases as per your network)
traffic_lights = {
    "joinedS_3150402309_662901136_cluster_4793209901_662901137": list(range(11)),
    "cluster_127658071_3185210184_3185210187": list(range(8)),
    "206595627": list(range(8)),
    "joinedS_2066330708_3150421787_3150421800_635117662": list(range(4))
}

# MARL parameters
state_size = 12  # Example state size; adjust based on your environment
action_size = max(len(phases) for phases in traffic_lights.values())
gamma = 0.9  # Discount factor
alpha = 0.01  # Learning rate
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.99
batch_size = 128
num_episodes = 100  # Number of training episodes
memory = deque(maxlen=2000)  # Replay memory

# Build the centralized DQN model
def build_model():
    model = Sequential([
        Dense(64, input_dim=state_size * len(traffic_lights), activation='relu'),
        Dense(64, activation='relu'),
        Dense(action_size * len(traffic_lights), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    return model

dqn_model = build_model()

def choose_actions(states):
    """Epsilon-greedy action selection for all traffic lights."""
    global epsilon
    if np.random.rand() <= epsilon:
        return {tl: random.choice(traffic_lights[tl]) for tl in traffic_lights.keys()}

    q_values = dqn_model.predict(states, verbose=0)
    actions = {}
    for i, tl in enumerate(traffic_lights.keys()):
        valid_actions = traffic_lights[tl]
        actions[tl] = valid_actions[np.argmax([q_values[0][i * action_size + a] for a in valid_actions])]
    return actions

def replay():
    """Replay experiences and train the centralized model."""
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for states, actions, rewards, next_states, done in minibatch:
        target = rewards
        if not done:
            target = rewards + gamma * np.max(dqn_model.predict(next_states, verbose=0)[0])
        target_f = dqn_model.predict(states, verbose=0)
        for i, tl in enumerate(traffic_lights.keys()):
            target_f[0][i * action_size + actions[tl]] = target
        dqn_model.fit(states, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def preprocess_states():
    """Preprocess the state representation for all traffic lights."""
    states = []
    for tl in traffic_lights.keys():
        phase = traci.trafficlight.getPhase(tl)
        state = np.zeros(state_size)
        state[phase] = 1
        states.extend(state)
    return np.reshape(states, [1, state_size * len(traffic_lights)])

def train_episode(episode_num, return_dict):
    """Train the model for a single episode using multiprocessing."""
    print(f"Starting Episode {episode_num + 1}")
    traci.start(["sumo", "-c", "test.sumocfg"])  # Your SUMO configuration file

    done = False
    while not done:
        states = preprocess_states()
        actions = choose_actions(states)

        for tl, action in actions.items():
            if action in traffic_lights[tl]:  # Ensure action is within the valid range
                traci.trafficlight.setPhase(tl, action)
        traci.simulationStep()  # Step forward in the simulation

        rewards = -1  # Placeholder: Define a real reward function
        next_states = preprocess_states()
        done = traci.simulation.getMinExpectedNumber() <= 0

        memory.append((states, actions, rewards, next_states, done))
        replay()

    traci.close()  # Close simulation after each episode
    return_dict[episode_num] = "Completed"  # Store status of episode

def train_marl():
    """Train the centralized MARL model using multiprocessing."""
    manager = Manager()
    return_dict = manager.dict()

    processes = []
    for episode in range(num_episodes):
        p = Process(target=train_episode, args=(episode, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to finish

    print("Training completed.")

def test_marl():
    """Test the centralized MARL model."""
    traci.start(["sumo", "-c", "test.sumocfg"])  # Your SUMO configuration file

    done = False
    while not done:
        states = preprocess_states()
        q_values = dqn_model.predict(states, verbose=0)
        actions = {}
        for i, tl in enumerate(traffic_lights.keys()):
            valid_actions = traffic_lights[tl]
            actions[tl] = valid_actions[np.argmax([q_values[0][i * action_size + a] for a in valid_actions])]

        for tl, action in actions.items():
            if action in traffic_lights[tl]:  # Ensure action is within the valid range
                traci.trafficlight.setPhase(tl, action)
        traci.simulationStep()

        done = traci.simulation.getMinExpectedNumber() <= 0

    traci.close()  # Close simulation after testing
    print("Testing completed.")

def main():
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train_marl()  # Train the MARL model
    elif mode == "test":
        test_marl()  # Test the MARL model
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")

if __name__ == "__main__":
    main()
