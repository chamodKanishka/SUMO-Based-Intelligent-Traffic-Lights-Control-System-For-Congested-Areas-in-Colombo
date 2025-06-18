# Name : Nanayakkara Weragoda Vidanalage Chamod Kanishka Chathuranga
# Student No: M23W0628
import numpy as np
import traci  # Make sure SUMO's TraCI module is installed
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from multiprocessing import Process, Queue

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
memory = deque(maxlen=2000)  # Replay memory

# Build the DQN model
def build_model():
    model = Sequential([
        Dense(32, input_dim=state_size, activation='relu'),
        Dense(32, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    return model

dqn_model = build_model()

def choose_action(state, tl):
    """Epsilon-greedy action selection for a specific traffic light."""
    if np.random.rand() <= epsilon:
        return random.choice(traffic_lights[tl])  # Random valid action
    q_values = dqn_model.predict(state, verbose=0)
    valid_actions = traffic_lights[tl]
    return valid_actions[np.argmax([q_values[0][a] for a in valid_actions])]

def replay():
    """Replay experiences and train the model."""
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(dqn_model.predict(next_state, verbose=0)[0])
        target_f = dqn_model.predict(state, verbose=0)
        target_f[0][action] = target
        dqn_model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def preprocess_state(tl):
    """Preprocess the state representation."""
    # Example: Encode traffic light state as one-hot vector
    phase = traci.trafficlight.getPhase(tl)
    state = np.zeros(state_size)
    state[phase] = 1
    return np.reshape(state, [1, state_size])

def train_traffic_light(tl, q):
    """Train a specific traffic light."""
    for episode in range(num_episodes):
        traci.start(["sumo", "-c", "test.sumocfg"])

        done = False
        while not done:
            state = preprocess_state(tl)
            action = choose_action(state, tl)
            if action in traffic_lights[tl]:
                traci.trafficlight.setPhase(tl, action)
            traci.simulationStep()

            reward = -1  # Placeholder: Define a real reward function
            next_state = preprocess_state(tl)
            done = traci.simulation.getMinExpectedNumber() <= 0

            memory.append((state, action, reward, next_state, done))
            replay()

        traci.close()
    q.put(f"Training completed for {tl}")

def train_dqn():
    """Train the DQN model using multiprocessing."""
    processes = []
    queue = Queue()
    for tl in traffic_lights.keys():
        p = Process(target=train_traffic_light, args=(tl, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        print(queue.get())

def test_traffic_light(tl):
    """Test a specific traffic light."""
    traci.start(["sumo", "-c", "test.sumocfg"])

    done = False
    while not done:
        state = preprocess_state(tl)
        action = np.argmax(dqn_model.predict(state, verbose=0)[0])
        if action in traffic_lights[tl]:
            traci.trafficlight.setPhase(tl, action)
        traci.simulationStep()

        done = traci.simulation.getMinExpectedNumber() <= 0

    traci.close()
    print(f"Testing completed for {tl}")

def test_dqn():
    """Test the trained DQN model using multiprocessing."""
    processes = []
    for tl in traffic_lights.keys():
        p = Process(target=test_traffic_light, args=(tl,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def main():
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train_dqn()
    elif mode == "test":
        test_dqn()
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")

if __name__ == "__main__":
    main()
