# Name : Nanayakkara Weragoda Vidanalage Chamod Kanishka Chathuranga
# Student No: M23W0628
import numpy as np
import traci  # Make sure SUMO's TraCI module is installed

# Define traffic lights and their phases
traffic_lights = {
    "joinedS_3150402309_662901136_cluster_4793209901_662901137": list(range(11)),
    "cluster_127658071_3185210184_3185210187": list(range(8)),
    "206595627": list(range(8)),
    "joinedS_2066330708_3150421787_3150421800_635117662": list(range(4))
}

# Initialize Q-table: {state: {traffic_light: [Q-values for each action]}}
q_table = {tl: np.zeros(len(phases)) for tl, phases in traffic_lights.items()}

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
min_epsilon = 0.01  # Minimum exploration rate
num_episodes = 100  # Number of training episodes
max_steps = 1000  # Max steps per episode

def choose_action(state, traffic_light):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.choice(len(traffic_lights[traffic_light]))  # Random action
    return np.argmax(q_table[traffic_light])  # Best action based on Q-values

def simulate_step():
    """Simulate one step in the SUMO environment."""
    traci.simulationStep()

def calculate_reward(traffic_light):
    """Calculate reward based on traffic conditions."""
    reward = 0
    # Penalize waiting time
    for lane in traci.trafficlight.getControlledLanes(traffic_light):
        reward -= traci.lane.getWaitingTime(lane)
    # Penalize teleport events (collisions or jams)
    reward -= 100 * traci.simulation.getStartingTeleportNumber()
    # Encourage higher average speeds
    vehicle_speeds = [traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()]
    if vehicle_speeds:
        reward += sum(vehicle_speeds) / len(vehicle_speeds)
    return reward

def train_q_learning():
    """Train Q-learning for traffic signal control."""
    global epsilon
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        traci.start(["sumo", "-c", "test.sumocfg"])  # Your SUMO configuration file

        done = False
        step = 0
        while not done and step < max_steps:
            step += 1
            # Iterate over all traffic lights
            for tl, phases in traffic_lights.items():
                state = traci.trafficlight.getPhase(tl)  # Current state of traffic light (phase)

                # Ensure that the state is within bounds of the Q-table
                if state >= len(traffic_lights[tl]):
                    state = 0  # Reset state to a valid phase

                action = choose_action(state, tl)  # Choose an action (phase change)
                
                # Set the traffic light phase (action)
                traci.trafficlight.setPhase(tl, action)
                
                simulate_step()  # Step forward in the simulation

                # Reward calculation
                reward = calculate_reward(tl)

                # Q-value update rule (Q-learning)
                next_state = traci.trafficlight.getPhase(tl)  # Next state after taking action
                if next_state >= len(traffic_lights[tl]):
                    next_state = 0  # Reset next state to a valid phase

                future_rewards = np.max(q_table[tl])  # Best future Q-value
                old_value = q_table[tl][state]  # Current Q-value

                # Update Q-value using the Q-learning formula
                q_table[tl][state] = old_value + alpha * (reward + gamma * future_rewards - old_value)

            # Check if simulation is done (no cars left)
            done = traci.simulation.getMinExpectedNumber() <= 0
        
        traci.close()  # Close the simulation after each episode
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1} completed. Epsilon: {epsilon}")

def test_q_learning():
    """Test Q-learning model after training."""
    traci.start(["sumo", "-c", "test.sumocfg"])  # Your SUMO configuration file

    done = False
    step = 0
    while not done and step < max_steps:
        step += 1
        # Iterate over all traffic lights
        for tl, phases in traffic_lights.items():
            state = traci.trafficlight.getPhase(tl)  # Current state of traffic light (phase)

            # Ensure that the state is within bounds of the Q-table
            if state >= len(traffic_lights[tl]):
                state = 0  # Reset state to a valid phase

            # Choose the best action based on Q-values (greedy policy)
            action = np.argmax(q_table[tl])

            # Set the traffic light phase (action)
            traci.trafficlight.setPhase(tl, action)
            
            simulate_step()  # Step forward in the simulation

        # Check if simulation is done (no cars left)
        done = traci.simulation.getMinExpectedNumber() <= 0

    traci.close()  # Close the simulation after testing
    print("Testing completed.")

def main():
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train_q_learning()  # Train the Q-learning model
    elif mode == "test":
        test_q_learning()  # Test the Q-learning model
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")

if __name__ == "__main__":
    main()
