# --- PART 1: IMPORTS ---
# Python libraries
import torch
import numpy as np
import simpy
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
import random # <-- This import is included

# Your other .py files
from dqn_agent import Agent                   # <-- From dqn_agent.py
from simulation import setup_shop_simulation  # <-- From simulation.py

# --- PART 2: JobShopEnv CLASS ---
class JobShopEnv(gym.Env):
    """
    A custom Gymnasium Environment that wraps the SimPy simulation.
    """
    def __init__(self, shop_config):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        # Use np.float32 for consistency with PyTorch
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # Store the config to pass to the simulation on reset
        self.shop_config = shop_config 
        
    def reset(self, seed=None, options=None):
        # We need to set the seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.env = simpy.Environment()
        self.shop = setup_shop_simulation(self.env, self, self.shop_config) 
        
        self.env.run(until=self.shop.next_decision_point)
        initial_state = self._get_state()
        info = {}
        return initial_state, info

    def step(self, action):
        self.shop.resolve_decision(action)
        self.env.run(until=self.shop.next_decision_point)
        
        new_state = self._get_state()
        reward = self.shop.calculate_reward()
        done = self.shop.is_finished
        
        truncated = False 
        info = {}
        
        return new_state, reward, done, truncated, info

    def _get_state(self):
        # get_feature_vector() already returns a np.float32 array.
        return self.shop.get_feature_vector()


# --- PART 3: TRAINING SCRIPT ---
def train(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning."""
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    print("Starting training...")
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset(seed=i_episode) # Pass seed for reproducibility
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated:
                break 
                
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
    print("Training finished.")
    return scores

# --- PART 4: MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # Define your shop configuration
    SHOP_CONFIG = {
        'machines': 3,
        'job_arrival_rate': 0.8,
        'processing_time_mean': 10,
        'due_date_factor': 3
    }
    
    print("Initializing environment and agent...")
    
    env = JobShopEnv(shop_config=SHOP_CONFIG)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Pass the seed for reproducible results
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    # Start the training
    scores = train(env, agent)
    
    # Save the model
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_model.pth')
    print("Model saved to dqn_model.pth")
    
    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score (Cumulative Reward)')
    plt.xlabel('Episode #')
    plt.title('DQN Training Scores')
    plt.savefig('results/training_scores.png')
    plt.close(fig) # Close the figure to free memory
    

    print("Training plot saved to training_scores.png")
