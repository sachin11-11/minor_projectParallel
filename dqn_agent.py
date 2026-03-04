import os
import numpy as np
import random
import pickle
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

import config


class DQNetwork(nn.Module):
    """Deep Q-Network neural network."""
    
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def size(self):
        """Return current size of the buffer."""
        return len(self.buffer)
    
    def get_data(self):
        """Get all data from the buffer for serialization."""
        return list(self.buffer)
    
    def load_data(self, data):
        """Load data into the buffer."""
        self.buffer = deque(data, maxlen=self.buffer.maxlen)


class DQNAgent:
    """DQN Agent for traffic light control."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        
        # Q-Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNetwork(state_size, action_size).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; otherwise, use greedy
        
        Returns:
            action: Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the Q-network using experience replay."""
        if self.replay_buffer.size() < self.batch_size:
            return 0.0  # Not enough samples to train
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save the Q-network."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def save_checkpoint(self, filepath, episode, metrics, replay_data=None):
        """
        Save a full training checkpoint with metrics and replay buffer.
        
        Args:
            filepath: Path to save the checkpoint (.pth file)
            episode: Current episode number (1-indexed)
            metrics: Dictionary containing training metrics
            replay_data: Optional replay buffer data (list of experiences)
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'metrics': metrics,
        }
        torch.save(checkpoint, filepath)
        
        # Save replay buffer separately (can be large)
        if replay_data is not None:
            buffer_path = filepath.replace('.pth', '_buffer.pkl')
            with open(buffer_path, 'wb') as f:
                pickle.dump(replay_data, f)
        
        print(f"Checkpoint saved to {filepath} (episode {episode})")
    
    def load_checkpoint(self, filepath):
        """
        Load a full training checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            episode: The episode number to resume from
            metrics: Dictionary containing training metrics
        """
        checkpoint = torch.load(filepath, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        episode = checkpoint.get('episode', 0)
        metrics = checkpoint.get('metrics', {})
        
        # Try to load replay buffer
        buffer_path = filepath.replace('.pth', '_buffer.pkl')
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                self.replay_buffer.load_data(buffer_data)
                print(f"Replay buffer loaded ({self.replay_buffer.size()} experiences)")
            except Exception as e:
                print(f"Warning: Could not load replay buffer: {e}")
        
        print(f"Checkpoint loaded from {filepath} (episode {episode})")
        return episode, metrics
    
    def load(self, filepath):
        """Load the Q-network."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")
