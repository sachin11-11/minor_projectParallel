"""
Standalone graph generation script.
Can regenerate all graphs from saved training metrics without retraining.
Usage: python3 generate_graphs.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config


def load_metrics():
    """Load training metrics from the checkpoint directory."""
    metrics_path = os.path.join(config.CHECKPOINT_DIR, "training_metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"ERROR: Metrics file not found at {metrics_path}")
        print("Please run train_dqn.py first.")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Loaded metrics from {metrics_path}")
    print(f"  Episodes trained: {metrics['last_episode']}")
    print(f"  Evaluation points: {len(metrics['eval_episodes'])}")
    
    return metrics


def generate_queue_length_graph(eval_episodes, eval_avg_queues):
    """Generate AVG QUEUE LENGTH VS EPISODES graph."""
    plt.figure(figsize=(14, 7))
    plt.plot(eval_episodes, eval_avg_queues, 'b-o', markersize=2, linewidth=1.5,
             label='Avg Queue Length')
    
    # Add trend line
    if len(eval_episodes) > 10:
        z = np.polyfit(eval_episodes, eval_avg_queues, 3)
        p = np.poly1d(z)
        plt.plot(eval_episodes, p(eval_episodes), 'r--', linewidth=2, alpha=0.7, label='Trend')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Queue Length', fontsize=12)
    plt.title('AVG QUEUE LENGTH VS EPISODES', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(config.GRAPH_DIR, "AVG_QUE_LENGTH_VS_EPISODES.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_waiting_time_graph(eval_episodes, eval_avg_waits):
    """Generate AVG WAITING TIME VS EPISODES graph."""
    plt.figure(figsize=(14, 7))
    plt.plot(eval_episodes, eval_avg_waits, 'g-o', markersize=2, linewidth=1.5,
             label='Avg Waiting Time')
    
    if len(eval_episodes) > 10:
        z = np.polyfit(eval_episodes, eval_avg_waits, 3)
        p = np.poly1d(z)
        plt.plot(eval_episodes, p(eval_episodes), 'r--', linewidth=2, alpha=0.7, label='Trend')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Waiting Time (seconds)', fontsize=12)
    plt.title('AVG WAITING TIME VS EPISODES', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(config.GRAPH_DIR, "AVG_WAITING_TIME_VS_EPISODES.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_reward_graphs(eval_episodes, eval_rewards):
    """Generate reward graphs at various milestones in REWARD_GRAPH subfolder."""
    os.makedirs(config.REWARD_GRAPH_DIR, exist_ok=True)
    
    milestones = [10, 20, 50, 100, 200, 300, 400, 500]
    
    for milestone in milestones:
        # Find how many eval points exist up to this milestone
        ep_idx = sum(1 for ep in eval_episodes if ep <= milestone)
        
        if ep_idx == 0:
            continue
        
        plot_episodes = eval_episodes[:ep_idx]
        plot_rewards = eval_rewards[:ep_idx]
        
        plt.figure(figsize=(14, 7))
        plt.plot(plot_episodes, plot_rewards, 'b-o', markersize=3, linewidth=1.5,
                 label='Episode Reward')
        
        # Cumulative average
        cumulative_avg = np.cumsum(plot_rewards) / np.arange(1, len(plot_rewards) + 1)
        plt.plot(plot_episodes, cumulative_avg, 'r-', linewidth=2, alpha=0.7,
                 label='Cumulative Average')
        
        # Moving average
        if len(plot_rewards) >= 5:
            window = min(10, len(plot_rewards))
            moving_avg = np.convolve(plot_rewards, np.ones(window)/window, mode='valid')
            ma_episodes = plot_episodes[window-1:]
            plt.plot(ma_episodes, moving_avg, 'g--', linewidth=2, alpha=0.7,
                     label=f'{window}-Point Moving Avg')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.title(f'REWARD FUNCTION (Episode 2 to Episode {plot_episodes[-1]})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        save_path = os.path.join(config.REWARD_GRAPH_DIR, f"reward_up_to_ep{milestone}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


def generate_training_rewards_graph(episode_rewards):
    """Generate overall training rewards graph."""
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, alpha=0.6,
             label='Episode Reward', linewidth=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards per Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    window_size = 20
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size, len(episode_rewards) + 1), moving_avg,
                 label=f'{window_size}-Episode Moving Average', color='red', linewidth=2)
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, alpha=0.3,
             label='Episode Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards with Moving Average')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(config.GRAPH_DIR, "training_rewards.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Generate all graphs from saved metrics."""
    print("=" * 70)
    print("GRAPH GENERATION")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    os.makedirs(config.REWARD_GRAPH_DIR, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics()
    if metrics is None:
        return
    
    eval_episodes = metrics['eval_episodes']
    eval_avg_queues = metrics['eval_avg_queues']
    eval_avg_waits = metrics['eval_avg_waits']
    eval_rewards = metrics['eval_rewards']
    episode_rewards = metrics['episode_rewards']
    
    # Generate graphs
    print("\nGenerating graphs...")
    
    print("\n1. Average Queue Length vs Episodes:")
    generate_queue_length_graph(eval_episodes, eval_avg_queues)
    
    print("\n2. Average Waiting Time vs Episodes:")
    generate_waiting_time_graph(eval_episodes, eval_avg_waits)
    
    print("\n3. Reward Graphs (milestones):")
    generate_reward_graphs(eval_episodes, eval_rewards)
    
    print("\n4. Training Rewards:")
    generate_training_rewards_graph(episode_rewards)
    
    print("\n" + "=" * 70)
    print("ALL GRAPHS GENERATED SUCCESSFULLY!")
    print(f"Output directory: {config.GRAPH_DIR}/")
    print(f"Reward graphs: {config.REWARD_GRAPH_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
