"""
Evaluate trained DQN agent vs Fixed-Time control.
Runs both controllers on a full-day simulation and generates comparison graphs.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sumo_env import SumoEnvironment
from dqn_agent import DQNAgent
import config


def run_fixed_time_control(env, flow_file):
    """
    Run full-day simulation with fixed-time traffic light control.
    Cycles through phases with fixed durations.
    
    Returns:
        avg_queue: Average queue length over the episode
        avg_wait: Average waiting time over the episode
        total_reward: Total reward accumulated
        step_rewards: List of rewards at each step
    """
    env.set_flow_file(flow_file)
    state = env.reset()
    rewards = []
    phase_duration = config.GREEN_PHASE_DURATION  # 15 seconds per phase
    current_phase = 0
    time_in_phase = 0
    
    done = False
    while not done:
        # Fixed-time: switch phase every phase_duration
        if time_in_phase >= phase_duration:
            current_phase = (current_phase + 1) % config.NUM_ACTIONS
            time_in_phase = 0
        
        next_state, reward, done, info = env.step(current_phase)
        rewards.append(reward)
        state = next_state
        time_in_phase += config.ACTION_DURATION
    
    avg_queue = env.get_episode_avg_queue_length()
    avg_wait = env.get_episode_avg_waiting_time()
    total_reward = sum(rewards)
    
    return avg_queue, avg_wait, total_reward, rewards


def run_dqn_control(env, agent, flow_file):
    """
    Run full-day simulation with trained DQN agent.
    
    Returns:
        avg_queue: Average queue length over the episode
        avg_wait: Average waiting time over the episode
        total_reward: Total reward accumulated
        step_rewards: List of rewards at each step
    """
    env.set_flow_file(flow_file)
    state = env.reset()
    rewards = []
    
    done = False
    while not done:
        action = agent.act(state, training=False)  # Greedy
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        state = next_state
    
    avg_queue = env.get_episode_avg_queue_length()
    avg_wait = env.get_episode_avg_waiting_time()
    total_reward = sum(rewards)
    
    return avg_queue, avg_wait, total_reward, rewards


def plot_comparison(fixed_results, dqn_results, save_dir):
    """
    Create comparison graphs for Fixed-Time vs DQN control.
    
    Args:
        fixed_results: dict with keys: avg_queue, avg_wait, total_reward, step_rewards
        dqn_results: dict with keys: avg_queue, avg_wait, total_reward, step_rewards
        save_dir: Directory to save the graphs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ========================================
    # 1. Bar chart: Avg Queue Length, Avg Waiting Time, Reward
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    categories = ['Fixed-Time', 'DQN Agent']
    colors = ['#3498db', '#e74c3c']
    
    # Avg Queue Length
    ax = axes[0]
    values = [fixed_results['avg_queue'], dqn_results['avg_queue']]
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Average Queue Length', fontsize=12)
    ax.set_title('Average Queue Length Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Avg Waiting Time
    ax = axes[1]
    values = [fixed_results['avg_wait'], dqn_results['avg_wait']]
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Average Waiting Time (s)', fontsize=12)
    ax.set_title('Average Waiting Time Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Total Reward
    ax = axes[2]
    values = [fixed_results['total_reward'], dqn_results['total_reward']]
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Reward Function Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "DQN_vs_FixedTime_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # ========================================
    # 2. Step-by-step reward comparison
    # ========================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Rewards over time
    ax = axes[0]
    fixed_rewards = fixed_results['step_rewards']
    dqn_rewards = dqn_results['step_rewards']
    
    ax.plot(range(len(fixed_rewards)), fixed_rewards, alpha=0.5, linewidth=0.8,
            label='Fixed-Time Control', color='#3498db')
    ax.plot(range(len(dqn_rewards)), dqn_rewards, alpha=0.5, linewidth=0.8,
            label='DQN Control', color='#e74c3c')
    ax.set_xlabel('Action Step', fontsize=12)
    ax.set_ylabel('Step Reward', fontsize=12)
    ax.set_title('Step-by-Step Reward Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Cumulative rewards
    ax = axes[1]
    fixed_cumul = np.cumsum(fixed_rewards)
    dqn_cumul = np.cumsum(dqn_rewards)
    ax.plot(range(len(fixed_cumul)), fixed_cumul, linewidth=2,
            label='Fixed-Time Control', color='#3498db')
    ax.plot(range(len(dqn_cumul)), dqn_cumul, linewidth=2,
            label='DQN Control', color='#e74c3c')
    ax.set_xlabel('Action Step', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Cumulative Reward Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "DQN_vs_FixedTime_rewards_over_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Compare fixed-time control vs trained DQN control."""
    
    print("=" * 70)
    print("TRAFFIC LIGHT CONTROL COMPARISON")
    print("Fixed-Time vs Trained DQN Agent")
    print("=" * 70)
    
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    
    # Use Day 1 flow for evaluation
    eval_flow = config.FLOW_FILES[0]
    print(f"\nEvaluation flow file: {eval_flow}")
    print(f"Episode length: {config.MAX_STEPS_PER_EPISODE} seconds (full day)")
    
    # ===== Run Fixed-Time Control =====
    print("\n[1/2] Running Fixed-Time Control (full day)...")
    env_fixed = SumoEnvironment(use_gui=False)
    fixed_queue, fixed_wait, fixed_reward, fixed_step_rewards = run_fixed_time_control(
        env_fixed, eval_flow)
    env_fixed.close()
    
    print(f"  Avg Queue Length: {fixed_queue:.2f}")
    print(f"  Avg Waiting Time: {fixed_wait:.2f}s")
    print(f"  Total Reward: {fixed_reward:.2f}")
    
    fixed_results = {
        'avg_queue': fixed_queue,
        'avg_wait': fixed_wait,
        'total_reward': fixed_reward,
        'step_rewards': fixed_step_rewards,
    }
    
    # ===== Run DQN Control =====
    print("\n[2/2] Running DQN Control (full day)...")
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"  ERROR: Trained model not found at {config.MODEL_SAVE_PATH}")
        print("  Please run train_dqn.py first to train the model.")
        return
    
    env_dqn = SumoEnvironment(use_gui=False)
    env_dqn.set_flow_file(eval_flow)
    state = env_dqn.reset()
    state_size = env_dqn.get_state_size()
    action_size = env_dqn.get_action_size()
    env_dqn.close()
    
    agent = DQNAgent(state_size, action_size)
    agent.load(config.MODEL_SAVE_PATH)
    
    env_dqn2 = SumoEnvironment(use_gui=False)
    dqn_queue, dqn_wait, dqn_reward, dqn_step_rewards = run_dqn_control(
        env_dqn2, agent, eval_flow)
    env_dqn2.close()
    
    print(f"  Avg Queue Length: {dqn_queue:.2f}")
    print(f"  Avg Waiting Time: {dqn_wait:.2f}s")
    print(f"  Total Reward: {dqn_reward:.2f}")
    
    dqn_results = {
        'avg_queue': dqn_queue,
        'avg_wait': dqn_wait,
        'total_reward': dqn_reward,
        'step_rewards': dqn_step_rewards,
    }
    
    # ===== Generate Comparison Graphs =====
    print("\n[3/3] Generating comparison graphs...")
    plot_comparison(fixed_results, dqn_results, config.GRAPH_DIR)
    
    # ===== Print Summary =====
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Fixed-Time':>15} {'DQN Agent':>15} {'Improvement':>15}")
    print("-" * 70)
    
    queue_improvement = ((fixed_queue - dqn_queue) / fixed_queue) * 100 if fixed_queue > 0 else 0
    wait_improvement = ((fixed_wait - dqn_wait) / fixed_wait) * 100 if fixed_wait > 0 else 0
    reward_improvement = ((dqn_reward - fixed_reward) / abs(fixed_reward)) * 100 if fixed_reward != 0 else 0
    
    print(f"{'Avg Queue Length':<25} {fixed_queue:>15.2f} {dqn_queue:>15.2f} {queue_improvement:>14.1f}%")
    print(f"{'Avg Waiting Time (s)':<25} {fixed_wait:>15.2f} {dqn_wait:>15.2f} {wait_improvement:>14.1f}%")
    print(f"{'Total Reward':<25} {fixed_reward:>15.2f} {dqn_reward:>15.2f} {reward_improvement:>14.1f}%")
    print("=" * 70)
    
    # Save results as JSON
    results = {
        'fixed_time': {
            'avg_queue_length': float(fixed_queue),
            'avg_waiting_time': float(fixed_wait),
            'total_reward': float(fixed_reward),
        },
        'dqn_agent': {
            'avg_queue_length': float(dqn_queue),
            'avg_waiting_time': float(dqn_wait),
            'total_reward': float(dqn_reward),
        },
        'improvements': {
            'queue_length_pct': float(queue_improvement),
            'waiting_time_pct': float(wait_improvement),
            'reward_pct': float(reward_improvement),
        }
    }
    
    results_path = os.path.join(config.GRAPH_DIR, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
