import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime

from sumo_env import SumoEnvironment
from dqn_agent import DQNAgent
import config


def evaluate_agent(env, agent, flow_file):
    """
    Run one evaluation episode with greedy policy.
    Returns avg queue length, avg waiting time, and total reward.
    """
    env.set_flow_file(flow_file)
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state, training=False)  # Greedy policy
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
    
    avg_queue = env.get_episode_avg_queue_length()
    avg_wait = env.get_episode_avg_waiting_time()
    
    return avg_queue, avg_wait, total_reward


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                   if f.startswith("checkpoint_ep") and f.endswith(".pth")]
    
    if not checkpoints:
        return None
    
    # Sort by episode number
    checkpoints.sort(key=lambda x: int(x.replace("checkpoint_ep", "").replace(".pth", "")))
    latest = checkpoints[-1]
    return os.path.join(checkpoint_dir, latest)


def train():
    """Main training loop for DQN traffic light control."""
    
    # Create directories for outputs
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    os.makedirs(config.REWARD_GRAPH_DIR, exist_ok=True)
    
    # Initialize environment and agent
    print("Initializing SUMO environment...")
    env = SumoEnvironment(use_gui=config.SUMO_GUI)
    
    # Use Day 1 flow for initial reset to get state/action sizes
    env.set_flow_file(config.FLOW_FILES[0])
    initial_state = env.reset()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Incoming lanes: {len(env.incoming_lanes)}")
    print(f"Outgoing lanes: {len(env.outgoing_lanes)}")
    env.close()
    
    # Initialize DQN agent
    print("Initializing DQN agent...")
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    eval_episodes = []        # Episode numbers where evaluation was done
    eval_avg_queues = []      # Avg queue length at each evaluation
    eval_avg_waits = []       # Avg waiting time at each evaluation
    eval_rewards = []         # Total reward at each evaluation
    start_episode = 0
    
    # Check for existing checkpoints to resume
    latest_checkpoint = find_latest_checkpoint(config.CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}. Loading...")
        start_episode, metrics = agent.load_checkpoint(latest_checkpoint)
        
        # Restore metrics
        episode_rewards = metrics.get('episode_rewards', [])
        episode_losses = metrics.get('episode_losses', [])
        eval_episodes = metrics.get('eval_episodes', [])
        eval_avg_queues = metrics.get('eval_avg_queues', [])
        eval_avg_waits = metrics.get('eval_avg_waits', [])
        eval_rewards = metrics.get('eval_rewards', [])
        
        print(f"Resuming from episode {start_episode + 1}...")
        print(f"  Loaded {len(episode_rewards)} episode rewards")
        print(f"  Loaded {len(eval_episodes)} evaluation results")
    
    print(f"\nStarting training for {config.EPISODES} episodes...")
    print(f"Each episode = {config.MAX_STEPS_PER_EPISODE} simulation seconds")
    print(f"Cycling through Day 1-4 flow files")
    print(f"Checkpoint every {config.CHECKPOINT_INTERVAL} episodes")
    print(f"Evaluation every {config.TEST_INTERVAL} episodes")
    print("=" * 70)
    
    for episode in range(start_episode, config.EPISODES):
        # Calculate slices per day to advance time across the day dynamically
        slices_per_day = 59400 // config.MAX_STEPS_PER_EPISODE
        if slices_per_day == 0: slices_per_day = 1
        
        # Select flow file and time slice
        day_idx = (episode // slices_per_day) % len(config.FLOW_FILES)
        slice_idx = episode % slices_per_day
        start_time = slice_idx * config.MAX_STEPS_PER_EPISODE
        
        flow_file = config.FLOW_FILES[day_idx]
        day_num = day_idx + 1
        
        # Format the slice hours for nice printing
        start_hr = start_time // 3600
        end_hr = (start_time + config.MAX_STEPS_PER_EPISODE) // 3600
        slice_hours = f"{start_hr:02d}:00-{end_hr:02d}:00"
        
        # Reset environment with the selected flow and advance SUMO clock
        env.set_flow_file(flow_file)
        state = env.reset(start_time=start_time)
        episode_reward = 0
        episode_loss = []
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{config.EPISODES} [Day {day_num} | Time {slice_hours}] | Epsilon: {agent.epsilon:.3f}")
        
        done = False
        while not done:
            # Select action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Print progress every 500 steps
            if step_count % 500 == 0:
                print(f"  Step {step_count} | Sim time: {info['step_count']}s | "
                      f"Queue: {info['queue_length']:.1f} | "
                      f"Wait: {info['waiting_time']:.1f}s")
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network periodically
        if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Record training metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # Get episode-level metrics from env
        train_avg_queue = env.get_episode_avg_queue_length()
        train_avg_wait = env.get_episode_avg_waiting_time()
        
        print(f"  COMPLETED | Reward: {episode_reward:.2f} | "
              f"Loss: {avg_loss:.4f} | Steps: {step_count} | "
              f"Avg Queue: {train_avg_queue:.2f} | Avg Wait: {train_avg_wait:.2f}s")
        
        # Close env after training episode
        env.close()
        
        # ===== EVALUATION every TEST_INTERVAL episodes =====
        if (episode + 1) % config.TEST_INTERVAL == 0:
            print(f"\n  >>> EVALUATING at episode {episode + 1}...")
            
            # Evaluate on Day 1 flow (consistent evaluation baseline)
            eval_flow = config.FLOW_FILES[0]
            eval_env = SumoEnvironment(use_gui=False)
            avg_q, avg_w, eval_reward = evaluate_agent(eval_env, agent, eval_flow)
            eval_env.close()
            
            eval_episodes.append(episode + 1)
            eval_avg_queues.append(avg_q)
            eval_avg_waits.append(avg_w)
            eval_rewards.append(eval_reward)
            
            print(f"  >>> EVAL RESULT | Avg Queue: {avg_q:.2f} | "
                  f"Avg Wait: {avg_w:.2f}s | Reward: {eval_reward:.2f}")
        
        # ===== CHECKPOINT every CHECKPOINT_INTERVAL episodes =====
        if (episode + 1) % config.CHECKPOINT_INTERVAL == 0:
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_losses': episode_losses,
                'eval_episodes': eval_episodes,
                'eval_avg_queues': eval_avg_queues,
                'eval_avg_waits': eval_avg_waits,
                'eval_rewards': eval_rewards,
            }
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_ep{episode + 1}.pth")
            agent.save_checkpoint(
                checkpoint_path,
                episode + 1,
                metrics,
                replay_data=agent.replay_buffer.get_data()
            )
            
            # Also save metrics as JSON for easy access
            metrics_path = os.path.join(config.CHECKPOINT_DIR, "training_metrics.json")
            json_metrics = {
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_losses': [float(l) for l in episode_losses],
                'eval_episodes': eval_episodes,
                'eval_avg_queues': [float(q) for q in eval_avg_queues],
                'eval_avg_waits': [float(w) for w in eval_avg_waits],
                'eval_rewards': [float(r) for r in eval_rewards],
                'last_episode': episode + 1,
            }
            with open(metrics_path, 'w') as f:
                json.dump(json_metrics, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    
    # Save final model
    agent.save(config.MODEL_SAVE_PATH)
    
    # Generate all graphs
    print("\nGenerating performance graphs...")
    generate_all_graphs(episode_rewards, eval_episodes, eval_avg_queues, 
                        eval_avg_waits, eval_rewards)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Episodes: {config.EPISODES}")
    print(f"Average Reward (All): {np.mean(episode_rewards):.2f}")
    print(f"Average Reward (Last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    if eval_avg_queues:
        print(f"Final Avg Queue Length: {eval_avg_queues[-1]:.2f}")
        print(f"Final Avg Waiting Time: {eval_avg_waits[-1]:.2f}s")
    print("=" * 70)
    
    return episode_rewards


def generate_all_graphs(episode_rewards, eval_episodes, eval_avg_queues, 
                        eval_avg_waits, eval_rewards):
    """Generate all required performance and reward graphs."""
    
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    os.makedirs(config.REWARD_GRAPH_DIR, exist_ok=True)
    
    # ========================================
    # 1. AVG QUEUE LENGTH VS EPISODES
    # ========================================
    plt.figure(figsize=(14, 7))
    plt.plot(eval_episodes, eval_avg_queues, 'b-o', markersize=2, linewidth=1.5, label='Avg Queue Length')
    
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
    
    # ========================================
    # 2. AVG WAITING TIME VS EPISODES
    # ========================================
    plt.figure(figsize=(14, 7))
    plt.plot(eval_episodes, eval_avg_waits, 'g-o', markersize=2, linewidth=1.5, label='Avg Waiting Time')
    
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
    
    # ========================================
    # 3. REWARD GRAPHS in REWARD_GRAPH subfolder
    # ========================================
    # Generate reward graphs at various milestones
    milestones = [10, 20, 50, 100, 200, 300, 400, 500]
    
    for milestone in milestones:
        if milestone > len(eval_episodes):
            # Use whatever data we have up to this point
            if not eval_episodes:
                continue
            # Find the closest episode index
            ep_idx = len(eval_episodes)
        else:
            # Find how many eval points exist up to this milestone
            ep_idx = sum(1 for ep in eval_episodes if ep <= milestone)
        
        if ep_idx == 0:
            continue
        
        # Get data up to this milestone
        plot_episodes = eval_episodes[:ep_idx]
        plot_rewards = eval_rewards[:ep_idx]
        
        plt.figure(figsize=(14, 7))
        plt.plot(plot_episodes, plot_rewards, 'b-o', markersize=3, linewidth=1.5, 
                 label='Episode Reward')
        
        # Add cumulative average
        cumulative_avg = np.cumsum(plot_rewards) / np.arange(1, len(plot_rewards) + 1)
        plt.plot(plot_episodes, cumulative_avg, 'r-', linewidth=2, alpha=0.7, 
                 label='Cumulative Average')
        
        # Add moving average if enough data
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
    
    # ========================================
    # 4. Training rewards over all episodes
    # ========================================
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


if __name__ == "__main__":
    train()
