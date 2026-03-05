"""
Parallel DQN Training with 4 SUMO Environment Workers.

Architecture:
  - 1 Learner process (main): DQN agent, replay buffer, training
  - 4 Worker processes: each runs its own SUMO+TraCI simulation

Workers hold local Q-network copies for action selection and send
transitions to the learner via multiprocessing.Queue. The learner
periodically broadcasts updated weights back to workers.

Usage:
    python parallel_train_dqn.py
"""

import os
import sys
import copy
import json
import time
import queue
import numpy as np
import multiprocessing as mp
from collections import OrderedDict

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from dqn_agent import DQNAgent, DQNetwork


# ─────────────────────────────────────────────────────────────────────
# WORKER PROCESS
# ─────────────────────────────────────────────────────────────────────

def compute_episode_params(global_episode):
    """
    Given a global episode number (0-indexed), compute the flow file
    and simulation start time.
    
    Returns:
        flow_file (str), start_time (int), day_num (int), slice_hours (str)
    """
    slices_per_day = 59400 // config.MAX_STEPS_PER_EPISODE
    if slices_per_day == 0:
        slices_per_day = 1

    day_idx = (global_episode // slices_per_day) % len(config.FLOW_FILES)
    slice_idx = global_episode % slices_per_day
    start_time = slice_idx * config.MAX_STEPS_PER_EPISODE

    flow_file = config.FLOW_FILES[day_idx]
    day_num = day_idx + 1

    start_hr = start_time // 3600
    end_hr = (start_time + config.MAX_STEPS_PER_EPISODE) // 3600
    slice_hours = f"{start_hr:02d}:00-{end_hr:02d}:00"

    return flow_file, start_time, day_num, slice_hours


def select_action(q_network, state, epsilon, action_size, device):
    """Epsilon-greedy action selection using a local Q-network."""
    import random
    if random.random() < epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_network(state_t)
        return q_values.argmax().item()


def worker_process(worker_id, num_workers, episodes_per_worker,
                   transition_queue, result_queue, weight_queue,
                   state_size, action_size,
                   start_global_ep=0, resumed_epsilon=None):
    """
    Worker process that runs SUMO episodes and sends transitions
    to the central learner.
    
    Args:
        worker_id: Unique integer ID (0 to NUM_WORKERS-1)
        num_workers: Total number of workers
        episodes_per_worker: How many episodes this worker should run
        transition_queue: Queue to send (s, a, r, s', done) to learner
        result_queue: Queue to send episode summaries to learner
        weight_queue: Queue to receive model weight updates from learner
        state_size: Size of the state vector
        action_size: Number of discrete actions
        start_global_ep: Starting global episode offset (for resume)
        resumed_epsilon: Epsilon to resume from (None = start fresh)
    """
    # Import here so each process gets its own traci module state
    from sumo_env import SumoEnvironment

    print(f"[Worker {worker_id}] Starting (PID {os.getpid()})")

    # Create environment with unique worker_id
    env = SumoEnvironment(use_gui=False, worker_id=worker_id)

    # Create local Q-network for action selection (CPU is fine)
    device = torch.device("cpu")
    local_net = DQNetwork(state_size, action_size).to(device)

    # Wait for initial weights from the learner
    print(f"[Worker {worker_id}] Waiting for initial weights...")
    initial_weights = weight_queue.get()
    local_net.load_state_dict(initial_weights)
    print(f"[Worker {worker_id}] Received initial weights, starting episodes")

    epsilon = resumed_epsilon if resumed_epsilon is not None else config.EPSILON_START

    for local_ep in range(episodes_per_worker):
        # Map local episode to a global episode number (staggered, offset by resume point)
        global_ep = start_global_ep + worker_id + local_ep * num_workers

        if global_ep >= config.EPISODES:
            break

        # Determine flow file and time slice for this episode
        flow_file, start_time, day_num, slice_hours = compute_episode_params(global_ep)

        print(f"[Worker {worker_id}] Episode {global_ep + 1}/{config.EPISODES} "
              f"[Day {day_num} | {slice_hours}] | Epsilon: {epsilon:.3f}")

        # Run episode
        env.set_flow_file(flow_file)
        state = env.reset(start_time=start_time)
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            action = select_action(local_net, state, epsilon, action_size, device)
            next_state, reward, done, info = env.step(action)

            # Send transition to the learner
            transition_queue.put((
                state.tolist(),
                int(action),
                float(reward),
                next_state.tolist(),
                bool(done)
            ))

            state = next_state
            episode_reward += reward
            step_count += 1
            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"[Worker {worker_id}] Episode {global_ep + 1} | Step {step_count} | "
                      f"Queue: {info['queue_length']:.1f} | Wait: {info['waiting_time']:.1f}s")

        # Decay epsilon after each episode
        epsilon = max(config.EPSILON_MIN, epsilon * config.EPSILON_DECAY)

        # Close env after episode
        env.close()

        # Get episode metrics
        avg_queue = env.get_episode_avg_queue_length()
        avg_wait = env.get_episode_avg_waiting_time()

        # Send episode summary to learner
        result_queue.put({
            'worker_id': worker_id,
            'global_episode': global_ep,
            'reward': episode_reward,
            'steps': step_count,
            'avg_queue': avg_queue,
            'avg_wait': avg_wait,
            'epsilon': epsilon,
        })

        # Check for weight updates from learner (non-blocking, drain queue)
        latest_weights = None
        try:
            while True:
                latest_weights = weight_queue.get_nowait()
        except queue.Empty:
            pass
        if latest_weights is not None:
            local_net.load_state_dict(latest_weights)

    print(f"[Worker {worker_id}] Finished all episodes")


# ─────────────────────────────────────────────────────────────────────
# LEARNER (MAIN PROCESS)
# ─────────────────────────────────────────────────────────────────────

def get_state_action_sizes():
    """Launch a quick SUMO session to determine state and action sizes."""
    from sumo_env import SumoEnvironment
    env = SumoEnvironment(use_gui=False, worker_id=99)
    env.set_flow_file(config.FLOW_FILES[0])
    env.reset()
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    env.close()
    return state_size, action_size


def run_parallel_training():
    """Main function: central learner with parallel SUMO workers."""

    # ── Setup directories ──
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    os.makedirs(config.REWARD_GRAPH_DIR, exist_ok=True)

    # ── Determine state/action sizes ──
    print("Determining state/action sizes...")
    state_size, action_size = get_state_action_sizes()
    print(f"State size: {state_size}, Action size: {action_size}")

    # ── Initialize DQN agent ──
    print("Initializing DQN agent (learner)...")
    agent = DQNAgent(state_size, action_size)

    # ── Check for existing checkpoint to resume ──
    start_global_ep = 0
    episode_rewards = []
    episode_losses = []

    from train_dqn import find_latest_checkpoint
    latest_checkpoint = find_latest_checkpoint(config.CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}. Loading...")
        start_global_ep, metrics = agent.load_checkpoint(latest_checkpoint)
        episode_rewards = metrics.get('episode_rewards', [])
        episode_losses = metrics.get('episode_losses', [])
        print(f"Resuming from episode {start_global_ep + 1}...")

    # ── Multiprocessing setup ──
    num_workers = config.NUM_WORKERS
    episodes_remaining = config.EPISODES - start_global_ep
    episodes_per_worker = (episodes_remaining + num_workers - 1) // num_workers

    transition_queue = mp.Queue(maxsize=2000)
    result_queue = mp.Queue()
    weight_queues = [mp.Queue() for _ in range(num_workers)]

    # Send initial weights to all workers
    initial_weights = copy.deepcopy(agent.q_network.state_dict())
    # Move tensors to CPU for cross-process transfer
    cpu_weights = OrderedDict(
        {k: v.cpu() for k, v in initial_weights.items()}
    )
    for wq in weight_queues:
        wq.put(copy.deepcopy(cpu_weights))

    # ── Launch worker processes ──
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(wid, num_workers, episodes_per_worker,
                  transition_queue, result_queue, weight_queues[wid],
                  state_size, action_size,
                  start_global_ep, agent.epsilon),
            daemon=True
        )
        p.start()
        workers.append(p)
        print(f"Launched worker {wid} (PID {p.pid})")

    # ── Learner loop ──
    print(f"\n{'=' * 70}")
    print(f"PARALLEL TRAINING: {config.EPISODES} episodes across {num_workers} workers")
    print(f"Each episode = {config.MAX_STEPS_PER_EPISODE}s ({config.MAX_STEPS_PER_EPISODE // 3600}h)")
    print(f"Weight sync every {config.WEIGHT_SYNC_INTERVAL} training steps")
    print(f"{'=' * 70}\n")

    completed_episodes = start_global_ep
    train_steps = 0
    last_weight_sync = 0
    training_start_time = time.time()

    # Episode results buffer (may arrive out of order from workers)
    episode_results = {}
    current_losses = []

    while completed_episodes < config.EPISODES:
        # ── Collect transitions (batch of up to 50 at a time) ──
        transitions_collected = 0
        while transitions_collected < 50:
            try:
                s, a, r, ns, d = transition_queue.get(timeout=1.0)
                agent.remember(
                    np.array(s, dtype=np.float32),
                    a, r,
                    np.array(ns, dtype=np.float32),
                    d
                )
                transitions_collected += 1

                # Train on each transition if buffer is ready
                if agent.replay_buffer.size() >= agent.batch_size:
                    loss = agent.replay()
                    if loss > 0:
                        current_losses.append(loss)
                    train_steps += 1

                    # Periodic weight sync to workers
                    if train_steps - last_weight_sync >= config.WEIGHT_SYNC_INTERVAL:
                        weights = OrderedDict(
                            {k: v.cpu() for k, v in agent.q_network.state_dict().items()}
                        )
                        for wq in weight_queues:
                            # Clear old weights first
                            try:
                                while True:
                                    wq.get_nowait()
                            except queue.Empty:
                                pass
                            wq.put(copy.deepcopy(weights))
                        last_weight_sync = train_steps

            except queue.Empty:
                break

        # ── Collect episode results ──
        while True:
            try:
                result = result_queue.get_nowait()
                global_ep = result['global_episode']
                episode_results[global_ep] = result
            except queue.Empty:
                break

        # ── Process completed episodes in order ──
        while completed_episodes in episode_results:
            result = episode_results.pop(completed_episodes)
            episode_rewards.append(result['reward'])
            
            avg_loss = float(np.mean(current_losses)) if current_losses else 0.0
            episode_losses.append(avg_loss)
            current_losses.clear()

            elapsed = time.time() - training_start_time
            print(f"  Episode {completed_episodes + 1}/{config.EPISODES} "
                  f"[Worker {result['worker_id']}] | "
                  f"Reward: {result['reward']:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Steps: {result['steps']} | "
                  f"Queue: {result['avg_queue']:.2f} | "
                  f"Wait: {result['avg_wait']:.2f}s | "
                  f"Eps: {result['epsilon']:.3f} | "
                  f"Time: {elapsed:.0f}s")

            completed_episodes += 1

            # Update target network periodically
            if completed_episodes % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                print(f"  >> Target network updated at episode {completed_episodes}")

            # Save checkpoint
            if completed_episodes % config.CHECKPOINT_INTERVAL == 0:
                metrics = {
                    'episode_rewards': episode_rewards,
                    'episode_losses': episode_losses,
                    'eval_episodes': [],
                    'eval_avg_queues': [],
                    'eval_avg_waits': [],
                    'eval_rewards': [],
                }
                cp_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"checkpoint_ep{completed_episodes}.pth"
                )
                agent.save_checkpoint(
                    cp_path, completed_episodes, metrics,
                    replay_data=agent.replay_buffer.get_data()
                )

                # Also save JSON metrics
                json_metrics = {
                    'episode_rewards': [float(r) for r in episode_rewards],
                    'episode_losses': [float(l) for l in episode_losses],
                    'last_episode': completed_episodes,
                    'train_steps': train_steps,
                    'elapsed_seconds': elapsed,
                }
                metrics_path = os.path.join(config.CHECKPOINT_DIR, "training_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(json_metrics, f, indent=2)

        # ── Check if all workers have exited ──
        all_done = all(not p.is_alive() for p in workers)
        if all_done and transition_queue.empty() and result_queue.empty():
            break

    # ── Cleanup ──
    total_time = time.time() - training_start_time
    print(f"\n{'=' * 70}")
    print("PARALLEL TRAINING COMPLETED!")
    print(f"{'=' * 70}")
    print(f"Total episodes: {completed_episodes}")
    print(f"Total training steps: {train_steps}")
    print(f"Total wall-clock time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
    if episode_rewards:
        print(f"Average Reward (All): {np.mean(episode_rewards):.2f}")
        print(f"Average Reward (Last 50): {np.mean(episode_rewards[-50:]):.2f}")
        print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")

    # Save final model
    agent.save(config.MODEL_SAVE_PATH)

    # Generate training reward graph
    generate_training_graphs(episode_rewards)

    # Wait for workers to fully exit
    for p in workers:
        p.join(timeout=10)

    print(f"{'=' * 70}")


def generate_training_graphs(episode_rewards):
    """Generate training performance graphs."""
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    os.makedirs(config.REWARD_GRAPH_DIR, exist_ok=True)

    if not episode_rewards:
        print("No episode rewards to plot.")
        return

    # ── Training rewards per episode ──
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
        moving_avg = np.convolve(episode_rewards,
                                   np.ones(window_size) / window_size, mode='valid')
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
    save_path = os.path.join(config.GRAPH_DIR, "training_rewards_parallel.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # ── Reward milestones ──
    milestones = [10, 50, 100, 200, 300, 400, 500]
    for milestone in milestones:
        if milestone > len(episode_rewards):
            continue
        plot_rewards = episode_rewards[:milestone]
        plt.figure(figsize=(14, 7))
        plt.plot(range(1, len(plot_rewards) + 1), plot_rewards, 'b-o',
                 markersize=2, linewidth=1, label='Episode Reward')
        cumulative_avg = np.cumsum(plot_rewards) / np.arange(1, len(plot_rewards) + 1)
        plt.plot(range(1, len(plot_rewards) + 1), cumulative_avg, 'r-',
                 linewidth=2, alpha=0.7, label='Cumulative Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Reward Function (Episode 1 to {milestone})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(config.REWARD_GRAPH_DIR,
                                  f"reward_up_to_ep{milestone}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required for safe multiprocessing on all platforms
    mp.set_start_method("spawn", force=True)
    run_parallel_training()
