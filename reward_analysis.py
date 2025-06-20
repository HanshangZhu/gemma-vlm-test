#!/usr/bin/env python3
"""
MetaWorld Reward Analysis - Understanding reward space and success criteria
"""

import numpy as np
import metaworld
import random
import matplotlib.pyplot as plt

def analyze_reward_structure(task_name='button-press-topdown-v3', n_episodes=100):
    """Analyze the reward structure of a MetaWorld task"""
    
    print(f"🔍 Analyzing Reward Structure for: {task_name}")
    print("="*60)
    
    # Setup environment
    mt10 = metaworld.MT10()
    env_cls = mt10.train_classes[task_name]
    env = env_cls()
    task = random.choice([task for task in mt10.train_tasks
                         if task.env_name == task_name])
    env.set_task(task)
    
    # Collect reward data
    rewards_per_episode = []
    successes = []
    max_rewards = []
    step_rewards = []
    
    print(f"\n📊 Running {n_episodes} episodes with random actions...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_rewards = []
        episode_success = False
        
        # Run episode with random actions
        for step in range(50):  # Max 50 steps per episode
            # Random action in [-1, 1]
            action = np.random.uniform(-1, 1, 4)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            step_rewards.append(reward)
            
            if info.get('success', False):
                episode_success = True
            
            if terminated or truncated:
                break
        
        total_reward = sum(episode_rewards)
        max_reward = max(episode_rewards) if episode_rewards else 0
        
        rewards_per_episode.append(total_reward)
        successes.append(episode_success)
        max_rewards.append(max_reward)
        
        if episode < 5:  # Print first few episodes for inspection
            print(f"Episode {episode+1}: Total={total_reward:.3f}, Max={max_reward:.3f}, Success={episode_success}")
    
    # Analysis
    print(f"\n📈 REWARD ANALYSIS")
    print("="*40)
    print(f"Episodes analyzed: {n_episodes}")
    print(f"Total steps: {len(step_rewards)}")
    
    # Success analysis
    success_count = sum(successes)
    success_rate = success_count / n_episodes * 100
    print(f"\n🎯 SUCCESS METRICS:")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{n_episodes})")
    
    if success_count > 0:
        successful_rewards = [rewards_per_episode[i] for i in range(n_episodes) if successes[i]]
        print(f"Successful episode rewards: {successful_rewards}")
        print(f"Min successful reward: {min(successful_rewards):.3f}")
        print(f"Max successful reward: {max(successful_rewards):.3f}")
        print(f"Avg successful reward: {np.mean(successful_rewards):.3f}")
    
    # Step reward analysis
    step_rewards = np.array(step_rewards)
    print(f"\n📊 STEP REWARD STATISTICS:")
    print(f"Min step reward: {step_rewards.min():.3f}")
    print(f"Max step reward: {step_rewards.max():.3f}")
    print(f"Mean step reward: {step_rewards.mean():.3f}")
    print(f"Std step reward: {step_rewards.std():.3f}")
    
    # Episode reward analysis
    episode_rewards = np.array(rewards_per_episode)
    print(f"\n📈 EPISODE REWARD STATISTICS:")
    print(f"Min episode reward: {episode_rewards.min():.3f}")
    print(f"Max episode reward: {episode_rewards.max():.3f}")
    print(f"Mean episode reward: {episode_rewards.mean():.3f}")
    print(f"Std episode reward: {episode_rewards.std():.3f}")
    
    # Reward distribution
    print(f"\n📊 REWARD DISTRIBUTION:")
    positive_steps = (step_rewards > 0).sum()
    zero_steps = (step_rewards == 0).sum()
    negative_steps = (step_rewards < 0).sum()
    
    print(f"Positive rewards: {positive_steps} ({positive_steps/len(step_rewards)*100:.1f}%)")
    print(f"Zero rewards: {zero_steps} ({zero_steps/len(step_rewards)*100:.1f}%)")
    print(f"Negative rewards: {negative_steps} ({negative_steps/len(step_rewards)*100:.1f}%)")
    
    # Quartiles
    quartiles = np.percentile(step_rewards, [25, 50, 75])
    print(f"\n📊 REWARD QUARTILES:")
    print(f"25th percentile: {quartiles[0]:.3f}")
    print(f"50th percentile (median): {quartiles[1]:.3f}")
    print(f"75th percentile: {quartiles[2]:.3f}")
    
    # Create visualization
    create_reward_visualization(step_rewards, episode_rewards, successes, task_name)
    
    return {
        'step_rewards': step_rewards,
        'episode_rewards': episode_rewards,
        'successes': successes,
        'success_rate': success_rate
    }

def create_reward_visualization(step_rewards, episode_rewards, successes, task_name):
    """Create visualization of reward distribution"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Step reward histogram
        ax1.hist(step_rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', label='Zero reward')
        ax1.set_xlabel('Step Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Step Reward Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode reward histogram
        ax2.hist(episode_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Episode Total Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Episode Reward Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success vs reward scatter
        success_mask = np.array(successes)
        colors = ['red' if not s else 'green' for s in successes]
        ax3.scatter(range(len(episode_rewards)), episode_rewards, c=colors, alpha=0.6)
        ax3.set_xlabel('Episode Number')
        ax3.set_ylabel('Episode Total Reward')
        ax3.set_title('Episode Rewards (Green=Success, Red=Failure)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Reward over time (first 200 steps)
        steps_to_show = min(200, len(step_rewards))
        ax4.plot(step_rewards[:steps_to_show], alpha=0.7, color='purple')
        ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Step Number')
        ax4.set_ylabel('Step Reward')
        ax4.set_title(f'Reward Over Time (First {steps_to_show} steps)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Reward Analysis: {task_name}', fontsize=16)
        plt.tight_layout()
        
        filename = f'reward_analysis_{task_name.replace("-", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved reward analysis to '{filename}'")
        
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")

def test_optimal_actions(task_name='button-press-topdown-v3'):
    """Test what happens with 'optimal' actions (towards button)"""
    print(f"\n🎯 Testing optimal-like actions for {task_name}")
    print("-"*50)
    
    # Setup environment
    mt10 = metaworld.MT10()
    env_cls = mt10.train_classes[task_name]
    env = env_cls()
    task = random.choice([task for task in mt10.train_tasks
                         if task.env_name == task_name])
    env.set_task(task)
    
    obs = env.reset()
    
    # Try some "smart" actions - small movements towards button
    test_actions = [
        [0.1, 0.0, -0.1, 0.0],   # Move forward, down
        [0.0, 0.1, -0.1, 0.0],   # Move right, down  
        [-0.1, 0.0, -0.1, 0.0],  # Move back, down
        [0.0, 0.0, -0.2, 1.0],   # Move down, open gripper
        [0.0, 0.0, 0.1, 1.0],    # Move up, open gripper
    ]
    
    print("Testing strategic actions:")
    for i, action in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        success = info.get('success', False)
        print(f"Action {i+1} {action}: reward={reward:.3f}, success={success}")
        
        if terminated or truncated or success:
            break
    
    return

def main():
    # Analyze multiple tasks
    tasks = ['button-press-topdown-v3', 'reach-v3', 'push-v3']
    
    for task in tasks:
        print("\n" + "="*80)
        try:
            results = analyze_reward_structure(task, n_episodes=50)
            test_optimal_actions(task)
        except Exception as e:
            print(f"Error analyzing {task}: {e}")
        print("="*80)

if __name__ == "__main__":
    main() 