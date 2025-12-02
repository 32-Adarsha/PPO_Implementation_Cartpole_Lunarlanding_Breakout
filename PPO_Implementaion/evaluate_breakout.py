import gymnasium as gym
import ale_py
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple
import argparse


gym.register_envs(ale_py)


class ActorCritic(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), 
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, stride=2), 
            nn.Tanh(),
            nn.Flatten(), 
            nn.Linear(2592, 256), 
            nn.Tanh(),
        )
        self.actor = nn.Sequential(nn.Linear(256, num_actions))
        self.critic = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.head(x)
        return self.actor(h), self.critic(h)


def make_env(render_mode=None):
    env = gym.make(
        "ALE/Breakout-v5", 
        frameskip=4, 
        repeat_action_probability=0.0,
        render_mode=render_mode
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def evaluate_model(
    model_path: str,
    num_episodes: int = 10,
    device: str = 'cuda',
    render: bool = False,
    greedy: bool = True
):

    if device == 'cuda' and not torch.cuda.is_available():
        print("  CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\n{'='*70}")
    print(f"PPO Breakout Model Evaluation")
    print(f"{'='*70}")
    print(f"\n Model: {model_path}")
    print(f" Episodes: {num_episodes}")
    print(f" Device: {device}")
    print(f" Action Selection: {'Greedy (argmax)' if greedy else 'Stochastic (sample)'}")
    print(f"  Render: {render}")
    

    print(f"\n Loading model...")
    model = ActorCritic(4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f" Model loaded successfully!")
    

    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)
    

    print(f"\n Starting evaluation...\n")
    
    episode_rewards = []
    episode_lengths = []
    action_counts = np.zeros(4)
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        with torch.no_grad():
            while not done:

                obs_tensor = torch.from_numpy(np.array(observation) / 255.0).unsqueeze(0).float().to(device)
                

                logits, value = model(obs_tensor)
                
                if greedy:

                    action = logits.argmax(dim=-1).item()
                else:

                    probs = torch.softmax(logits, dim=-1)
                    distribution = torch.distributions.categorical.Categorical(probs)
                    action = distribution.sample().item()
                

                action_counts[action] += 1
                

                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1:2d}/{num_episodes} | "
              f"Reward: {episode_reward:6.1f} | "
              f"Length: {episode_length:5d} steps")
    
    env.close()
    

    print(f"\n{'='*70}")
    print(f" Evaluation Results")
    print(f"{'='*70}")
    print(f"\n Reward Statistics:")
    print(f"   Mean:   {np.mean(episode_rewards):6.2f}")
    print(f"   Std:    {np.std(episode_rewards):6.2f}")
    print(f"   Min:    {np.min(episode_rewards):6.2f}")
    print(f"   Max:    {np.max(episode_rewards):6.2f}")
    print(f"   Median: {np.median(episode_rewards):6.2f}")
    
    print(f"\n Episode Length Statistics:")
    print(f"   Mean:   {np.mean(episode_lengths):6.2f} steps")
    print(f"   Std:    {np.std(episode_lengths):6.2f} steps")
    print(f"   Min:    {np.min(episode_lengths):6.0f} steps")
    print(f"   Max:    {np.max(episode_lengths):6.0f} steps")
    
    print(f"\n Action Distribution:")
    total_actions = action_counts.sum()
    for name, count in zip(action_names, action_counts):
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"   {name:5s}: {pct:5.1f}% ({int(count):,} actions)")
    
    print(f"\n{'='*70}\n")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'action_counts': action_counts,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model on Breakout')
    parser.add_argument('--model', type=str, default='models/best_model_358.pth',
                        help='Path to model file (default: models/best_model_358.pth)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (default: False)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic action selection instead of greedy (default: False)')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        device=args.device,
        render=args.render,
        greedy=not args.stochastic
    )


if __name__ == "__main__":
    main()
