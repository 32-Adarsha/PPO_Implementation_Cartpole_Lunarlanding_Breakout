import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        

        self.policy = nn.Linear(hidden_dim, action_dim)
        

        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.policy(features), self.value(features)


def evaluate_agent(model_path: str, num_episodes: int = 5, render: bool = True):
    print(f"Loading model from: {model_path}")
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path))
    model.eval()
 
    

    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    total_rewards = []
    
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float()
                logits, value = model(obs_tensor)
                

                action = torch.argmax(logits).item()
            

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.02)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d} | Reward: {episode_reward:6.1f} | Steps: {steps:4d}")
    
    env.close()
    

    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(total_rewards):6.2f}")
    print(f"Min Reward:     {np.min(total_rewards):6.1f}")
    print(f"Max Reward:     {np.max(total_rewards):6.1f}")
    print(f"Std Reward:     {np.std(total_rewards):6.2f}")
    print(f"{'='*50}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate PPO CartPole agent')
    parser.add_argument('--model', type=str, default='models/solved_model.pth',
                        help='Path to model file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PPO CartPole Evaluation")
    print("=" * 50)
    print()
    
    evaluate_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
