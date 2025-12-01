import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        

        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
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
    print(" Model loaded\n")
    

    if render:
        env = gym.make('LunarLander-v3', render_mode='human')
    else:
        env = gym.make('LunarLander-v3')
    
    total_rewards = []
    successful_landings = 0
    
    print(f"Running {num_episodes} episodes...\n")
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:

            with torch.no_grad():
                obs_tensor = torch.from_numpy(observation).float()
                logits, value = model(obs_tensor)
                

                action = torch.argmax(logits).item()
            

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.02)
        
        total_rewards.append(episode_reward)
        

        if episode_reward >= 200:
            successful_landings += 1
            status = " SUCCESS"
        elif episode_reward >= 0:
            status = "  PARTIAL"
        else:
            status = " CRASH"
        
        print(f"Episode {episode + 1:2d} | Reward: {episode_reward:7.2f} | Steps: {steps:4d} | {status}")
    
    env.close()
    

    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average Reward:      {np.mean(total_rewards):7.2f}")
    print(f"Min Reward:          {np.min(total_rewards):7.2f}")
    print(f"Max Reward:          {np.max(total_rewards):7.2f}")
    print(f"Std Reward:          {np.std(total_rewards):7.2f}")
    print(f"Successful Landings: {successful_landings}/{num_episodes} ({successful_landings/num_episodes*100:.1f}%)")
    print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate PPO LunarLander agent')
    parser.add_argument('--model', type=str, default='models/lunarlander_solved_model.pth',
                        help='Path to model file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPO LunarLander Evaluation")
    print("=" * 60)
    print()
    
    evaluate_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
