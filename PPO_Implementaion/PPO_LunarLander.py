import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
import os
import json


@dataclass
class PPOConfig:
    num_environments: int = 8
    steps_per_iter: int = 256
    epochs: int = 10
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda_value: float = 0.95
    device: str = 'cpu'
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_iterations: int = 5000
    learning_rate: float = 3e-4
    gradient_clip: float = 0.5
    graduation_threshold: float = 200.0
    
    def print_config(self):
        print(f"\n Configuration:")
        print(f"   Environments: {self.num_environments}")
        print(f"   Steps per iteration: {self.steps_per_iter}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Device: {self.device}")
        print(f"   Total iterations: {self.max_iterations:,}")
        print(f"   Graduation threshold: {self.graduation_threshold}")


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


class ParallelEnvironments:
    ACTION_NAMES = ['NOOP', 'LEFT_ENGINE', 'MAIN_ENGINE', 'RIGHT_ENGINE']
    
    def __init__(self, num_environments: int):
        self.num_environments = num_environments
        self.envs = [gym.make('LunarLander-v3') for _ in range(num_environments)]
        self.observations = [None] * num_environments
        self.episode_rewards = [0] * num_environments
        
        for i in range(num_environments):
            self.reset(i)
    
    def reset(self, environment_id: int):
        self.episode_rewards[environment_id] = 0
        observation, _ = self.envs[environment_id].reset()
        self.observations[environment_id] = observation
        return observation
    
    def step(self, environment_id: int, action: int):
        observation, reward, terminated, truncated, info = self.envs[environment_id].step(action)
        done = terminated or truncated
        self.episode_rewards[environment_id] += reward
        self.observations[environment_id] = observation
        return observation, reward, done, info


class ExperienceBuffer:
    def __init__(self, num_environments: int, steps: int, observation_dim: int, device: str):
        self.num_environments = num_environments
        self.steps = steps
        self.device = device
        
        self.observations = torch.zeros((num_environments, steps, observation_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_environments, steps), dtype=torch.long, device=device)
        self.log_probabilities = torch.zeros((num_environments, steps), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_environments, steps + 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_environments, steps), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_environments, steps), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((num_environments, steps), dtype=torch.float32, device=device)
    
    def store(self, environment_id, t, observation, action, log_probability, value, reward, done):
        self.observations[environment_id, t] = observation
        self.actions[environment_id, t] = action
        self.log_probabilities[environment_id, t] = log_probability
        self.values[environment_id, t] = value
        self.rewards[environment_id, t] = reward
        self.dones[environment_id, t] = done
    
    def compute_gae(self, environment_id, gamma, gae_lambda_value):
        for t in reversed(range(self.steps)):
            next_non_terminal = 1.0 - self.dones[environment_id, t]
            delta = (
                self.rewards[environment_id, t] +
                gamma * self.values[environment_id, t + 1] * next_non_terminal -
                self.values[environment_id, t]
            )
            
            if t == self.steps - 1:
                self.advantages[environment_id, t] = delta
            else:
                self.advantages[environment_id, t] = (
                    delta + gamma * gae_lambda_value * self.advantages[environment_id, t + 1] * next_non_terminal
                )
    
    def get_data(self):
        return (
            self.advantages.reshape(-1),
            self.observations.reshape(-1, self.observations.shape[-1]),
            self.actions.reshape(-1),
            self.log_probabilities.reshape(-1),
            self.values[:, :self.steps].reshape(-1)
        )


class PPOTrainer:
    def __init__(self, configuration: PPOConfig):
        self.configuration = configuration
        

        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        

        print(" Creating environments...")
        self.envs = ParallelEnvironments(configuration.num_environments)
        print(f" Created {configuration.num_environments} environments")
        

        print("\n Creating Actor-Critic model...")
        self.model = ActorCritic().to(configuration.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configuration.learning_rate)
        print(" Model created")
        
        configuration.print_config()
        

        self.episode_count = 0
        self.recent_rewards = deque(maxlen=100)
        self.all_rewards = []
        self.best_reward = -float('inf')
        self.action_counts = np.zeros(4)
    
    def collect_experience(self, buffer: ExperienceBuffer):
        episodes_done = 0
        graduated = False
        
        for environment_id in range(self.configuration.num_environments):
            with torch.no_grad():
                for t in range(self.configuration.steps_per_iter):

                    observation = torch.from_numpy(
                        self.envs.observations[environment_id]
                    ).float().to(self.configuration.device)
                    

                    logits, value = self.model(observation)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = distribution.sample()
                    log_probability = distribution.log_prob(action)
                    

                    self.action_counts[action.item()] += 1
                    

                    next_obs, reward, done, info = self.envs.step(environment_id, action.item())
                    

                    buffer.store(environment_id, t, observation, action, log_probability, value.squeeze(), reward, done)
                    

                    if done:
                        episodes_done += 1
                        if self.handle_episode_end(environment_id):
                            graduated = True
                            break
                
                if graduated:
                    break
                

                final_obs = torch.from_numpy(
                    self.envs.observations[environment_id]
                ).float().to(self.configuration.device)
                _, final_value = self.model(final_obs)
                buffer.values[environment_id, self.configuration.steps_per_iter] = final_value.squeeze()
                

                buffer.compute_gae(environment_id, self.configuration.gamma, self.configuration.gae_lambda_value)
        
        return episodes_done, graduated
    
    def handle_episode_end(self, environment_id: int):
        episode_reward = self.envs.episode_rewards[environment_id]
        self.episode_count += 1
        self.recent_rewards.append(episode_reward)
        self.all_rewards.append(episode_reward)
        

        if self.episode_count % 10 == 0:
            avg_reward = np.mean(self.recent_rewards)
            print(f"Episode {self.episode_count:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(100): {avg_reward:7.2f}")
            

            if avg_reward >= self.configuration.graduation_threshold:
                print(f"\n{'='*60}")
                print(f" SOLVED! Average reward: {avg_reward:.2f}")
                print(f"{'='*60}\n")
                torch.save(self.model.state_dict(), "models/lunarlander_solved_model.pth")
                return True
        

        # if self.episode_count % 100 == 0:
        #     self.print_action_distribution()
        

        # if episode_reward > self.best_reward:
        #     self.best_reward = episode_reward
        #     torch.save(self.model.state_dict(), f"models/best_model_{self.best_reward:.0f}.pth")
        #     print(f" New best reward: {self.best_reward:.2f}")
        

        self.envs.reset(environment_id)
        return False
    
    def print_action_distribution(self):
        total = self.action_counts.sum()
        if total > 0:
            print(f"\n Action Distribution (Episode {self.episode_count}):")
            for i, (name, count) in enumerate(zip(ParallelEnvironments.ACTION_NAMES, self.action_counts)):
                pct = (count / total * 100)
                print(f"   {name:12s}: {pct:5.1f}% ({int(count):,} actions)")
            print()
            self.action_counts = np.zeros(4)
    
    def update_policy(self, buffer: ExperienceBuffer):
        advantages, observations, actions, old_log_probs, old_values = buffer.get_data()
        

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.configuration.epochs):

            indices = torch.randperm(len(advantages), device=self.configuration.device)
            
            for start in range(0, len(advantages), self.configuration.batch_size):
                end = min(start + self.configuration.batch_size, len(advantages))
                batch_idx = indices[start:end]
                

                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_values = old_values[batch_idx]
                

                logits, values = self.model(batch_obs)
                values = values.squeeze(-1)
                distribution = torch.distributions.Categorical(logits=logits)
                log_probabilities = distribution.log_prob(batch_actions)
                

                returns = batch_advantages + batch_old_values
                

                ratio = torch.exp(log_probabilities - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                

                value_loss = F.mse_loss(values, returns)
                

                entropy_loss = -distribution.entropy().mean()
                

                loss = (
                    policy_loss +
                    self.configuration.value_coefficient * value_loss +
                    self.configuration.entropy_coefficient * entropy_loss
                )
                

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configuration.gradient_clip)
                self.optimizer.step()
    
    def train(self):
        print("\n Starting training...\n")
        
        for iteration in range(self.configuration.max_iterations):

            buffer = ExperienceBuffer(
                self.configuration.num_environments,
                self.configuration.steps_per_iter,
                8,
                self.configuration.device
            )
            

            episodes_done, graduated = self.collect_experience(buffer)
            

            if graduated:
                print("\n Training complete - LunarLander SOLVED!")
                self.save_training_log()
                return
            

            self.update_policy(buffer)
        
        print("\n Training complete!")
        torch.save(self.model.state_dict(), "models/lunarlander_final_model_env_8.pth")
        print(" Final model saved")
        

        self.save_training_log()
    
    def save_training_log(self):
        log_data = {
            'rewards': self.all_rewards,
            'configuration': {
                'num_environments': self.configuration.num_environments,
                'steps_per_iter': self.configuration.steps_per_iter,
                'gae_lambda_value': self.configuration.gae_lambda_value,
                'entropy_coefficient': self.configuration.entropy_coefficient,
                'learning_rate': self.configuration.learning_rate,
                'batch_size': self.configuration.batch_size,
                'epochs': self.configuration.epochs
            }
        }
        
        with open('logs/lunarlander_training_env_8.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        print(" Training log saved to logs/lunarlander_training_env_1.json")


def main():
    print("=" * 60)
    print("PPO for LunarLander-v3")
    print("=" * 60)
    
    configuration = PPOConfig(
        num_environments=8,
        steps_per_iter=256,
        epochs=10,
        batch_size=128,
        max_iterations=5000
    )
    
    trainer = PPOTrainer(configuration)
    trainer.train()


if __name__ == "__main__":
    main()
