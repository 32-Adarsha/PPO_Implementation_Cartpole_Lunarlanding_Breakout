import gymnasium as gym
import ale_py
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
import os
from typing import Tuple, List, Optional


gym.register_envs(ale_py)


@dataclass
class PPOConfig:
    num_environments: int = 8
    T: int = 128
    K: int = 4
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda_value: float = 0.95
    device: str = 'cuda'
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    nb_iterations: int = 40000
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5
    
    def __post_init__(self):
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("  CUDA not available, falling back to CPU")
            self.device = 'cpu'
    
    def print_config(self):
        print(f"\n Configuration:")
        print(f"   Parallel environments: {self.num_environments}")
        print(f"   Steps per iteration: {self.T}")
        print(f"   PPO epochs: {self.K}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Entropy coefficient: {self.ent_coef}")
        print(f"   Device: {self.device}")
        print(f"   Total iterations: {self.nb_iterations:,}")
        print(f"   Total steps: {self.nb_iterations * self.T * self.num_environments:,}")


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


class ParallelEnvironments:
    ACTION_NAMES = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    def __init__(self, num_environments: int):
        self.num_environments = num_environments
        self.envs = [self._make_env() for _ in range(num_environments)]
        self.observations = [None for _ in range(num_environments)]
        self.current_life = [None for _ in range(num_environments)]
        self.done = [False for _ in range(num_environments)]
        self.total_rewards = [0 for _ in range(num_environments)]

        for environment_id in range(num_environments):
            self.reset_env(environment_id)

    def _make_env(self, record_video: bool = False, video_name: str = "") -> gym.Env:
        env = gym.make(
            "ALE/Breakout-v5", 
            frameskip=4, 
            repeat_action_probability=0.0,
            render_mode="rgb_array" if record_video else None
        )
        
        if record_video:
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(
                env,
                video_folder="videos_highscores",
                episode_trigger=lambda x: True,
                name_prefix=video_name
            )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    def reset_env(self, environment_id: int):
        self.total_rewards[environment_id] = 0
        observation, _ = self.envs[environment_id].reset()
        


        num_moves = random.randint(5, 15)
        for _ in range(num_moves):

            move_action = random.choice([0, 2, 3])
            observation, reward, terminated, truncated, info = self.envs[environment_id].step(move_action)
            self.total_rewards[environment_id] += reward
            if 'lives' in info:
                self.current_life[environment_id] = info['lives']
        
        self.observations[environment_id] = observation

    def step(self, environment_id: int, action: int) -> Tuple:
        next_obs, reward, terminated, truncated, info = self.envs[environment_id].step(action)
        

        done = False
        if 'lives' in info and info['lives'] < self.current_life[environment_id]:
            done = True
            self.current_life[environment_id] = info['lives']
        
        dead = terminated or truncated
        self.done[environment_id] = done
        self.total_rewards[environment_id] += reward
        self.observations[environment_id] = next_obs
        
        return next_obs, reward, dead, done, info
    
    def record_video(
        self, 
        actorcritic: ActorCritic, 
        device: str, 
        reward_score: float
    ) -> float:
        print(f"\n Recording video for high score: {reward_score:.0f}...")
        

        video_env = self._make_env(record_video=True, video_name=f"score_{reward_score:.0f}")
        
        observation, _ = video_env.reset()
        
        total_reward = 0
        done = False
        steps = 0
        max_steps = 1000
        
        with torch.no_grad():
            while not done and steps < max_steps:
                obs_tensor = torch.from_numpy(np.array(observation) / 255.0).unsqueeze(0).float().to(device)
                logits, _ = actorcritic(obs_tensor)
                

                if reward_score < 10:

                    probs = torch.softmax(logits, dim=-1)
                    distribution = torch.distributions.categorical.Categorical(probs)
                    action = distribution.sample().item()
                else:

                    action = logits.argmax(dim=-1).item()
                
                observation, reward, terminated, truncated, _ = video_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
        
        video_env.close()
        
        if steps >= max_steps:
            print(f"  Video timeout after {max_steps} steps (agent may not be trained yet)")
        
        print(f" Video saved! Episode reward: {total_reward:.0f} ({steps} steps)")
        print(f"   Saved to: videos_highscores/score_{reward_score:.0f}-*.mp4\n")
        
        return total_reward


class ExperienceBuffer:
    def __init__(self, num_environments: int, T: int, device: str):
        self.num_environments = num_environments
        self.T = T
        self.device = device
        
        self.advantages = torch.zeros((num_environments, T), dtype=torch.float32, device=device)
        self.states = torch.zeros((num_environments, T, 4, 84, 84), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_environments, T), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((num_environments, T), dtype=torch.float32, device=device)
        self.state_values = torch.zeros((num_environments, T+1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_environments, T), dtype=torch.float32, device=device)
        self.is_terminal = torch.zeros((num_environments, T), dtype=torch.float16, device=device)
    
    def store_step(
        self, 
        environment_id: int, 
        t: int, 
        observation: torch.Tensor, 
        action: int,
        log_probability: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool
    ):
        self.states[environment_id, t] = observation
        self.actions[environment_id, t] = action
        self.logprobs[environment_id, t] = log_probability
        self.state_values[environment_id, t] = value
        self.rewards[environment_id, t] = reward
        self.is_terminal[environment_id, t] = done
    
    def compute_gae(self, environment_id: int, gamma: float, gae_lambda_value: float):
        for t in range(self.T - 1, -1, -1):
            next_non_terminal = 1.0 - self.is_terminal[environment_id, t]
            delta_t = (
                self.rewards[environment_id, t] + 
                gamma * self.state_values[environment_id, t+1] * next_non_terminal - 
                self.state_values[environment_id, t]
            )
            
            if t == (self.T - 1):
                A_t = delta_t
            else:
                A_t = delta_t + gamma * gae_lambda_value * self.advantages[environment_id, t+1] * next_non_terminal
            
            self.advantages[environment_id, t] = A_t
    
    def get_flattened_data(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.advantages.reshape(-1),
            self.states.reshape(-1, 4, 84, 84),
            self.actions.reshape(-1),
            self.logprobs.reshape(-1),
            self.state_values[:, :self.T].reshape(-1)
        )


class PPOTrainer:
    def __init__(self, configuration: PPOConfig):
        self.configuration = configuration
        

        os.makedirs('plots', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('videos_highscores', exist_ok=True)
        

        print(" Creating environments...")
        self.envs = ParallelEnvironments(configuration.num_environments)
        print(f" Created {configuration.num_environments} parallel environments")
        

        print("\n Creating Actor-Critic model...")
        self.actorcritic = ActorCritic(4).to(configuration.device)
        self.optimizer = torch.optim.Adam(
            self.actorcritic.parameters(), 
            lr=configuration.learning_rate
        )
        

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=1.0, 
            end_factor=0.1, 
            total_iters=configuration.nb_iterations
        )
        
        print(f" Model created")
        configuration.print_config()
        

        self.max_reward = 0
        self.total_rewards = [[] for _ in range(configuration.num_environments)]
        self.smoothed_rewards = [[] for _ in range(configuration.num_environments)]
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.all_episode_rewards = []
        self.action_counts = np.zeros(4)
    
    def collect_experience(self, buffer: ExperienceBuffer) -> int:
        episodes_completed = 0
        
        for environment_id in range(self.configuration.num_environments):
            with torch.no_grad():
                for t in range(self.configuration.T):

                    observation = torch.from_numpy(
                        np.array(self.envs.observations[environment_id]) / 255.0
                    ).unsqueeze(0).float().to(self.configuration.device)
                    

                    logits, value = self.actorcritic(observation)
                    logits, value = logits.squeeze(0), value.squeeze(0)
                    m = torch.distributions.categorical.Categorical(logits=logits)


                    action = m.sample().item()
                    

                    self.action_counts[action] += 1

                    log_probability = m.log_prob(torch.tensor([action]).to(self.configuration.device))
                    

                    _, reward, dead, done, _ = self.envs.step(environment_id, action)
                    

                    reward = np.sign(reward)


                    buffer.store_step(environment_id, t, observation, action, log_probability, value, reward, done)


                    if dead:
                        episodes_completed += 1
                        self._handle_episode_end(environment_id, t)


                final_obs = torch.from_numpy(
                    np.array(self.envs.observations[environment_id]) / 255.0
                ).unsqueeze(0).float().to(self.configuration.device)
                buffer.state_values[environment_id, self.configuration.T] = self.actorcritic(final_obs)[1].squeeze(0)


                buffer.compute_gae(environment_id, self.configuration.gamma, self.configuration.gae_lambda_value)
        
        return episodes_completed
    
    def _handle_episode_end(self, environment_id: int, t: int):
        episode_reward = self.envs.total_rewards[environment_id]
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.all_episode_rewards.append(episode_reward)
        

        if self.episode_count % 10 == 0:
            avg_reward = np.mean(self.episode_rewards)
            total_steps = self.current_iteration * self.configuration.T * self.configuration.num_environments + environment_id * self.configuration.T + t
            print(f"Episode {self.episode_count:4d} | "
                  f"Steps: {total_steps:8d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(100): {avg_reward:6.2f}")
        

        # if self.episode_count % 100 == 0 and self.episode_count > 0:
        #     self._print_action_distribution()
        

        if episode_reward > self.max_reward:
            self.max_reward = episode_reward
            torch.save(self.actorcritic.state_dict(), f"models/best_model_{self.max_reward:.0f}.pth")
            print(f" New best reward: {self.max_reward:.0f}")
            

            self.envs.record_video(self.actorcritic, self.configuration.device, self.max_reward)

        self.total_rewards[environment_id].append(episode_reward)
        self.envs.reset_env(environment_id)
    
    def _print_action_distribution(self):
        total_actions = self.action_counts.sum()
        if total_actions > 0:
            print(f"\n Action Distribution (Episode {self.episode_count}):")
            for i, (name, count) in enumerate(zip(ParallelEnvironments.ACTION_NAMES, self.action_counts)):
                pct = (count / total_actions * 100)
                print(f"   {name:5s}: {pct:5.1f}% ({int(count):,} actions)")
            

            max_pct = (self.action_counts.max() / total_actions * 100)
            if max_pct > 80:
                dominant_action = ParallelEnvironments.ACTION_NAMES[self.action_counts.argmax()]
                print(f"     WARNING: Agent heavily favors {dominant_action} ({max_pct:.1f}%)")
                print(f"     This suggests the agent may not be learning properly!")
            print()
            

            self.action_counts = np.zeros(4)
    
    def update_policy(self, buffer: ExperienceBuffer, iteration: int):
        flat_advantages, flat_states, flat_actions, flat_old_logprobs, flat_old_values = buffer.get_flattened_data()
        
        for epoch in range(self.configuration.K):

            indices = torch.randperm(len(flat_advantages), device=self.configuration.device)
            
            for start in range(0, len(flat_advantages), self.configuration.batch_size):
                end = min(start + self.configuration.batch_size, len(flat_advantages))
                mb_indices = indices[start:end]


                mb_advantages = flat_advantages[mb_indices]
                mb_states = flat_states[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_old_logprobs = flat_old_logprobs[mb_indices]
                mb_old_values = flat_old_values[mb_indices]


                logits, value = self.actorcritic(mb_states)
                value = value.squeeze(-1)
                m = torch.distributions.categorical.Categorical(logits=logits)
                log_probability = m.log_prob(mb_actions)
                

                returns = mb_advantages + mb_old_values


                ratio = torch.exp(log_probability - mb_old_logprobs)
                alpha = 1.0 - iteration / self.configuration.nb_iterations
                clip_range = 0.1 * alpha
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()


                value_loss1 = F.mse_loss(returns, value, reduction='none')
                value_clipped = mb_old_values + torch.clamp(value - mb_old_values, -clip_range, clip_range)
                value_loss2 = F.mse_loss(returns, value_clipped, reduction='none')
                value_loss = torch.max(value_loss1, value_loss2).mean()


                entropy_loss = -m.entropy().mean()


                loss = policy_loss + self.configuration.vf_coef * value_loss + self.configuration.ent_coef * entropy_loss


                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(), self.configuration.max_grad_norm)
                self.optimizer.step()
    
    def plot_progress(self):
        plt.figure(figsize=(10, 6))
        for environment_id in range(self.configuration.num_environments):
            if len(self.total_rewards[environment_id]) > 0:
                self.smoothed_rewards[environment_id].append(np.mean(self.total_rewards[environment_id]))
                plt.plot(self.smoothed_rewards[environment_id], alpha=0.6)
        
        self.total_rewards = [[] for _ in range(self.configuration.num_environments)]
        plt.title("Average Reward on Breakout")
        plt.xlabel("Training Epochs (x400 iterations)")
        plt.ylabel("Average Reward per Episode")
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/training_progress.png')
        plt.close()
    
    def train(self):
        print("\n Starting training...\n")
        
        for iteration in range(self.configuration.nb_iterations):
            self.current_iteration = iteration
            

            buffer = ExperienceBuffer(self.configuration.num_environments, self.configuration.T, self.configuration.device)
            

            self.collect_experience(buffer)
            

            if (iteration % 400 == 0) and iteration > 0:
                self.plot_progress()
            

            self.update_policy(buffer, iteration)
            

            self.scheduler.step()
        
        print("\n Training complete!")
        torch.save(self.actorcritic.state_dict(), "models/Breakout_final_model.pth")
        print(" Final model saved to models/final_model.pth")


def main():
    print("=" * 70)
    print("PPO for Breakout")
    print("=" * 70)
    print()
    

    configuration = PPOConfig(
        num_environments=8,
        T=128,
        K=4,
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        nb_iterations=40000
    )
    

    trainer = PPOTrainer(configuration)
    trainer.train()


if __name__ == "__main__":
    main()
