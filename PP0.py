import os
import json
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "global_config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)



class NN(nn.Module):
    def __init__(self , input_size , hidden_layers , output_size , act_fun = nn.ReLU) -> None:
        super().__init__()

        layers = []
        _in = input_size
        for h in hidden_layers: 
            layers.append(nn.Linear(_in , h));
            layers.append(act_fun())
            _in = h

        layers.append(nn.Linear(_in , output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class ActorCritic(nn.Module):
    def __init__(self, Input_Dim , Hidden_Dim , Output_Dim):
        super().__init__()

        #    NOTE:  Need to add dropout 
        self.actor = NN(Input_Dim , Hidden_Dim , Output_Dim);
        self.critic = NN(Input_Dim , Hidden_Dim , 1);

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


class PPOAgent():
    def __init__(self ,  cfg) -> None:
        #  NOTE: I will put all the configuration for the  PPOAgent 
        self.hidden_layers  = cfg['hidden_layers']
        self.input_size = cfg['input_size'] 
        self.output_size = cfg['output_size']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.enthropy_coff = cfg['enthropy_coff']
        self.learing_rate = cfg['learing_rate']
        self.max_step = cfg['max_step']
        self.max_episode = cfg['max_episode'] 
        self.batch_size = cfg['batch_size']
        self.ppo_step = cfg['ppo_step'] 
        self.n_Trials = cfg['N_Trials']
        self.reward_threshold = cfg['reward_threshold']

        self.agent = ActorCritic(self.input_size , self.hidden_layers , self.output_size) 
        self.optimizer = optim.Adam(self.agent.parameters() , lr=self.learing_rate)

        self.env = cfg['env']

    
    def cal_reward(self, rewards):
        dis_reward = []
        prev_reward = 0
        for reward in reversed(rewards):
            prev_reward = self.gamma * prev_reward + reward
            dis_reward.insert(0, prev_reward)
        dis_reward = torch.tensor(dis_reward, dtype=torch.float32)
        return dis_reward
   
    
    def cal_advantage(self, rewards , values):
        advantages = rewards - values
        # NOTE: Normalized advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
        
    def cal_surrogate_loss(self , log_prob_old ,log_prob_new , advantages):
        advantages = advantages.detach()
        policy_ratio = ( log_prob_new - log_prob_old).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(policy_ratio, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2).mean()
        return surrogate_loss

   
    def cal_loss(self , surrogate_loss , entropy , rewards , pred_value):
        entropy_bonus = self.enthropy_coff * entropy.mean()
        policy_loss = -(surrogate_loss + entropy_bonus)
        value_loss = F.smooth_l1_loss(pred_value, rewards)  
        return policy_loss, value_loss


    def init_training(self):
        states = []
        actions = []
        actions_log_probability = []
        values = []
        rewards = []
        done = False
        episode_reward = 0
        return states, actions, actions_log_probability, values, rewards, done, episode_reward

    def sample_trajectory(self):
        states, actions, actions_log_probability, values, rewards, done, episode_reward = self.init_training()
        state = self.env.reset()
        self.agent.train()
        # NOTE : Could be replaced with max_step
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            states.append(state)
            action_pred, value_pred = self.agent(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            state, reward, done, _ = self.env.step(action.item())
            actions.append(action)
            actions_log_probability.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)
            episode_reward += reward

        states = torch.cat(states)
        actions = torch.cat(actions)
        actions_log_probability = torch.cat(actions_log_probability)
        values = torch.cat(values).squeeze(-1)
        dis_rewards = self.cal_reward(rewards)
        advantages = self.cal_advantage(dis_rewards, values)
        return episode_reward, states, actions, actions_log_probability, advantages, dis_rewards

    

    def update(self , states , actions , old_log_p , advantages  , rewards ):
        total_policy_loss = 0;
        total_value_loss = 0;

        actions = actions.detach()
        old_log_p = old_log_p.detach()
        advantages = advantages.detach()


        training_dataset = TensorDataset(states,actions,old_log_p,advantages,rewards)
        batch_dataset = DataLoader(training_dataset,batch_size=self.batch_size,shuffle=False)

        for _ in range(self.ppo_step):
            for _ , (batch_states , batch_action , batch_old_log_p , batch_advantage , batch_rewards) in enumerate(batch_dataset):
                action_pred , value_pred = self.agent(batch_states)
                value_pred = value_pred.squeeze(-1)
                action_prob = F.softmax(action_pred , dim=-1)
                action_dist = torch.distributions.Categorical(action_prob)
                new_log_p = action_dist.log_prob(batch_action)
                enthropy = action_dist.entropy()
                surrogate_loss = self.cal_surrogate_loss(batch_old_log_p , new_log_p , batch_advantage)
                policy_loss , value_loss = self.cal_loss(surrogate_loss , enthropy , batch_rewards , value_pred)

                self.optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        return total_policy_loss/self.ppo_step , total_value_loss/self.ppo_step

    def evaluate(self):
        self.agent.eval()
        rewards = []
        done = False
        episode_reward = 0
        state = self.env.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = self.agent(state)
                action_prob = F.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _ = self.env.step(action.item())
            episode_reward += reward
        return episode_reward



    def train(self):
        #  NOTE: train logic
        train_rewards =[]
        test_rewards = []
        policy_losses = []
        value_losses = []

        for episode in range(self.max_episode):
            rewards , states , actions , old_log_p , advantage , dis_rewards = self.sample_trajectory()
            policy_loss , value_loss = self.update(states , actions , old_log_p , advantage , dis_rewards ) 
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            train_rewards.append(rewards)
            test_reward = self.evaluate()
            test_rewards.append(test_reward)
            mean_train_rewards = np.mean(train_rewards[-self.n_Trials:])
            mean_test_rewards = np.mean(test_rewards[-self.n_Trials:])
            mean_abs_policy_loss = np.mean(np.abs(policy_losses[-self.n_Trials:]))
            mean_abs_value_loss = np.mean(np.abs(value_losses[-self.n_Trials:]))
            if episode % 10 == 0:
                print(f'Episode: {episode:3} | \
                    Mean Train Rewards: {mean_train_rewards:3.1f} \
                    | Mean Test Rewards: {mean_test_rewards:3.1f} \
                    | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                    | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
            if mean_test_rewards >= self.reward_threshold :
                print(f'Reached reward threshold in {episode} episodes')
                break





