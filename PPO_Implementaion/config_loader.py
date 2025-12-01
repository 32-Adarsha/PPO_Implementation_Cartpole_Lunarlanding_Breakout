import json
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PPOConfig:
    num_environments: int
    steps_per_iter: int
    epochs: int
    batch_size: int
    gamma: float
    gae_lambda_value: float
    learning_rate: float
    value_coefficient: float
    entropy_coefficient: float
    gradient_clip: float
    max_iterations: int
    clip_range: float
    device: str = 'cpu'
    
    def print_config(self):
        print(f"\n Configuration:")
        print(f"   Environments: {self.num_environments}")
        print(f"   Steps per iteration: {self.steps_per_iter}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Gamma: {self.gamma}")
        print(f"   GAE Lambda: {self.gae_lambda_value}")
        print(f"   Value coefficient: {self.value_coefficient}")
        print(f"   Entropy coefficient: {self.entropy_coefficient}")
        print(f"   Gradient clip: {self.gradient_clip}")
        print(f"   Clip range: {self.clip_range}")
        print(f"   Device: {self.device}")
        print(f"   Total iterations: {self.max_iterations:,}")


def load_config(config_path: str = "global_config.json", environment: str = "cartpole") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        global_config = json.load(f)
    
    if environment not in global_config["environments"]:
        available = list(global_config["environments"].keys())
        raise ValueError(f"Environment '{environment}' not found. Available: {available}")
    
    env_config = global_config["environments"][environment]
    training_config = global_config["training"]
    paths_config = global_config["paths"]
    
    return {
        "env_config": env_config,
        "training_config": training_config,
        "paths_config": paths_config
    }


def create_ppo_config(environment: str = "cartpole", config_path: str = "global_config.json") -> PPOConfig:
    config_data = load_config(config_path, environment)
    
    hyperparams = config_data["env_config"]["hyperparameters"]
    device = config_data["training_config"]["device"]
    
    return PPOConfig(
        num_environments=hyperparams["num_environments"],
        steps_per_iter=hyperparams["steps_per_iter"],
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        gae_lambda_value=hyperparams["gae_lambda_value"],
        learning_rate=hyperparams["learning_rate"],
        value_coefficient=hyperparams["value_coefficient"],
        entropy_coefficient=hyperparams["entropy_coefficient"],
        gradient_clip=hyperparams["gradient_clip"],
        max_iterations=hyperparams["max_iterations"],
        clip_range=hyperparams["clip_range"],
        device=device
    )


def get_env_info(environment: str = "cartpole", config_path: str = "global_config.json") -> Dict[str, Any]:
    config_data = load_config(config_path, environment)
    env_config = config_data["env_config"]
    
    return {
        "env_name": env_config["env_name"],
        "observation_dim": env_config.get("observation_dim"),
        "observation_shape": env_config.get("observation_shape"),
        "action_dim": env_config["action_dim"],
        "hidden_dim": env_config.get("hidden_dim"),
        "solved_threshold": env_config["solved_threshold"]
    }


def setup_directories(config_path: str = "global_config.json"):
    with open(config_path, 'r') as f:
        global_config = json.load(f)
    
    paths = global_config["paths"]
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        print(f" Created directory: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Configuration Utility")
    print("=" * 60)
    
    environments = ["cartpole", "lunarlander", "breakout"]
    
    for env in environments:
        print(f"\n{env.upper()} Configuration:")
        print("-" * 60)
        
        try:
            configuration = create_ppo_config(env)
            env_info = get_env_info(env)
            
            print(f"Environment: {env_info['env_name']}")
            print(f"Solved Threshold: {env_info['solved_threshold']}")
            configuration.print_config()
        except Exception as e:
            print(f"Error loading {env}: {e}")
    
    print("\n" + "=" * 60)
    print("Setting up directories...")
    setup_directories()
    print("=" * 60)
