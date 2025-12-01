import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
from typing import List, Dict, Tuple, Optional


def load_training_data(log_file: str) -> Tuple[List[float], Dict]:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Training log not found: {log_file}")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    return data.get('rewards', []), data.get('configuration', {})


def smooth_rewards(rewards: List[float], window: int = 10) -> np.ndarray:
    if len(rewards) < window:
        return np.array(rewards)
    
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    return smoothed


def detect_environment(log_file: str) -> Tuple[str, float, str]:
    filename = os.path.basename(log_file).lower()
    
    if 'cartpole' in filename:
        return 'CartPole-v1', 490.0, 'blue'
    elif 'lunarlander' in filename or 'lunar' in filename:
        return 'LunarLander-v3', 200.0, 'red'
    elif 'breakout' in filename:
        return 'Breakout', 40.0, 'orange'
    else:
        return 'Unknown Environment', 0.0, 'gray'


def create_plot(
    rewards: List[float],
    configuration: Dict,
    env_name: str,
    solved_threshold: float,
    output_path: str,
    color: str = 'blue',
    window: int = 10
):
    if not rewards:
        print(f"Error: No reward data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = smooth_rewards(rewards, window=window)
    
    ax.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.5, label='Raw Rewards')
    ax.plot(episodes[:len(smoothed)], smoothed, color=color, linewidth=2.5, label=f'Smoothed ({window}-ep avg)')
    
    if solved_threshold > 0:
        ax.axhline(y=solved_threshold, color='green', linestyle='--', linewidth=2, label=f'Solved ({solved_threshold})')
    
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'{env_name} Training Progress', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    combined_text = (
        f"Training Statistics:\n"
        f"Episodes: {len(rewards)}\n"
        f"Final Avg: {final_avg:.2f}\n"
        f"Max: {max_reward:.2f}\n"
        f"Min: {min_reward:.2f}\n"
        f"Solved: {'Yes' if solved_threshold > 0 and final_avg >= solved_threshold else 'No'}\n"
    )
    
    if configuration:
        combined_text += (
            f"\nHyperparameters:\n"
            f"Envs: {configuration.get('num_environments', 'N/A')}\n"
            f"Steps: {configuration.get('steps_per_iter', 'N/A')}\n"
            f"Epochs: {configuration.get('epochs', 'N/A')}\n"
            f"Batch: {configuration.get('batch_size', 'N/A')}\n"
            f"LR: {configuration.get('learning_rate', 'N/A')}\n"
            f"GAE λ: {configuration.get('gae_lambda_value', 'N/A')}\n"
            f"Entropy: {configuration.get('entropy_coefficient', 'N/A')}"
        )
    
    ax.text(0.02, 0.98, combined_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.5),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot PPO training progress from a JSON log file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_training.py logs/cartpole_training.json
  python plot_training.py logs/lunarlander_training.json -o my_plot.png
  python plot_training.py logs/cartpole_training.json --window 20
        """
    )
    
    parser.add_argument('log_file', type=str,
                        help='Path to training log JSON file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output plot filename (default: auto-generated from input)')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Environment name (default: auto-detect from filename)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Solved threshold (default: auto-detect from filename)')
    parser.add_argument('--color', type=str, default=None,
                        help='Plot color (default: auto-detect from filename)')
    parser.add_argument('--window', type=int, default=10,
                        help='Smoothing window size (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPO Training Progress Visualization")
    print("=" * 60)
    
    try:
        print(f"\nLoading: {args.log_file}")
        rewards, configuration = load_training_data(args.log_file)
        print(f"Loaded {len(rewards)} episodes")
        
        env_name, solved_threshold, color = detect_environment(args.log_file)
        
        if args.env_name:
            env_name = args.env_name
        if args.threshold is not None:
            solved_threshold = args.threshold
        if args.color:
            color = args.color
        
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(os.path.basename(args.log_file))[0]
            output_path = f"{base_name}_plot.png"
        
        print(f"Environment: {env_name}")
        print(f"Solved threshold: {solved_threshold}")
        print(f"Generating plot...")
        
        create_plot(
            rewards=rewards,
            configuration=configuration,
            env_name=env_name,
            solved_threshold=solved_threshold,
            output_path=output_path,
            color=color,
            window=args.window
        )
        
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
