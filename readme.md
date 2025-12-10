# PPO Implementation for Gymnasium Environments

This repository contains a Proximal Policy Optimization (PPO) implementation for training reinforcement learning agents on various Gymnasium environments including CartPole, LunarLander, and Breakout.

## Project Structure

```
PPO/
├── PPO_Implementaion/          # Main implementation directory
│   ├── PPO_CartPole.py         # CartPole training script
│   ├── PPO_LunarLander.py      # LunarLander training script
│   ├── PPO_Breakout.py         # Breakout training script
│   ├── evaluate_cartpole.py    # CartPole evaluation script
│   ├── evaluate_lunarlander.py # LunarLander evaluation script
│   ├── evaluate_model.py       # Breakout evaluation script
│   ├── plot_training.py        # Training visualization script
│   └── config_loader.py        # Configuration loader utility
├── models/                     # Saved model checkpoints
├── logs/                       # Training logs (JSON format)
├── plots/                      # Training plots
├── global_config.json          # Global configuration file
└── README.md                   # This file
```

## Installation

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install gymnasium torch numpy matplotlib ale-py
   ```

> **Note:** Make sure to activate the virtual environment (`source venv/bin/activate`) before running any training or evaluation scripts.

## Training

### CartPole
```bash
cd PPO_Implementaion
python PPO_CartPole.py
```

### LunarLander
```bash
cd PPO_Implementaion
python PPO_LunarLander.py
```

### Breakout
```bash
cd PPO_Implementaion
python PPO_Breakout.py
```

## PPO Breakout Demo

![PPO Breakout Demo](/Short_Video/breakout.gif)

Training logs and model checkpoints will be saved automatically in the `logs/` and `models/` directories.

## Evaluating Models

### CartPole Evaluation

**Basic evaluation (5 episodes with rendering):**
```bash
cd PPO_Implementaion
python evaluate_cartpole.py
```

**Custom evaluation:**
```bash
python evaluate_cartpole.py --model ../models/solved_model.pth --episodes 10 --no-render
```

**Available arguments:**
- `--model`: Path to the model file (default: `models/solved_model.pth`)
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--no-render`: Disable visual rendering for faster evaluation

**Example output:**
```
Episode  1 | Reward:  500.0 | Steps:  500
Episode  2 | Reward:  500.0 | Steps:  500
...
==================================================
Evaluation Summary (5 episodes)
==================================================
Average Reward: 500.00
Min Reward:     500.0
Max Reward:     500.0
Std Reward:       0.00
==================================================
```

---

### LunarLander Evaluation

**Basic evaluation (5 episodes with rendering):**
```bash
cd PPO_Implementaion
python evaluate_lunarlander.py
```

**Custom evaluation:**
```bash
python evaluate_lunarlander.py --model ../models/lunarlander_solved_model.pth --episodes 10 --no-render
```

**Available arguments:**
- `--model`: Path to the model file (default: `models/lunarlander_solved_model.pth`)
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--no-render`: Disable visual rendering for faster evaluation

**Example output:**
```
Episode  1 | Reward:  245.67 | Steps:  234 |  SUCCESS
Episode  2 | Reward:  189.23 | Steps:  267 |  PARTIAL
...
============================================================
Evaluation Summary (5 episodes)
============================================================
Average Reward:      215.45
Min Reward:          189.23
Max Reward:          245.67
Std Reward:           23.12
Successful Landings: 4/5 (80.0%)
============================================================
```

**Success criteria:**
- **SUCCESS**: Reward ≥ 200 (successful landing)
- **PARTIAL**: Reward ≥ 0 (partial success)
- **CRASH**: Reward < 0 (crashed)

---

### Breakout Evaluation

**Basic evaluation (10 episodes, greedy policy):**
```bash
cd PPO_Implementaion
python evaluate_model.py
```

**Custom evaluation:**
```bash
python evaluate_model.py --model ../Breakout_Model/best_model_358.pth --episodes 20 --device cpu --render
```

**Available arguments:**
- `--model`: Path to the model file (default: `models/best_model_358.pth`)
- `--episodes`: Number of episodes to evaluate (default: 10)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--render`: Enable visual rendering (default: False)
- `--stochastic`: Use stochastic action selection instead of greedy (default: False)

**Example output:**
```
======================================================================
PPO Breakout Model Evaluation
======================================================================

 Model: models/best_model_358.pth
 Episodes: 10
 Device: cpu
 Action Selection: Greedy (argmax)
 Render: False

 Starting evaluation...

Episode  1/10 | Reward:   42.0 | Length:  3456 steps
Episode  2/10 | Reward:   38.0 | Length:  3201 steps
...
======================================================================
 Evaluation Results
======================================================================

 Reward Statistics:
   Mean:    40.50
   Std:      3.21
   Min:     35.00
   Max:     45.00
   Median:  41.00

 Episode Length Statistics:
   Mean:   3289.40 steps
   Std:     234.56 steps
   Min:    2987 steps
   Max:    3678 steps

 Action Distribution:
   NOOP :  12.3% (4,056 actions)
   FIRE :   8.7% (2,867 actions)
   RIGHT:  39.5% (13,012 actions)
   LEFT :  39.5% (13,011 actions)

======================================================================
```

**Action selection modes:**
- **Greedy (default)**: Uses `argmax` to select the action with highest probability
- **Stochastic**: Samples actions from the policy distribution (use `--stochastic` flag)

---

## Visualizing Training Results

To generate training plots from saved logs:

```bash
cd PPO_Implementaion
python plot_training.py
```

This will create plots showing:
- Episode rewards (raw and smoothed)
- Solved threshold line
- Training hyperparameters
- Training statistics

Plots are saved in the `plots/` directory.

## Configuration

All hyperparameters are defined in `global_config.json`. You can modify:
- Learning rates
- Batch sizes
- Network architectures
- Environment-specific parameters
- Training iterations

## Available Models

The `models/` directory contains trained checkpoints:
- `solved_model.pth` - CartPole solved model
- `cartpole_final_model_Env_1.pth` - CartPole final checkpoint
- `lunarlander_solved_model.pth` - LunarLander solved model
- `lunarlander_final_model_env_1.pth` - LunarLander final checkpoint

## Tips for Evaluation

1. **Quick testing**: Use `--no-render` flag for faster evaluation without visualization
2. **Statistical significance**: Run at least 10-20 episodes for reliable performance metrics
3. **Greedy vs Stochastic**: Greedy policy typically performs better but stochastic can reveal exploration behavior
4. **Device selection**: Use `--device cpu` if CUDA is not available or for debugging
5. **Rendering**: Enable `--render` to visually inspect agent behavior (slows down evaluation)

## Performance Benchmarks

| Environment   | Solved Threshold | Typical Training Time | Expected Eval Performance |
|---------------|------------------|----------------------|---------------------------|
| CartPole      | 490.0            | ~5-10 minutes        | 500.0 (perfect)           |
| LunarLander   | 200.0            | ~30-60 minutes       | 200-250                   |
| Breakout      | 40.0             | ~4-8 hours           | 35-45                     |

## Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Use `--device cpu` or reduce batch size in `global_config.json`

**Issue**: Module not found errors
- **Solution**: Ensure virtual environment is activated and all dependencies are installed

**Issue**: Model file not found
- **Solution**: Check the path to the model file and ensure it exists in the `models/` directory

**Issue**: Rendering not working
- **Solution**: Ensure you have a display available. On headless servers, use `--no-render`

## License

This project is for educational purposes as part of CS591 coursework.
