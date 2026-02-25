import gymnasium as gym
import torch as th
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

custom_net_arch = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[64, 64], vf=[64, 64]))

tmp_path = "/tmp/sb3_log/"

results = {}

def train_bench(algo_name, algo_class):
    env = gym.make("CartPole-v1", render_mode="human")
    env = Monitor(env)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    if algo_name == "DQN":
        
        custom_net_arch = [64, 64]
    else:
        custom_net_arch = dict(pi=[64, 64], vf=[64, 64])
    
    policy_kwargs = dict(net_arch=custom_net_arch)
    
    
    # Initialize Model
    model = algo_class(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=0
    )
    model.set_logger(new_logger)
    print(f"🚀 Training {algo_name}...")
    model.learn(total_timesteps=2000, tb_log_name=algo_name)

    latest_stats = model.logger.name_to_value
    
    
    loss = latest_stats.get("train/loss") or latest_stats.get("train/value_loss")
 
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    results[algo_name] = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "loss": loss
    }
    env.close()

# 4. Execute Benchmark
algos = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
for name, cls in algos.items():
    train_bench(name, cls)

print(results)