import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()
print(obs,info)

print("Starting Observation:", obs)

print("Action Space:", env.action_space)

eps_over = False
t_reward = 0

while not eps_over:
    action = env.action_space.sample()
    
    obs,reward,terminated,truncated,info = env.step(action)

    t_reward += reward
    eps_over = terminated or truncated    

print("Episode Over:", t_reward)
env.close()
