from sac_agent import Agent
import torch
import gym
import numpy as np

max_episode_num = 30000
env = gym.make('Humanoid-v2')

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

is_train = False
# is_train = True

if is_train is True:
    agent = Agent(env, device)
    agent.train(max_episode_num=max_episode_num, max_time=5000)
else:
    agent = Agent(env, device, write_mode=False)
    agent.load_model(agent.save_model_path + 'humanoid_v2_sac_8000000.pth')
    avg_episode_reward = 0.
    num_test = 5
    episode_rewards = []
    for _ in range(num_test):    
        state = env.reset()
        episode_reward = 0.
        for _ in range(10000):
            action = agent.actor.get_action(torch.tensor([state]).to(device).float(), stochastic=False)
            env.render()         
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break 
        episode_rewards.append(episode_reward)

    print('avg_episode_reward : ', np.mean(episode_rewards))