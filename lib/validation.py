import numpy as np

import torch

from lib import environ


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        have_position = False
        position_steps = 0
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            obs, reward, done, _ = env.step(action_idx)

            if reward is None:
              reward = 0

            if (action == environ.Actions.Lay or action == environ.Actions.Back) and have_position:
                have_position = True
                position_steps = 0
            elif action == environ.Actions.Close:
                profit = reward
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position_steps = 0
                have_position = False
            elif env._state.have_position:
                position_steps += 1
               
            total_reward += reward
            episode_steps += 1

            if done:
                if have_position is True:                  
                    profit = reward
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                    position_steps = 0
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return { key: np.mean(vals) for key, vals in stats.items() }
