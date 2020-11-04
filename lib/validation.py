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
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            if(env._state.back_value != 0.0):
                close = env._state._cur_close(environ.Actions.Back)
            else:
                close = env._state._cur_close(environ.Actions.Lay)

            close_price = close

            if action == environ.Actions.Lay and position is None:
                position = env._state._cur_close(environ.Actions.Lay)
                position_steps = 0
            if action == environ.Actions.Back and position is None:
                position = env._state._cur_close(environ.Actions.Back)
                position_steps = 0
            elif action == environ.Actions.Close and position is not None:
                close_type = environ.Actions.Lay if env._state.back_value + env._state.lay_value > 0 else environ.Actions.Back
                close_price = env._state._prices.lay_price[env._state._offset] if close_type == environ.Actions.Lay else env._state._prices.back_price[env._state._offset]
                close_size = (env._state.back_value + env._state.lay_value) / close_price

                if close_type == environ.Actions.Lay:
                  env._state.lay_value = env._state.lay_value - env._state._prices.lay_price[env._state._offset] * close_size
                  reward = 1 - env._state.open_price / close_price
                  print("reward:", round(reward*100, 3),"bet: Lay", "open_price", 
                    env._state.open_price, "close_price", close_price, 
                    "selection_id", 
                    env._state._prices.selection_id[env._state._offset].decode(),
                    "seconds_to_start", env._state._prices.seconds_to_start[env._state._offset],
                    "offset", env._state._offset,
                    "keep", env._state.keep_duration,
                    "open_second", env._state.open_second,
                    "seconds_length", env._state.open_second - env._state._prices.seconds_to_start[env._state._offset] 
                  )
                else:
                  #смена знака для размера закрывающей ставки
                  close_size = - close_size
                  env._state.back_value =+ env._state._prices.back_price[env._state._offset] * close_size
                  reward = env._state.open_price / close_price - 1
                  print("reward:", round(reward*100, 3),"bet: Back", "open_price",
                      env._state.open_price, "close_price", close_price, 
                      "selection_id", 
                      env._state._prices.selection_id[env._state._offset].decode(),
                      "seconds_to_start", env._state._prices.seconds_to_start[env._state._offset],
                      "offset", env._state._offset,
                      "keep", env._state.keep_duration,
                      "open_second", env._state.open_second,
                      "seconds_length", env._state.open_second - env._state._prices.seconds_to_start[env._state._offset]
                  )

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                  close_type = environ.Actions.Lay if env._state.back_value + env._state.lay_value > 0 else environ.Actions.Back
                  close_price = env._state._prices.lay_price[env._state._offset] if close_type == environ.Actions.Lay else env._state._prices.back_price[env._state._offset]
                  close_size = (env._state.back_value + env._state.lay_value) / close_price

                  if close_type == environ.Actions.Lay:
                    #env._state.lay_value = env._state.lay_value - env._state._prices.lay_price[env._state._offset] * close_size
                    profit = 1 - env._state.open_price / close_price
                  else:
                    #смена знака для размера закрывающей ставки
                    close_size = - close_size
                    #env._state.back_value =+ env._state._prices.back_price[env._state._offset] * close_size
                  profit = env._state.open_price / close_price - 1

                  stats['order_profits'].append(profit)
                  stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return { key: np.mean(vals) for key, vals in stats.items() }
