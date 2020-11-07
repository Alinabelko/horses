import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import collections

from . import data

DEFAULT_BARS_COUNT = 100

MINIMAL_BET = 4.0
KEEP_REWARD = 0.00
OPEN_BET_PENALTY = 0.00
CLOSE_BET_PENALTY = 0.00
ZERO_BET_PENALTY = 0.0

class Actions(enum.Enum):
    Skip = 0
    Lay = 1
    Back = 2
    Close = 3

class State:
    def __init__(self, bars_count, reset_on_close=False, reward_on_close=True, is_validation=False):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.open_bet_penalty = OPEN_BET_PENALTY
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.is_validation = is_validation

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.lay_value = 0.0
        self.back_value = 0.0
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset
        self.keep_duration = 0
        self.open_second = 0
        self.previous_profit = 0

    @property
    def shape(self):
        # [prices] * bars + have_position + keep + rel_profit
        return (15*self.bars_count + 1 + 1 + 1, )

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.seconds_to_start[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.bet_type[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.avg_price[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.price_size[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind1[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind2[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind3[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind4[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind5[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind6[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind7[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind8[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.ind9[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.stack_back_proportion[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.stack_lay_proportion[self._offset + bar_idx]
            shift += 1
        res[shift] = float(self.keep_duration)
        shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            if(self.bet_type == Actions.Lay):
                close = self._cur_close(Actions.Back)
                res[shift] = 1 - self.open_price /  close
            else:
                close = self._cur_close(Actions.Lay)              
                res[shift] = self.open_price / close - 1
        return res


    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        reward = 0.0
        done = False

        if action == Actions.Back and not self.have_position:
            self.bet_type = Actions.Back
            self.have_position = True
            self.open_second = self._prices.seconds_to_start[self._offset]
            self.open_price = self._prices.back_price[self._offset]
            self.back_value += self._prices.back_price[self._offset] * MINIMAL_BET
            reward -= self.open_bet_penalty
            self.previous_profit = 0
        if action == Actions.Lay and not self.have_position:
            self.bet_type = Actions.Lay
            self.have_position = True
            self.open_second = self._prices.seconds_to_start[self._offset]
            self.open_price = self._prices.lay_price[self._offset]
            self.lay_value -= self._prices.lay_price[self._offset] * MINIMAL_BET
            reward -= self.open_bet_penalty
            self.previous_profit = 0

        if action == Actions.Close and self.have_position:
            close_type = Actions.Lay if (self.back_value + self.lay_value) > 0 else Actions.Back
            close_price = self._prices.lay_price[self._offset] if close_type == Actions.Lay else self._prices.back_price[self._offset]
            close_size = (self.back_value + self.lay_value) / close_price
            if close_type == Actions.Lay:
              self.lay_value -= self._prices.lay_price[self._offset] * close_size
              reward = self.open_price / close_price - 1           
              if(reward == 0):
                reward -= ZERO_BET_PENALTY
            else:
              #смена знака для размера закрывающей ставки
              close_size = - close_size
              self.back_value += self._prices.back_price[self._offset] * close_size
              reward = 1 - self.open_price / close_price                   
              if(reward == 0):
                reward -= ZERO_BET_PENALTY
            self.print_log(reward, close_price)
            reward -= CLOSE_BET_PENALTY
            self.have_position = False
            self.open_price = 0.0
            self.keep_duration = 0
            self.previous_profit = 0
            self.open_second = 0
            self.bet_type = Actions.Close

        if self.have_position:
          profit = 0
          if(self.bet_type == Actions.Back):
              close = self._cur_close(Actions.Lay)
              profit = self.open_price / close - 1
          elif(self.bet_type == Actions.Lay):
              close = self._cur_close(Actions.Back)
              profit = 1 - self.open_price / close
          reward += profit - self.previous_profit
          reward += KEEP_REWARD
          self.keep_duration += 1
          self.previous_profit = profit

        self._offset += 1 

        done = True if self._prices.seconds_to_start[self._offset] <= 10 or self._offset >= self._prices.seconds_to_start.shape[0]-1 else False

        if done:
          if(self.have_position):
            close_type = Actions.Lay if self.back_value + self.lay_value > 0 else Actions.Back
            close_price = self._prices.lay_price[self._offset] if close_type == Actions.Lay else self._prices.back_price[self._offset]
            close_size = (self.back_value + self.lay_value) / close_price
            if close_type == Actions.Lay:
              self.lay_value -= self._prices.lay_price[self._offset] * close_size
              reward = self.open_price / close_price - 1
              if(reward == 0):
                reward -= ZERO_BET_PENALTY
              self.print_log(reward, close_price)
            else:
              #смена знака для размера закрывающей ставки
              close_size = - close_size
              self.back_value += self._prices.back_price[self._offset] * close_size
              reward = 1 - self.open_price / close_price             
              if(reward == 0):
                reward -= ZERO_BET_PENALTY
              self.print_log(reward, close_price)
              reward -= CLOSE_BET_PENALTY
          self.have_position = False
          self.bet_type = Actions.Close
          self.back_value = 0.0
          self.lay_value = 0.0
        
        return reward, done

    def _cur_close(self, bet_type):
        if(bet_type == Actions.Back):
          return self._prices.back_price[self._offset]      
        return self._prices.lay_price[self._offset]

    def print_log(self, reward, close_price):
      if(not self.is_validation):
        print("rew:", round(reward*100, 3),
                  self.bet_type,
                  self.open_price, "-", close_price,
                  "seconds", 
                  self.open_second, "-", 
                  self._prices.seconds_to_start[self._offset],                    
                  "offset", self._offset,
                  "keep", self.keep_duration,
                  "selection_id", 
                  self._prices.selection_id[self._offset].decode(),
                   )
      else:
        print("val:", round(reward*100, 3),
                  self.bet_type,
                  self.open_price, "-", close_price,  
                  "seconds", 
                  self.open_second, "-", 
                  self._prices.seconds_to_start[self._offset]      ,    
                  "offset", self._offset,
                  "keep", self.keep_duration,
                  "selection_id", 
                  self._prices.selection_id[self._offset].decode(),
                   )
      return

class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        return (19, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.seconds_to_start[self._offset-ofs:self._offset+1]
        res[1] = self._prices.bet_type[self._offset-ofs:self._offset+1]
        res[2] = self._prices.avg_price[self._offset-ofs:self._offset+1]
        res[3] = self._prices.price_size[self._offset-ofs:self._offset+1]
        res[4] = self._prices.ind1[self._offset-ofs:self._offset+1]
        res[5] = self._prices.ind2[self._offset-ofs:self._offset+1]
        res[6] = self._prices.ind3[self._offset-ofs:self._offset+1]
        res[7] = self._prices.ind4[self._offset-ofs:self._offset+1]
        res[8] = self._prices.ind5[self._offset-ofs:self._offset+1]
        res[9] = self._prices.ind6[self._offset-ofs:self._offset+1]
        res[10] = self._prices.ind7[self._offset-ofs:self._offset+1]
        res[11] = self._prices.ind8[self._offset-ofs:self._offset+1]
        res[12] = self._prices.ind9[self._offset-ofs:self._offset+1]
        res[13] = self._prices.stack_back_proportion[self._offset-ofs:self._offset+1]
        res[14] = self._prices.stack_lay_proportion[self._offset-ofs:self._offset+1]
        res[16] = float(self.keep_duration)
        res[17] = float(self.have_position)
        if not self.have_position:
            res[18] = 0.0
        else:
            if(self.bet_type == Actions.Lay):
                close = self._cur_close(Actions.Back)
                res[18] = 1 - self.open_price /  close
            else:
                close = self._cur_close(Actions.Lay)              
                res[18] = self.open_price / close - 1
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=OPEN_BET_PENALTY, reset_on_close=False, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False, is_validation=False):
        assert isinstance(prices, dict)
        self._prices = prices
        self.is_validation = is_validation
        if state_1d:
            self._state = State1D(bars_count, reset_on_close, reward_on_close=False, is_validation = is_validation)
        else:
            self._state = State(bars_count, reset_on_close, reward_on_close=False, is_validation = is_validation)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = bars           
            while(prices.seconds_to_start[offset] > 600 or prices.seconds_to_start[offset] < 100):
              offset = self.np_random.choice(prices.avg_price.shape[0]-bars) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)