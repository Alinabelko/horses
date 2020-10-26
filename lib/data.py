import os
import csv
import glob
import numpy as np
import collections
import pandas as pd
from pathlib import Path

Prices = collections.namedtuple('Prices', field_names=['seconds_to_start', 'selection_id', 'bet_type',
 'price', 'price_size', 'stack_sum_back', 'stack_sum_lay', 'back_price', 'lay_price'])

def read_csv(file_name, sep=';', filter_data=True, fix_open_price=False):
  print("Reading", file_name)
  df = pd.read_csv(file_name, sep=sep, decimal=',')
  df.columns = ['SECONDS_TO_START', 'BET_TYPE',
                'PRICE', 'PRICE_SIZE', 
                'STACK_SUM_BACK', 'STACK_SUM_LAY',
                'BACK_PRICE', 'LAY_PRICE']
  df['SECONDS_TO_START'] = df['SECONDS_TO_START'].astype('int')
  df['SELECTION_ID'] = Path(file_name).stem
  df = df.sort_values(by=['SECONDS_TO_START'], ascending=False )
  
  print("Read done, got %d rows" % (len(df)))
  return Prices(seconds_to_start=np.array(df['SECONDS_TO_START'], dtype=np.float32),
                  selection_id=np.array(df['SELECTION_ID'], dtype=np.string_),
                  bet_type=np.array(df['BET_TYPE'], dtype=np.bool_),
                  price=np.array(df['PRICE'], dtype=np.float32),
                  price_size=np.array(df['PRICE_SIZE'], dtype=np.float32),
                  stack_sum_back=np.array(df['STACK_SUM_BACK'], dtype=np.float32),
                  stack_sum_lay=np.array(df['STACK_SUM_LAY'], dtype=np.float32),
                  back_price=np.array(df['BACK_PRICE'], dtype=np.float32),
                  lay_price=np.array(df['LAY_PRICE'], dtype=np.float32))


def prices_to_relative(prices):
    """
    Convert prices to relative
    """
    assert isinstance(prices, Prices)
    rp = 1/prices.price
    rps = prices.price_size / prices.price
    return Prices(seconds_to_start=prices.seconds_to_start,
                  selection_id=prices.selection_id, bet_type=prices.bet_type,
                  price=rp, price_size=rps, stack_sum_back=prices.stack_sum_back,
                  stack_sum_lay=prices.stack_sum_lay,
                  back_price=prices.back_price,
                  lay_price=prices.lay_price)


def load_relative(csv_file):
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result