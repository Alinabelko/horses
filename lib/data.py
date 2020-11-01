import os
import csv
import glob
import numpy as np
import collections
import pandas as pd
from pathlib import Path

Prices = collections.namedtuple('Prices', 
field_names=['seconds_to_start', 'selection_id', 'bet_type',
                'ind1', 'ind2', 'ind3', 'ind4', 'ind5',
                'ind6', 'ind7', 'ind8', 'ind9',
                'avg_price', 'price_size', 'stack_back_proportion',
                'stack_lay_proportion', 'back_price', 'lay_price'])

def read_csv(file_name, sep=';', filter_data=True, fix_open_price=False):
  df = pd.read_csv(file_name, sep=sep, decimal=',')
  df.columns = ['SECONDS_TO_START', 'BET_TYPE',
                'AVG_PRICE', 
                'PRICE_SIZE', 
                'IND1', 'IND2', 'IND3', 'IND4', 'IND5',
                'IND6', 'IND7', 'IND8', 'IND9',
                'STACK_BACK_PROPORTION', 'STACK_LAY_PROPORTION',
                'BACK_PRICE', 'LAY_PRICE']
  df['SECONDS_TO_START'] = df['SECONDS_TO_START'].astype('int')
  df['SELECTION_ID'] = Path(file_name).stem
  df = df.sort_values(by=['SECONDS_TO_START'], ascending=False )
  
  print("Read done, got %d rows" % (len(df)))
  return Prices(seconds_to_start=np.array(df['SECONDS_TO_START'], dtype=np.float32),
                  selection_id=np.array(df['SELECTION_ID'], dtype=np.string_),
                  bet_type=np.array(df['BET_TYPE'], dtype=np.bool_),
                  avg_price=np.array(df['AVG_PRICE'], dtype=np.float32),
                  ind1=np.array(df['IND1'], dtype=np.float32),
                  ind2=np.array(df['IND2'], dtype=np.float32),
                  ind3=np.array(df['IND3'], dtype=np.float32),
                  ind4=np.array(df['IND4'], dtype=np.float32),
                  ind5=np.array(df['IND5'], dtype=np.float32),
                  ind6=np.array(df['IND6'], dtype=np.float32),
                  ind7=np.array(df['IND7'], dtype=np.float32),
                  ind8=np.array(df['IND8'], dtype=np.float32),
                  ind9=np.array(df['IND9'], dtype=np.float32),
                  price_size=np.array(df['PRICE_SIZE'], dtype=np.float32),
                  stack_back_proportion=np.array(df['STACK_BACK_PROPORTION'], dtype=np.float32),
                  stack_lay_proportion=np.array(df['STACK_LAY_PROPORTION'], dtype=np.float32),
                  back_price=np.array(df['BACK_PRICE'], dtype=np.float32),
                  lay_price=np.array(df['LAY_PRICE'], dtype=np.float32))


def load_relative(csv_file):
    return read_csv(csv_file)


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result