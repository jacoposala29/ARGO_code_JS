import os
import pandas as pd
import pickle as pkl
from process_jtwc import *

abbr = 'io'
name = 'INDIANOCEAN'

df_lst = []
dr = f'../../../jtwc/{abbr}/'
for fn in os.listdir(dr):
    print(fn)
    raw = pd.read_table(dr+fn, header=None, delimiter=',', usecols=range(11))
    df_lst.append(convert_df(raw))

df = pd.concat(df_lst)

pkl.dump(df, open(f'JTWC_{name}.pkl', 'wb'))
df.to_csv(f'JTWC_{name}.csv')
