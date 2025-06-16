!pip install ace_tools

import pandas as pd
import numpy as np
import itertools

from simuloop_module import simuloop_1 as simuloop
from simuhete import simuhete
from scipy.optimize import dual_annealing


#初期値の読み込み
param_lines = []
with open('/Users/user/Desktop/replication/5368_data and programs_0/start_values.txt') as f:
    sim_num = str(int(f.readline().strip()))  # 1行目：sim_num
    for line in f:
        param_lines.extend([float(x) for x in line.strip().split()])
#データの読み込み
datairline = pd.read_stata("/Users/user/Desktop/replication/5368_data and programs_0/CilibertoTamerEconometrica.dta").set_index("market")

#企業数とシミュレーション数、outcomeの組み合わせ数を定義
k = 6
r = 100
total = 2**k


#説明変数と目的変数の獲得
X = datairline.iloc[:, k:]
y = datairline.iloc[:, :k] 

#マーケット数とカラム数の獲得          
rowX = m = X.shape[0] 
colX  = X.shape[1]

#それぞれの市場毎に全ての組み合わせを作る。
##(0,1)^6
index = list(itertools.product([0, 1], repeat=k))
index = np.array(index)  
##(0,1)^6を市場分だけ引き延ばす
repindex = np.tile(index, (X.shape[0], 1))

#CCPを持ってくる
#prob = np.loadtxt('conddensityorder.raw')

#誤差項の追加
epsi_firm = np.random.normal(0,4,m*k*r).reshape(m,k*r)
epsi_market = np.random.normal(0,4,m*r).reshape(m,r)
epsi_market = np.repeat(epsi_market, repeats=k, axis=1)

epsi = epsi_firm + epsi_market

#シミュレーション
iteration = 1
param0 = np.array(param_lines)
oldresultsasa = np.concatenate(([0, 0, iteration, r], param0))

df = pd.DataFrame([oldresultsasa])
log_path = f'oldresultsasa{sim_num}.csv'
df.to_csv(log_path, index=False)

bounds = [(-30, 30)] * len(param0)

result = dual_annealing(
    simuhete,
    bounds=bounds,
    x0=param0,  # ← 初期値をここで指定
    args=(datairline, prob, repindex, k, r, rowX, oldresultsasa, iteration, log_path)
)
param = result.x
fval = result.fun