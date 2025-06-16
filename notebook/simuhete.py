import  numpy as np
import itertools
import pickle
import os
from simuloop_module import simuloop_1 as simuloop
import pandas as pd


def simuhete(param,datairline, prob, repindex, k, r, rowX, oldresultsasa, iteration,log_path):
    total = 2**k
    #説明変数と目的変数の獲得
    X = datairline.iloc[:, k:]
    y = datairline.iloc[:, :k] 

    #マーケット数とカラム数の獲得          
    rowX = X.shape[0] 
    colX = X.shape[1]

    #それぞれパラメーターの獲得
    paraconstant = param[0]
    paramX = param[1:9]
    paraheterog1 = param[9]
    paraheterog2 = param[10]
    parafirm1 = param[11:17]

    #マーケット単位の利潤に与える効果
    roleX = X.iloc[:, :8] @ paramX
    roleX = np.tile(roleX.to_numpy().reshape(-1, 1), (1, k))

    #異質性
    roleheter = X.iloc[:, 8:14].to_numpy() * paraheterog1
    roleheter += X.iloc[:, 14:20].to_numpy() * paraheterog2

    #各アウトカム毎に反事実的な利潤を与えたい
    index = list(itertools.product([0, 1], repeat=k))
    index = np.array(index)  # ⬅️ NumPy配列に変換
    total = index.shape[0]
    repindex = np.tile(index, (X.shape[0], 1))

    #異質性な競争効果
    onothereffect = np.zeros((total * X.shape[0], k))


    for i in range(k):
        effect = repindex[:, i:i+1] * parafirm1[i]
        effect = np.tile(effect, (1, k))
        effect[:, i] = 0  # 自分自身の影響は除外
        onothereffect += effect

    #形を揃えてあげる
    owneffect = np.ones((total, k)) * paraconstant
    repi_roleX = np.repeat(roleX, repeats=2 ** k, axis=0)
    repi_roleheter = np.repeat(roleheter, repeats=2 ** k, axis=0)
    repi_owneffect = np.repeat(owneffect, repeats=rowX, axis=0)

    common = repi_owneffect + repi_roleheter + onothereffect + repi_roleX
    
    #均衡確率の下限と上限を求める
    m=rowX
    epsi = np.random.normal(0,1,m*k*r).reshape(m,k*r)
    meanlow, meanupp, temp = simuloop(common,m,k,r,epsi,repindex)

    #目的関数の算出
    sand_matrix = np.where(
        prob[:rowX, :] > meanupp,
        (prob[:rowX, :] - meanupp)**2,
        0
        ) + np.where(
        prob[:rowX, :] < meanlow,
        (prob[:rowX, :] - meanlow)**2,
        0
        )
    ##均衡が存在しなかった場合は補正する。
    neverpure = np.min(np.isfinite(meanupp), axis=1)
    findneverpure = np.where(neverpure == 0)[0]
    if findneverpure.size > 0:
        sand_matrix = np.delete(sand_matrix, findneverpure, axis=0)

    ##最終的な目的関数
    sand = np.sum(sand_matrix)
    iteration += 1
    #過去の結果を読み込む
    if os.path.exists(log_path):
        oldresultsasa = pd.read_csv(log_path).values
    else:
        oldresultsasa = np.empty((0, 4 + len(param)))

    #新しい結果を1行としてまとめる
    new_result = np.concatenate([[temp, sand, iteration, r], param])
    oldresultsasa = np.vstack([oldresultsasa, new_result])

    # 保存（毎回上書き）
    columns = ['temp', 'sand', 'iteration', 'r'] + [f'param_{i+1}' for i in range(len(param))]
    pd.DataFrame(oldresultsasa, columns=columns).to_csv(log_path, index=False)

    return sand  



