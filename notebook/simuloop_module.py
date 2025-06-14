"""
使用パッケージ numpy
    Parameters
    ----------
    common : ndarray of shape (m * k**2, k)
        観察部分の利得関数
    m : int
        市場数
    k : int
        企業数
    r : int
        シミュレーション回数
    epsi : ndarray of shape (m, k * r)
        誤差項
    repindex : ndarray of shape (m * k**2, k)
        outcomeの集合

    Returns
    -------
    meanlow : ndarray of shape (m, k**2)
        各市場毎のLower bound probabilities
    meanupp : ndarray of shape (m, k**2)
        各市場毎のUpper boud probabilities
    temp : float
        均衡が存在しない市場の割合
"""


import numpy as np

def simuloop_1(common,m,k,r,epsi,repindex):
    total=2**k

    #複数均衡数とユニークな均衡をカウントする
    sumupp = np.zeros((m, total))
    sumlow = np.zeros((m, total))

    #上をシミュレーション回数で割ることで割合にするので作成
    totcount = r * np.ones((m, 1))


    #均衡が存在しないケースを考慮し、tempにその情報を保存していく（これを最終的にtotocountから引くことで割合を正当化）
    temp = np.zeros((m, 1))

    #シミュレーションの実行
    for i in np.arange(r):
        ##誤差項加えるパート
        #シミレーションごとに誤差を獲得（積分）。
        epsitemp = epsi[:, k * i : k * (i + 1)] 
        epsitemp = np.repeat(epsitemp, repeats=2 ** k, axis=0)
        #観測できる利潤に加える
        equil = epsitemp + common  

        ##均衡を探すパート
        #近郊では逸脱、もしくは実際のアクションが0以上なら１を取り、それ以外では0をとる。
        #それがrepindexと一致してれば、プレイヤーが最適反応をとっているということ
        equil = ((equil>=0).astype(int) == repindex).astype(int)

        #最適反応をとっているプレイヤー数の獲得
        #それが全体のプレイヤー数と一致するならそれは均衡である。
        #vectorequilにはあるoutcomが均衡かどうかが市場ごとに入っている
        sumequil = np.sum(equil, axis=1)
        vectorequil = (sumequil == 2).astype(int)
        
        #各市場における均衡数を獲得する。
        #助長なやり方であるが、累積和を取り、total毎にスライス、その差分を取ることで市場毎の均衡数を獲得。
        cumsumvectorequil = np.cumsum(vectorequil)
        sumvectorequil = cumsumvectorequil[total - 1::total].reshape(-1, 1)
        if sumvectorequil.shape[0] > 1:
            sumvectorequil[1:] = np.diff(sumvectorequil, axis=0)
        
        #upperは複数均衡を考慮してもいいので、そのままカウントする。
        #ただし形は(m, total)で記載していく
        upp = vectorequil.reshape(m, total)

        #lowは複数均衡が発生したらカウントできない。そのため.whereで複数均衡が存在している市場を特定し0にする
        low = upp.copy()
        lowtobezero = np.where(sumvectorequil > 1)[0]
        low[lowtobezero, :] = 0

        ##均衡数が０の対処
        #存在するのであればその数だけ減らす。またtenpにカウントすることで最終的に、均衡が存在しない数を表示できる。
        upptobezero = np.where(sumvectorequil == 0)[0]
        if upptobezero.size > 0:
            count = (sumvectorequil == 0).astype(int).reshape(-1, 1)
            temp += count
            totcount -= count
        
        #シミュレーション毎に加算
        sumupp += upp
        sumlow += low


    # 下限確率と上限確率の計算
    meanlow = sumlow / totcount 
    meanupp = sumupp / totcount  

    # 均衡なし市場の割合
    temp = np.sum(temp) / (r * m)
    return meanlow , meanupp,temp


