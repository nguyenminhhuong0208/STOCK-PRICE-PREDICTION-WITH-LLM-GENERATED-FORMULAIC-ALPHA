import numpy as np
import pandas as pd
import ppscore as pps

# Giả sử df có các cột: open, high, low, close, volume
# Thêm returns, vwap, adv20
df = pd.read_csv('../data/final/VCB_final.csv')
df = df.copy()
df["returns"] = df["close"].pct_change()
df["vwap"] = (df["close"] * df["volume"]).rolling(10).mean()
df["adv20"] = df["volume"].rolling(20).mean()

# ===== Helper functions =====
def rank(x):
    return x.rank(pct=True)

def delta(x, n):
    return x.diff(n)

def ts_rank(x, window):
    return x.rolling(window).apply(lambda s: rank(s).iloc[-1] if not s.isnull().all() else np.nan)

def ts_argmax(x, window):
    return x.rolling(window).apply(lambda s: np.argmax(s) if not s.isnull().all() else np.nan)

def ts_min(x, window):
    return x.rolling(window).min()

def ts_max(x, window):
    return x.rolling(window).max()


# ============ ALPHA DEFINITIONS =============

# Alpha 1
df["X1"] = np.where(df["returns"] < 0,
                    df["returns"].rolling(20).std(),
                    df["close"])
df["Alpha1"] = rank(ts_argmax(df["X1"]**2, 5)) - 0.5

# Alpha 2
df["Alpha2"] = -1 * df.apply(
    lambda r: np.nan, axis=1)  # placeholder, vectorized below

df["Alpha2"] = -1 * df["volume"].apply(np.log).diff(2).rolling(6).corr(
    ((df["close"] - df["open"]) / df["open"]).rank()
).rank()

# Alpha 3
df["Alpha3"] = -1 * df["open"].rolling(10).corr(df["volume"].rank())

# Alpha 4
df["Alpha4"] = -1 * ts_rank(rank(df["low"]), 9)

# Alpha 5
df["Alpha5"] = rank(df["open"] - df["vwap"].rolling(10).mean()) * \
               (-1 * abs(rank(df["close"] - df["vwap"])))

# Alpha 6
df["Alpha6"] = -1 * df["open"].rolling(10).corr(df["volume"])

# Alpha 7
cond = df["adv20"] < df["volume"]
signal = -1 * ts_rank(abs(delta(df["close"], 7)), 60) * np.sign(delta(df["close"], 7))
df["Alpha7"] = np.where(cond, signal, -1)

# Alpha 8
tmp = df["open"].rolling(5).sum() * df["returns"].rolling(5).sum()
df["Alpha8"] = -1 * rank(tmp - tmp.shift(10))

# Alpha 9
c1 = 0 < ts_min(delta(df["close"], 1), 5)
c2 = ts_max(delta(df["close"], 1), 5) < 0
df["Alpha9"] = np.where(c1, delta(df["close"], 1),
                 np.where(c2, delta(df["close"], 1),
                        -1 * delta(df["close"], 1)))

# Alpha 10
c1 = 0 < ts_min(delta(df["close"], 1), 4)
c2 = ts_max(delta(df["close"], 1), 4) < 0
df["tmp10"] = np.where(c1, delta(df["close"], 1),
                np.where(c2, delta(df["close"], 1),
                       -1 * delta(df["close"], 1)))
df["Alpha10"] = rank(df["tmp10"])


# ===================== PPS SCORE ======================

alpha_cols = [f"Alpha{i}" for i in range(1, 11)]
pps_scores = {}

for a in alpha_cols:
    pps_scores[a] = pps.score(df, a, "close")["ppscore"]

pps_scores_df = pd.DataFrame.from_dict(pps_scores, orient="index", columns=["PPS"])
print(pps_scores_df.sort_values("PPS", ascending=False))
