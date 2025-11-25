# === FULL ALPHA SCRIPT (Alpha11 -> Alpha101) - best-effort mapping ===
import numpy as np
import pandas as pd

df = pd.read_csv('../data/final/VCB_final.csv')
# Ensure copies to avoid in-place surprises
df = df.copy()

# If df not prepared, create baseline columns
if 'df' not in globals():
    raise RuntimeError("Please load your DataFrame as variable `df` before running this script.")

# Basic derived series if missing
if 'returns' not in df.columns:
    df['returns'] = df['close'].pct_change()
if 'vwap' not in df.columns:
    # approximate vwap as rolling mean(close*volume)/rolling mean(volume) if volume available
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['vwap'] = (df['close'] * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
    else:
        df['vwap'] = df['close']
if 'adv20' not in df.columns:
    df['adv20'] = df['volume'].rolling(20).mean()

# ---------- Helper functions ----------
def rank(s):
    # global percentile rank (series)
    return s.rank(pct=True, na_option='keep')

def delta(s, n=1):
    return s.diff(n)

def ts_rank(s, window):
    return s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if x.size>0 and not np.all(np.isnan(x)) else np.nan, raw=False)

def ts_argmax(s, window):
    return s.rolling(window).apply(lambda x: np.nan if x.size==0 or np.all(np.isnan(x)) else float(np.nanargmax(x)), raw=True)

def ts_argmin(s, window):
    return s.rolling(window).apply(lambda x: np.nan if x.size==0 or np.all(np.isnan(x)) else float(np.nanargmin(x)), raw=True)

def ts_max(s, window):
    return s.rolling(window).max()

def ts_min(s, window):
    return s.rolling(window).min()

def rolling_cov(a, b, window):
    return a.rolling(window).cov(b)

def rolling_corr(a, b, window):
    return a.rolling(window).corr(b)

def decay_linear(series, window):
    def apply_decay(x):
        x = np.array(x, dtype=float)
        if np.all(np.isnan(x)): return np.nan
        w = np.arange(1, len(x)+1)
        return np.nansum(x * w) / np.nansum(w)
    return series.rolling(window).apply(apply_decay, raw=True)

def scale(series):
    s = series.copy()
    mu = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return (s - mu) * 0.0
    return (s - mu) / sd

def signed_power(x, p):
    return np.sign(x) * (np.abs(x) ** p)

# convenience alias
o = df['open']; h = df['high']; l = df['low']; c = df['close']; v = df['volume']; r = df['returns']; vw = df['vwap']; adv20 = df['adv20']

# Container
computed = []

# --- Start computing alphas (11 -> 101) ---
# Many formulas are approximations / best-effort from the textual input you provided.
# Where formula is unclear, leave NaN and comment.
# Alpha11:
df['Alpha11'] = (rank(ts_max(vw - c, 3)) + rank(ts_min(vw - c, 3))) * rank(delta(v, 3))
computed.append('Alpha11')

# Alpha12:
df['Alpha12'] = np.sign(delta(v,1)) * (-1.0 * delta(c,1))
computed.append('Alpha12')

# Alpha13..Alpha16: not provided clearly -> placeholders
df['Alpha13'] = np.nan
df['Alpha14'] = np.nan
df['Alpha15'] = np.nan
df['Alpha16'] = np.nan
computed += ['Alpha13','Alpha14','Alpha15','Alpha16']

# Alpha17: (-1 * rank(covariance(rank(close), rank(volume), 5)))
df['Alpha17'] = -1.0 * rank( rolling_cov(rank(c), rank(v), 5) )
computed.append('Alpha17')

# Next formula in your list mapped to Alpha18:
# ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
df['Alpha18'] = (-1.0 * rank(delta(r,3))) * rolling_corr(o, v, 10)
computed.append('Alpha18')

# Alpha19: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
tmp = rolling_corr(rank(h), rank(v), 3)
df['Alpha19'] = -1.0 * tmp.rank(pct=True).rolling(3).sum()
computed.append('Alpha19')

# Alpha20: complex chunk mapped earlier; using the combined expression in your list:
df['Alpha20'] = (-1.0 * rank(ts_rank(c,10))) * rank(delta(delta(c,1),1)) * rank(ts_rank(v/adv20,5))
computed.append('Alpha20')

# Alpha21: long conditional from your input. Implement exactly as ternary chain
# ((((mean(close,8) + std(close,8)) < mean(close,2)) ? -1 : ((mean(close,2) < (mean(close,8)-std(close,8))) ? 1 : ((1 < volume/adv20 or volume/adv20 ==1) ? 1 : -1))))
ma8 = c.rolling(8).mean()
std8 = c.rolling(8).std()
ma2 = c.rolling(2).mean()
ratio_vol = v / adv20
cond1 = ( (ma8 + std8) < ma2 )
cond2 = ( ma2 < (ma8 - std8) )
df['Alpha21'] = np.where(cond1, -1.0, np.where(cond2, 1.0, np.where((ratio_vol>1) | (ratio_vol==1), 1.0, -1.0)))
computed.append('Alpha21')

# Alpha22-23-24: some formulas present; pick ones found later:
# Alpha22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
df['Alpha22'] = -1.0 * ( delta( rolling_corr(h, v, 5), 5 ) * rank(c.rolling(20).std()) )
computed.append('Alpha22')

# Alpha23: (((sum(high,20)/20) < high) ? (-1 * delta(high,2)) : 0)
df['Alpha23'] = np.where( (h.rolling(20).mean() < h), -1.0 * delta(h,2), 0.0 )
computed.append('Alpha23')

# Alpha24: long conditional about 100-day mean; implement as given
# if (delta(mean(close,100),100)/delay(close,100)) <= 0.05 -> -1*(close - ts_min(close,100)) else -1*delta(close,3)
ma100 = c.rolling(100).mean()
delta_ma100 = delta(ma100,100)
delay_close_100 = c.shift(100)
frac = delta_ma100 / delay_close_100
df['Alpha24'] = np.where( (frac < 0.05) | (frac == 0.05), -1.0 * (c - ts_min(c,100)), -1.0 * delta(c,3) )
computed.append('Alpha24')

# Alpha25:
df['Alpha25'] = rank(((-1.0 * r) * adv20 * vw * (h - c)))
computed.append('Alpha25')

# Alpha26-29: some formulas present later; implement those we can map:
# Alpha26 & 27 & 28 placeholders if not explicit
df['Alpha26'] = np.nan
df['Alpha27'] = np.nan
df['Alpha28'] = np.nan
computed += ['Alpha26','Alpha27','Alpha28']

# Alpha29 mapping pieces:
# -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5),5),3)
df['Alpha29'] = -1.0 * ts_max( rolling_corr( ts_rank(v,5), ts_rank(h,5), 5 ), 3 )
computed.append('Alpha29')

# The next ambiguous ones (min(product(...),5)+ ts_rank(delay(-1*returns,6),5)) are unclear - create Alpha30 mapping below.

# Alpha30:
# ((1.0 - rank(sign(close-delay(close,1))+sign(delay(close,1)-delay(close,2))+sign(delay(close,2)-delay(close,3)))) * sum(volume,5) ) / sum(volume,20)
s1 = np.sign(c - c.shift(1)) + np.sign(c.shift(1) - c.shift(2)) + np.sign(c.shift(2) - c.shift(3))
df['Alpha30'] = ((1.0 - rank(s1)) * v.rolling(5).sum()) / (v.rolling(20).sum())
computed.append('Alpha30')

# Alpha31:
df['Alpha31'] = rank(rank(rank(decay_linear(-1.0 * rank(rank(delta(c,10))), 10)))) + rank(-1.0 * delta(c,3)) + np.sign(scale(rolling_corr(adv20, l, 12)))
computed.append('Alpha31')

# Alpha32:
df['Alpha32'] = scale((c.rolling(7).mean() - c)) + (20.0 * scale( rolling_corr(vw, c.shift(5), 230) ))
computed.append('Alpha32')

# Alpha33-35: implement ones present
# Alpha33 placeholder
df['Alpha33'] = np.nan
computed.append('Alpha33')

# Alpha34:
# rank(-1 * ((1 - (open/close))**1))
df['Alpha34'] = rank(-1.0 * (1.0 - (o / c)))
computed.append('Alpha34')

# Alpha35:
# rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))
df['Alpha35'] = rank((1.0 - rank(r.rolling(2).std() / r.rolling(5).std())) + (1.0 - rank(delta(c,1))))
computed.append('Alpha35')

# Alpha36: long weighted combination - implement as expression given
df['Alpha36'] = (
    (2.21 * rank( rolling_corr((c - o), delta(v,1).shift(1), 15) )) +
    (0.7 * rank((o - c))) +
    (0.73 * rank(ts_rank(-1.0 * r.shift(1), 6))) +
    rank(abs( rolling_corr(vw, adv20, 6) )) +
    (0.6 * rank(((c.rolling(200).mean() - o) * (c - o))))
)
computed.append('Alpha36')

# Alpha37-39: mixture: fill best-effort from provided pieces
df['Alpha37'] = np.nan
df['Alpha38'] = np.nan
# Alpha39 pieces:
df['Alpha39'] = rank( rolling_corr( (o - c).shift(1), c, 200 ) ) + rank(o - c)
computed += ['Alpha37','Alpha38','Alpha39']

# Another trio: (-1 * rank(Ts_Rank(close,10))) * rank((close/open))
df['Alpha40'] = (-1.0 * rank(ts_rank(c,10))) * rank(c / o)
computed.append('Alpha40')

# Alpha41-43:
df['Alpha41'] = np.nan
# Alpha42: (((high*low)^0.5) - vwap)
df['Alpha42'] = ( (h * l) ** 0.5 ) - vw
# Alpha43: (rank(vwap - close)/ rank(vwap + close))
df['Alpha43'] = rank(vw - c) / rank(vw + c)
computed += ['Alpha41','Alpha42','Alpha43']

# Alpha44:
df['Alpha44'] = -1.0 * rolling_corr(h, rank(v), 5)
computed.append('Alpha44')

# Alpha45:
df['Alpha45'] = -1.0 * ( rank( ( (delta(c,5).rolling(20).sum()) if False else ( (c.shift(5).rolling(20).sum())/20 ) ) ) if False else 0 ) 
# The original Alpha45 is complex - leave placeholder explanation
df['Alpha45'] = np.nan
computed.append('Alpha45')

# Alpha46: conditional using delayed closes differences
d20_10 = (c.shift(20) - c.shift(10)) / 10
d10_0 = (c.shift(10) - c) / 10
expr = d20_10 - d10_0
df['Alpha46'] = np.where(expr > 0.25, -1.0, np.where(expr < 0, 1.0, (-1.0 * 1.0 * (c - c.shift(1)))))
computed.append('Alpha46')

# Alpha47: formula present
df['Alpha47'] = (((rank(1.0 / c) * v) / adv20) * ((h * rank(h - c)) / (h.rolling(5).sum() / 5.0))) - rank(vw - vw.shift(5))
computed.append('Alpha47')

# Alpha48 not in list; skip. Alpha49:
expr49 = ((c.shift(20) - c.shift(10)) / 10) - ((c.shift(10) - c) / 10)
df['Alpha49'] = np.where(expr49 < (-1.0 * 0.1), 1.0, (-1.0 * 1.0 * (c - c.shift(1))))
computed.append('Alpha49')

# Alpha50:
df['Alpha50'] = -1.0 * ts_max(rank( rolling_corr(rank(v), rank(vw), 5) ), 5)
computed.append('Alpha50')

# Alpha51 similar to 49 with threshold 0.05
df['Alpha51'] = np.where(expr49 < (-1.0 * 0.05), 1.0, (-1.0 * 1.0 * (c - c.shift(1))))
computed.append('Alpha51')

# Alpha52:
df['Alpha52'] = ((-1.0 * ts_min(l,5) + ts_min(l,5).shift(5)) * rank(((r.rolling(240).sum() - r.rolling(20).sum()) / 220.0))) * ts_rank(v,5)
computed.append('Alpha52')

# Alpha53:
df['Alpha53'] = -1.0 * delta( (((c - l) - (h - c)) / (c - l + 1e-12)), 9 )
computed.append('Alpha53')

# Alpha54-55:
df['Alpha54'] = np.nan
df['Alpha55'] = (-1.0 * ((l - c) * (o ** 5))) / ((l - h) * (c ** 5 + 1e-12))
computed += ['Alpha54','Alpha55']

# Alpha56 (typoed as lpha#57 etc): implement correlation rank thing
df['Alpha56'] = -1.0 * rolling_corr(rank((c - ts_min(c,12)) / (ts_max(h,12) - ts_min(l,12) + 1e-12)), rank(v), 6)
computed.append('Alpha56')

# Alpha57..59: some given
df['Alpha57'] = np.nan
df['Alpha58'] = (0 - (1 * ((c - vw) / decay_linear(rank(ts_argmax(c,30)), 2))))
computed += ['Alpha57','Alpha58']

# Alpha60:
df['Alpha60'] = (0 - (1 * ((2 * scale(rank(((((c - l) - (h - c)) / (h - l + 1e-12)) * v)))) - scale(rank(ts_argmax(c,10)))) ))
computed.append('Alpha60')

# Alpha61:
df['Alpha61'] = ( rank(vw - ts_min(vw, 16)) < rank( rolling_corr(vw, df['adv20'].rolling(180).mean() if False else adv20, 18) ) ).astype(float)
computed.append('Alpha61')

# Alpha62 best-effort:
df['Alpha62'] = ( rank( rolling_corr(vw, adv20.rolling(22).sum(), 10) ) < rank( (rank(o) + rank(o)) < (rank((h + l)/2) + rank(h)) ) ).astype(float) * -1.0
computed.append('Alpha62')

# A63 missing; Alpha64..Alpha66 provide numeric transforms; we approximate them as placeholders or best-effort:
df['Alpha63'] = np.nan
df['Alpha64'] = -1.0 * ( rank( rolling_corr( (o * 0.178404 + l * (1 - 0.178404)).rolling(13).sum(), adv20.rolling(13).sum(), 17) ) < rank( delta( (((h + l)/2)*0.178404 + vw*(1-0.178404)), 4 ) ) ).astype(float)
df['Alpha65'] = ( rank( rolling_corr((o * 0.00817205 + vw*(1-0.00817205)), adv20.rolling(9).sum(), 6) ) < rank( o - ts_min(o,13) ) ).astype(float) * -1.0
df['Alpha66'] = -1.0 * ( rank( decay_linear(delta(vw,3), 7)) + ts_rank(decay_linear((((l*0.96633) + (l*(1-0.96633)) - vw) / (o - ((h + l)/2 + 1e-12))), 11), 7) )
computed += ['Alpha63','Alpha64','Alpha65','Alpha66']

# Alpha68,71,72... many with decimal params - implement as best-effort placeholders or approximations:
df['Alpha67'] = np.nan
df['Alpha68'] = -1.0 * ( ts_rank( rolling_corr(rank(h), rank(df['volume'].rolling(15).mean() if False else v), 9), 14) < rank(delta((c*0.518371 + l*(1-0.518371)),1)) ).astype(float)
df['Alpha69'] = np.nan
df['Alpha70'] = np.nan
df['Alpha71'] = np.maximum( ts_rank(decay_linear( rolling_corr(ts_rank(c,3), ts_rank(adv20,12), 18), 4), 16), ts_rank(decay_linear(rank(((l + o) - (vw + vw)))**2, 16), 4) )
df['Alpha72'] = rank(decay_linear( rolling_corr((h + l)/2, adv20, 9), 10 )) / rank(decay_linear( rolling_corr(ts_rank(vw,4), ts_rank(v,18), 7), 3 ))
df['Alpha73'] = -1.0 * np.maximum( rank(decay_linear(delta(vw,5), 3)), ts_rank(decay_linear(( delta((o*0.147155 + l*(1-0.147155)), 2) / (o*0.147155 + l*(1-0.147155) + 1e-12)) * -1, 3), 16) )
computed += ['Alpha67','Alpha68','Alpha69','Alpha70','Alpha71','Alpha72','Alpha73']

# Alpha74..Alpha76 approximations
df['Alpha74'] = (rank( rolling_corr(c, adv20.rolling(37).sum(), 15) ) < rank( rolling_corr( rank(h*0.0261661 + vw*(1-0.0261661)), rank(v), 11 ) ) ).astype(float) * -1.0
df['Alpha75'] = ( rank( rolling_corr(vw, v, 4) ) < rank( rolling_corr(rank(l), rank(df['volume'].rolling(50).mean() if False else v), 12) ) ).astype(float)
computed += ['Alpha74','Alpha75']

# Alpha77:
df['Alpha77'] = np.minimum( rank(decay_linear((((h + l)/2 + h) - (vw + h)), 20)), rank(decay_linear( rolling_corr((h + l)/2, adv20, 3), 5) ) )
computed.append('Alpha77')

# Alpha78:
df['Alpha78'] = rank( rolling_corr( (l*0.352233 + vw*(1-0.352233)).rolling(20).sum(), adv20.rolling(20).sum(), 6) ) ** rank( rolling_corr(rank(vw), rank(v), 6) )
computed.append('Alpha78')

# Alpha79..82 not provided clearly -> placeholders
for i in [79,80,81,82]:
    df[f'Alpha{i}'] = np.nan
    computed.append(f'Alpha{i}')

# Alpha83:
df['Alpha83'] = ( rank( ( (h - l) / (c.rolling(5).mean()) ).shift(2) ) * rank(rank(v)) ) / ( ((h - l) / c.rolling(5).mean()) / (vw - c + 1e-12) )
computed.append('Alpha83')

# Alpha84:
df['Alpha84'] = signed_power( ts_rank(vw - ts_max(vw, 15), 21), delta(c,5) )
computed.append('Alpha84')

# Alpha85:
df['Alpha85'] = ( rank( rolling_corr(h*0.876703 + c*(1-0.876703), adv20.rolling(30).sum(), 10) ) ** rank( rolling_corr(ts_rank((h + l)/2,3), ts_rank(v,10), 7) ) )
computed.append('Alpha85')

# Alpha86:
df['Alpha86'] = ( ts_rank( rolling_corr(c, adv20.rolling(20).sum(), 6), 20) < rank((o + c) - (vw + o)) ).astype(float) * -1.0
computed.append('Alpha86')

# Alpha88 (87 maybe missing) approximate:
df['Alpha87'] = np.nan
df['Alpha88'] = np.minimum( rank(decay_linear(((rank(o) + rank(l)) - (rank(h) + rank(c))), 8)), ts_rank(decay_linear( rolling_corr(ts_rank(c,8), ts_rank(adv20,20), 8), 6), 2) )
computed += ['Alpha87','Alpha88']

# Alpha92:
df['Alpha92'] = np.minimum( ts_rank(decay_linear((((h + l)/2 + c) < (l + o)).astype(float), 15), 18), ts_rank(decay_linear( rolling_corr(rank(l), rank(v.rolling(30).mean() if False else v), 8), 6), 6) )
computed.append('Alpha92')

# Alpha94..96 approximations:
df['Alpha94'] = ( rank(vw - ts_min(vw, 12)) ** ts_rank( rolling_corr(ts_rank(vw,20), ts_rank(adv20,6), 18), 2 ) ) * -1.0
df['Alpha95'] = ( rank(o - ts_min(o,12)) < ts_rank( rank( rolling_corr((h + l)/2, adv20.rolling(19).sum(), 13) ** 5 ), 11 ) ).astype(float)
df['Alpha96'] = -1.0 * np.maximum( ts_rank(decay_linear( rolling_corr(rank(vw), rank(v), 4), 4), 8), ts_rank(decay_linear(ts_argmax( rolling_corr(ts_rank(c,7), ts_rank(adv20,4), 3), 12), 14), 13) )
computed += ['Alpha94','Alpha95','Alpha96']

# Alpha98:
df['Alpha98'] = rank(decay_linear(rolling_corr(vw, df['volume'].rolling(5).sum() if False else adv20.rolling(5).sum(), 4), 7)) - rank(decay_linear(ts_rank(ts_argmin( rolling_corr(rank(o), rank(v.rolling(15).mean() if False else v), 20), 8), 6), 8))
computed.append('Alpha98')

# Alpha99:
df['Alpha99'] = ( rank( rolling_corr((h + l)/2, adv20.rolling(60).sum(), 9) ) < rank( rolling_corr(l, v, 6) ) ).astype(float) * -1.0
computed.append('Alpha99')

# Alpha101:
df['Alpha101'] = (c - o) / ((h - l) + 0.001)
computed.append('Alpha101')

# Many alphas intentionally left NaN where original input was ambiguous or missing.
# For traceability, mark any missing Alphas between 11..101
for i in range(1, 102):
    col = f'Alpha{i}'
    if col not in df.columns:
        df[col] = np.nan
    # keep computed list updated
    if col not in computed:
        computed.append(col)

# Optional: fill or standardize alphas (you can comment out if you prefer raw)
alpha_cols = [f'Alpha{i}' for i in range(1, 102)]
# Fill extremely early NaNs due to rolling windows with NaN (optional)
# df[alpha_cols] = df[alpha_cols].fillna(method='ffill').fillna(0)

# Quick diagnostics
print("Alphas created:", len(alpha_cols))
print("NaN fraction per alpha (sample):")
print(df[alpha_cols].isna().mean().sort_values().head(20))

print("\nUnique values per alpha (sample):")
print(df[alpha_cols].nunique().sort_values().head(20))

# Attempt to compute PPS for these alphas if ppscore installed
try:
    import ppscore as pps
    pps_results = {}
    for a in alpha_cols:
        # skip all-NaN or constant
        if df[a].isna().all() or df[a].nunique() <= 1:
            pps_results[a] = 0.0
            continue
        try:
            score = pps.score(df[[a, 'close']].dropna(), a, 'close')['ppscore']
            pps_results[a] = float(score)
        except Exception as e:
            pps_results[a] = 0.0
    pps_df = pd.Series(pps_results).sort_values(ascending=False)
    print("\nTop PPS scores (if computed):")
    print(pps_df.head(20))
except Exception as e:
    print("\nppscore not available or failed to compute PPS. Install ppscore and re-run if you want PPS.")
    # to install: pip install ppscore

# Save alphas to df_alphas variable for convenience
df_alphas = df[alpha_cols]

# End of script
