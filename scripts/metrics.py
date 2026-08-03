# -*- coding: utf-8 -*-
"""绩效指标函数集"""
import numpy as np, pandas as pd

TRADING_DAYS = 252

def cagr_from_wealth(w, days):
    return w ** (TRADING_DAYS / max(days, 1)) - 1

def annualized_ret(r):
    r = r.dropna()
    if len(r) == 0: return np.nan
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1

def annualized_vol(r):
    r = r.dropna()
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan

def max_drawdown(w):
    """w: 财富序列(任意起点) -> (mdd, 峰值日期, 谷值日期)"""
    w = w.dropna()
    if len(w) == 0: return np.nan, None, None
    cummax = w.cummax()
    dd = w / cummax - 1
    trough = dd.idxmin()
    peak = w[:trough].idxmax()
    return dd.min(), peak, trough

def drawdown_series(w):
    w = w.dropna()
    return w / w.cummax() - 1

def sharpe(r, rf=0.018):
    r = r.dropna()
    if len(r) < 2: return np.nan
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    ex = r - rf_d
    sd = ex.std(ddof=1)
    return ex.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan

def calmar(ann_ret, mdd):
    return ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan

def rolling_sharpe(r, window=252, rf=0.018):
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    ex = r - rf_d
    rs = ex.rolling(window).mean() / ex.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return rs

def summary(r, name="", rf=0.018, periods=None):
    """输出一组指标; periods: {'牛':(s,e),...} 额外分区间统计"""
    r = r.dropna()
    w = (1 + r).cumprod()
    ann = annualized_ret(r)
    vol = annualized_vol(r)
    mdd, peak_d, trough_d = max_drawdown(w)
    s = sharpe(r, rf)
    c = calmar(ann, mdd)
    total = w.iloc[-1] - 1
    win = (r > 0).mean()
    out = {"name": name, "n_days": len(r), "start": str(r.index.min().date()), "end": str(r.index.max().date()),
           "total_ret": total, "cagr": ann, "vol": vol, "max_dd": mdd,
           "mdd_peak": str(peak_d.date()) if peak_d is not None else None,
           "mdd_trough": str(trough_d.date()) if trough_d is not None else None,
           "sharpe": s, "calmar": c, "win_rate": win, "final_wealth": w.iloc[-1]}
    if periods:
        out["periods"] = {}
        for pn, (ps, pe) in periods.items():
            sub = r[(r.index >= ps) & (r.index <= pe)]
            if len(sub) > 2:
                sw = (1 + sub).cumprod()
                pmdd, _, _ = max_drawdown(sw)
                out["periods"][pn] = {"cagr": annualized_ret(sub), "vol": annualized_vol(sub),
                                      "max_dd": pmdd, "total": sw.iloc[-1] - 1}
    return out

def fmt_pct(x, digits=2):
    return f"{x*100:.{digits}f}%" if x == x else "nan"

def fmt_num(x, digits=2):
    return f"{x:.{digits}f}" if x == x else "nan"
