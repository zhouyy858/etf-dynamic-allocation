# -*- coding: utf-8 -*-
"""v35 单纯纳斯达克ETF(159941)波段方法研究: 15种波段方法 x 每日调仓 x 双窗口
严格无未来(signal_lag=1, 溢价T-2); 现金层 75% 511010 + 25% 逆回购; 对比 v26 基线; fee 万5主口径"""
import sys, os, json, copy
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SKILL}/scripts")
import numpy as np, pandas as pd

CFG = json.load(open(f"{SKILL}/references/final_cfg_v26.json"))
from data_prep import build_panel, read_table, rets_from, TRADING_DAYS
from engine import run_backtest, evaluate
from strategy import DynamicStrategy, SignalSet

bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
REPO = 0.022
BOND_PCT = CFG.get("cash_bond_pct", 0.75)
FEE5 = 0.0005; FEE15 = 0.00015

PX = pd.read_csv(f"{SKILL}/assets/data/qdii_price_159941.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
NAV = pd.read_csv(f"{SKILL}/assets/data/159941_nav.csv", parse_dates=["date"]).set_index("date")
NAV = NAV[~NAV.index.duplicated(keep="last")].sort_index()["unit_nav"]
PREM_RAW = (PX / NAV - 1.0).drop(pd.Timestamp("2022-07-04"), errors="ignore").clip(-0.10, 0.15)
PREM = PREM_RAW.shift(2)

def prem_at(dt):
    v = PREM.reindex(pd.DatetimeIndex([dt]), method="ffill")
    if len(v.dropna()) == 0:
        return None
    return float(v.dropna().iloc[0])

SIG = SignalSet(R, lag=1)
SIGr = SignalSet(Rr, lag=1)

class P:
    def __init__(self, Rs, sig):
        lvl = (1 + Rs["159941"].dropna()).cumprod().reindex(Rs.index).ffill()
        self.lvl = lvl
        self.sig = sig
        self.idx = pd.Series(np.arange(len(lvl)), index=lvl.index)
        self.ma = {N: lvl.rolling(N, min_periods=min(N // 2, 20)).mean() for N in (10, 20, 50, 60, 120, 200)}
        self.mom = {N: lvl.pct_change(N) for N in (63, 126, 252)}
        self.dd = {N: (lvl / lvl.rolling(N, min_periods=min(N // 2, 60)).max() - 1.0) for N in (60, 252)}
        self.hh20 = lvl.rolling(20).max(); self.ll10 = lvl.rolling(10).min()
        self.vol20 = lvl.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS)
    def i(self, dt):
        return max(0, int(self.idx.reindex(pd.DatetimeIndex([dt]), method="ffill").iloc[0]) - 1)

PP = P(R, SIG); PPr = P(Rr, SIGr)

def f_buyhold(p, dt): return 1.0
def f_ma(p, dt, N=20):
    i = p.i(dt)
    return 1.0 if p.lvl.iloc[i] > p.ma[N].iloc[i] else 0.0
def f_ma2(p, dt, N1=20, N2=60):
    i = p.i(dt); l = p.lvl.iloc[i]
    return 1.0 if (l > p.ma[N1].iloc[i] and p.ma[N1].iloc[i] > p.ma[N2].iloc[i]) else 0.0
def f_ma_graded(p, dt):
    i = p.i(dt); l = p.lvl.iloc[i]
    if l > p.ma[20].iloc[i]: return 1.0
    if l > p.ma[60].iloc[i]: return 0.6
    if l > p.ma[120].iloc[i]: return 0.3
    return 0.0
def f_mom(p, dt, N=126):
    i = p.i(dt)
    return 1.0 if p.mom[N].iloc[i] > 0 else 0.0
def f_dd(p, dt, N=252, thr=0.10):
    i = p.i(dt)
    return 1.0 if p.dd[N].iloc[i] > -thr else 0.0
def f_turtle(p, dt):
    i = p.i(dt); l = p.lvl.iloc[i]
    if l > p.hh20.iloc[i - 1]: return 1.0
    if l < p.ll10.iloc[i - 1]: return 0.0
    return np.nan
def f_voltgt(p, dt, target=0.15):
    i = p.i(dt)
    v = p.vol20.iloc[i]
    if np.isnan(v) or v <= 0: return 1.0
    return min(1.0, target / v)
def f_rsrs(p, dt, thr=0.0):
    i = p.i(dt)
    return 1.0 if float(p.sig.rsrs["159941"].iloc[i]) >= thr else 0.0
def f_ind(p, dt, name="macd"):
    i = p.i(dt)
    return 1.0 if float(p.sig.ind["159941"][name].iloc[i]) >= 0.5 else 0.0
def premium_adj(w, dt, p_thr=(0.02, 0.05), p_cut=(0.5, 0.2)):
    v = prem_at(dt)
    if v is None: return w
    if v > p_thr[1]: return w * p_cut[1]
    if v > p_thr[0]: return w * p_cut[0]
    return w
def f_combo(p, dt, momN=126, ddN=252, ddthr=0.10, use_prem=True):
    i = p.i(dt)
    w = 1.0
    if p.mom[momN].iloc[i] <= 0: w = 0.0
    elif p.dd[ddN].iloc[i] <= -ddthr: w = 0.0
    if use_prem: w = premium_adj(w, dt)
    return w

def simulate(Rs, start, tgt_fn, freq="daily", fee=FEE5, label=""):
    R2 = Rs[["159941"]].copy()
    R2 = R2[R2.index >= start].ffill().fillna(0.0)
    dates = R2.index; n = len(dates)
    bond_ = bond.reindex(R2.index).ffill().fillna(0.0)
    repo_d = (1 + REPO) ** (1 / TRADING_DAYS) - 1
    w = 0.0
    rets = np.zeros(n); to_day = np.zeros(n); ws = np.zeros(n); hold = 0.0
    for i in range(n):
        dt = dates[i]
        r = float(R2["159941"].iloc[i]) if not np.isnan(R2["159941"].iloc[i]) else 0.0
        pf_ret = w * r + (1 - w) * (BOND_PCT * float(bond_.iloc[i]) + (1 - BOND_PCT) * repo_d)
        if freq == "daily" or dt.weekday() == 4:
            t = tgt_fn(dt)
            if t is np.nan: t = w
            t = float(np.clip(t, 0.0, 1.0))
            d = t - w
            to_day[i] = abs(d)
            pf_ret -= abs(d) * fee
            w = t
        rets[i] = pf_ret; ws[i] = w; hold += w
    rs = pd.Series(rets, index=dates)
    Wt = (1 + rs).cumprod()
    wdf = pd.DataFrame({"159941": ws, "cash": 1 - ws}, index=dates)
    out = {"name": label, "rets": rs, "wealth": Wt, "weights": wdf, "turnover": float(to_day.sum())}
    e = evaluate(out)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"],
                to=e["turnover"], hold=hold / n)

def run_v26(Rs, ps, fee=FEE5):
    ds = DynamicStrategy(Rs, cfg=CFG)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=CFG.get("min_delta", 0.02),
                       repo=CFG.get("repo_rate", 0.022), tranche_weights=CFG.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=CFG.get("cash_bond_pct", 0.0),
                       rebal_weekday=CFG.get("rebal_weekday", 4), rebal_freq=CFG.get("rebal_freq", "weekly"),
                       strict=True, fee=fee)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"])

CASES = [
    ("buyhold 满仓", lambda dt: f_buyhold(PP, dt), lambda dt: f_buyhold(PPr, dt)),
    ("ma10 单线", lambda dt: f_ma(PP, dt, 10), lambda dt: f_ma(PPr, dt, 10)),
    ("ma20 单线", lambda dt: f_ma(PP, dt, 20), lambda dt: f_ma(PPr, dt, 20)),
    ("ma60 单线", lambda dt: f_ma(PP, dt, 60), lambda dt: f_ma(PPr, dt, 60)),
    ("ma120 单线", lambda dt: f_ma(PP, dt, 120), lambda dt: f_ma(PPr, dt, 120)),
    ("ma20/60 双线", lambda dt: f_ma2(PP, dt, 20, 60), lambda dt: f_ma2(PPr, dt, 20, 60)),
    ("ma50/200 双线", lambda dt: f_ma2(PP, dt, 50, 200), lambda dt: f_ma2(PPr, dt, 50, 200)),
    ("ma 分级(20/60/120)", lambda dt: f_ma_graded(PP, dt), lambda dt: f_ma_graded(PPr, dt)),
    ("mom63 动量", lambda dt: f_mom(PP, dt, 63), lambda dt: f_mom(PPr, dt, 63)),
    ("mom126 动量", lambda dt: f_mom(PP, dt, 126), lambda dt: f_mom(PPr, dt, 126)),
    ("mom252 动量", lambda dt: f_mom(PP, dt, 252), lambda dt: f_mom(PPr, dt, 252)),
    ("dd60 止损-8%", lambda dt: f_dd(PP, dt, 60, 0.08), lambda dt: f_dd(PPr, dt, 60, 0.08)),
    ("dd252 止损-10%", lambda dt: f_dd(PP, dt, 252, 0.10), lambda dt: f_dd(PPr, dt, 252, 0.10)),
    ("dd252 止损-15%", lambda dt: f_dd(PP, dt, 252, 0.15), lambda dt: f_dd(PPr, dt, 252, 0.15)),
    ("海龟20/10", lambda dt: f_turtle(PP, dt), lambda dt: f_turtle(PPr, dt)),
    ("波动率目标15%", lambda dt: f_voltgt(PP, dt, 0.15), lambda dt: f_voltgt(PPr, dt, 0.15)),
    ("RSRS门控 thr0", lambda dt: f_rsrs(PP, dt, 0.0), lambda dt: f_rsrs(PPr, dt, 0.0)),
    ("RSRS门控 thr0.05", lambda dt: f_rsrs(PP, dt, 0.05), lambda dt: f_rsrs(PPr, dt, 0.05)),
    ("MACD门控", lambda dt: f_ind(PP, dt, "macd"), lambda dt: f_ind(PPr, dt, "macd")),
    ("RSI门控", lambda dt: f_ind(PP, dt, "rsi14"), lambda dt: f_ind(PPr, dt, "rsi14")),
    ("combo mom126+dd+prem", lambda dt: f_combo(PP, dt, 126, 252, 0.10, True), lambda dt: f_combo(PPr, dt, 126, 252, 0.10, True)),
    ("combo mom252+prem", lambda dt: f_combo(PP, dt, 252, 252, 0.10, True), lambda dt: f_combo(PPr, dt, 252, 252, 0.10, True)),
    ("ma60 + 溢价门控", lambda dt: premium_adj(f_ma(PP, dt, 60), dt), lambda dt: premium_adj(f_ma(PPr, dt, 60), dt)),
]
rows = []
p0 = run_v26(R, "2014-06-23"); r0 = run_v26(Rr, "2025-04-23")
rows.append(dict(name="A v26组合(基线)", proxy=p0, real=r0))
print(f"{'A v26组合(基线)':<28} | proxy {p0['cagr']:6.2f}/{p0['mdd']:7.2f}/Cal{p0['calmar']:.2f}/TO{p0['to']:6.0f} | real {r0['cagr']:6.2f}/{r0['mdd']:7.2f}/Cal{r0['calmar']:.2f}/TO{r0['to']:6.0f}", flush=True)
for name, fp, fr in CASES:
    p = simulate(R, "2014-06-23", fp, freq="daily", fee=FEE5, label=name)
    r = simulate(Rr, "2025-04-23", fr, freq="daily", fee=FEE5, label=name)
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<28} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:6.0f}/持{p['hold']:.0%} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:6.0f}/持{r['hold']:.0%}", flush=True)
json.dump({"rows": rows}, open(f"{SKILL}/out/exp_v35_nasdaq_swing.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {SKILL}/out/exp_v35_nasdaq_swing.json")
