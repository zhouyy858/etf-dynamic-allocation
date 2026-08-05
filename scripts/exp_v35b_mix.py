# -*- coding: utf-8 -*-
"""v35b: 15%国内底仓(159232 7.5% + 515100 7.5%固定不动) + 85%纳指波段(159941) 混合方案
每日调仓(周五周频变体另测), 严格无未来(signal_lag=1, 溢价T-2), 现金层75%511010+25%逆回购"""
import sys, json, numpy as np, pandas as pd
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SKILL}/scripts")
from data_prep import build_panel, read_table, rets_from, TRADING_DAYS
from engine import evaluate
from strategy import SignalSet

bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
REPO = 0.022; BOND_PCT = 0.75; FEE = 0.0005
BASE = {"159232": 0.075, "515100": 0.075}

PX = pd.read_csv(f"{SKILL}/assets/data/qdii_price_159941.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
NAV = pd.read_csv(f"{SKILL}/assets/data/159941_nav.csv", parse_dates=["date"]).set_index("date")
NAV = NAV[~NAV.index.duplicated(keep="last")].sort_index()["unit_nav"]
PREM = (PX / NAV - 1.0).drop(pd.Timestamp("2022-07-04"), errors="ignore").clip(-0.10, 0.15).shift(2)
def prem_at(dt):
    v = PREM.reindex(pd.DatetimeIndex([dt]), method="ffill")
    return None if len(v.dropna()) == 0 else float(v.dropna().iloc[0])

def make_p(Rs):
    sig = SignalSet(Rs, lag=1)
    lvl = (1 + Rs["159941"].dropna()).cumprod().reindex(Rs.index).ffill()
    idx = pd.Series(np.arange(len(lvl)), index=lvl.index)
    return dict(sig=sig, lvl=lvl, idx=idx,
                ma={N: lvl.rolling(N, min_periods=min(N // 2, 20)).mean() for N in (20, 60, 120)},
                mom={N: lvl.pct_change(N) for N in (63, 126, 252)},
                dd={N: (lvl / lvl.rolling(N, min_periods=min(N // 2, 60)).max() - 1.0) for N in (60, 252)},
                vol20=lvl.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS))
def i_of(p, dt): return max(0, int(p["idx"].reindex(pd.DatetimeIndex([dt]), method="ffill").iloc[0]) - 1)

def premium_adj(w, dt, thr=(0.02, 0.05), cut=(0.5, 0.2)):
    v = prem_at(dt)
    if v is None: return w
    if v > thr[1]: return w * cut[1]
    if v > thr[0]: return w * cut[0]
    return w

def sig_ma(p, dt, N=20):
    i = i_of(p, dt); return 1.0 if p["lvl"].iloc[i] > p["ma"][N].iloc[i] else 0.0
def sig_rsrs(p, dt, thr=0.05):
    i = i_of(p, dt); return 1.0 if float(p["sig"].rsrs["159941"].iloc[i]) >= thr else 0.0
def sig_vol(p, dt, target=0.15):
    i = i_of(p, dt); v = p["vol20"].iloc[i]
    if np.isnan(v) or v <= 0: return 1.0
    return min(1.0, target / v)
def sig_combo(p, dt):
    i = i_of(p, dt)
    w = 1.0
    if p["mom"][252].iloc[i] <= 0: w = 0.0
    elif p["dd"][252].iloc[i] <= -0.10: w = 0.0
    return premium_adj(w, dt)

def simulate(Rs, start, sig_fn, freq="daily", max_w=0.85, label=""):
    R2 = Rs[["159232", "515100", "159941"]].copy()
    R2 = R2[R2.index >= start].ffill().fillna(0.0)
    dates = R2.index; n = len(dates)
    bond_ = bond.reindex(R2.index).ffill().fillna(0.0)
    repo_d = (1 + REPO) ** (1 / TRADING_DAYS) - 1
    p = make_p(Rs)
    w = 0.0; rets = np.zeros(n); to = np.zeros(n); ws = np.zeros(n)
    for i in range(n):
        dt = dates[i]
        r232, r100, r941 = float(R2["159232"].iloc[i]), float(R2["515100"].iloc[i]), float(R2["159941"].iloc[i])
        cash = 1.0 - BASE["159232"] - BASE["515100"] - w
        pf_ret = BASE["159232"] * r232 + BASE["515100"] * r100 + w * r941 + cash * (BOND_PCT * float(bond_.iloc[i]) + (1 - BOND_PCT) * repo_d)
        if freq == "daily" or dt.weekday() == 4:
            t = float(np.clip(sig_fn(dt) * max_w, 0.0, max_w))
            d = t - w; to[i] = abs(d); pf_ret -= abs(d) * FEE; w = t
        rets[i] = pf_ret; ws[i] = w
    rs = pd.Series(rets, index=dates); Wt = (1 + rs).cumprod()
    wdf = pd.DataFrame({"159232": BASE["159232"], "515100": BASE["515100"], "159941": ws, "cash": 1 - BASE["159232"] - BASE["515100"] - ws}, index=dates)
    e = evaluate({"name": label, "rets": rs, "wealth": Wt, "weights": wdf, "turnover": float(to.sum())})
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"], hold=float(ws.mean()))

PP = make_p(R); PPr = make_p(Rr)
CASES = [
    ("M1 底仓15+buyhold85", lambda dt: 1.0),
    ("M2 底仓15+ma20波段", lambda dt: sig_ma(PP, dt, 20)),
    ("M3 底仓15+RSRS0.05波段", lambda dt: sig_rsrs(PP, dt, 0.05)),
    ("M4 底仓15+波动率15%", lambda dt: sig_vol(PP, dt, 0.15)),
    ("M5 底仓15+mom252+dd+prem", lambda dt: sig_combo(PP, dt)),
]
rows = []
for name, fn in CASES:
    p = simulate(R, "2014-06-23", fn); r = simulate(Rr, "2025-04-23", fn)
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<24} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:6.0f}/持{p['hold']:.0%} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:6.0f}/持{r['hold']:.0%}", flush=True)
json.dump({"rows": rows}, open(f"{SKILL}/out/exp_v35b_mix.json", "w"), ensure_ascii=False, indent=1)
print(f"[ok] {SKILL}/out/exp_v35b_mix.json")
