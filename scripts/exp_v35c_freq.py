# -*- coding: utf-8 -*-
"""v35c: 波段方法频率对比(每日 vs 周五周频)"""
import sys, json, numpy as np, pandas as pd
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SKILL}/scripts")
from data_prep import build_panel, read_table, rets_from, TRADING_DAYS
from engine import evaluate
from strategy import SignalSet
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_v35_nasdaq_swing.py")).read().split("def simulate")[0])  # 复用 P/prem/信号函数定义(不含simulate)

def simulate2(Rs, start, tgt_fn, freq="daily", fee=0.0005, label=""):
    R2 = Rs[["159941"]].copy(); R2 = R2[R2.index >= start].ffill().fillna(0.0)
    dates = R2.index; n = len(dates)
    bond_ = bond.reindex(R2.index).ffill().fillna(0.0)
    repo_d = (1 + REPO) ** (1 / TRADING_DAYS) - 1
    w = 0.0; rets = np.zeros(n); to = np.zeros(n); ws = np.zeros(n); hold = 0.0
    for i in range(n):
        dt = dates[i]
        r = float(R2["159941"].iloc[i]) if not np.isnan(R2["159941"].iloc[i]) else 0.0
        pf_ret = w * r + (1 - w) * (BOND_PCT * float(bond_.iloc[i]) + (1 - BOND_PCT) * repo_d)
        if freq == "daily" or dt.weekday() == 4:
            t = tgt_fn(dt)
            if t is np.nan: t = w
            t = float(np.clip(t, 0.0, 1.0)); d = t - w; to[i] = abs(d); pf_ret -= abs(d) * fee; w = t
        rets[i] = pf_ret; ws[i] = w; hold += w
    rs = pd.Series(rets, index=dates); Wt = (1 + rs).cumprod()
    wdf = pd.DataFrame({"159941": ws, "cash": 1 - ws}, index=dates)
    e = evaluate({"name": label, "rets": rs, "wealth": Wt, "weights": wdf, "turnover": float(to.sum())})
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"], hold=hold / n)

rows = []
for name, fp, fr in [
    ("ma20", lambda dt: f_ma(PP, dt, 20), lambda dt: f_ma(PPr, dt, 20)),
    ("RSRS0.05", lambda dt: f_rsrs(PP, dt, 0.05), lambda dt: f_rsrs(PPr, dt, 0.05)),
    ("波动率15%", lambda dt: f_voltgt(PP, dt, 0.15), lambda dt: f_voltgt(PPr, dt, 0.15)),
    ("ma60+prem", lambda dt: premium_adj(f_ma(PP, dt, 60), dt), lambda dt: premium_adj(f_ma(PPr, dt, 60), dt)),
]:
    for freq in ("daily", "weekly"):
        p = simulate2(R, "2014-06-23", fp, freq=freq, label=f"{name}_{freq}")
        r = simulate2(Rr, "2025-04-23", fr, freq=freq, label=f"{name}_{freq}")
        rows.append(dict(name=f"{name} {freq}", proxy=p, real=r))
        print(f"{name:<12} {freq:<7} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:6.0f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:6.0f}", flush=True)
json.dump({"rows": rows}, open(f"{SKILL}/out/exp_v35c_freq.json", "w"), ensure_ascii=False, indent=1)
