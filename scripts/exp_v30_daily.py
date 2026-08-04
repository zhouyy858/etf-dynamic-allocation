# -*- coding: utf-8 -*-
"""v30 每日小幅步进(严格口径): 每天向目标靠近 max_step, 先计提当日收益后收盘成交(accrual=pre),
现金层 75% 511010 + 25% 逆回购, v26其余参数不动, 无未来函数(signal_lag=1)
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")
CFG = json.load(open(f"{SKILL_REF}/final_cfg_v26.json"))
from data_prep import build_panel, read_table, rets_from, TRADING_DAYS
from engine import evaluate
from strategy import DynamicStrategy

bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
SLOTS = ["159232", "515100", "159941", "513500", "159952"]
FEE = 0.0005
REPO = 0.022
CASH_BOND_PCT = CFG.get("cash_bond_pct", 0.75)
MIN_STEP = 0.0002

def run_daily(Rs, start, max_step, mode="daily_clip", gap_frac=0.2):
    R2 = Rs.copy(); R2 = R2[R2.index >= start].ffill().fillna(0.0)
    ds = DynamicStrategy(R2, cfg=CFG)
    tf = ds.target_fn(); df = ds.daily_fn()
    dates = R2.index; n = len(dates)
    bond_ = bond.reindex(R2.index).ffill().fillna(0.0)
    repo_d = (1 + REPO) ** (1 / TRADING_DAYS) - 1
    w = np.zeros(len(SLOTS))
    rets = np.zeros(n); to_day = np.zeros(n)
    pf_ctx = []
    for i in range(n):
        dt = dates[i]
        ctx = {"pf_rets": pd.Series(pf_ctx, index=dates[:i])}
        ctx["equity"] = float(w.sum()); ctx["weights"] = w.copy()
        # 1) 先计提当日收益(accrual=pre): 按昨日权重
        r = R2.iloc[i].values
        g = w * (1.0 + r)
        cash_pre = 1.0 - w.sum()
        c = cash_pre * (1.0 + CASH_BOND_PCT * float(bond_.iloc[i]) + (1 - CASH_BOND_PCT) * repo_d)
        factor = float(g.sum() + c)
        w = g / factor
        # 2) 收盘成交: T-1信号目标(无未来)
        e0 = df(dt, R2.iloc[:i], ctx); t0 = tf(dt, R2.iloc[:i], ctx)
        cand = e0 if e0 is not None else t0
        if cand is not None:
            tgt = np.array([cand[s] for s in SLOTS], dtype=float)
            total = tgt.sum()
            if total > 1.0 + 1e-9:
                tgt = tgt / total
            delta = tgt - w
            if mode == "daily_clip":
                step = np.clip(delta, -max_step, max_step)
            else:
                step = np.clip(delta * gap_frac, -max_step, max_step)
            if np.abs(step).max() >= MIN_STEP:
                fee_t = float(np.abs(step).sum()) * FEE
                w = w + step
                w = w / (1.0 + fee_t)  # 手续费从组合扣
                to_day[i] = float(np.abs(step).sum())
        # 组合日收益(成交不产生当日收益=pre口径, 费用单独扣)
        pf_ret = (w * (1.0 + r)).sum() + (1.0 - w.sum()) * (1.0 + CASH_BOND_PCT * float(bond_.iloc[i]) + (1 - CASH_BOND_PCT) * repo_d) - 1.0
        if to_day[i] > 0:
            pf_ret -= to_day[i] * FEE
        rets[i] = pf_ret
        pf_ctx.append(pf_ret)
    rs = pd.Series(rets, index=dates)
    Wt = (1 + rs).cumprod()
    wdf = pd.DataFrame(np.zeros((n, len(SLOTS))), index=dates, columns=SLOTS)
    wdf["cash"] = 1.0 - wdf[SLOTS].sum(axis=1)
    out = {"name": f"daily_{mode}_{max_step}", "rets": rs, "wealth": Wt, "weights": wdf,
           "turnover": float(to_day.sum())}
    e = evaluate(out)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"])

def main():
    cases = [("D2 daily_clip 0.1%/日", dict(max_step=0.001)),
             ("D3 daily_clip 0.2%/日", dict(max_step=0.002)),
             ("D4 daily_clip 0.5%/日", dict(max_step=0.005)),
             ("D5 daily_prop 10%/日(上限1%)", dict(max_step=0.01, mode="daily_prop", gap_frac=0.1)),
             ("D6 daily_prop 20%/日(上限1%)", dict(max_step=0.01, mode="daily_prop", gap_frac=0.2))]
    rows = []
    for label, kw in cases:
        p = run_daily(R, "2014-06-23", **kw); r = run_daily(Rr, "2025-04-23", **kw)
        rows.append(dict(name=label, proxy=p, real=r))
        print(f"{label:<24} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:6.0f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:6.0f}", flush=True)
    json.dump({"rows": rows}, open(f"{OUT}/exp_v30_daily.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v30_daily.json")

if __name__ == "__main__":
    main()
