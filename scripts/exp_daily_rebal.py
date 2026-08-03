# -*- coding: utf-8 -*-
"""日频微调执行纪律实验 (对比 每周三+三周三笔1/3 vs 每天小幅步进)
max_step=每日每仓最大权重变动; mode:
  daily_clip : 每天向目标靠近, 每仓最多 max_step (用户方案: 0.1%/0.2%)
  daily_prop : 每天移动缺口的一定比例 gap_frac (保底 min_step, 上限 max_step)
不修改周频引擎; 输出 proxy+real 指标/换手/平均偏离度/最大执行时滞
"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest, evaluate, TRADING_DAYS
from strategy import DynamicStrategy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = json.load(open("out/final_cfg_v15.json"))
SLOTS = ["159232", "515100", "159941", "513500", "159952"]
FEE = 0.0005
REPO = 0.022

def run_weekly(R, cfg, start):
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, name="W", min_delta=0.02, repo=REPO)
    return evaluate(res), res, ds

def run_daily(R, cfg, start, max_step, min_delta=0.0002, mode="daily_clip", gap_frac=0.2, min_step=0.0005):
    """每日步进: 每天计算目标(常规+紧急风控), 按 mode 决定当日步长, 收盘成交扣费"""
    R = R.copy()
    if start: R = R[R.index >= start]
    R = R.ffill().fillna(0.0)
    ds = DynamicStrategy(R, cfg=cfg)
    tf = ds.target_fn(); df = ds.daily_fn()
    dates = R.index; n = len(dates)
    w = np.zeros(len(SLOTS))
    repo_d = (1 + REPO) ** (1 / TRADING_DAYS) - 1
    rets = np.zeros(n); to_day = np.zeros(n); dev_hist = np.zeros(n)
    w_hist_slots = np.zeros((n, len(SLOTS)))
    pf_ctx = []
    for i in range(n):
        dt = dates[i]
        ctx = ({"pf_rets": pd.Series(pf_ctx, index=dates[:len(pf_ctx)])} if pf_ctx else {"pf_rets": pd.Series(dtype=float)})
        ctx["equity"] = float(w.sum()); ctx["weights"] = w.copy()
        tgt = None
        if i == 0:
            t0 = tf(dt, R.iloc[:i], ctx)
            if t0 is not None:
                tgt = np.array([t0[s] for s in SLOTS], dtype=float)
        else:
            t0 = tf(dt, R.iloc[:i], ctx)
            e0 = df(dt, R.iloc[:i], ctx)
            cand = e0 if e0 is not None else t0
            if cand is not None:
                tgt = np.array([cand[s] for s in SLOTS], dtype=float)
        if tgt is not None:
            total = tgt.sum()
            if total > 1.0 + 1e-9:
                tgt = tgt / total
            delta = tgt - w
            step = np.zeros(len(SLOTS))
            if mode == "daily_clip":
                step = np.clip(delta, -max_step, max_step)
            else:  # daily_prop
                step = np.clip(delta * gap_frac, -max_step, max_step)
                step = np.where(np.abs(step) < min_step, 0.0, step)
            if i > 0 and np.abs(step).max() < min_delta:
                step[:] = 0.0
            w = w + step
            to_day[i] = np.abs(step).sum()
        cash = 1.0 - w.sum()
        r = R.iloc[i].values
        g = w * (1.0 + r)
        c = cash * (1.0 + repo_d)
        fee_today = to_day[i] * FEE
        pf_ret = float(g.sum() + c - 1.0 - fee_today)
        factor = 1.0 + pf_ret
        w = g / factor; cash = c / factor
        w_hist_slots[i] = w / (w.sum() + cash)
        rets[i] = pf_ret
        pf_ctx.append(pf_ret)
        if tgt is not None:
            dev_hist[i] = float(np.abs(tgt - w[:len(SLOTS)] / (w.sum() + cash) ).sum())
    rs = pd.Series(rets, index=dates)
    Wt = (1 + rs).cumprod()
    wdf = pd.DataFrame(w_hist_slots, index=dates, columns=SLOTS)
    wdf["cash"] = 1.0 - wdf[SLOTS].sum(axis=1)
    out = {"name": f"daily_{mode}_{max_step:.4f}", "rets": rs, "wealth": Wt, "weights": wdf,
           "turnover": float(to_day.sum()), "avg_dev": float(dev_hist.mean())}
    ev = evaluate(out, periods=None)
    return ev, out, ds

def main():
    rows = []
    R_P, _ = build_panel("proxy"); R_R, _ = build_panel("real")
    cases = []
    for label, R, start in [("proxy", R_P, "2014-06-23"), ("real", R_R, "2025-04-23")]:
        ew, resw, dsw = run_weekly(R, BASE, start)
        rows.append({"variant": f"周频三笔_{label}", "cagr": ew["cagr"]*100, "mdd": ew["max_dd"]*100,
                     "sharpe": ew["sharpe"], "calmar": ew["calmar"], "to": ew["turnover"],
                     "cash": ew["avg_cash"]*100})
        for ms in [0.001, 0.002, 0.005]:
            ed, _, _ = run_daily(R, BASE, start, max_step=ms)
            rows.append({"variant": f"日频clip{ms*100:.1f}%_{label}", "cagr": ed["cagr"]*100, "mdd": ed["max_dd"]*100,
                         "sharpe": ed["sharpe"], "calmar": ed["calmar"], "to": ed["turnover"], "cash": ed["avg_cash"]*100})
        for gf in [0.1, 0.2]:
            ed, _, _ = run_daily(R, BASE, start, max_step=0.01, mode="daily_prop", gap_frac=gf)
            rows.append({"variant": f"日频prop{int(gf*100)}%_{label}", "cagr": ed["cagr"]*100, "mdd": ed["max_dd"]*100,
                         "sharpe": ed["sharpe"], "calmar": ed["calmar"], "to": ed["turnover"], "cash": ed["avg_cash"]*100})
    for r in rows:
        print(f"{r['variant']:22s} CAGR {r['cagr']:6.2f}% MDD {r['mdd']:7.2f}% Sharpe {r['sharpe']:.2f} Calmar {r['calmar']:.2f} TO {r['to']:6.1f} cash {r['cash']:5.1f}%")
    json.dump(rows, open(os.path.join(OUT, "exp_daily_rebal.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_daily_rebal.json")

if __name__ == "__main__":
    main()
