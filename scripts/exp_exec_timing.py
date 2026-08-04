# -*- coding: utf-8 -*-
"""成交时点 A/B (v25): 收盘成交(当前) vs 开盘成交近似(post计提=最早边界) vs 次日收盘成交
信号统一用 T-1 收盘 (无未来函数); 差异仅在执行时点
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
CFG = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v25.json")))

def run(Rs, ps, pe, bond, accrual, exec_lag):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    ds = DynamicStrategy(Rs, cfg=CFG)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name="DYN", min_delta=CFG.get("min_delta", 0.02),
                       repo=CFG.get("repo_rate", 0.022), tranche_weights=CFG.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=CFG.get("cash_bond_pct", 0.0),
                       rebal_weekday=CFG.get("rebal_weekday", 4), rebal_freq=CFG.get("rebal_freq", "weekly"),
                       accrual_mode=accrual, exec_lag=exec_lag, strict=(accrual=="pre" and exec_lag==0))
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], to=e["turnover"])

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    modes = [
        ("收盘成交(当前v25)", "pre", 0),
        ("开盘成交≈(post计提)", "post", 0),
        ("次日收盘成交", "pre", 1),
    ]
    print("===== proxy 全历史 (2014-06-23起) =====")
    print(f"{'口径':<22} | {'CAGR':>7} {'MDD':>7} {'Calmar':>6} {'换手':>6}")
    rows = {}
    for nm, am, el in modes:
        p = run(R, "2014-06-23", None, bond, am, el)
        r = run(Rr, "2025-04-23", None, bond, am, el)
        rows[nm] = {"proxy": p, "real": r}
        print(f"{nm:<22} | {p['cagr']:6.2f}% {p['mdd']:6.2f}% {p['calmar']:6.2f} {p['to']:6.1f}")
    print("\n===== real 窗口 (2025-04-23起) =====")
    for nm, am, el in modes:
        r = rows[nm]["real"]
        print(f"{nm:<22} | {r['cagr']:6.2f}% {r['mdd']:6.2f}% {r['calmar']:6.2f} {r['to']:6.1f}")
    json.dump(rows, open(f"{OUT}/exp_exec_timing.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_exec_timing.json")

if __name__ == "__main__":
    main()
