# -*- coding: utf-8 -*-
"""收益-回撤约束前沿: 全局权益激进系数 f (state_map growth/defense 等比缩放)
展示 MDD 预算 ↔ CAGR 的取舍, 供用户在"可接受回撤"与"收益"间选择
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v23.json")))

def scale_state(cfg, f, max_eq=0.98):
    c = copy.deepcopy(cfg)
    c["state_map"] = {k: [max(4, min(98, round(g * f))), max(1, min(40, round(d * f)))]
                      for k, (g, d) in c["state_map"].items()}
    c["max_eq"] = max_eq
    return c

def run(cfg, R, bond, start):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], cash=e["avg_cash"]*100)

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    rows = []
    for f in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]:
        for me in [0.98, 1.0]:
            c = scale_state(BASE, f, max_eq=me)
            p = run(c, R, bond, "2014-06-23")
            r_ = run(c, Rr, bond, "2025-04-23")
            rows.append({"f": f, "max_eq": me, "p_cagr": p["cagr"], "p_mdd": p["mdd"],
                         "p_calmar": p["calmar"], "p_cash": p["cash"],
                         "r_cagr": r_["cagr"], "r_mdd": r_["mdd"], "r_calmar": r_["calmar"]})
    print(f"{'f':>5} {'max_eq':>6} | {'proxy CAGR':>9} {'MDD':>7} {'Calmar':>7} {'现金':>5} | {'real CAGR':>9} {'MDD':>7} {'Calmar':>7}")
    for x in rows:
        print(f"{x['f']:>5.2f} {x['max_eq']:>6.2f} | {x['p_cagr']:>8.2f}% {x['p_mdd']:>6.2f}% {x['p_calmar']:>7.2f} {x['p_cash']:>4.0f}% | "
              f"{x['r_cagr']:>8.2f}% {x['r_mdd']:>6.2f}% {x['r_calmar']:>7.2f}")
    json.dump(rows, open(f"{OUT}/exp_frontier.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_frontier.json")

if __name__ == "__main__":
    main()
