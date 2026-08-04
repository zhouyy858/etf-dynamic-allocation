# -*- coding: utf-8 -*-
"""CN快刹车触发阈值平台形态: 0.05~0.08 细扫 + 与min_delta交互"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v25.json")))

def run(cfg, Rs, ps, bond):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], to=e["turnover"])

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    bp = run(BASE, R, "2014-06-23", bond); br = run(BASE, Rr, "2025-04-23", bond)
    print(f"{'配置':<26} | {'proxy':>20} | {'real':>20}")
    print(f"{'基线v25(0.07)':<26} | {bp['cagr']:6.2f}/{bp['mdd']:6.2f}/Cal{bp['calmar']:.2f} | {br['cagr']:6.2f}/{br['mdd']:6.2f}/Cal{br['calmar']:.2f}")
    rows=[]
    for thr in [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]:
        for md in [0.035, 0.04]:
            c = copy.deepcopy(BASE)
            c["market_dd"] = {"CN": [thr, 0.05, 0.18, 0.14], "US": BASE["market_dd"]["US"]}
            c["min_delta"] = md
            p = run(c, R, "2014-06-23", bond); r = run(c, Rr, "2025-04-23", bond)
            rows.append(dict(thr=thr, md=md, proxy=p, real=r))
            print(f"CN_thr={thr:.3f} md={md:.3f}".ljust(26) + f" | {p['cagr']:6.2f}/{p['mdd']:6.2f}/Cal{p['calmar']:.2f} | {r['cagr']:6.2f}/{r['mdd']:6.2f}/Cal{r['calmar']:.2f}")
    json.dump({"base_proxy": bp, "base_real": br, "rows": rows}, open(f"{OUT}/exp_v26_cnfb.json","w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v26_cnfb.json")

if __name__ == "__main__":
    main()
