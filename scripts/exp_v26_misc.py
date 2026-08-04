# -*- coding: utf-8 -*-
"""v26 收尾复核: 微扫未覆盖的剩余策略轴(defense_momentum/corr_risk_cut/premium_cut/vol_buf)"""
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
    print(f"{'配置':<30} | {'proxy':>20} | {'real':>20}")
    print(f"{'基线v25':<30} | {bp['cagr']:6.2f}/{bp['mdd']:6.2f}/Cal{bp['calmar']:.2f} | {br['cagr']:6.2f}/{br['mdd']:6.2f}/Cal{br['calmar']:.2f}")
    rows=[dict(name="基线v25", proxy=bp, real=br)]
    opts = {
        "dm_t=3.5":      {"defense_momentum_t": 3.5},
        "dm_t=4.5":      {"defense_momentum_t": 4.5},
        "dm_win=60":     {"defense_momentum_win": 60},
        "dm_win=100":    {"defense_momentum_win": 100},
        "corr_cut 0.85/0.65": {"corr_risk_cut": [0.85, 0.65]},
        "corr_cut 0.75/0.55": {"corr_risk_cut": [0.75, 0.55]},
        "prem_cut 0.7/0.4/0.2": {"premium_cut": [0.7, 0.4, 0.2]},
        "prem_cut 0.5/0.25/0.1": {"premium_cut": [0.5, 0.25, 0.1]},
        "corr_win=30":   {"corr_risk_win": 30},
        "corr_win=60":   {"corr_risk_win": 60},
    }
    for name, kv in opts.items():
        c = copy.deepcopy(BASE); c.update(kv)
        p = run(c, R, "2014-06-23", bond); r = run(c, Rr, "2025-04-23", bond)
        rows.append(dict(name=name, proxy=p, real=r))
        print(f"{name:<30} | {p['cagr']:6.2f}/{p['mdd']:6.2f}/Cal{p['calmar']:.2f} | {r['cagr']:6.2f}/{r['mdd']:6.2f}/Cal{r['calmar']:.2f}")
    json.dump({"base_proxy": bp, "base_real": br, "rows": rows}, open(f"{OUT}/exp_v26_misc.json","w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v26_misc.json")

if __name__ == "__main__":
    main()
