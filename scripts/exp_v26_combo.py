# -*- coding: utf-8 -*-
"""v26 组合验证: 微扫候选轴组合 + 平台形态细扫 + premium_tilt cap/max
严格口径: proxy 2014-06-23 起 / real 2025-04-23 起, 与前序微扫一致
"""
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

def both(cfg, R, Rr, bond, ps="2014-06-23", prs="2025-04-23"):
    p = run(cfg, R, ps, bond)
    r = run(cfg, Rr, prs, bond)
    return p, r

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")

    bp, br = both(BASE, R, Rr, bond)
    print(f"{'配置':<46} | {'proxy':>20} | {'real':>20}")
    print(f"{'基线v25':<46} | {bp['cagr']:6.2f}/{bp['mdd']:6.2f}/Cal{bp['calmar']:.2f} | {br['cagr']:6.2f}/{br['mdd']:6.2f}/Cal{br['calmar']:.2f}")
    rows = [dict(name="基线v25", proxy=bp, real=br)]

    combos = {
        "gate_win=110":          {"gate_win": 110},
        "gate_win=105":          {"gate_win": 105},
        "gate_win=115":          {"gate_win": 115},
        "CN_fb=0.06":            {"market_dd": {"CN": [0.06,0.05,0.18,0.14], "US": BASE["market_dd"]["US"]}},
        "min_delta=0.04":        {"min_delta": 0.04},
        "A+B (gate110+CN.06)":   {"gate_win": 110, "market_dd": {"CN": [0.06,0.05,0.18,0.14], "US": BASE["market_dd"]["US"]}},
        "A+C (gate110+md.04)":   {"gate_win": 110, "min_delta": 0.04},
        "B+C (CN.06+md.04)":     {"min_delta": 0.04, "market_dd": {"CN": [0.06,0.05,0.18,0.14], "US": BASE["market_dd"]["US"]}},
        "A+B+C":                 {"gate_win": 110, "min_delta": 0.04, "market_dd": {"CN": [0.06,0.05,0.18,0.14], "US": BASE["market_dd"]["US"]}},
        "ptilt_cap=0.04":        {"premium_tilt_cap": 0.04},
        "ptilt_cap=0.06":        {"premium_tilt_cap": 0.06},
        "ptilt_max=0.4":         {"premium_tilt_max": 0.4},
        "ptilt_max=0.6":         {"premium_tilt_max": 0.6},
        "ptilt_thr=0.015":       {"premium_tilt_thr": 0.015},
        "ptilt_thr=0.03":        {"premium_tilt_thr": 0.03},
        "hyst 0.66/0.21":        {"hyst_down": 0.21},
    }
    for name, kv in combos.items():
        c = copy.deepcopy(BASE); c.update(kv)
        p, r = both(c, R, Rr, bond)
        rows.append(dict(name=name, proxy=p, real=r))
        print(f"{name:<46} | {p['cagr']:6.2f}/{p['mdd']:6.2f}/Cal{p['calmar']:.2f} | {r['cagr']:6.2f}/{r['mdd']:6.2f}/Cal{r['calmar']:.2f}")
    json.dump({"base_proxy": bp, "base_real": br, "rows": rows},
              open(f"{OUT}/exp_v26_combo.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v26_combo.json")

if __name__ == "__main__":
    main()
