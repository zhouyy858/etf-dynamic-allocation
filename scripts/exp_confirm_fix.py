# -*- coding: utf-8 -*-
"""A/B: _confirm_score 采样日 周三(旧硬编码) vs 周五(与v23调仓日一致)"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate, SLOTS
from strategy import DynamicStrategy

OUT = os.path.join(os.getcwd(), "out")
R_PROXY, W_PROXY = build_panel("proxy")
R_REAL, W_REAL = build_panel("real")
REAL_START = "2025-04-23"

def run_one(R, cfg, label, start=None):
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, name=label, min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    ev = evaluate(res, periods={} if start else None)
    return dict(cagr=ev["cagr"], vol=ev["vol"], max_dd=ev["max_dd"], sharpe=ev["sharpe"],
                calmar=ev["calmar"], turnover=ev["turnover"])

if __name__ == "__main__":
    cfg = json.load(open("references/final_cfg_v23.json"))
    out = {}
    for wd, tag in [(2, "wed(旧)"), (4, "fri(新)")]:
        c = dict(cfg); c["confirm_weekday"] = wd
        out[tag] = {"proxy": run_one(R_PROXY, c, "proxy", start="2014-06-23"), "real": run_one(R_REAL, c, "real", start=REAL_START)}
        print(tag, json.dumps(out[tag], ensure_ascii=False))
    json.dump(out, open(os.path.join(OUT, "exp_confirm_fix.json"), "w"), ensure_ascii=False, indent=1)
    print("saved out/exp_confirm_fix.json")
