# -*- coding: utf-8 -*-
"""v25 vs v26 共振熊(带合成指数信号 a_mkt_override) 修正重跑"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from stress_test import synthetic_resonance

CFG25 = json.load(open(f"{SKILL_REF}/final_cfg_v25.json"))
CFG26 = json.load(open(f"{SKILL_REF}/final_cfg_v26.json"))
bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
Rr, _ = build_panel("real")
synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)

def run(cfg, Rs, ps, pe, a_mkt=None):
    ds = DynamicStrategy(Rs, cfg=cfg, a_mkt_override=a_mkt)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], to=e["turnover"])

a25 = run(CFG25, synth, "2024-09-02", "2026-07-31", s_idx)
a26 = run(CFG26, synth, "2024-09-02", "2026-07-31", s_idx)
print(f"共振熊(带合成指数信号):")
print(f"  v25: MDD={a25['mdd']:.2f}% CAGR={a25['cagr']:.2f}% Cal={a25['calmar']:.2f} TO={a25['to']:.0f}%")
print(f"  v26: MDD={a26['mdd']:.2f}% CAGR={a26['cagr']:.2f}% Cal={a26['calmar']:.2f} TO={a26['to']:.0f}%")
json.dump({"v25": a25, "v26": a26}, open(f"{OUT}/exp_v26_resonance.json", "w"), ensure_ascii=False, indent=1)
print("[ok]")
