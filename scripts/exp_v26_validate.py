# -*- coding: utf-8 -*-
"""v26 全量验证: 全窗口 + WFO + 压力情景(v25 vs v26 同口径)"""
import sys, os, json, copy
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
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")

def run(cfg, Rs, ps, pe=None):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"],
                to=e["turnover"], cash=e["avg_cash"]*100)

def show(tag, p25, p26, r25, r26):
    print(f"\n===== {tag} =====")
    print(f"{'窗口':<16}{'v25 CAGR':>9}{'v25 MDD':>9}{'v25 Cal':>8} | {'v26 CAGR':>9}{'v26 MDD':>9}{'v26 Cal':>8}")
    for name, a, b in [("proxy全历史", p25, p26), ("real窗口", r25, r26)]:
        print(f"{name:<16}{a['cagr']:8.2f}%{a['mdd']:8.2f}%{a['calmar']:8.2f} | {b['cagr']:8.2f}%{b['mdd']:8.2f}%{b['calmar']:8.2f}")
    return {"proxy": {"v25": p25, "v26": p26}, "real": {"v25": r25, "v26": r26}}

out = {}
p25 = run(CFG25, R, "2014-06-23"); p26 = run(CFG26, R, "2014-06-23")
r25 = run(CFG25, Rr, "2025-04-23"); r26 = run(CFG26, Rr, "2025-04-23")
out["full"] = show("全窗口", p25, p26, r25, r26)

# WFO: 训练 2014-2021 / OOS 2022+
w25_tr = run(CFG25, R, "2014-06-23", "2021-12-31"); w26_tr = run(CFG26, R, "2014-06-23", "2021-12-31")
w25_te = run(CFG25, R, "2022-01-01"); w26_te = run(CFG26, R, "2022-01-01")
out["wfo"] = {"v25": {"train": w25_tr, "test": w25_te}, "v26": {"train": w26_tr, "test": w26_te}}
print("\n===== WFO =====")
print(f"{'':10}{'v25 train':>22}{'v25 OOS':>22}{'v26 train':>22}{'v26 OOS':>22}")
for k in ["cagr", "mdd", "calmar"]:
    print(f"{k:<10}{w25_tr[k]:>15.2f}{w25_te[k]:>15.2f}{w26_tr[k]:>15.2f}{w26_te[k]:>15.2f}")

# 压力情景
synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
SCEN = [
    ("牛市_2019-2021", "2019-01-04", "2021-02-18", R, None),
    ("震荡市_2023-2024", "2023-01-03", "2024-08-30", R, None),
    ("熊市_2015股灾", "2015-06-15", "2016-02-29", R, None),
    ("熊市_2021-2022", "2021-02-19", "2022-10-31", R, None),
    ("共振熊_US-30_CN-20", "2024-09-02", "2026-07-31", synth, s_idx),
]
out["stress"] = {}
print("\n===== 压力情景 (max_dd%) =====")
for name, ps, pe, Rs, am in SCEN:
    a25 = run(CFG25, Rs, ps, pe); a26 = run(CFG26, Rs, ps, pe)
    out["stress"][name] = {"v25": a25, "v26": a26}
    print(f"{name:<22} v25 MDD={a25['mdd']:6.2f}% Cal={a25['calmar']:5.2f} | v26 MDD={a26['mdd']:6.2f}% Cal={a26['calmar']:5.2f}")

json.dump(out, open(f"{OUT}/exp_v26_validate.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v26_validate.json")
