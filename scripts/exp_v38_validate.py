# -*- coding: utf-8 -*-
"""v27候选(sb_recover 8->12)全套验证: 双窗口+OOS+7情景压力+WFO训练/测试
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from stress_test import synthetic_resonance

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f"{SKILL}/references/final_cfg_v26.json"))
CAND = copy.deepcopy(CFG); CAND["speed_brake_recover"] = 12
bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")

def run(cfg, Rs, ps, pe=None):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
                       tranche_weights=cfg.get("tranche_weights"), cash_bond_rets=bond,
                       cash_bond_pct=cfg.get("cash_bond_pct", 0.0), rebal_weekday=cfg.get("rebal_weekday", 4),
                       rebal_freq=cfg.get("rebal_freq", "weekly"), strict=True)
    return evaluate(res)

out = {}
def cmp(tag, cfg_a, cfg_b, Rs, ps, pe=None):
    a, b = run(cfg_a, Rs, ps, pe), run(cfg_b, Rs, ps, pe)
    out[tag] = {"base": {"calmar": a["calmar"], "cagr": a["cagr"]*100, "mdd": a["max_dd"]*100, "sharpe": a["sharpe"]},
                "cand": {"calmar": b["calmar"], "cagr": b["cagr"]*100, "mdd": b["max_dd"]*100, "sharpe": b["sharpe"]}}
    print(f"{tag:28s} base {a['calmar']:5.2f} ({a['cagr']*100:6.2f}/{a['max_dd']*100:6.2f})   cand {b['calmar']:5.2f} ({b['cagr']*100:6.2f}/{b['max_dd']*100:6.2f})")

print("== 双窗口 + OOS ==")
cmp("proxy 全历史", CFG, CAND, R, "2014-06-23")
cmp("real 真实窗口", CFG, CAND, Rr, "2025-04-23")
cmp("OOS 2022-", CFG, CAND, R, "2022-01-04")

print("\n== 7情景压力 ==")
synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
SCEN = [("牛市_2019-2021", "2019-01-04", "2021-02-18", R), ("牛市_2025-2026", "2025-04-23", "2026-07-31", Rr),
        ("震荡_2023-2024", "2023-01-03", "2024-08-30", R), ("熊市_2015", "2015-06-15", "2016-02-29", R),
        ("熊市_2018", "2018-01-02", "2019-01-03", R), ("熊市_2021-22", "2021-02-19", "2022-10-31", R),
        ("共振熊_合成", "2024-09-02", "2026-07-31", synth)]
for nm, ps, pe, Rs in SCEN:
    cmp(nm, CFG, CAND, Rs, ps, pe)

print("\n== WFO 训练/测试(proxy) ==")
cmp("WFO训练 2014-2021", CFG, CAND, R, "2014-06-23", "2021-12-31")
cmp("WFO测试 2022-2026", CFG, CAND, R, "2022-01-04", "2026-07-31")

json.dump(out, open(f"{SKILL}/out/exp_v38_validate.json", "w"), ensure_ascii=False, indent=1)
