# -*- coding: utf-8 -*-
"""v31 收益增强探索(不换标的): 牛市仓位平台(state_map 9/8档+max_eq) + vol_target灵敏度
平台扫描防过拟合: 相邻值稳定 + 双窗口同向 + MDD可控
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")
CFG = json.load(open(f"{SKILL_REF}/final_cfg_v26.json"))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")

def run(cfg, Rs, ps):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"], cash=e["avg_cash"]*100)

def C(**kw):
    c = copy.deepcopy(CFG); c.update(kw); return c

def sm(g9, d9, g8, d8):
    sm_ = dict(CFG["state_map"])
    sm_["9"] = [g9, d9]; sm_["8"] = [g8, d8]
    return sm_

cases = [
    ("A 基线v26(9:81/8:85)", C()),
    ("B 仓位小平台(84/88)", C(state_map=sm(84,3,88,6))),
    ("C 仓位中平台(87/91)", C(state_map=sm(87,3,91,6))),
    ("D 仓位大平台(90/94+max_eq1.0)", C(state_map=sm(90,3,94,6), max_eq=1.0)),
    ("E vol_target 0.22", C(vol_target=0.22)),
    ("F vol_target 0.25+scale_hi1.15", C(vol_target=0.25, vol_scale_hi=1.15)),
    ("G B+E组合", C(state_map=sm(84,3,88,6), vol_target=0.22)),
]
rows = []
for name, cfg in cases:
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<28} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/cash{p['cash']:4.1f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/cash{r['cash']:4.1f}", flush=True)
json.dump({"rows": rows}, open(f"{OUT}/exp_v31_enhance.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v31_enhance.json")
