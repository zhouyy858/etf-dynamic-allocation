# -*- coding: utf-8 -*-
"""v37 卖出冷却期机制(知乎"带止盈年化65%"核心理念: 卖出后等回落再买):
标的弹性被清到地板后 N 周内锁地板, 防止追高/反复横跳; 严格双窗口, fee万5"""
import sys, os, json, copy
SKILL = "/Users/mac/.codex/skills/etf-dynamic-allocation"
sys.path.insert(0, f"{SKILL}/scripts")
import numpy as np, pandas as pd
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
CFG = json.load(open(f"{SKILL}/references/final_cfg_v26.json"))

def run(cfg, Rs, ps):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"])

def C(**kw):
    c = copy.deepcopy(CFG); c.update(kw); return c

cases = [
    ("A 基线v26", C()),
    ("B 冷却4周", C(cool_off_weeks=4)),
    ("C 冷却8周", C(cool_off_weeks=8)),
    ("D 冷却12周", C(cool_off_weeks=12)),
    ("E 冷却16周", C(cool_off_weeks=16)),
]
rows = []
for name, cfg in cases:
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<14} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:5.0f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:5.0f}", flush=True)
json.dump({"rows": rows}, open("/tmp/exp_v37_cooloff.json", "w"), ensure_ascii=False, indent=1)
print("[ok] /tmp/exp_v37_cooloff.json")
