# -*- coding: utf-8 -*-
"""v30 调仓规则探索(用户取消"每周三周三笔"约束): tranche比例/频率/周几
v26其余参数不动, 严格口径双窗口(accrual=pre, exec_lag=0, 收盘成交)
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
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"])

def C(**kw):
    c = copy.deepcopy(CFG); c.update(kw); return c

cases = []
# A. 分笔比例 (每周五, weekly)
for tw, label in [([1.0], "A1 1笔(基线)"), ([0.5,0.5], "A2 2笔均分"), ([1/3,1/3,1/3], "A3 3笔均分"),
                  ([0.6,0.4], "A4 2笔6/4"), ([0.7,0.3], "A5 2笔7/3"), ([0.4,0.3,0.3], "A6 3笔4/3/3"),
                  ([0.25,0.25,0.25,0.25], "A7 4笔均分")]:
    cases.append((label, C(tranche_weights=tw)))
# B. 频率 (1笔)
for freq, label in [("biweekly", "B1 隔周1笔"), ("monthly", "B2 月度1笔")]:
    cases.append((label, C(rebal_freq=freq)))
# C. 周几 (每周1笔)
for wd, label in [(0, "C1 周一"), (1, "C2 周二"), (2, "C3 周三"), (3, "C4 周四")]:
    cases.append((label, C(rebal_weekday=wd)))

rows = []
for label, cfg in cases:
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append(dict(name=label, proxy=p, real=r))
    print(f"{label:<16} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:5.0f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:5.0f}", flush=True)
json.dump({"rows": rows}, open(f"{OUT}/exp_v30_schedule.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v30_schedule.json")
