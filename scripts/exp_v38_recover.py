# -*- coding: utf-8 -*-
"""v38 探索: 深熊锁解除机制调优 (防2021-02式"锁解除→满仓重入→第二波打穿")
  轴1: recov(解除阈值, market_dd第4元素) CN 0.12-0.20 / US 0.13-0.17
  轴2: 解除逻辑 OR(base: dd>-recov 或 站上SMA120且双均线多头) vs AND(两个条件都要)
  轴3: 重入斜坡: 锁解除后权益上限按 50%/(1-recv_ramp) 逐步恢复 N 周 (新机制)
  验证: 严格口径双窗口(proxy 2014-06-23起 / real 2025-04-23起) + OOS(proxy 2022-01-04起)
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

CFG = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references", "final_cfg_v26.json")))
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
bond = rets_from(read_table("511010_nav.csv"), "cum_nav")

def run(cfg, Rs, ps):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
                       tranche_weights=cfg.get("tranche_weights"), cash_bond_rets=bond,
                       cash_bond_pct=cfg.get("cash_bond_pct", 0.0), rebal_weekday=cfg.get("rebal_weekday", 4),
                       rebal_freq=cfg.get("rebal_freq", "weekly"), strict=True)
    e = evaluate(res)
    return e["calmar"], e["cagr"] * 100, e["max_dd"] * 100

rows = []
def add(name, cfg):
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append({"name": name, "proxy": {"calmar": p[0], "cagr": p[1], "mdd": p[2]},
                 "real": {"calmar": r[0], "cagr": r[1], "mdd": r[2]}})
    print(f"{name:22s} proxy Cal {p[0]:.2f} ({p[1]:.2f}/{p[2]:.2f})  real Cal {r[0]:.2f} ({r[1]:.2f}/{r[2]:.2f})")

base = copy.deepcopy(CFG)
add("A 基线v26", base)

# 轴1: recov 扫描 (CN 第4元素; US 同步按比例 0.15/0.18*0.12..)
for cn in [0.12, 0.13, 0.15, 0.16, 0.18, 0.20]:
    c = copy.deepcopy(CFG)
    c["market_dd"] = {"CN": [0.07, 0.05, 0.18, cn], "US": [0.13, 0.12, 0.24, 0.15]}
    add(f"CN recov {cn}", c)
for us in [0.13, 0.17, 0.19]:
    c = copy.deepcopy(CFG)
    c["market_dd"] = {"CN": [0.07, 0.05, 0.18, 0.14], "US": [0.13, 0.12, 0.24, us]}
    add(f"US recov {us}", c)

json.dump({"rows": rows}, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "exp_v38_recover.json"), "w"), ensure_ascii=False)
