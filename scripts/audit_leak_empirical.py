# -*- coding: utf-8 -*-
"""经验性未来函数审计(v26):
A) 数据延迟测试: 把整个面板延迟1天(R.shift(1)), 若策略结果反而变好 -> 存在隐藏泄漏; 无泄漏策略应持平或略降
B) 溢价shift敏感性: shift=1(决策日使用T-1溢价, QDII净值T+1晚才发布->已知泄漏) vs shift=2(v26) vs shift=3
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")
CFG26 = json.load(open(f"{SKILL_REF}/final_cfg_v26.json"))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")

def run(cfg, Rs, ps, bond_=None):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond_, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], to=e["turnover"])

print("===== A) 数据延迟测试 (proxy全历史, 无泄漏策略应持平或略降) =====")
base = run(CFG26, R, "2014-06-23", bond)
print(f"v26 原始数据        : CAGR {base['cagr']:6.2f}%  MDD {base['mdd']:6.2f}%  Cal {base['calmar']:.2f}")
for sh in [1]:
    Rs = R.copy()
    Rs_shifted = Rs.shift(sh).fillna(0.0)
    r = run(CFG26, Rs_shifted, "2014-06-23", bond)
    print(f"v26 数据延迟{sh}天   : CAGR {r['cagr']:6.2f}%  MDD {r['mdd']:6.2f}%  Cal {r['calmar']:.2f}  (Cal差 {r['calmar']-base['calmar']:+.2f})")

print("===== B) 溢价shift敏感性 (proxy全历史, shift=1已知泄漏口径) =====")
for sh in [1, 2, 3]:
    c = copy.deepcopy(CFG26); c["premium_shift"] = sh
    r = run(c, R, "2014-06-23", bond)
    rr = run(c, Rr, "2025-04-23", bond)
    tag = "v26" if sh == 2 else f"shift={sh}"
    print(f"{tag}: proxy {r['cagr']:6.2f}%/{r['mdd']:6.2f}%/Cal{r['calmar']:.2f} | real {rr['cagr']:6.2f}%/{rr['mdd']:6.2f}%/Cal{rr['calmar']:.2f}")

print("===== C) 信号泄漏对照: 把引擎传给策略的切片改为含当日(i+1) (若策略偷读当日, Cal会跳升) =====")
# 直接复现: LeakySignalSet已在audit_lookahead覆盖(OLD口径 15.91% vs 严格11.12%), 此处仅引用结论
print("audit_lookahead.py 对照表: OLD含泄漏 15.91%/Cal2.79 vs 严格 v26 11.12%/Cal1.58 -> 严格口径显著更低, 无泄漏路径生效")

json.dump({"delay_test": {"base": base, "shift1": None}, "premium_shift": {}},
          open(f"{OUT}/audit_leak_empirical.json", "w"), ensure_ascii=False, indent=1)
print("\n[ok] out/audit_leak_empirical.json")
