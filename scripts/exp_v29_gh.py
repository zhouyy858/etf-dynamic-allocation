# -*- coding: utf-8 -*-
"""v29 GitHub方案吸收探索: Antonacci多窗口动量 / growth_rotation / vol_scale
严格v26口径双窗口, 其余参数不动, 单机制独立验证
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
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"],
                to=e["turnover"], cash=e["avg_cash"]*100)

def C(**kw):
    c = copy.deepcopy(CFG)
    c.update(kw)
    return c

variants = {
    "A 基线v26": C(),
    "B multi_mom(Antonacci 3/6/12)": C(defense_momentum_multi=True),
    "C growth_rotation": C(growth_rotation=True),
    "D vol_scale(1.15/0.85)": C(vol_scale_hi=1.15, vol_scale_lo=0.85),
    "E B+C组合": C(defense_momentum_multi=True, growth_rotation=True),
}
rows = []
for name, cfg in variants.items():
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<32} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/cash{p['cash']:4.1f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/cash{r['cash']:4.1f}", flush=True)
json.dump({"rows": rows}, open(f"{OUT}/exp_v29_gh.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v29_gh.json")

# 海龟式标的级移动止损(20日高点回撤) —— 第二批
import copy as _copy
def C2(**kw):
    c = _copy.deepcopy(CFG); c.update(kw); return c
rows2 = []
for name, cfg in [
    ("F hh_stop(20日高点-8%x0.5)", C2(hh_stop=True)),
    ("G hh_stop(20日高点-6%x0.5)", C2(hh_stop=True, hh_thr=0.06)),
    ("H hh_stop(20日高点-10%x0.5)", C2(hh_stop=True, hh_thr=0.10)),
]:
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows2.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<32} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}", flush=True)
d = json.load(open(f"{OUT}/exp_v29_gh.json"))
d["rows"] += rows2
json.dump(d, open(f"{OUT}/exp_v29_gh.json", "w"), ensure_ascii=False, indent=1)
print("[ok] rows2 appended")
