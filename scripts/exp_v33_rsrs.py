# -*- coding: utf-8 -*-
"""v33 RSRS 机制验证(来自 94fsckbzfd/quant-etf-strategy V8.7, close-only版):
R3a 绝对门控只滤成长 / R3b 全标的口径 / R1 防御动量RSRS替代 / R1b 0.5混合
严格v26口径双窗口, 平台扫描(阈值0.0/0.05/0.10)
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

cases = [
    ("A 基线v26", C()),
    ("B R3a 门控成长 thr0.05", C(rsrs_gate=True)),
    ("C R3b 门控全部 thr0.05", C(rsrs_gate=True, rsrs_gate_all=True)),
    ("D R1 防御动量纯RSRS", C(rsrs_defense=True, rsrs_defense_mix=1.0)),
    ("E R1b 防御动量0.5混合", C(rsrs_defense=True, rsrs_defense_mix=0.5)),
    ("F R3a 阈值0.0", C(rsrs_gate=True, rsrs_thr=0.0)),
    ("G R3a 阈值0.10", C(rsrs_gate=True, rsrs_thr=0.10)),
]
rows = []
for name, cfg in cases:
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<26} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:5.0f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:5.0f}", flush=True)
json.dump({"rows": rows}, open(f"{OUT}/exp_v33_rsrs.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v33_rsrs.json")
