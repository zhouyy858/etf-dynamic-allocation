# -*- coding: utf-8 -*-
"""v34 常用技术指标面板验证(MACD/RSI14/KDJ/BOLL%B/CCI14/TRIX12/BIAS24/WILLR14/MOM20/ZSCORE60):
10指标 x 4用法(门控成长/门控全部/防御动量纯指标/防御动量0.5混合), 严格v26口径双窗口
指标全部close-only(数据无high/low/volume), 强度归一化0~1中性0.5, 由_idx结构防线保证无未来函数
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

INDS = ["macd", "rsi14", "kdj", "boll_pctb", "cci14", "trix12", "bias24", "willr14", "mom20", "zscore60"]
cases = [("A 基线v26", C())]
for nm in INDS:
    cases += [
        (f"{nm} 门控成长", C(ind_gate=True, ind_name=nm, ind_thr=0.5, ind_cut=0.5)),
        (f"{nm} 门控全部", C(ind_gate=True, ind_gate_all=True, ind_name=nm, ind_thr=0.5, ind_cut=0.5)),
        (f"{nm} 防御纯指标", C(ind_defense=True, ind_name=nm, ind_defense_mix=1.0)),
        (f"{nm} 防御0.5混合", C(ind_defense=True, ind_name=nm, ind_defense_mix=0.5)),
    ]
rows = []
for name, cfg in cases:
    p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<26} | proxy {p['cagr']:6.2f}/{p['mdd']:7.2f}/Cal{p['calmar']:.2f}/TO{p['to']:5.0f} | real {r['cagr']:6.2f}/{r['mdd']:7.2f}/Cal{r['calmar']:.2f}/TO{r['to']:5.0f}", flush=True)
json.dump({"rows": rows}, open(f"{OUT}/exp_v34_indicators.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v34_indicators.json")
