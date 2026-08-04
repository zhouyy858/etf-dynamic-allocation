# -*- coding: utf-8 -*-
"""v27候选探索: 剔除159232自由现金流 vs 加入券商ETF(512880, 中证全指证券公司)
严格v26口径双窗口(proxy 2014-06-23起 / real 2025-04-23起), 无未来函数
券商数据: proxy=index_sz399975(中证全指证券公司, csindex 2011+); real=512880场内价(2016+)
槽位替换保持策略结构不变(防御动量桶/CN成长桶), 换的是底层标的
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

def broker_rets(layer):
    if layer == "proxy":
        df = read_table("index_sz399975.csv")
        return rets_from(df, "close")
    df = read_table("512880_price.csv")
    return rets_from(df, "close")

def swap(R, slot, new):
    R2 = R.copy()
    R2[slot] = new.reindex(R2.index).ffill().fillna(0.0)
    return R2

def run(cfg, Rs, ps, bond_=None):
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond_ or bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"])

BR = {layer: broker_rets(layer) for layer in ["proxy", "real"]}

variants = {}
variants["A 基线v26(含159232)"] = (CFG, R, Rr)
c_excl = copy.deepcopy(CFG); c_excl["exclude"] = ["159232"]
variants["B 剔除159232(4资产)"] = (c_excl, R, Rr)
c_swap232 = copy.deepcopy(CFG)
Rp232 = swap(R, "159232", BR["proxy"]); Rr232 = swap(Rr, "159232", BR["real"])
variants["C 159232->券商"] = (c_swap232, Rp232, Rr232)
Rp500 = swap(R, "515100", BR["proxy"]); Rr500 = swap(Rr, "515100", BR["real"])
variants["D 515100->券商"] = (c_swap232, Rp500, Rr500)
Rp952 = swap(R, "159952", BR["proxy"]); Rr952 = swap(Rr, "159952", BR["real"])
variants["E 159952->券商(CN成长)"] = (c_swap232, Rp952, Rr952)

print(f"{'变体':<24} | {'proxy CAGR/MDD/Cal':>26} | {'real CAGR/MDD/Cal':>26}")
print(f"{'A 基线v26':<24} | {'':>26} | {'':>26}")
rows = []
for name, (cfg, Rp, Rr_) in variants.items():
    p = run(cfg, Rp, "2014-06-23")
    r = run(cfg, Rr_, "2025-04-23")
    rows.append(dict(name=name, proxy=p, real=r))
    print(f"{name:<24} | {p['cagr']:6.2f}/{p['mdd']:6.2f}/Cal{p['calmar']:.2f} | {r['cagr']:6.2f}/{r['mdd']:6.2f}/Cal{r['calmar']:.2f}")
json.dump({"base_proxy": rows[0]["proxy"], "base_real": rows[0]["real"], "rows": rows},
          open(f"{OUT}/exp_v27_broker.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v27_broker.json")
