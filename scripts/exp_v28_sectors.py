# -*- coding: utf-8 -*-
"""v28候选探索: 银行512800/酒512690/医疗512170/新能源车515030/医药512010/有色512400/军工512660
严格v26口径双窗口(proxy 2014-06-23起 / real 2025-04-23起), 无未来函数
proxy=csindex指数(2012/2015起), real=腾讯场内价(上市日起)
每个候选分别替换 159232/515100/159952 三个槽位, 策略结构不变
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

CAND = [
    ("512800", "银行",   "399986"),
    ("512690", "酒",     "399987"),
    ("512170", "医疗",   "399989"),
    ("515030", "新能源车", "930997"),
    ("512010", "医药",   "000913"),
    ("512400", "有色",   "000819"),
    ("512660", "军工",   "399967"),
]
SLOTS = ["159232", "515100", "159952"]

def cand_rets(fund, idx, layer):
    if layer == "proxy":
        return rets_from(read_table(f"index_{idx}.csv"), "close")
    return rets_from(read_table(f"{fund}_price.csv"), "close")

def swap(R, slot, new):
    R2 = R.copy()
    R2[slot] = new.reindex(R2.index).ffill().fillna(0.0)
    return R2

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

base_p = run(CFG, R, "2014-06-23")
base_r = run(CFG, Rr, "2025-04-23")
print(f"基线 proxy {base_p['cagr']:.2f}/{base_p['mdd']:.2f}/Cal{base_p['calmar']:.2f} | real {base_r['cagr']:.2f}/{base_r['mdd']:.2f}/Cal{base_r['calmar']:.2f}", flush=True)

out = {"base_proxy": base_p, "base_real": base_r, "cands": []}
for fund, name, idx in CAND:
    P = cand_rets(fund, idx, "proxy"); Rl = cand_rets(fund, idx, "real")
    # 资产统计
    p_full = P.copy()
    stats = {
        "cagr_proxy": (np.prod(1 + p_full.loc["2014-06-23":]) ** (252/len(p_full.loc["2014-06-23":])) - 1) * 100,
        "mdd_proxy": np.exp(p_full.loc["2014-06-23":].cumsum()).pipe(lambda w: (w/w.cummax()-1).min()) * 100,
        "vol_proxy": p_full.loc["2014-06-23":].std() * np.sqrt(252) * 100,
        "corr_159232": float(P.reindex(R.index).ffill().corr(R["159232"])),
        "corr_515100": float(P.reindex(R.index).ffill().corr(R["515100"])),
        "corr_159952": float(P.reindex(R.index).ffill().corr(R["159952"])),
        "corr_159941": float(P.reindex(R.index).ffill().corr(R["159941"])),
    }
    entry = {"fund": fund, "name": name, "idx": idx, "stats": stats, "swaps": {}}
    for slot in SLOTS:
        cfg = copy.deepcopy(CFG)
        Rp = swap(R, slot, P); Rrl = swap(Rr, slot, Rl)
        pp = run(cfg, Rp, "2014-06-23"); rr = run(cfg, Rrl, "2025-04-23")
        entry["swaps"][slot] = {"proxy": pp, "real": rr}
        print(f"{name}({fund}) -> {slot}: proxy {pp['cagr']:.2f}/{pp['mdd']:.2f}/Cal{pp['calmar']:.2f} | real {rr['cagr']:.2f}/{rr['mdd']:.2f}/Cal{rr['calmar']:.2f}", flush=True)
    out["cands"].append(entry)

json.dump(out, open(f"{OUT}/exp_v28_sectors.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[ok] {OUT}/exp_v28_sectors.json")
