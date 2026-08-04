# -*- coding: utf-8 -*-
"""v22b growth_split 分量扰动专项审计 (anti-overfit)
逐分量 ±1/±2/±3pp 调整(另一分量补偿), 检查 Calmar 悬崖
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v22b.json")))

def run(cfg):
    from data_prep import build_panel, read_table, rets_from
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    R, _ = build_panel("proxy")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start="2014-06-23", name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], to=e["turnover"])

def move(split, s_from, s_to, dpp):
    g = dict(split)
    g[s_from] = max(0.01, g[s_from] - dpp)
    g[s_to] = g[s_to] + dpp
    assert abs(sum(g.values()) - 1.0) < 1e-6, g
    return g

def main():
    R, _ = __import__("data_prep").build_panel("proxy")  # 预热面板
    base = run(BASE)
    print(f"基线 v22b: CAGR {base['cagr']:.2f}% MDD {base['mdd']:.2f}% Calmar {base['calmar']:.2f} TO {base['to']:.0f}\n")
    rows = []
    bull0 = dict(BASE["growth_split_bull"])  # 35/55/10
    bear0 = dict(BASE["growth_split_bear"])  # 50/30/20
    combos = []
    for comp, other in [("159941", "513500"), ("159941", "159952"), ("159952", "513500")]:
        for dpp in (1, 2, 3):
            combos.append(("bull", comp, other, dpp))
            combos.append(("bull", other, comp, dpp))
    for comp, other in [("159952", "159941"), ("159952", "513500"), ("159941", "513500")]:
        for dpp in (5, 10):
            combos.append(("bear", comp, other, dpp))
    for tag, comp, other, dpp in combos:
        c = copy.deepcopy(BASE)
        if tag == "bull":
            c["growth_split_bull"] = move(bull0, comp, other, dpp / 100.0)
        else:
            c["growth_split_bear"] = move(bear0, comp, other, dpp / 100.0)
        r = run(c)
        delta = r["calmar"] - base["calmar"]
        rows.append({"tag": tag, "comp": comp, "other": other, "dpp": dpp, **r, "d_calmar": delta})
        flag = " <<< 悬崖" if delta < -0.10 else (" <<< 改善" if delta > 0.02 else "")
        print(f"{tag:5s} {comp}->{other} {dpp:+2d}pp | CAGR {r['cagr']:6.2f}% MDD {r['mdd']:6.2f}% "
              f"Calmar {r['calmar']:.3f} ({delta:+.3f}) TO {r['to']:4.0f}{flag}")
    json.dump({"base": base, "rows": rows}, open(f"{OUT}/exp_v22b_split_audit.json", "w"), ensure_ascii=False, indent=1)
    n_cliff = sum(1 for r in rows if r["d_calmar"] < -0.10)
    n_imp = sum(1 for r in rows if r["d_calmar"] > 0.02)
    print(f"\n悬崖(Calmar降>0.10)次数: {n_cliff}/{len(rows)}; 改善(升>0.02)次数: {n_imp}")
    print("[ok] out/exp_v22b_split_audit.json")

if __name__ == "__main__":
    main()
