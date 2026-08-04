# -*- coding: utf-8 -*-
"""premium_tilt 参数平台扫描: thr/cap/max 网格, 选稳定平台防过拟合"""
import sys, os, json, copy, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v24.json")))

def run(cfg, Rs, ps, pe, bond):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"])

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    rows = []
    for thr, cap, mx in itertools.product([0.01, 0.015, 0.02, 0.03], [0.04, 0.05, 0.06], [0.4, 0.5, 0.7]):
        c = copy.deepcopy(BASE)
        c.update(premium_tilt=True, premium_tilt_thr=thr, premium_tilt_cap=cap, premium_tilt_max=mx)
        p = run(c, R, "2014-06-23", None, bond)
        r = run(c, Rr, "2025-04-23", None, bond)
        tr = run(c, R, "2014-06-23", "2021-12-31", bond)
        te = run(c, R, "2022-01-01", None, bond)
        rows.append(dict(thr=thr, cap=cap, mx=mx,
                         pc=p["cagr"], pm=p["mdd"], pcal=p["calmar"],
                         rc=r["cagr"], rm=r["mdd"], rcal=r["calmar"],
                         trcal=tr["calmar"], tecal=te["calmar"]))
    print(f"{'thr':>5} {'cap':>5} {'mx':>4} | {'proxy':>20} | {'real':>20} | WFO tr/te")
    for x in sorted(rows, key=lambda x: -x["pcal"]):
        print(f"{x['thr']:>5.3f} {x['cap']:>5.3f} {x['mx']:>4.2f} | {x['pc']:6.2f}%/{x['pm']:6.2f}%/Cal{x['pcal']:.3f} | "
              f"{x['rc']:6.2f}%/{x['rm']:6.2f}%/Cal{x['rcal']:.3f} | {x['trcal']:.3f}/{x['tecal']:.3f}")
    json.dump(rows, open(f"{OUT}/exp_v24b_tilt_scan.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v24b_tilt_scan.json  (n={len(rows)})")

if __name__ == "__main__":
    main()
