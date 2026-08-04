# -*- coding: utf-8 -*-
"""v26 候选: v25平台参数微扫(逐轴单测) + 护栏内组合微调
全部严格无未来函数口径; 每个配置只变一个轴
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v25.json")))

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
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"], to=e["turnover"])

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    AXES = {
        "vol_target": [("vol_target", v) for v in [0.17, 0.18, 0.19, 0.20, 0.21]],
        "min_delta": [("min_delta", v) for v in [0.03, 0.035, 0.04, 0.045, 0.05]],
        "hyst_up": [("hyst_up", v) for v in [0.55, 0.60, 0.66, 0.72, 0.78]],
        "hyst_down": [("hyst_down", v) for v in [0.18, 0.21, 0.24, 0.27, 0.30]],
        "gate_win": [("gate_win", v) for v in [90, 100, 110, 120, 130]],
        "floor_us": [("floor_pct", {"cn": 0.0, "us": v}) for v in [5.0, 7.5, 10.0, 12.5]],
        "speed_brake_thr": [("speed_brake_thr", v) for v in [-0.03, -0.035, -0.04, -0.05]],
        "dd_eq_cap": [("dd_eq_cap", v) for v in
                      [[[-0.06,88],[-0.1,78],[-0.14,68],[-0.18,58]],
                       [[-0.1,82],[-0.14,72],[-0.18,62],[-0.22,52]],
                       [[-0.08,80],[-0.12,70],[-0.16,60],[-0.2,50]],
                       [[-0.08,90],[-0.12,80],[-0.16,70],[-0.2,60]]]],
        "CN_fb": [("market_dd", {"CN": [v, 0.05, 0.18, 0.14], "US": BASE["market_dd"]["US"]}) for v in [0.06, 0.07, 0.08]],
        "corr_thr": [("corr_risk_thr", v) for v in [[0.25,0.37],[0.3,0.42],[0.35,0.47]]],
        "premium_thr": [("premium_thr", v) for v in [[0.04,0.07,0.11],[0.05,0.08,0.12],[0.06,0.09,0.13]]],
        "grow_bull": [("growth_split_bull", v) for v in
                      [{"159952":0.34,"159941":0.56,"513500":0.10},{"159952":0.34,"159941":0.55,"513500":0.11},
                       {"159952":0.35,"159941":0.55,"513500":0.10},{"159952":0.35,"159941":0.54,"513500":0.11},
                       {"159952":0.36,"159941":0.54,"513500":0.10},{"159952":0.36,"159941":0.53,"513500":0.11}]],
        "grow_bear": [("growth_split_bear", v) for v in
                      [{"159952":0.45,"159941":0.35,"513500":0.20},{"159952":0.50,"159941":0.30,"513500":0.20},
                       {"159952":0.55,"159941":0.25,"513500":0.20},{"159952":0.50,"159941":0.25,"513500":0.25},
                       {"159952":0.45,"159941":0.30,"513500":0.25}]],
    }
    print(f"{'轴':<14} {'取值':<34} | {'proxy':>20} | {'real':>20}")
    rows = []
    base_p = run(BASE, R, "2014-06-23", None, bond)
    base_r = run(BASE, Rr, "2025-04-23", None, bond)
    print(f"{'基线v25':<14} {'':<34} | {base_p['cagr']:6.2f}%/{base_p['mdd']:6.2f}%/Cal{base_p['calmar']:.2f} | "
          f"{base_r['cagr']:6.2f}%/{base_r['mdd']:6.2f}%/Cal{base_r['calmar']:.2f}")
    for ax, opts in AXES.items():
        for key, val in opts:
            c = copy.deepcopy(BASE); c[key] = copy.deepcopy(val)
            p = run(c, R, "2014-06-23", None, bond)
            r = run(c, Rr, "2025-04-23", None, bond)
            lab = str(val)[:32]
            rows.append(dict(ax=ax, val=lab, proxy=p, real=r))
            print(f"{ax:<14} {lab:<34} | {p['cagr']:6.2f}%/{p['mdd']:6.2f}%/Cal{p['calmar']:.2f} | "
                  f"{r['cagr']:6.2f}%/{r['mdd']:6.2f}%/Cal{r['calmar']:.2f}")
    json.dump({"base_proxy": base_p, "base_real": base_r, "rows": rows},
              open(f"{OUT}/exp_v26_micro.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v26_micro.json")

if __name__ == "__main__":
    main()
