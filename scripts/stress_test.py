# -*- coding: utf-8 -*-
"""三情景压力测试 + 跨市场相关性 + 极端行情检验 (v10最终配置, 每周三1/3三周完成)
情景: 牛市(2019-2021 / 真实窗口2025-2026) / 震荡(2023-2024) / 熊市(2015股灾 / 2018熊市)
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest, evaluate, fmt_eval, SLOTS
from strategy import DynamicStrategy
from metrics import max_drawdown, annualized_ret

OUT = "out"
FIG = "plots"
CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "final_cfg_v10.json")
R, W = build_panel("proxy")
Rr, Wr = build_panel("real")
CFG = json.load(open(CFG_FILE)) if os.path.exists(CFG_FILE) else {}

BENCHMARKS = {
    "B1等权20": {s: 20 for s in SLOTS},
    "B2保守防御": {"159232": 25, "515100": 25, "159941": 20, "513500": 20, "159952": 10},
    "B3均衡": {"159232": 15, "515100": 15, "159941": 25, "513500": 20, "159952": 25},
    "B4成长进攻": {"159232": 10, "515100": 10, "159941": 30, "513500": 20, "159952": 30},
    "B5价值60/成长40": {"159232": 30, "515100": 30, "159941": 15, "513500": 15, "159952": 10},
}

SCENARIOS = {
    "牛市_2019-2021": ("2019-01-04", "2021-02-18", R),
    "牛市_真实窗口2025-2026": ("2025-04-23", "2026-07-31", Rr),
    "震荡市_2023-2024": ("2023-01-03", "2024-08-30", R),
    "熊市_2015股灾": ("2015-06-15", "2016-02-29", R),
    "熊市_2018": ("2018-01-02", "2019-01-03", R),
    "熊市_2021-2022": ("2021-02-19", "2022-10-31", R),
}

def run_scenario(name, ps, pe, R):
    ds = DynamicStrategy(R, cfg=CFG)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(), start=ps, end=pe, name="DYN v10")
    rows = {"DYN v10": evaluate(res)}
    for bname, bw in BENCHMARKS.items():
        rb = run_backtest(R, fixed_weights=bw, start=ps, end=pe, name=bname, min_delta=0.0002)
        rows[bname] = evaluate(rb)
    return rows

def main():
    results = {}
    for name, (ps, pe, R) in SCENARIOS.items():
        results[name] = run_scenario(name, ps, pe, R)
        print(f"\n===== {name} ({ps} ~ {pe}) =====")
        for k, e in results[name].items():
            print(f"  {k:<16} CAGR={e['cagr']*100:7.2f}%  MDD={e['max_dd']*100:6.2f}%  Sharpe={e['sharpe']:.2f}  Calmar={e['calmar']:.2f}  total={e['total_ret']*100:8.2f}%")
    with open(f"{OUT}/stress_test.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("\n[ok] out/stress_test.json")

if __name__ == "__main__":
    main()
