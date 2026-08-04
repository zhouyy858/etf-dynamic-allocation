# -*- coding: utf-8 -*-
"""v21 严格口径底仓网格搜索 (未来函数修正后重校准)
口径: signal_lag=1(前日信号) + premium_shift=2(溢价T-2) + 每周三3周三笔 + 现金50%国债ETF
评分: 全历史 Calmar 优先, 约束真实窗口 CAGR>=20% 且 MDD>=-10%
输出: 终端排名表 + out/exp_v21_search.json
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(f"{SKILL_REF}/final_cfg_v20.json"))
bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
REPO = 0.022
TW = [1.0 / 3.0] * 3

def run_one(cn, us, R, start, tag):
    cfg = dict(CFG)
    cfg["floor_pct"] = {"cn": cn, "us": us}
    cfg["signal_lag"] = 1
    cfg["premium_shift"] = 2
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name=tag, min_delta=0.02, repo=REPO,
                       tranche_weights=TW, cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0))
    return evaluate(res)

def main():
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    grid = [0.0, 2.5, 5.0, 7.5, 10.0]
    rows = []
    for cn, us in itertools.product(grid, grid):
        e = run_one(cn, us, R, "2014-06-23", "px")
        er = run_one(cn, us, Rr, "2025-04-23", "rl")
        rows.append({"cn_floor": cn, "us_floor": us, "floor_total": cn + us,
                     "proxy_cagr": e["cagr"], "proxy_mdd": e["max_dd"], "proxy_sharpe": e["sharpe"],
                     "proxy_calmar": e["calmar"], "proxy_turnover": e["turnover"],
                     "real_cagr": er["cagr"], "real_mdd": er["max_dd"], "real_sharpe": er["sharpe"],
                     "real_calmar": er["calmar"], "real_turnover": er["turnover"],
                     "pass": (er["cagr"] >= 0.20 and er["max_dd"] >= -0.10)})
        print(f"cn={cn:5.1f} us={us:5.1f} | proxy {e['cagr']*100:6.2f}%/{e['max_dd']*100:6.2f}% Calmar {e['calmar']:.2f} | "
              f"real {er['cagr']*100:6.2f}%/{er['max_dd']*100:6.2f}% Calmar {er['calmar']:.2f} "
              f"{'PASS' if rows[-1]['pass'] else ''}", flush=True)
    ok = [r for r in rows if r["pass"]]
    pool = sorted(ok if ok else rows, key=lambda r: (-r["proxy_calmar"], r["proxy_mdd"]))
    print("\n===== 排名(前12) =====")
    for i, r in enumerate(pool[:12]):
        print(f"{i+1:2d}. cn={r['cn_floor']:5.1f} us={r['us_floor']:5.1f} (总{r['floor_total']:4.1f}) | "
              f"proxy {r['proxy_cagr']*100:6.2f}%/{r['proxy_mdd']*100:6.2f}% Calmar {r['proxy_calmar']:.2f} | "
              f"real {r['real_cagr']*100:6.2f}%/{r['real_mdd']*100:6.2f}% Calmar {r['real_calmar']:.2f}")
    json.dump({"pool": rows, "ranked": pool}, open(f"{OUT}/exp_v21_search.json", "w"),
              ensure_ascii=False, indent=2, default=str)
    print(f"\n[ok] {OUT}/exp_v21_search.json  ({len(rows)}组, {len(ok)}组达标)")

if __name__ == "__main__":
    main()
