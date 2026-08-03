# -*- coding: utf-8 -*-
"""底仓比例网格搜索 (v17 基础上, 仅覆盖 floor_pct)
网格: 国内底仓 cn x 海外底仓 us (各 7.5%~20%)
评分纪律: 全历史 Calmar 优先, 约束真实窗口 CAGR>=20% 且 MDD>=-10%
输出: 排名表 + out/exp_base_search.json
"""
import sys, os, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(f"{SKILL_REF}/final_cfg_v17.json"))
REPO = 0.022
MIN_DELTA = 0.02
TW = CFG.get("tranche_weights")

def run_one(cn, us, R, start, end, tag):
    cfg = dict(CFG)
    cfg["floor_pct"] = {"cn": cn, "us": us}
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=end, name=tag, min_delta=MIN_DELTA, repo=REPO,
                       tranche_weights=TW)
    e = evaluate(res)
    return e

def main():
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    grid = [0.0, 2.5, 5.0, 7.5, 10.0]
    rows = []
    for cn, us in itertools.product(grid, grid):
        e = run_one(cn, us, R, "2014-06-23", None, "px")
        er = run_one(cn, us, Rr, "2025-04-23", None, "rl")
        rows.append({
            "cn_floor": cn, "us_floor": us, "floor_total": cn + us,
            "proxy_cagr": e["cagr"], "proxy_mdd": e["max_dd"], "proxy_sharpe": e["sharpe"],
            "proxy_calmar": e["calmar"], "proxy_turnover": e["turnover"],
            "real_cagr": er["cagr"], "real_mdd": er["max_dd"], "real_sharpe": er["sharpe"],
            "real_calmar": er["calmar"], "real_turnover": er["turnover"],
            "pass": (er["cagr"] >= 0.20 and er["max_dd"] >= -0.10),
        })
        print(f"cn={cn:5.1f} us={us:5.1f} | proxy {e['cagr']*100:6.2f}%/{e['max_dd']*100:6.2f}% Calmar {e['calmar']:.2f} | "
              f"real {er['cagr']*100:6.2f}%/{er['max_dd']*100:6.2f}% Calmar {er['calmar']:.2f} "
              f"{'PASS' if rows[-1]['pass'] else ''}", flush=True)
    # 排序: 先按约束过滤, 再全历史 Calmar 降序
    ok = [r for r in rows if r["pass"]]
    pool = sorted(ok if ok else rows, key=lambda r: (-r["proxy_calmar"], r["proxy_mdd"]))
    print("\n===== 排名(前12) =====")
    for i, r in enumerate(pool[:12]):
        print(f"{i+1:2d}. cn={r['cn_floor']:5.1f} us={r['us_floor']:5.1f} (总{r['floor_total']:4.1f}) | "
              f"proxy {r['proxy_cagr']*100:6.2f}%/{r['proxy_mdd']*100:6.2f}% Calmar {r['proxy_calmar']:.2f} | "
              f"real {r['real_cagr']*100:6.2f}%/{r['real_mdd']*100:6.2f}% Calmar {r['real_calmar']:.2f}")
    json.dump({"pool": rows, "ranked": pool}, open(f"{OUT}/exp_base_search.json", "w"),
              ensure_ascii=False, indent=2, default=str)
    print(f"\n[ok] {OUT}/exp_base_search.json  ({len(rows)}组, {len(ok)}组达标)")

if __name__ == "__main__":
    main()
