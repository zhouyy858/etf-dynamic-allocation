# -*- coding: utf-8 -*-
"""第四轮搜索: 新执行规则(每周三1/3、三周调整完)下的参数再优化
评分: 全历史Calmar优先, 约束真实窗口CAGR>=20% 且 MDD>=-10%"""
import sys, os, json, random
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest
from strategy import DynamicStrategy
from metrics import annualized_ret, max_drawdown, sharpe

R_PROXY, W_PROXY = build_panel("proxy")
R_REAL, W_REAL = build_panel("real")
START = "2014-06-23"
REAL_START = "2025-04-23"

def rand_cfg(rng):
    g9 = rng.choice([70, 74, 78, 82, 86])
    state_map = {9: (g9, rng.choice([3, 4, 5]))}
    ratios = rng.choice([[0.92, 0.90, 0.88, 0.86, 0.80, 0.78, 0.75, 0.70, 0.62],
                         [0.90, 0.88, 0.85, 0.80, 0.78, 0.72, 0.68, 0.65, 0.60],
                         [0.95, 0.93, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75, 0.68]])
    g = g9
    for sc in range(8, -1, -1):
        g = max(round(g * ratios[8 - sc]), 4)
        d = 30 - round((30 - 4) * (sc / 9.0))
        state_map[sc] = (g, d)
    return {
        "state_map": state_map,
        "market_dd": {
            "CN": (rng.choice([0.07, 0.08, 0.09]), rng.choice([0.08, 0.10, 0.12]),
                   rng.choice([0.20, 0.22, 0.24]), rng.choice([0.10, 0.12])),
            "US": (rng.choice([0.09, 0.10, 0.12]), rng.choice([0.12, 0.15, 0.18]),
                   rng.choice([0.24, 0.26, 0.28]), rng.choice([0.12, 0.15])),
        },
        "dd_eq_cap": rng.choice([[[-0.12, 78], [-0.18, 62], [-0.25, 45]],
                                 [[-0.10, 75], [-0.15, 60], [-0.20, 45]],
                                 [[-0.12, 80], [-0.18, 65], [-0.25, 50]],
                                 [[-0.10, 80], [-0.16, 65], [-0.22, 50]]]),
        "hyst_up": round(rng.uniform(0.45, 0.75), 2),
        "hyst_down": round(rng.uniform(0.12, 0.22), 2),
        "min_cash": rng.choice([0.04, 0.05, 0.06, 0.08]),
        "max_eq": rng.choice([0.93, 0.95, 0.97, 0.99]),
        "vol_target": rng.choice([0.15, 0.16, 0.18, 0.20]),
        "gate_win": rng.choice([120, 120]),
        "growth_split_bull": {"159952": rng.choice([0.32, 0.34, 0.36]),
                              "159941": rng.choice([0.44, 0.46, 0.48]),
                              "513500": 0.20},
    }

def run_one(R, cfg, start):
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(), start=start)
    r, w = res["rets"], res["wealth"]
    mdd = max_drawdown(w)[0]
    return {"cagr": annualized_ret(r), "mdd": mdd, "sharpe": sharpe(r),
            "calmar": (annualized_ret(r) / abs(mdd) if mdd < 0 else np.nan),
            "cash": float(res["weights"]["cash"].mean()), "turnover": res["turnover"]}

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 23
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        cfg = rand_cfg(rng)
        try:
            m = run_one(R_PROXY, cfg, START)
            m2 = run_one(R_REAL, cfg, REAL_START)
        except Exception as e:
            print(f"[err] {k}: {str(e)[:80]}"); continue
        m["real_cagr"] = m2["cagr"]; m["real_mdd"] = m2["mdd"]; m["real_sharpe"] = m2["sharpe"]
        m["real_calmar"] = m2["calmar"]
        m["cfg"] = cfg
        rows.append(m)
    df = pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "cfg"} for r in rows])
    df.to_csv("out/search4_results.csv", index=False)
    with open("out/search4_configs.json", "w") as f:
        json.dump([r["cfg"] for r in rows], f, ensure_ascii=False)
    print(f"=== n={len(rows)} 达标(real CAGR>=20% 且 MDD>=-10%) Calmar top 12 ===")
    ok = df[(df["real_cagr"] >= 0.20) & (df["real_mdd"] >= -0.10)]
    top = ok.sort_values("calmar", ascending=False).head(12)
    print(top[["cagr", "mdd", "sharpe", "calmar", "real_cagr", "real_mdd", "real_sharpe", "real_calmar", "turnover"]].round(4).to_string())
    print("\n=== 全部 Calmar top 10 ===")
    print(df.sort_values("calmar", ascending=False).head(10)[["cagr", "mdd", "sharpe", "calmar", "real_cagr", "real_mdd", "real_calmar"]].round(4).to_string())

if __name__ == "__main__":
    main()
