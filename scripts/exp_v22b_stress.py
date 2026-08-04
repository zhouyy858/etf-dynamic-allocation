# -*- coding: utf-8 -*-
"""v22b 压力测试: v22 vs C3 全情景对比 (严格口径)
情景: 牛市19-21 / 真实牛25-26 / 震荡23-24 / 2015股灾 / 2018熊 / 2021-22熊 / 合成共振熊
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")

def load(name):
    return json.load(open(f"{SKILL_REF}/{name}"))

CFG22 = load("final_cfg_v22.json")
CFG22B = copy.deepcopy(CFG22)
CFG22B.update(growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10},
              growth_split_bear={"159952": 0.50, "159941": 0.30, "513500": 0.20},
              min_delta=0.035, version="v22b")
REPO = 0.022

def scale_k(rs, target):
    lo, hi = -2.0, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        v = (1 + mid * rs.fillna(0.0)).prod() - 1.0
        if v > target: hi = mid
        else: lo = mid
    return (lo + hi) / 2

def synthetic_resonance(R, window, us_target, cn_target, idx_target):
    from data_prep import DATA_DIR
    ps, pe = window
    Rs = R.loc[ps:pe].copy()
    us_avg = (Rs["159941"] + Rs["513500"]) / 2
    cn_avg = (Rs["159232"] + Rs["515100"] + Rs["159952"]) / 3
    k_us = scale_k(us_avg, us_target); k_cn = scale_k(cn_avg, cn_target)
    idx = pd.read_csv(os.path.join(DATA_DIR, "index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
    idx_w = idx.loc[ps:pe].pct_change().dropna().reindex(Rs.index).ffill().fillna(0.0)
    k_idx = scale_k(idx_w, idx_target)
    synth = Rs.copy()
    for s in ["159941", "513500"]: synth[s] = Rs[s] * k_us
    for s in ["159232", "515100", "159952"]: synth[s] = Rs[s] * k_cn
    lvl0 = float(idx.loc[ps:ps].iloc[0]) if (idx.loc[ps:ps]).shape[0] else 1000.0
    s_idx = (1 + k_idx * idx_w).cumprod() * lvl0
    print(f"  [合成] US累计{(1+k_us*us_avg).prod()-1:+.1%} CN累计{(1+k_cn*cn_avg).prod()-1:+.1%} HS300累计{(1+k_idx*idx_w).prod()-1:+.1%}")
    return synth, s_idx

def run(cfg, ps, pe, Rs, a_mkt):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    from data_prep import read_table, rets_from
    ds = DynamicStrategy(Rs, cfg=cfg, a_mkt_override=a_mkt)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav") if cfg.get("cash_bond_pct") else None
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name="DYN", min_delta=cfg.get("min_delta", 0.02), repo=REPO,
                       tranche_weights=cfg.get("tranche_weights"), cash_bond_rets=bond,
                       cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], total=e["total_ret"]*100)

def main():
    from data_prep import build_panel
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
    SCEN = [
        ("牛市_2019-2021", R, None, "2019-01-04", "2021-02-18"),
        ("牛市_真实25-26", Rr, None, "2025-04-23", "2026-07-31"),
        ("震荡_2023-2024", R, None, "2023-01-03", "2024-08-30"),
        ("熊_2015股灾", R, None, "2015-06-15", "2016-02-29"),
        ("熊_2018", R, None, "2018-01-02", "2019-01-03"),
        ("熊_2021-2022", R, None, "2021-02-19", "2022-10-31"),
        ("共振熊_合成", synth, s_idx, "2024-09-02", "2026-07-31"),
    ]
    results = {}
    print(f"{'情景':<18} | {'v22 CAGR/MDD/Calmar':32s} | {'v22b(C3) CAGR/MDD/Calmar':32s}")
    for name, Rs, am, ps, pe in SCEN:
        r22 = run(CFG22, ps, pe, Rs, am)
        rb = run(CFG22B, ps, pe, Rs, am)
        results[name] = {"v22": r22, "v22b_C3": rb}
        print(f"{name:<18} | {r22['cagr']:6.2f}%/{r22['mdd']:6.2f}%/{r22['calmar']:5.2f}       | "
              f"{rb['cagr']:6.2f}%/{rb['mdd']:6.2f}%/{rb['calmar']:5.2f}")
    json.dump(results, open(f"{OUT}/exp_v22b_stress.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v22b_stress.json")

if __name__ == "__main__":
    main()
