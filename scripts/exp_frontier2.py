# -*- coding: utf-8 -*-
"""前沿补充: 激进版(f=1.1/1.2/1.3)压力情景 + cash_bond_pct 扫描
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v23.json")))

def scale_state(cfg, f):
    c = copy.deepcopy(cfg)
    c["state_map"] = {k: [max(4, min(98, round(g * f))), max(1, min(40, round(d * f)))]
                      for k, (g, d) in c["state_map"].items()}
    return c

def run(cfg, Rs, ps, pe, bond, a_mkt=None):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    ds = DynamicStrategy(Rs, cfg=cfg, a_mkt_override=a_mkt)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, calmar=e["calmar"])

def synthetic_resonance(R, window, us_target, cn_target, idx_target):
    from data_prep import DATA_DIR
    ps, pe = window
    Rs = R.loc[ps:pe].copy()
    us_avg = (Rs["159941"] + Rs["513500"]) / 2
    cn_avg = (Rs["159232"] + Rs["515100"] + Rs["159952"]) / 3
    def scale_k(rs, tgt):
        lo, hi = -2.0, 2.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (1 + mid * rs.fillna(0.0)).prod() - 1.0 > tgt: hi = mid
            else: lo = mid
        return (lo + hi) / 2
    k_us = scale_k(us_avg, us_target); k_cn = scale_k(cn_avg, cn_target)
    idx = pd.read_csv(os.path.join(DATA_DIR, "index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
    idx_w = idx.loc[ps:pe].pct_change().dropna().reindex(Rs.index).ffill().fillna(0.0)
    k_idx = scale_k(idx_w, idx_target)
    synth = Rs.copy()
    for s in ["159941", "513500"]: synth[s] = Rs[s] * k_us
    for s in ["159232", "515100", "159952"]: synth[s] = Rs[s] * k_cn
    lvl0 = float(idx.loc[ps:ps].iloc[0]) if (idx.loc[ps:ps]).shape[0] else 1000.0
    s_idx = (1 + k_idx * idx_w).cumprod() * lvl0
    return synth, s_idx

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)

    print("===== 激进版压力情景 (f=1.0/1.1/1.2/1.3) =====")
    scen = [("2015股灾", R, "2015-06-15", "2016-02-29", None), ("2021-22熊", R, "2021-02-19", "2022-10-31", None),
            ("共振熊", synth, "2024-09-02", "2026-07-31", s_idx), ("19-21牛", R, "2019-01-04", "2021-02-18", None)]
    results = {}
    for f in [1.0, 1.1, 1.2, 1.3]:
        c = scale_state(BASE, f)
        row = {}
        print(f"\n-- f={f} --")
        for nm, Rs, ps, pe, am in scen:
            r = run(c, Rs, ps, pe, bond, am)
            row[nm] = r
            print(f"  {nm:<10} CAGR={r['cagr']:7.2f}% MDD={r['mdd']:6.2f}% Cal={r['calmar']:5.2f}")
        results[f] = row

    print("\n===== cash_bond_pct 扫描 (v23基座) =====")
    for cbp in [0.0, 0.25, 0.5, 0.75, 1.0]:
        c = copy.deepcopy(BASE); c["cash_bond_pct"] = cbp
        p = run(c, R, "2014-06-23", None, bond)
        rr = run(c, Rr, "2025-04-23", None, bond)
        print(f"  cash_bond_pct={cbp:.2f} | proxy {p['cagr']:6.2f}%/{p['mdd']:6.2f}%/Cal{p['calmar']:.2f} | "
              f"real {rr['cagr']:6.2f}%/{rr['mdd']:6.2f}%/Cal{rr['calmar']:.2f}")
        results[f"cbp_{cbp}"] = {"proxy": p, "real": rr}
    json.dump(results, open(f"{OUT}/exp_frontier2.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[ok] {OUT}/exp_frontier2.json")

if __name__ == "__main__":
    main()
