# -*- coding: utf-8 -*-
"""v24 候选: cash_bond_pct 0.5->0.75/1.0 验证 (WFO + 压力 + 扰动)
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v23.json")))

def variant(cbp):
    c = copy.deepcopy(BASE); c["cash_bond_pct"] = cbp; c["version"] = f"v24_cbp{cbp}"; return c

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

def synth_res(Rr):
    from data_prep import DATA_DIR
    ps, pe = "2024-09-02", "2026-07-31"
    Rs = Rr.loc[ps:pe].copy()
    us = (Rs["159941"] + Rs["513500"]) / 2; cn = (Rs["159232"] + Rs["515100"] + Rs["159952"]) / 3
    def sk(rs, t):
        lo, hi = -2.0, 2.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (1 + mid * rs.fillna(0.0)).prod() - 1.0 > t: hi = mid
            else: lo = mid
        return (lo + hi) / 2
    k_us, k_cn = sk(us, -0.30), sk(cn, -0.20)
    idx = pd.read_csv(os.path.join(DATA_DIR, "index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
    iw = idx.loc[ps:pe].pct_change().dropna().reindex(Rs.index).ffill().fillna(0.0)
    k_i = sk(iw, -0.25)
    s = Rs.copy()
    for x in ["159941", "513500"]: s[x] = Rs[x] * k_us
    for x in ["159232", "515100", "159952"]: s[x] = Rs[x] * k_cn
    lvl0 = float(idx.loc[ps:ps].iloc[0])
    return s, (1 + k_i * iw).cumprod() * lvl0

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    synth, s_idx = synth_res(Rr)

    VAR = {"v23_cbp0.5": BASE, "v24_cbp0.75": variant(0.75), "v24_cbp1.0": variant(1.0)}
    print("===== 全历史 + real + WFO =====")
    print(f"{'配置':<14} | {'proxy':>22} | {'real':>22} | {'WFO训练':>22} | {'WFO_OOS':>22}")
    rows = {}
    for tag, c in VAR.items():
        p = run(c, R, "2014-06-23", None, bond)
        r = run(c, Rr, "2025-04-23", None, bond)
        tr = run(c, R, "2014-06-23", "2021-12-31", bond)
        te = run(c, R, "2022-01-01", None, bond)
        rows[tag] = {"proxy": p, "real": r, "train": tr, "oos": te}
        print(f"{tag:<14} | {p['cagr']:6.2f}%/{p['mdd']:6.2f}%/Cal{p['calmar']:.2f} | "
              f"{r['cagr']:6.2f}%/{r['mdd']:6.2f}%/Cal{r['calmar']:.2f} | "
              f"{tr['cagr']:6.2f}%/{tr['mdd']:6.2f}%/Cal{tr['calmar']:.2f} | "
              f"{te['cagr']:6.2f}%/{te['mdd']:6.2f}%/Cal{te['calmar']:.2f}")
    print("\n===== 压力情景 =====")
    scen = [("2015股灾", R, "2015-06-15", "2016-02-29", None), ("2021-22熊", R, "2021-02-19", "2022-10-31", None),
            ("共振熊", synth, "2024-09-02", "2026-07-31", s_idx), ("19-21牛", R, "2019-01-04", "2021-02-18", None)]
    for tag, c in VAR.items():
        line = f"{tag:<14}"
        for nm, Rs, ps, pe, am in scen:
            r = run(c, Rs, ps, pe, bond, am)
            line += f" | {nm}: {r['mdd']:6.2f}%"
        print(line)
    json.dump(rows, open(f"{OUT}/exp_v24_cash.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v24_cash.json")

if __name__ == "__main__":
    main()
