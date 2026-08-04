# -*- coding: utf-8 -*-
"""v24b 候选: QDII相对溢价倾斜 + 债券层品种对比
premium_tilt: 美股桶内 纳指/标普 按相对溢价向低溢价者倾斜(折溢价风险对冲)
bond: 511010 国债ETF vs 511380 国开债ETF 作为现金债券层
全部严格无未来函数口径(溢价面板已shift=2)
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BASE = json.load(open(os.path.join(HERE, "..", "references", "final_cfg_v24.json")))

def variant(**kw):
    c = copy.deepcopy(BASE); c.update(kw)
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
    bond511 = rets_from(read_table("511010_nav.csv"), "cum_nav")
    bond380 = rets_from(read_table("511380_nav.csv"), "cum_nav")
    synth, s_idx = synth_res(Rr)

    VAR = {
        "v24标准": BASE,
        "v24+tilt(2/5/50%)": variant(premium_tilt=True),
        "v24+tilt强(1.5/4/70%)": variant(premium_tilt=True, premium_tilt_thr=0.015, premium_tilt_cap=0.04, premium_tilt_max=0.7),
        "v24+511380": variant(cash_bond_code="511380"),
        "v24+tilt+511380": variant(premium_tilt=True, cash_bond_code="511380"),
    }
    print("===== proxy / real / WFO =====")
    print(f"{'配置':<24} | {'proxy':>22} | {'real':>22} | {'WFO训练':>22} | {'WFO_OOS':>22}")
    rows = {}
    for tag, c in VAR.items():
        bnd = bond380 if c.get("cash_bond_code") == "511380" else bond511
        p = run(c, R, "2014-06-23", None, bnd)
        r = run(c, Rr, "2025-04-23", None, bnd)
        tr = run(c, R, "2014-06-23", "2021-12-31", bnd)
        te = run(c, R, "2022-01-01", None, bnd)
        rows[tag] = {"proxy": p, "real": r, "train": tr, "oos": te}
        print(f"{tag:<24} | {p['cagr']:6.2f}%/{p['mdd']:6.2f}%/Cal{p['calmar']:.2f} | "
              f"{r['cagr']:6.2f}%/{r['mdd']:6.2f}%/Cal{r['calmar']:.2f} | "
              f"{tr['cagr']:6.2f}%/{tr['mdd']:6.2f}%/Cal{tr['calmar']:.2f} | "
              f"{te['cagr']:6.2f}%/{te['mdd']:6.2f}%/Cal{te['calmar']:.2f}")
    print("\n===== 压力情景 =====")
    scen = [("2015股灾", R, "2015-06-15", "2016-02-29", None), ("2021-22熊", R, "2021-02-19", "2022-10-31", None),
            ("共振熊", synth, "2024-09-02", "2026-07-31", s_idx), ("19-21牛", R, "2019-01-04", "2021-02-18", None)]
    for tag, c in VAR.items():
        bnd = bond380 if c.get("cash_bond_code") == "511380" else bond511
        line = f"{tag:<24}"
        for nm, Rs, ps, pe, am in scen:
            r = run(c, Rs, ps, pe, bnd, am)
            line += f" | {nm}: {r['mdd']:6.2f}%"
        print(line)
    json.dump(rows, open(f"{OUT}/exp_v24b_premium_tilt.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v24b_premium_tilt.json")

if __name__ == "__main__":
    main()
