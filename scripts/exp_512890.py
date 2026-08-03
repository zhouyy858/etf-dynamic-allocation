# -*- coding: utf-8 -*-
"""对比 515100(红利低波100/930955) vs 512890(红利低波/H30269) 在相同策略逻辑(v18: floor5+5, 周三1笔)下的表现
替换方式: 仅替换红利低波列的收益序列, 其余4只与策略参数完全不变
- proxy层: 515100用index_930955, 512890用index_h30269(中证官网, 2004起)
- real层: 515100用515100_nav, 512890用512890_nav(腾讯qfq, 2019-01-18起)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data_prep import build_panel, read_table, rets_from, DATA_DIR
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(f"{HERE}/../references/final_cfg_v18.json"))
REPO = 0.022
PERIODS = {
    "2015股灾": ("2015-06-15", "2016-02-29"),
    "2018熊市": ("2018-01-02", "2019-01-03"),
    "2019-2021牛市": ("2019-01-04", "2021-02-18"),
    "2021-2022熊市": ("2021-02-19", "2022-10-31"),
    "2023-2024震荡": ("2023-01-03", "2024-08-30"),
    "2024Q4-2026双牛": ("2024-09-02", "2026-07-31"),
}

def swap_series(R, slot, new_rets):
    R2 = R.copy()
    r2 = new_rets.reindex(R2.index).ffill()
    R2[slot] = r2
    return R2

def h30269_rets():
    df = read_table("index_h30269.csv")
    return rets_from(df, "close")

def nav512890_rets():
    df = read_table("512890_nav.csv")
    return rets_from(df, "cum_nav")

def run(R, start, tag, am=None):
    ds = DynamicStrategy(R, cfg=CFG, a_mkt_override=am)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name=tag, min_delta=0.02, repo=REPO,
                       tranche_weights=CFG.get("tranche_weights"))
    return evaluate(res, periods=PERIODS)

def fmt(e):
    return f"CAGR {e['cagr']*100:6.2f}%  MDD {e['max_dd']*100:6.2f}%  Sharpe {e['sharpe']:.2f}  Calmar {e['calmar']:5.2f}  TO {e['turnover']:6.1f}"

def main():
    R, _ = build_panel("proxy")          # 515100=930955
    Rr, _ = build_panel("real")          # 515100=515100_nav
    # 替换面板
    Rp = swap_series(R, "515100", h30269_rets())          # 512890=H30269
    Rr2 = swap_series(Rr, "515100", nav512890_rets())     # 512890=512890_nav(2019起)
    results = {}
    print("===== 全历史 proxy 2014-06-23 起 (515100=930955 vs 512890=H30269) =====")
    for tag, RR in [("515100", R), ("512890", Rp)]:
        e = run(RR, "2014-06-23", tag)
        results.setdefault(tag, {})["proxy_full"] = e
        print(f"  {tag}: {fmt(e)}")
    print("\n===== 分阶段 (proxy, 515100 vs 512890) =====")
    print(f"{'阶段':<14}{'515100 CAGR/MDD':>24}{'512890 CAGR/MDD':>24}")
    e1 = results["515100"]["proxy_full"]; e2 = results["512890"]["proxy_full"]
    for pn in PERIODS:
        a, b = e1["periods"][pn], e2["periods"][pn]
        print(f"{pn:<14}{a['cagr']*100:8.2f}%/{a['max_dd']*100:6.2f}%{b['cagr']*100:14.2f}%/{b['max_dd']*100:6.2f}%")
    print("\n===== proxy 2019-01-18 起 (512890上市后, 全代理) =====")
    for tag, RR in [("515100", R), ("512890", Rp)]:
        e = run(RR, "2019-01-18", tag)
        results[tag]["proxy_2019"] = e
        print(f"  {tag}: {fmt(e)}")
    print("\n===== real 2025-04-23 起 (真实净值) =====")
    for tag, RR in [("515100", Rr), ("512890", Rr2)]:
        e = run(RR, "2025-04-23", tag)
        results[tag]["real"] = e
        print(f"  {tag}: {fmt(e)}")
    print("\n===== 压力测试 (proxy代理) =====")
    from stress_test import synthetic_resonance
    synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
    synth2, s_idx2 = synthetic_resonance(Rr2, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)  # 512890面板重新合成(公平)
    scen = [("2015股灾", R, "2015-06-15", "2016-02-29", None),
            ("2018熊市", R, "2018-01-02", "2019-01-03", None),
            ("2021-22熊", R, "2021-02-19", "2022-10-31", None),
            ("2019-21牛", R, "2019-01-04", "2021-02-18", None),
            ("共振熊US-30/CN-20", None, "2024-09-02", "2026-07-31", None)]
    print(f"{'情景':<18}{'515100 CAGR/MDD/Cal':>30}{'512890 CAGR/MDD/Cal':>30}")
    for name, Rs, ps, pe, am in scen:
        if name == "共振熊US-30/CN-20":
            ea, eb = run(synth, ps, tag="a", am=s_idx), run(synth2, ps, tag="b", am=s_idx2)
        else:
            Rps = swap_series(Rs, "515100", h30269_rets())
            ea, eb = run(Rs, ps, tag="a", am=am), run(Rps, ps, tag="b", am=am)
        fa = f"{ea['cagr']*100:7.2f}%/{ea['max_dd']*100:6.2f}%/{ea['calmar']:4.2f}"
        fb = f"{eb['cagr']*100:7.2f}%/{eb['max_dd']*100:6.2f}%/{eb['calmar']:4.2f}"
        print(f"{name:<18}{fa:>30}{fb:>30}")
        results.setdefault("stress", {})[name] = {"515100": fa, "512890": fb}
    simple = {}
    for tag, d in results.items():
        simple[tag] = {kk: ({x: vv[x] for x in ["cagr","max_dd","sharpe","calmar","turnover"]} if isinstance(vv, dict) and "cagr" in vv else vv) for kk, vv in d.items()}
    json.dump(simple, open(f"{OUT}/exp_512890.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[ok] {OUT}/exp_512890.json")

if __name__ == "__main__":
    main()
