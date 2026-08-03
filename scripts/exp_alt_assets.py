# -*- coding: utf-8 -*-
"""候选资产探索: 单资产统计 + 相同逻辑(v18)替换回测
候选: 510300沪深300/510500中证500/510050上证50/512100中证1000/512480半导体/588000科创50/
      513180恒生科技/518880黄金/511010国债/510880上证红利/515080中证红利/511380可转债/
      指数: 000852中证1000/000832中证转债/000688科创50/000922中证红利
替换方式: swap到对应槽位(CN成长=159952 / CN防御=515100 / CN价值=159232), 其余不变
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
# 候选: (key, 名称, 数据文件, 槽位, 起点说明)
CANDS = [
    ("510300", "沪深300ETF", "510300_nav.csv", "159952", "2012-05"),
    ("510500", "中证500ETF", "510500_nav.csv", "159952", "2013-03"),
    ("510050", "上证50ETF", "510050_nav.csv", "159952", "2005-02"),
    ("000852", "中证1000(指数)", "index_000852.csv", "159952", "2004-01"),
    ("512480", "半导体ETF", "512480_nav.csv", "159952", "2019-06"),
    ("588000", "科创50ETF", "588000_nav.csv", "159952", "2020-11"),
    ("000688", "科创50(指数)", "index_000688.csv", "159952", "2019-12"),
    ("513180", "恒生科技ETF", "513180_nav.csv", "159952", "2021-05"),
    ("510880", "上证红利ETF", "510880_nav.csv", "515100", "2007-01"),
    ("000922", "中证红利(指数)", "index_000922.csv", "515100", "2004-01"),
    ("515080", "中证红利ETF", "515080_nav.csv", "515100", "2019-12"),
    ("518880", "黄金ETF", "518880_nav.csv", "515100", "2013-07"),
    ("511010", "国债ETF", "511010_nav.csv", "515100", "2013-03"),
    ("511380", "可转债ETF", "511380_nav.csv", "515100", "2020-04"),
    ("000832", "中证转债(指数)", "index_000832.csv", "515100", "2004-01"),
]
CANDS2 = [  # 额外替换 159232 价值槽
    ("510880", "上证红利ETF", "510880_nav.csv", "159232", "2007-01"),
    ("000922", "中证红利(指数)", "index_000922.csv", "159232", "2004-01"),
    ("511010", "国债ETF", "511010_nav.csv", "159232", "2013-03"),
]
def load_rets(fn):
    if fn.startswith("index_"):
        df = read_table(fn)
        return rets_from(df, "close")
    df = read_table(fn)
    return rets_from(df, "cum_nav")

def swap_series(R, slot, new_rets):
    R2 = R.copy()
    R2[slot] = new_rets.reindex(R2.index).ffill()
    return R2

def run(R, start, tag, am=None):
    ds = DynamicStrategy(R, cfg=CFG, a_mkt_override=am)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name=tag, min_delta=0.02, repo=REPO,
                       tranche_weights=CFG.get("tranche_weights"))
    return evaluate(res), res

def main():
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    results = {"stats": {}, "swaps": {}}
    # 1) 单资产统计: 2014-06-23~2026-07-31 共同窗口
    W0 = "2014-06-23"
    pool = {"159232": load_rets("index_932365.csv"), "515100": load_rets("index_930955.csv"),
            "159941": load_rets("270042_nav.csv"), "513500": load_rets("513500_nav.csv"),
            "159952": load_rets("index_sz399006.csv")}
    for k, name, fn, slot, start in CANDS + [("512890", "红利低波ETF", "512890_nav.csv", "515100", "2019-01")]:
        pool[k] = load_rets(fn)
    print("===== 单资产统计 (2014-06-23 ~ 2026-07-31 共同窗口, 部分标的自起始) =====")
    print(f"{'资产':<14}{'CAGR':>8}{'MDD':>9}{'Vol':>7}{'Sharpe':>8}{'与DYN相关':>10}")
    dync, res_dyn = run(R, W0, "DYN")
    dynr = res_dyn["rets"]
    for k, (name, s) in {**{k2: (n, load_rets(f)) for k2, n, f, _, _ in CANDS},
                         "159232": ("自由现金流", load_rets("index_932365.csv")),
                         "515100": ("红利低波100", load_rets("index_930955.csv")),
                         "159941": ("纳指100", load_rets("270042_nav.csv")),
                         "513500": ("标普500", load_rets("513500_nav.csv")),
                         "159952": ("创业板指", load_rets("index_sz399006.csv"))}.items():
        rs = s[s.index >= W0].dropna()
        if len(rs) < 100: continue
        w = (1 + rs).cumprod()
        mdd = (w / w.cummax() - 1).min()
        cagr = (1 + rs).prod() ** (252 / len(rs)) - 1
        vol = rs.std() * np.sqrt(252)
        sharpe = (cagr - REPO) / vol if vol > 0 else np.nan
        corr = rs.reindex(dynr.index).dropna().corr(dynr.reindex(rs.index).dropna())
        results["stats"][k] = {"name": name, "cagr": cagr, "mdd": mdd, "vol": vol, "sharpe": sharpe, "corr_dyn": corr}
        print(f"{name:<14}{cagr*100:7.2f}%{mdd*100:8.2f}%{vol*100:6.2f}%{sharpe:8.2f}{corr:10.2f}")
    # 2) 替换回测
    print("\n===== 替换回测 A组: proxy 全历史 2014-06-23 起 =====")
    base, _ = run(R, W0, "base")
    print(f"基准(v18原组合): {base['cagr']*100:6.2f}%/{base['max_dd']*100:6.2f}%/Calmar {base['calmar']:.2f}")
    for k, name, fn, slot, start in CANDS + CANDS2:
        if k == "000852" and slot == "159952" or k in ("512890",):  # 512890已测; 000852重复项
            continue
        rs = load_rets(fn)
        if rs.index.min() > pd.Timestamp(W0):
            continue
        Rp = swap_series(R, slot, rs)
        e, _ = run(Rp, W0, f"{k}->{slot}")
        results["swaps"][f"{k}->{slot}"] = {"name": name, "cagr": e["cagr"], "mdd": e["max_dd"], "calmar": e["calmar"],
                                            "sharpe": e["sharpe"], "turnover": e["turnover"]}
        print(f"{name}({k})->{slot}: {e['cagr']*100:6.2f}%/{e['max_dd']*100:6.2f}%/Calmar {e['calmar']:.2f}")
    # 3) B组: 各自起点窗口 (与同期基准对比)
    print("\n===== 替换回测 B组: 各自起点窗口 (同期对比) =====")
    for k, name, fn, slot, start in CANDS:
        if k == "512890": continue
        rs = load_rets(fn)
        st = rs.index.min() + pd.Timedelta(days=5)
        eb, _ = run(R, st, "base_same")
        Rp = swap_series(R, slot, rs)
        ea, _ = run(Rp, st, f"{k}")
        print(f"{name}({k})->{slot} [{st.date()}]: 替换 {ea['cagr']*100:6.2f}%/{ea['max_dd']*100:6.2f}%/C{ea['calmar']:.2f} | 同期基准 {eb['cagr']*100:6.2f}%/{eb['max_dd']*100:6.2f}%/C{eb['calmar']:.2f}")
        results["swaps"].setdefault(f"{k}->{slot}_win", {})["same_win"] = {"start": str(st.date()), "alt_cagr": ea["cagr"], "alt_mdd": ea["max_dd"], "alt_calmar": ea["calmar"], "base_cagr": eb["cagr"], "base_mdd": eb["max_dd"], "base_calmar": eb["calmar"]}
    # 4) real 窗口全部候选
    print("\n===== 替换回测 real 2025-04-23 起 (真实净值) =====")
    base_r, _ = run(Rr, "2025-04-23", "base")
    print(f"基准: {base_r['cagr']*100:6.2f}%/{base_r['max_dd']*100:6.2f}%/Calmar {base_r['calmar']:.2f}")
    for k, name, fn, slot, start in CANDS:
        rs = load_rets(fn)
        Rrp = swap_series(Rr, slot, rs)
        e, _ = run(Rrp, "2025-04-23", f"{k}")
        results["swaps"].setdefault(f"{k}->{slot}_real", {})["real"] = {"cagr": e["cagr"], "mdd": e["max_dd"], "calmar": e["calmar"]}
        print(f"{name}({k})->{slot}: {e['cagr']*100:6.2f}%/{e['max_dd']*100:6.2f}%/Calmar {e['calmar']:.2f}")
    json.dump(results, open(f"{OUT}/exp_alt_assets.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[ok] {OUT}/exp_alt_assets.json")

if __name__ == "__main__":
    main()
