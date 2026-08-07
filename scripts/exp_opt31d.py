# -*- coding: utf-8 -*-
"""v31d 候选预注册评估(2026-08-07, 数据至2026-08-06): 市场宽度 + vol_target复核
单机制×6, 严格口径(signal_lag=1/溢价T-2/先计提后成交, 无未来函数)
第一关: 双窗口Calmar同向+0.15 且 proxy MDD>-10% / real MDD>-4.1% -> 平台+OOS
候选来源: ①breadth_gate 市场宽度(7个A股指数站上SMA占比, 学术 Momentum+Breadth+Correlation);
②vol_target/vol_buf 邻域复核(每周重训未覆盖轴)。
用法: python3 exp_opt31d.py  (输出 out/exp_opt31d.json)"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

OUT = os.path.join(os.getcwd(), "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(HERE, "references", "final_cfg_v30.json")
PROXY_START, REAL_START = "2014-06-23", "2025-04-23"
ACCEPT = {"dcal": 0.15, "proxy_mdd": -0.10, "real_mdd": -0.041}

CANDS = {
    "base_v30": {},
    "B1_breadth20_50": {"breadth_gate": True, "breadth_win": 20, "breadth_thr": 0.5, "breadth_cut": 0.7},
    "B2_breadth20_40": {"breadth_gate": True, "breadth_win": 20, "breadth_thr": 0.4, "breadth_cut": 0.6},
    "B3_breadth100_50": {"breadth_gate": True, "breadth_win": 100, "breadth_thr": 0.5, "breadth_cut": 0.7},
    "T1_voltarget18": {"vol_target": 0.18},
    "T2_voltarget20": {"vol_target": 0.20},
    "T3_volbuf13": {"vol_buf": 1.30},
}

def run_cfg(cfg, R, start):
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    return evaluate(res), res

def metrics(cfg, R, start):
    e, _ = run_cfg(cfg, R, start)
    return {"cagr": e["cagr"], "mdd": e["max_dd"], "cal": e["calmar"],
            "sharpe": e["sharpe"], "cash": e["avg_cash"], "to": e["turnover"]}

def main():
    base = json.load(open(CFG_FILE))
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    rows = {}
    survivors = []
    for name, ov in CANDS.items():
        cfg = copy.deepcopy(base)
        cfg.update(ov)
        p = metrics(cfg, R, PROXY_START); r = metrics(cfg, Rr, REAL_START)
        rows[name] = {"override": ov, "proxy": p, "real": r}
        print(f"{name:<22} proxy Cal {p['cal']:5.2f} MDD {p['mdd']*100:6.2f}% | real Cal {r['cal']:5.2f} MDD {r['mdd']*100:6.2f}%")
    bp, br = rows["base_v30"]["proxy"], rows["base_v30"]["real"]
    print(f"\n基线 v30: proxy Cal {bp['cal']:.2f} MDD {bp['mdd']*100:.2f}% | real Cal {br['cal']:.2f} MDD {br['mdd']*100:.2f}%")
    for name in rows:
        if name == "base_v30":
            continue
        r = rows[name]
        dcp, dcr = r["proxy"]["cal"] - bp["cal"], r["real"]["cal"] - br["cal"]
        r["dcal_p"], r["dcal_r"] = dcp, dcr
        ok = (dcp >= ACCEPT["dcal"] and dcr >= ACCEPT["dcal"] and
              r["proxy"]["mdd"] > ACCEPT["proxy_mdd"] and r["real"]["mdd"] > ACCEPT["real_mdd"])
        flag = " ★通过一关" if ok else ("  (proxy单窗)" if dcp >= ACCEPT["dcal"] else
                ("  (real单窗)" if dcr >= ACCEPT["dcal"] else ""))
        print(f"{name:<22} ΔCal proxy {dcp:+.3f} / real {dcr:+.3f}  MDD p {r['proxy']['mdd']*100:.2f}% / r {r['real']['mdd']*100:.2f}%{flag}")
        if ok:
            survivors.append(name)
    json.dump(rows, open(os.path.join(OUT, "exp_opt31d.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n[一关通过] {len(survivors)} 个: {survivors}")
    print("\n已保存 out/exp_opt31d.json")

if __name__ == "__main__":
    main()
