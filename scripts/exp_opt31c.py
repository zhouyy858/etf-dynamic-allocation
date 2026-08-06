# -*- coding: utf-8 -*-
"""v31c 候选预注册评估(2026-08-06, 数据至2026-08-05): GitHub调研启发 第二波
单机制×5, 严格口径(signal_lag=1/溢价T-2/先计提后成交, 无未来函数)
第一关: 双窗口Calmar同向+0.15 且 proxy MDD>-10% / real MDD>-4.1% -> 平台+OOS
候选来源: ①adx_gate ADX趋势强度调节(沪深300 close-only ADX, 借鉴ivanyinjc五态识别);
②reversal_filter 短期过度延伸降权(20日涨幅惩罚, 借鉴ZaidShk 1个月反转过滤);
③gold_crisis 现金层危机凸性(CN回撤<-5%时现金层50%转518880黄金, 借鉴ZaidShk flight-to-quality)。
用法: python3 exp_opt31c.py  (输出 out/exp_opt31c.json)"""
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
    "V1_adx_low": {"adx_gate": True, "adx_win": 14, "adx_bands": [10.0, 15.0, 25.0],
                   "adx_cuts": [0.7, 0.85, 1.0]},
    "V2_adx_full": {"adx_gate": True, "adx_win": 14, "adx_bands": [10.0, 15.0, 25.0, 35.0],
                    "adx_cuts": [0.6, 0.8, 1.0, 1.05]},
    "V3_rev_8pct": {"reversal_filter": True, "rev_thr": 0.08, "rev_span": 0.10, "rev_min_k": 0.4},
    "V4_rev_10pct": {"reversal_filter": True, "rev_thr": 0.10, "rev_span": 0.10, "rev_min_k": 0.5},
    "V5_gold": {"gold_crisis": True, "gold_dd": -0.15, "gold_rec": -0.08, "gold_pct": 0.5},
}

def run_cfg(cfg, R, start):
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    gold = rets_from(read_table("518880_nav.csv"), "cum_nav")
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True, cash_gold_rets=gold, gold_override_fn=ds.gold_pct_fn())
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
        print(f"{name:<16} proxy Cal {p['cal']:5.2f} MDD {p['mdd']*100:6.2f}% | real Cal {r['cal']:5.2f} MDD {r['mdd']*100:6.2f}%")
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
        print(f"{name:<16} ΔCal proxy {dcp:+.3f} / real {dcr:+.3f}  MDD p {r['proxy']['mdd']*100:.2f}% / r {r['real']['mdd']*100:.2f}%{flag}")
        if ok:
            survivors.append(name)
    json.dump(rows, open(os.path.join(OUT, "exp_opt31c.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n[一关通过] {len(survivors)} 个: {survivors}")
    for name in survivors:
        verify(base, R, Rr, name)
    print("\n已保存 out/exp_opt31c.json")

OOS_START = "2022-01-04"
PLAT = {"win": 0.10, "cliff": 0.5, "min_ok": 3}
OOS_BAD = 0.05

def verify(base, R, Rr, name):
    """第二关平台扫描(gold三轴) + 第三关OOS"""
    cfg = copy.deepcopy(base)
    ov = {"gold_crisis": True, "gold_dd": -0.15, "gold_rec": -0.08, "gold_pct": 0.5}
    cfg.update(ov)
    bp = metrics(base, R, PROXY_START)["cal"]; br = metrics(base, Rr, REAL_START)["cal"]
    oo_b = metrics(base, R, OOS_START)["cal"]
    axes = {
        "gold_dd": {"-0.10": -0.10, "-0.12": -0.12, "-0.15": -0.15,
                    "-0.18": -0.18, "-0.20": -0.20},
        "gold_pct": {"0.3": 0.3, "0.5": 0.5, "0.7": 0.7, "1.0": 1.0, "0.2": 0.2},
        "gold_rec": {"-0.05": -0.05, "-0.08": -0.08, "-0.10": -0.10,
                     "-0.12": -0.12, "-0.15": -0.15},
    }
    all_ok = True
    for axis, vals in axes.items():
        ok_n, prev = 0, None
        for lab, vv in vals.items():
            cc = copy.deepcopy(cfg)
            cc[axis] = vv
            pp = metrics(cc, R, PROXY_START)["cal"]; rr = metrics(cc, Rr, REAL_START)["cal"]
            dpp, drr = pp - bp, rr - br
            print(f"  [平台:{axis}] {lab:<6} proxy {pp:.3f}({dpp:+.3f}) real {rr:.3f}({drr:+.3f})")
            if prev is not None and abs(pp - prev) > PLAT["cliff"]:
                print(f"  [平台] 悬崖(落差{abs(pp - prev):.2f}>0.5), 平台失败")
                all_ok = False
                break
            prev = pp
            if dpp > PLAT["win"] and drr > PLAT["win"]:
                ok_n += 1
        if ok_n < PLAT["min_ok"]:
            print(f"  [平台:{axis}] 仅{ok_n}点同向>0.10(<{PLAT['min_ok']}), 平台失败")
            all_ok = False
        else:
            print(f"  [平台:{axis}] {ok_n}点同向>0.10, OK")
    if not all_ok:
        print("  [二关] 平台不满足 → 拒绝, 维持 v30")
        return
    oo = metrics(cfg, R, OOS_START)["cal"]
    print(f"  [OOS] V5 OOS Cal {oo:.3f} vs 基线 {oo_b:.3f}")
    if oo < oo_b - OOS_BAD:
        print("  [OOS] 劣化>0.05, 拒绝")
        return
    print("  ★★★ 三关全过 → 建议升级 v31 ★★★")

if __name__ == "__main__":
    main()
