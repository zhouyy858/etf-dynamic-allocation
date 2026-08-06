# -*- coding: utf-8 -*-
"""v31b 候选预注册评估(2026-08-06, 数据至2026-08-05): GitHub调研启发 收益端2 + 回撤控制端2
单机制×4, 严格口径(signal_lag=1/溢价T-2/先计提后成交, 无未来函数)
第一关: 双窗口Calmar同向+0.15 且 proxy MDD>-10% / real MDD>-4.1% -> 第二关平台扫描+第三关OOS
三关全过才建议升级 v31; 全部证伪则维持 v30。
候选来源: ①growth_iv 波动率倒数加权(growth桶内, ∝(1/σ)^t, 借鉴IdealAuror全天候反波动加权);
②vol_gate 离散波动率门控(创业板20日年化波动分档降仓, 借鉴zhangsensen波动率门控100/70/40/10)。
拥挤度惩罚候选(ericxuzhesheng)因代理层(2014-2022指数)无成交额数据、无法对齐proxy窗口, 暂缓。
用法: python3 exp_opt31b.py  (输出 out/exp_opt31b.json)"""
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
    "V1_growth_iv_t1.0": {"growth_iv": True, "growth_iv_win": 60, "growth_iv_t": 1.0},
    "V2_growth_iv_t0.5": {"growth_iv": True, "growth_iv_win": 60, "growth_iv_t": 0.5},
    "V3_vol_gate_304050": {"vol_gate": True, "vol_gate_win": 20,
                           "vol_gate_bands": [0.30, 0.40, 0.50], "vol_gate_cuts": [0.7, 0.4, 0.1]},
    "V4_vol_gate_354555": {"vol_gate": True, "vol_gate_win": 20,
                           "vol_gate_bands": [0.35, 0.45, 0.55], "vol_gate_cuts": [0.7, 0.4, 0.1]},
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
    json.dump(rows, open(os.path.join(OUT, "exp_opt31b.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n[一关通过] {len(survivors)} 个: {survivors}")
    for name in survivors:
        verify(base, R, Rr, name)
    print("\n已保存 out/exp_opt31b.json")

OOS_START = "2022-01-04"
PLAT = {"win": 0.10, "cliff": 0.5, "min_ok": 3}
OOS_BAD = 0.05

def verify(base, R, Rr, name):
    """第二关平台扫描(vol_gate三轴) + 第三关OOS"""
    cfg = copy.deepcopy(base)
    ov = {"vol_gate": True, "vol_gate_win": 20,
          "vol_gate_bands": [0.30, 0.40, 0.50], "vol_gate_cuts": [0.7, 0.4, 0.1]}
    cfg.update(ov)
    bp = metrics(base, R, PROXY_START)["cal"]; br = metrics(base, Rr, REAL_START)["cal"]
    oo_b = metrics(base, R, OOS_START)["cal"]
    axes = {
        "bands平移": {
            "[0.28,0.38,0.48]": [0.28, 0.38, 0.48], "[0.30,0.40,0.50]": [0.30, 0.40, 0.50],
            "[0.32,0.42,0.52]": [0.32, 0.42, 0.52], "[0.25,0.35,0.45]": [0.25, 0.35, 0.45],
            "[0.35,0.45,0.55]": [0.35, 0.45, 0.55]},
        "cuts平移": {
            "[0.6,0.35,0.05]": [0.6, 0.35, 0.05], "[0.7,0.4,0.1]": [0.7, 0.4, 0.1],
            "[0.8,0.5,0.15]": [0.8, 0.5, 0.15], "[0.5,0.3,0.05]": [0.5, 0.3, 0.05],
            "[0.9,0.6,0.2]": [0.9, 0.6, 0.2]},
        "win": {"15": 15, "20": 20, "25": 25, "30": 30},
    }
    all_ok = True
    for axis, vals in axes.items():
        ok_n, prev = 0, None
        for lab, vv in vals.items():
            cc = copy.deepcopy(cfg)
            if axis == "win":
                cc["vol_gate_win"] = vv
            elif axis == "bands平移":
                cc["vol_gate_bands"] = vv
            else:
                cc["vol_gate_cuts"] = vv
            pp = metrics(cc, R, PROXY_START)["cal"]; rr = metrics(cc, Rr, REAL_START)["cal"]
            dpp, drr = pp - bp, rr - br
            print(f"  [平台:{axis}] {lab:<16} proxy {pp:.3f}({dpp:+.3f}) real {rr:.3f}({drr:+.3f})")
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
    print(f"  [OOS] V3 OOS Cal {oo:.3f} vs 基线 {oo_b:.3f}")
    if oo < oo_b - OOS_BAD:
        print("  [OOS] 劣化>0.05, 拒绝")
        return
    print("  ★★★ 三关全过 → 建议升级 v31 ★★★")

if __name__ == "__main__":
    main()
