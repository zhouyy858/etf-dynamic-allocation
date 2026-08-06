# -*- coding: utf-8 -*-
"""v31 候选预注册评估(2026-08-06, 数据至2026-08-05): A回撤控制端/B收益端/C现金层
单机制×12, 严格口径(signal_lag=1/溢价T-2/先计提后成交, 无未来函数)
第一关: 双窗口Calmar同向+0.15 且 proxy MDD>-10% / real MDD>-4.1% -> 第二关平台扫描+第三关OOS
三关全过才建议升级 v31(版本号只增不减, 不覆盖 v30); 全部证伪则维持 v30。
用法: python3 scripts/exp_opt31.py  (输出 out/exp_opt31.json)"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

OUT = os.path.join(os.getcwd(), "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(HERE, "references", "final_cfg_v30.json")
PROXY_START, REAL_START, OOS_START = "2014-06-23", "2025-04-23", "2022-01-04"
ACCEPT = {"dcal": 0.15, "proxy_mdd": -0.10, "real_mdd": -0.041}
PLAT = {"win": 0.10, "cliff": 0.5, "min_ok": 3}
OOS_BAD = 0.05

CANDS = {
    "base_v30": {},
    "A1_dd_eq_cap紧": {"dd_eq_cap": [[-0.06, 85], [-0.10, 75], [-0.14, 65], [-0.18, 55]]},
    "A2_dd_eq_cap紧低": {"dd_eq_cap": [[-0.06, 80], [-0.10, 70], [-0.14, 60], [-0.18, 50]]},
    "A3_sb_ddthr-2%": {"speed_brake_dd_thr": -0.02},
    "A4_sb_ddthr-5%": {"speed_brake_dd_thr": -0.05},
    "B1_reentry1pp": {"reentry_step": 1.0},
    "B2_reentry2pp": {"reentry_step": 2.0},
    "B3_state9_85_3": {"state_map": {"0": [5, 30], "1": [15, 29], "2": [25, 28], "3": [36, 26],
                                      "4": [39, 21], "5": [44, 16], "6": [66, 14], "7": [65, 10],
                                      "8": [88, 6], "9": [85, 3]}},
    "B4_growth32_61": {"growth_split_bull": {"159952": 0.32, "159941": 0.61, "513500": 0.07}},
    "C1_bond80": {"cash_bond_pct": 0.80},
    "C2_bond85": {"cash_bond_pct": 0.85},
    "C3_repo2.5%": {"repo_rate": 0.025},
    "C4_repo1.9%": {"repo_rate": 0.019},
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
    rows, survivors = {}, []
    for name, ov in CANDS.items():
        cfg = copy.deepcopy(base)
        cfg.update(ov)
        p = metrics(cfg, R, PROXY_START); r = metrics(cfg, Rr, REAL_START)
        row = {"override": ov, "proxy": p, "real": r,
               "dcal_p": p["cal"] - 0, "dcal_r": 0}  # 占位, 基线算完再填
        rows[name] = row
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
    json.dump(rows, open(f"{OUT}/exp_opt31.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n[一关通过] {len(survivors)} 个: {survivors}")
    for name in survivors:
        ov = rows[name]["override"]
        cfg = copy.deepcopy(base); cfg.update(ov)
        print(f"\n=== 二关/三关: {name} ({ov}) ===")
        verify(cfg, R, Rr, ov, name)
    print("\n[ok] out/exp_opt31.json")

def verify(cfg, R, Rr, ov, name):
    """第二关平台扫描 + 第三关OOS(只对通过一关的候选)"""
    key = next(iter(ov))
    v = ov[key]
    base = json.load(open(CFG_FILE))
    bp = metrics(base, R, PROXY_START)["cal"]; br = metrics(base, Rr, REAL_START)["cal"]
    oo_b = metrics(base, R, OOS_START)["cal"]
    if isinstance(v, list) and key != "dd_eq_cap":
        print(f"  [平台] {key} 列表参数, 手工验证, 跳过自动扫描")
        return
    # 平台扫描: 对数值轴取候选±2步(步长按轴)
    if key == "dd_eq_cap":
        # 阶梯类: 整体平移±1pp 验证平台
        steps = [-0.01, 0.01]
        vals = {"shift": steps}
        for s in steps:
            shifted = [[thr + s, eq] for thr, eq in v]
            cc = copy.deepcopy(cfg); cc[key] = shifted
            pp = metrics(cc, R, PROXY_START)["cal"]; rr = metrics(cc, Rr, REAL_START)["cal"]
            print(f"  [平台] 整体{s:+.1%}pp: proxy {pp:.3f}({pp-bp:+.3f}) real {rr:.3f}({rr-br:+.3f})")
        print(f"  [OOS] {key}={v} OOS Cal {metrics(cfg, R, OOS_START)['cal']:.3f} vs 基线 {oo_b:.3f}")
        return
    if isinstance(v, (int, float)):
        step = 0.01 if key in ("reentry_step", "speed_brake_dd_thr") else 0.5
        vals = {}
        for k in (1, 2):
            vals[f"{v + step * k:.2g}"] = v + step * k
            vals[f"{v - step * k:.2g}"] = v - step * k
        ok_n, prev = 0, None
        for lab, vv in sorted(vals.items(), key=lambda x: x[1]):
            cc = copy.deepcopy(cfg); cc[key] = vv
            pp = metrics(cc, R, PROXY_START)["cal"]; rr = metrics(cc, Rr, REAL_START)["cal"]
            dpp, drr = pp - bp, rr - br
            print(f"  [平台] {key}={lab}: proxy {pp:.3f}({dpp:+.3f}) real {rr:.3f}({drr:+.3f})")
            if prev is not None and abs(pp - prev) > PLAT["cliff"]:
                print(f"  [平台] 悬崖(落差{abs(pp-prev):.2f}>0.5), 平台失败")
                return
            prev = pp
            if dpp > PLAT["win"] and drr > PLAT["win"]:
                ok_n += 1
        if ok_n < PLAT["min_ok"]:
            print(f"  [平台] 仅{ok_n}点同向>0.10(<{PLAT['min_ok']}), 平台失败")
            return
        oo = metrics(cfg, R, OOS_START)["cal"]
        print(f"  [OOS] {key}={v} OOS Cal {oo:.3f} vs 基线 {oo_b:.3f}")
        if oo < oo_b - OOS_BAD:
            print("  [OOS] 劣化>0.05, 拒绝")
            return
        print("  ★★★ 三关全过 → 建议升级 v31 ★★★")

if __name__ == "__main__":
    main()
