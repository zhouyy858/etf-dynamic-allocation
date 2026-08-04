# -*- coding: utf-8 -*-
"""v22b 候选全维度对比 (严格无未来函数口径, 与 search_v22 一致)
对比 v22 基线 vs 3 个 v22b 候选:
  C1 = gs 35/55/10 + bear 50/30/20
  C2 = gs 35/55/10 + min_delta 0.035
  C3 = gs 35/55/10 + bear 50/30/20 + min_delta 0.035
输出: out/exp_v22b_round4.json (全历史proxy + real窗口 + 关键市场窗口)
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)

def load(name):
    return json.load(open(f"{SKILL_REF}/{name}"))

CFG22 = load("final_cfg_v22.json")

def variant(tag, **kw):
    c = copy.deepcopy(CFG22)
    c.update(kw)
    c["version"] = tag
    return c

C1 = variant("v22b_C1", growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10},
             growth_split_bear={"159952": 0.50, "159941": 0.30, "513500": 0.20})
C2 = variant("v22b_C2", growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10},
             min_delta=0.035)
C3 = variant("v22b_C3", growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10},
             growth_split_bear={"159952": 0.50, "159941": 0.30, "513500": 0.20},
             min_delta=0.035)

VARIANTS = {"v22": CFG22, "C1_gs35_bear5030": C1, "C2_gs35_md35": C2, "C3_all": C3}

def run(cfg, start, end, R, bond):
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=end, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"],
                total=e["total_ret"]*100, to=e["turnover"], cash=e["avg_cash"]*100)

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")

    WINDOWS = {
        "全历史_proxy": (R, "2014-06-23", "2026-07-31"),
        "real窗口": (Rr, "2025-04-23", "2026-07-31"),
        "OOS_2022-2026": (R, "2022-01-01", "2026-07-31"),
        "熊_2015股灾": (R, "2015-06-15", "2016-02-29"),
        "熊_2018": (R, "2018-01-02", "2019-01-03"),
        "牛_2019-2021": (R, "2019-01-04", "2021-02-18"),
        "熊_2021-2022": (R, "2021-02-19", "2022-10-31"),
        "震荡_2023-2024": (R, "2023-01-03", "2024-08-30"),
        "牛_2024Q4-2026OOS": (R, "2024-09-02", "2026-07-31"),
    }

    results = {}
    for wname, (Rs, ps, pe) in WINDOWS.items():
        row = {}
        for tag, cfg in VARIANTS.items():
            row[tag] = run(cfg, ps, pe, Rs, bond)
        results[wname] = row
        print(f"\n===== {wname} ({ps} ~ {pe}) =====")
        for tag in VARIANTS:
            r = row[tag]
            print(f"  {tag:<16} CAGR={r['cagr']:7.2f}% MDD={r['mdd']:6.2f}% Sharpe={r['sharpe']:5.2f} "
                  f"Calmar={r['calmar']:5.2f} TO={r['to']:5.0f}% cash={r['cash']:4.0f}%")
    json.dump(results, open(f"{OUT}/exp_v22b_round4.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v22b_round4.json")

if __name__ == "__main__":
    main()
