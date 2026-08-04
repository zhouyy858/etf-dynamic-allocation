# -*- coding: utf-8 -*-
"""v22b WFO 样本内/样本外验证 (严格口径)
训练窗口 2014-06~2021-12 vs 测试窗口 2022-01~2026-07 (OOS 含2024Q4-2026双牛)
对比 v22 vs C1/C2/C3: 若候选在 OOS 窗口优势消失或恶化, 判定过拟合
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out")

def load(name):
    return json.load(open(f"{SKILL_REF}/{name}"))

CFG22 = load("final_cfg_v22.json")

def variant(tag, **kw):
    c = copy.deepcopy(CFG22); c.update(kw); c["version"] = tag; return c

VARIANTS = {
    "v22": CFG22,
    "C1_gs35_bear5030": variant("v22b_C1", growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10},
                                growth_split_bear={"159952": 0.50, "159941": 0.30, "513500": 0.20}),
    "C2_gs35_md35": variant("v22b_C2", growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10}, min_delta=0.035),
    "C3_all": variant("v22b_C3", growth_split_bull={"159952": 0.35, "159941": 0.55, "513500": 0.10},
                      growth_split_bear={"159952": 0.50, "159941": 0.30, "513500": 0.20}, min_delta=0.035),
}

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
                total=e["total_ret"]*100, to=e["turnover"])

def main():
    from data_prep import build_panel, read_table, rets_from
    R, _ = build_panel("proxy")
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    TRAIN, TEST = ("2014-06-23", "2021-12-31"), ("2022-01-01", "2026-07-31")
    print(f"{'配置':<16} | {'训练 2014-2021':44s} | {'测试 2022-2026 (OOS)':44s}")
    print(f"{'':16s} | {'CAGR':>7s}{'MDD':>8s}{'Sharpe':>8s}{'Calmar':>8s}{'TO':>6s} | {'CAGR':>7s}{'MDD':>8s}{'Sharpe':>8s}{'Calmar':>8s}{'TO':>6s}")
    res = {}
    for tag, cfg in VARIANTS.items():
        tr = run(cfg, *TRAIN, R, bond)
        te = run(cfg, *TEST, R, bond)
        res[tag] = {"train": tr, "test": te}
        print(f"{tag:<16} | {tr['cagr']:7.2f}{tr['mdd']:8.2f}{tr['sharpe']:8.2f}{tr['calmar']:8.2f}{tr['to']:6.0f} | "
              f"{te['cagr']:7.2f}{te['mdd']:8.2f}{te['sharpe']:8.2f}{te['calmar']:8.2f}{te['to']:6.0f}")
    json.dump(res, open(f"{OUT}/exp_v22b_wfo.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[ok] {OUT}/exp_v22b_wfo.json")

if __name__ == "__main__":
    main()
