# -*- coding: utf-8 -*-
"""WFO 样本内/样本外验证: 训练窗口2014-2021选参, 2022-2026测试(含2024Q4-2026双牛)
检验 v15 全样本参数是否过拟合: 若训练窗口最优参数在测试窗口明显优于v15, 则v15有过拟合嫌疑
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

BASE = json.load(open("out/final_cfg_v15.json"))
R_P, _ = build_panel("proxy")
TRAIN = ("2014-06-23", "2021-12-31")
TEST = ("2022-01-01", "2026-07-31")

def run(cfg, start, end):
    ds = DynamicStrategy(R_P, cfg=cfg)
    res = run_backtest(R_P, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=end, name="DYN", min_delta=0.02, repo=0.022)
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"], to=e["turnover"])

def growth_scale(cfg, f):
    c = copy.deepcopy(cfg)
    c["state_map"] = {k: [max(4, min(98, round(g*f))), d] for k, (g, d) in c["state_map"].items()}
    return c

grid = {}
grid["v15基线"] = copy.deepcopy(BASE)
for gs in [0.9, 1.05, 1.1, 1.2]:
    grid[f"growth x{gs}"] = growth_scale(BASE, gs)
grid["hyst0.70/0.12"] = {**BASE, "hyst_up": 0.70, "hyst_down": 0.12}
grid["vol_target0.15"] = {**BASE, "vol_target": 0.15, "vol_buf": 1.0}

print(f"{'配置':22s} | {'训练窗口 2014-2021':42s} | {'测试窗口 2022-2026 (OOS)':42s}")
print(f"{'':22s} | {'CAGR':>7s}{'MDD':>8s}{'Sharpe':>8s}{'Calmar':>8s}{'TO':>6s} | {'CAGR':>7s}{'MDD':>8s}{'Sharpe':>8s}{'Calmar':>8s}{'TO':>6s}")
res_all = {}
for name, cfg in grid.items():
    tr = run(cfg, *TRAIN)
    te = run(cfg, *TEST)
    res_all[name] = {"train": tr, "test": te}
    print(f"{name:22s} | {tr['cagr']:7.2f}{tr['mdd']:8.2f}{tr['sharpe']:8.2f}{tr['calmar']:8.2f}{tr['to']:6.1f} | {te['cagr']:7.2f}{te['mdd']:8.2f}{te['sharpe']:8.2f}{te['calmar']:8.2f}{te['to']:6.1f}")
json.dump(res_all, open("out/wfo_test.json", "w"), ensure_ascii=False, indent=1)
print("[ok] out/wfo_test.json")
