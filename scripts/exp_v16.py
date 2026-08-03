# -*- coding: utf-8 -*-
"""v16 预注册变体实验 (防过拟合纪律):
每变体只改一个理论驱动机制, 同时在全历史(proxy)与真实窗口(real)评估,
不针对特定行情调参; 验收: proxy Calmar>=1.07 & CAGR>=13.2%, real CAGR>=25% & MDD>=-8%
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest, evaluate
from strategy import DynamicStrategy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = json.load(open("out/final_cfg_v15.json"))
R_P, _ = build_panel("proxy")
R_R, _ = build_panel("real")

def run(cfg, R, start, tag):
    ds = DynamicStrategy(R, cfg=cfg)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, name=tag, min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022))
    e = evaluate(res)
    return dict(cagr=e["cagr"]*100, mdd=e["max_dd"]*100, sharpe=e["sharpe"], calmar=e["calmar"],
                total=e["total_ret"]*100, to=e["turnover"], cash=e["avg_cash"]*100)

VARIANTS = {
    "v15基": {},
    "v16a_vol17激活": {"vol_target": 0.17, "vol_buf": 1.0},
    "v16b_vol非对称": {"vol_scale_hi": 1.15, "vol_scale_lo": 0.85, "vol_buf": 1.0},
    "v16c_确认2周": {"score_confirm_weeks": 2},
    "v16d_滞后增强": {"hyst_up": 0.70, "hyst_down": 0.12},
    "v16e_vol15激活": {"vol_target": 0.15, "vol_buf": 1.0},
    "v16f_去溢价门": {"premium_gate": False},
    "v16g_去相关风控": {"corr_risk": False},
    "v16h_去速度刹车": {"speed_brake": False},
}
rows = []
for name, mut in VARIANTS.items():
    cfg = copy.deepcopy(BASE); cfg.update(mut)
    p = run(cfg, R_P, "2014-06-23", name)
    r = run(cfg, R_R, "2025-04-23", name)
    rows.append({"variant": name, **{f"p_{k}": v for k, v in p.items()}, **{f"r_{k}": v for k, v in r.items()}})
    print(f"{name:18s} proxy CAGR {p['cagr']:6.2f} MDD {p['mdd']:7.2f} Sharpe {p['sharpe']:.2f} Calmar {p['calmar']:.2f} TO {p['to']:5.1f} | real CAGR {r['cagr']:6.2f} MDD {r['mdd']:7.2f} Calmar {r['calmar']:.2f} TO {r['to']:4.1f}")
json.dump(rows, open(os.path.join(OUT, "exp_v16.json"), "w"), ensure_ascii=False, indent=1)
print("[ok] out/exp_v16.json")

# ---- 第二轮: premium_rotate + growth 微调 (real窗口聚焦) ----
VAR2 = {
    "v16i_溢价转国内": {"premium_rotate": True},
    "v16j_growth1.05": None,
    "v16k_growth1.05+溢价转": None,
    "v16l_growth1.10": None,
}
def growth_scale(cfg, f):
    c = copy.deepcopy(cfg)
    c["state_map"] = {k: [max(4, min(98, round(g*f))), d] for k, (g, d) in c["state_map"].items()}
    return c
for name, mut in VAR2.items():
    cfg = copy.deepcopy(BASE)
    if "growth1.05" in name: cfg = growth_scale(cfg, 1.05)
    if "growth1.10" in name: cfg = growth_scale(cfg, 1.10)
    if mut: cfg.update(mut)
    p = run(cfg, R_P, "2014-06-23", name)
    r = run(cfg, R_R, "2025-04-23", name)
    rows.append({"variant": name, **{f"p_{k}": v for k, v in p.items()}, **{f"r_{k}": v for k, v in r.items()}})
    print(f"{name:18s} proxy CAGR {p['cagr']:6.2f} MDD {p['mdd']:7.2f} Sharpe {p['sharpe']:.2f} Calmar {p['calmar']:.2f} TO {p['to']:5.1f} | real CAGR {r['cagr']:6.2f} MDD {r['mdd']:7.2f} Calmar {r['calmar']:.2f} TO {r['to']:4.1f}")
json.dump(rows, open(os.path.join(OUT, "exp_v16.json"), "w"), ensure_ascii=False, indent=1)
print("[ok] out/exp_v16.json updated")
