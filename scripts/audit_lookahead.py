# -*- coding: utf-8 -*-
"""未来函数审计(可复现): 时序口径对照
外部审查发现原v17-v20回测含"当日收盘信号+当日收盘成交+成交仓位计提当日收益"超前口径。
本脚本用同一引擎的两种计提模式(accrual_mode=pre/post) + 信号滞后(signal_lag) + 成交滞后(exec_lag)
量化各口径差异, 并验证旧口径可复现已发布的v20数字(proxy 16.20%/-5.78%, real 28.10%/-5.34%)。

口径:
  OLD 旧发布口径   : accrual=post + signal_lag=0 + exec_lag=0 + 1笔成交  (v20发布值)
  B   严格(v21)   : accrual=pre  + signal_lag=1 + exec_lag=0 + 3周三笔  (周三早盘用周二数据, 收盘成交)
  A0  同日信号对照 : accrual=pre  + signal_lag=0 + exec_lag=0 + 1笔成交  (仅修计提, 保留同日信号)
  C   次日成交对照 : accrual=pre  + signal_lag=0 + exec_lag=1 + 1笔成交
  D   双滞后对照   : accrual=pre  + signal_lag=1 + exec_lag=1 + 1笔成交
用法: python3 scripts/audit_lookahead.py
输出: 终端对照表 + out/audit_lookahead.json
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from engine import run_backtest, evaluate
from strategy import DynamicStrategy
from stress_test import synthetic_resonance

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
CFG20 = json.load(open(f"{SKILL_REF}/final_cfg_v20.json"))
CFG21 = json.load(open(f"{SKILL_REF}/final_cfg_v21.json"))
bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
R, _ = build_panel("proxy"); Rr, _ = build_panel("real")

def run(R, start, tag, cfg, accrual="pre", signal_lag=0, exec_lag=0, tw=None):
    c = dict(cfg); c["signal_lag"] = signal_lag
    ds = DynamicStrategy(R, cfg=c)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=start, end=None, name=tag, min_delta=0.02, repo=0.022,
                       tranche_weights=tw, cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       exec_lag=exec_lag, accrual_mode=accrual)
    return evaluate(res)

print("===== 未来函数口径对照 · proxy 全历史 2014-06-23 起 =====")
MODES = [
    ("OLD 旧发布口径(含未来函数)", dict(CFG20), "post", 0, 0, [1.0]),
    ("A0 仅修计提·同日信号",       dict(CFG20), "pre",  0, 0, [1.0]),
    ("B  前日信号+当日成交(1笔)",   dict(CFG20), "pre",  1, 0, [1.0]),
    ("C  当日信号+次日成交",        dict(CFG20), "pre",  0, 1, [1.0]),
    ("D  前日信号+次日成交",        dict(CFG20), "pre",  1, 1, [1.0]),
    ("v21 前日信号+3周三笔(定稿)",  dict(CFG21), "pre",  1, 0, [1/3]*3),
]
res = {}
for name, cfg, am, sl, el, tw in MODES:
    e = run(R, "2014-06-23", "p", cfg, am, sl, el, tw)
    res[name] = {"proxy": {m: e[m] for m in ["cagr", "max_dd", "sharpe", "calmar", "turnover"]}}
    print(f"  {name:<30} CAGR {e['cagr']*100:6.2f}%  MDD {e['max_dd']*100:6.2f}%  Sharpe {e['sharpe']:.2f}  "
          f"Calmar {e['calmar']:4.2f}  TO {e['turnover']:6.1f}")

print("\n===== 真实窗口 2025-04-23 起 =====")
for name, cfg, am, sl, el, tw in MODES:
    e = run(Rr, "2025-04-23", "r", cfg, am, sl, el, tw)
    res[name]["real"] = {m: e[m] for m in ["cagr", "max_dd", "sharpe", "calmar", "turnover"]}
    print(f"  {name:<30} CAGR {e['cagr']*100:6.2f}%  MDD {e['max_dd']*100:6.2f}%  Calmar {e['calmar']:4.2f}")

# 验证旧口径可复现v20发布值
assert abs(res["OLD 旧发布口径(含未来函数)"]["proxy"]["cagr"] - 0.1620) < 0.005, "OLD proxy 复现失败"
assert abs(res["OLD 旧发布口径(含未来函数)"]["real"]["cagr"] - 0.2810) < 0.01, "OLD real 复现失败"
print("\n[ok] 旧口径复现v20发布值 (16.20%/28.10%) 通过")

print("\n===== 压力测试 (严格口径B vs 旧口径) =====")
synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)
scen = [("2015股灾", R, "2015-06-15", "2016-02-29", None), ("2018熊市", R, "2018-01-02", "2019-01-03", None),
        ("2021-22熊", R, "2021-02-19", "2022-10-31", None), ("2019-21牛", R, "2019-01-04", "2021-02-18", None),
        ("共振熊", synth, "2024-09-02", "2026-07-31", s_idx)]
print(f"{'情景':<10}{'OLD CAGR/MDD':>20}{'严格B CAGR/MDD':>22}")
for name, Rs, ps, pe, am in scen:
    cfg = dict(CFG20)
    a = run(Rs, ps, "old", cfg, "post", 0, 0, [1.0])
    b = run(Rs, ps, "b", cfg, "pre", 1, 0, [1.0])
    print(f"{name:<10}{a['cagr']*100:10.2f}%/{a['max_dd']*100:6.2f}%{b['cagr']*100:12.2f}%/{b['max_dd']*100:6.2f}%")
    res.setdefault("stress", {})[name] = {"OLD": [round(a["cagr"], 4), round(a["max_dd"], 4)],
                                          "B": [round(b["cagr"], 4), round(b["max_dd"], 4)]}

json.dump(res, open(f"{OUT}/audit_lookahead.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\n[ok] out/audit_lookahead.json")
