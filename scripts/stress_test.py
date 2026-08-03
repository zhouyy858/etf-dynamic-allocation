# -*- coding: utf-8 -*-
"""压力测试: 三情景(牛市/震荡/熊市) + 新增跨境共振熊市(合成冲击)
对比 DYN v10(旧) vs DYN v15(3周三笔) vs DYN v17(1笔) vs DYN v18(1笔+底仓5+5)
每周三决策、DYN v17 当日一笔成交(原分3周三笔各1/3已被替代)
合成共振: 取真实双牛窗口2024-09~2026-07, 人为将美股收益缩放到窗口累计-30%、
A股缩放到-20%、沪深300缩放到-25%(信号一致), 检验极端共振下的防守
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import run_backtest, evaluate, SLOTS
from strategy import DynamicStrategy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"); os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
CFG10 = json.load(open(f"{SKILL_REF}/final_cfg_v10.json"))
CFG14 = json.load(open(f"{SKILL_REF}/final_cfg_v15.json"))
CFG17 = json.load(open(f"{SKILL_REF}/final_cfg_v17.json"))
CFG18 = json.load(open(f"{SKILL_REF}/final_cfg_v18.json"))
REPO = 0.022

BENCHMARKS = {
    "B1等权20": {s: 20 for s in SLOTS},
    "B2保守防御": {"159232": 25, "515100": 25, "159941": 20, "513500": 20, "159952": 10},
    "B3均衡": {"159232": 15, "515100": 15, "159941": 25, "513500": 20, "159952": 25},
    "B4成长进攻": {"159232": 10, "515100": 10, "159941": 30, "513500": 20, "159952": 30},
    "B5价值60/成长40": {"159232": 30, "515100": 30, "159941": 15, "513500": 15, "159952": 10},
}

def scale_k(rs, target):
    """二分求k: prod(1+k*r)-1 = target (窗口内多为正收益, prod单调增)"""
    lo, hi = -2.0, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        v = (1 + mid * rs.fillna(0.0)).prod() - 1.0
        if v > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

def synthetic_resonance(R, window, us_target, cn_target, idx_target):
    """构造共振熊市: US累计-30%, A股累计-20%, HS300累计-25%"""
    ps, pe = window
    Rs = R.loc[ps:pe].copy()
    us_avg = (Rs["159941"] + Rs["513500"]) / 2
    cn_avg = (Rs["159232"] + Rs["515100"] + Rs["159952"]) / 3
    k_us = scale_k(us_avg, us_target)
    k_cn = scale_k(cn_avg, cn_target)
    from data_prep import DATA_DIR
    idx = pd.read_csv(os.path.join(DATA_DIR, "index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
    idx_w = idx.loc[ps:pe].pct_change().dropna()
    idx_w = idx_w.reindex(Rs.index).ffill().fillna(0.0)
    k_idx = scale_k(idx_w, idx_target)
    synth = Rs.copy()
    for s in ["159941", "513500"]:
        synth[s] = Rs[s] * k_us
    for s in ["159232", "515100", "159952"]:
        synth[s] = Rs[s] * k_cn
    lvl0 = float(idx.loc[ps:ps].iloc[0]) if (idx.loc[ps:ps]).shape[0] else 1000.0
    s_idx = (1 + k_idx * idx_w).cumprod() * lvl0
    print(f"  [合成] k_us={k_us:.3f}(窗口累计{(1+k_us*us_avg).prod()-1:+.2%}) "
          f"k_cn={k_cn:.3f}(累计{(1+k_cn*cn_avg).prod()-1:+.2%}) k_idx={k_idx:.3f}(累计{(1+k_idx*idx_w).prod()-1:+.2%})")
    return synth, s_idx

def run_scenario(name, ps, pe, R, a_mkt=None, dyn_cfg=None, min_delta=0.002, label="DYN", repo=0.018, tweights=None):
    ds = DynamicStrategy(R, cfg=dyn_cfg, a_mkt_override=a_mkt)
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=pe, name=label, min_delta=min_delta, repo=repo, tranche_weights=tweights)
    rows = {label: evaluate(res)}
    for bname, bw in BENCHMARKS.items():
        rb = run_backtest(R, fixed_weights=bw, start=ps, end=pe, name=bname, min_delta=0.0002)
        rows[bname] = evaluate(rb)
    return rows

def main():
    R, _ = build_panel("proxy")
    Rr, _ = build_panel("real")
    synth, s_idx = synthetic_resonance(Rr, ("2024-09-02", "2026-07-31"), -0.30, -0.20, -0.25)

    SCENARIOS = [
        ("牛市_2019-2021", "2019-01-04", "2021-02-18", R, None),
        ("牛市_真实窗口2025-2026", "2025-04-23", "2026-07-31", Rr, None),
        ("震荡市_2023-2024", "2023-01-03", "2024-08-30", R, None),
        ("熊市_2015股灾", "2015-06-15", "2016-02-29", R, None),
        ("熊市_2018", "2018-01-02", "2019-01-03", R, None),
        ("熊市_2021-2022", "2021-02-19", "2022-10-31", R, None),
        ("共振熊市_合成US-30+A股-20", "2024-09-02", "2026-07-31", synth, s_idx),
    ]

    results = {}
    for name, ps, pe, Rs, am in SCENARIOS:
        print(f"\n===== {name} ({ps} ~ {pe}) =====")
        r10 = run_scenario(name, ps, pe, Rs, a_mkt=am, dyn_cfg=CFG10, min_delta=0.002, label="DYN v10")
        r14 = run_scenario(name, ps, pe, Rs, a_mkt=am, dyn_cfg=CFG14, min_delta=0.02, label="DYN v15", repo=REPO)
        r17 = run_scenario(name, ps, pe, Rs, a_mkt=am, dyn_cfg=CFG17, min_delta=0.02, label="DYN v17", repo=REPO, tweights=CFG17.get("tranche_weights"))
        r18 = run_scenario(name, ps, pe, Rs, a_mkt=am, dyn_cfg=CFG18, min_delta=0.02, label="DYN v18", repo=REPO, tweights=CFG18.get("tranche_weights"))
        merged = {}
        for k, e in r10.items():
            merged[k] = e
        for k, e in r14.items():
            merged["v15_" + k] = e
        for k, e in r17.items():
            merged[k] = e  # 覆盖同名基准行(值相同), 新增 DYN v17 主行
        for k, e in r18.items():
            merged[k] = e  # 新增 DYN v18 主行
        results[name] = merged
        for k in ["DYN v10", "DYN v15", "DYN v17", "DYN v18"] + list(BENCHMARKS.keys()):
            if k in merged:
                e = merged[k]
                print(f"  {k:<18} CAGR={e['cagr']*100:7.2f}%  MDD={e['max_dd']*100:6.2f}%  "
                      f"Sharpe={e['sharpe']:.2f}  Calmar={e['calmar']:.2f}  total={e['total_ret']*100:8.2f}%")
            else:
                e = merged.get("v15_" + k)
                if e:
                    print(f"  v15_{k:<14} CAGR={e['cagr']*100:7.2f}%  MDD={e['max_dd']*100:6.2f}%  "
                          f"Sharpe={e['sharpe']:.2f}  Calmar={e['calmar']:.2f}  total={e['total_ret']*100:8.2f}%")
    json.dump(results, open(f"{OUT}/stress_test.json", "w"), ensure_ascii=False, indent=2, default=str)
    print("\n[ok] out/stress_test.json")

if __name__ == "__main__":
    main()
