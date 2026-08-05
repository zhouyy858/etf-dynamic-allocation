# -*- coding: utf-8 -*-
"""v42 探索: QDII溢价成本实测(2026-08-05, 研究未定稿)
问题: 回测收益按QDII单位净值计, 但实盘买的是场内价(含溢价/折价)。
本实验用 qdii_price_*.csv 场内价重跑 v29, 量化"净值口径 vs 场内价口径"的偏差,
并分解溢价门控在两种口径下的真实价值, 以及真实窗口收益归因。

发现摘要(详见 out/exp_v42_premium_cost.json):
1. 真实窗口(2025-04-23~2026-07-31): 净值 28.15%/-3.49%/Cal8.06 → 场内价 25.22%/-4.06%/Cal6.21
   (同harness对照; 发布值28.82%/-3.59%为官方口径)。溢价修正后 CAGR -2.9pp、MDD +0.6pp。
2. 全历史(2014-06-23起, 剔除份额拆分伪收益日): 净值 11.70%/-10.46%/1.12 → 场内价 10.17%/-12.13%/0.84。
   MDD加深主要是2015-16 QDII折价段与2022+溢价摆动, 非数据噪声。
3. 溢价门控价值(真实窗口): 净值空间 +10.5pp CAGR(17.6→28.2%) vs 场内价空间 +2.2pp(23.0→25.2%)
   —— 发布口径(v26: real Cal 6.60→7.49)在净值空间测量, 高估门控实盘价值约4-5倍;
   全历史场内价口径门控收益≈0(Sharpe +0.09), 门控价值是2025-26高溢价regime依赖的。
4. 归因(真实窗口): US成长年化12.4-12.7pp / CN成长7.8-10.3pp / CN防御2.1-2.2pp / 现金层0.8-1.0pp
   —— 收益主体是权益beta暴露(牛市), 现金层收益贡献小(其价值在流动性/风控)。
结论: 发布数字应注明"净值口径"; 前向验证/影子组合必须用场内价口径。
"""
import os, json
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "assets", "data")
OUT = os.path.join(ROOT, "out", "exp_v42_premium_cost.json")
CFG = json.load(open(os.path.join(ROOT, "references", "final_cfg_v29.json")))

# 份额折算/拆分伪收益日(场内价未复权, 剔除单日伪收益)
DROP_RET = {"159941": ["2022-07-05"], "513500": ["2022-03-30"]}


def price_rets(code):
    px = pd.read_csv(f"{DATA}/qdii_price_{code}.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
    r = px.pct_change().dropna()
    return r.drop([pd.Timestamp(d) for d in DROP_RET[code]], errors="ignore")


def prem_series(code):
    px = pd.read_csv(f"{DATA}/qdii_price_{code}.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
    nav = pd.read_csv(f"{DATA}/{code}_nav.csv", parse_dates=["date"]).set_index("date")["unit_nav"].sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    df = pd.concat([px.rename("px"), nav.rename("nav")], axis=1).dropna()
    if code == "159941":
        df = df.drop("2022-07-04", errors="ignore")
    return df["px"] / df["nav"] - 1.0


def prem_stats(p):
    return {"mean": float(p.mean()), "median": float(p.median()), "std": float(p.std()),
            "min": float(p.min()), "max": float(p.max()), "last": float(p.iloc[-1]),
            "pct_gt_5": float((p > 0.05).mean()), "pct_gt_3": float((p > 0.03).mean()),
            "n": int(len(p))}


def ar1_half_life(p):
    x = p.values[:-1]; y = p.values[1:]
    a = np.polyfit(x, y, 1)[0]
    hl = float(np.log(0.5) / np.log(a)) if 0 < a < 1 else float("inf")
    return {"ar1": float(a), "half_life_days": hl}


def main():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from data_prep import build_panel
    from strategy import DynamicStrategy
    from engine import run_backtest, evaluate

    R_orig, _ = build_panel("proxy")
    R_px = R_orig.copy()
    for code in ("159941", "513500"):
        R_px[code] = price_rets(code).reindex(R_px.index).combine_first(R_orig[code])
    nav511 = pd.read_csv(f"{DATA}/511010_nav.csv", parse_dates=["date"]).set_index("date")["cum_nav"].sort_index()
    bond = nav511.pct_change().dropna()

    def run_dyn(R, cfgx, start, end):
        ds = DynamicStrategy(R, cfg=cfgx)
        res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                           start=start, end=end, name="DYN",
                           min_delta=cfgx.get("min_delta", 0.02), repo=cfgx.get("repo_rate", 0.022),
                           tranche_weights=cfgx.get("tranche_weights"), cash_bond_rets=bond,
                           cash_bond_pct=cfgx.get("cash_bond_pct", 0.0),
                           rebal_weekday=cfgx.get("rebal_weekday", 2), rebal_freq=cfgx.get("rebal_freq", "weekly"))
        return evaluate(res), res

    cfg_off = dict(CFG); cfg_off["premium_gate"] = False; cfg_off["premium_tilt"] = False

    out = {"version": "v42", "date": "2026-08-05", "note": "QDII溢价成本实测: 净值口径vs场内价口径",
           "premium": {}, "backtests": {}, "attribution": {}}

    # 1) 溢价序列统计
    for code in ("159941", "513500"):
        p = prem_series(code)
        seg = {}
        for lab, sl in [("full", None), ("since2022", "2022-08-01"), ("real_window", "2025-04-23")]:
            s = p if sl is None else p[p.index >= sl]
            if len(s):
                seg[lab] = prem_stats(s)
        seg["ar1"] = ar1_half_life(p)
        out["premium"][code] = seg

    # 2) 净值 vs 场内价 回测(门控开/关 × 两窗口)
    combos = [("nav_gate_on", R_orig, CFG), ("nav_gate_off", R_orig, cfg_off),
              ("px_gate_on", R_px, CFG), ("px_gate_off", R_px, cfg_off)]
    for wlab, s, e in [("real", "2025-04-23", "2026-07-31"), ("full", "2014-06-23", "2026-07-31")]:
        out["backtests"].setdefault(wlab, {})
        for lab, R, c in combos:
            d, _ = run_dyn(R, c, s, e)
            out["backtests"][wlab][lab] = {k: float(d[k]) for k in
                                           ("cagr", "vol", "max_dd", "sharpe", "calmar", "avg_cash", "turnover")}

    # 3) 归因(真实窗口, 日终权重×次日收益)
    for lab, R in [("nav", R_orig), ("px", R_px)]:
        _, res = run_dyn(R, CFG, "2025-04-23", "2026-07-31")
        wdf = res["weights"]; rets = res["rets"]
        r = R.reindex(rets.index).fillna(0.0)
        cash_y = (1 - CFG["cash_bond_pct"]) * CFG["repo_rate"] / 252 + \
                 CFG["cash_bond_pct"] * bond.reindex(rets.index).fillna(0.0)
        w = wdf.drop(columns=["cash"]).shift(1); wc = wdf["cash"].shift(1)
        blocks = {"us_growth": ["159941", "513500"], "cn_growth": ["159952"],
                  "cn_defense": ["159232", "515100"], "cash": None}
        contrib = {}
        for b, cols in blocks.items():
            if cols is None:
                contrib[b] = float((wc * cash_y).sum())
            else:
                contrib[b] = float(sum((w[c] * r[c]).sum() for c in cols))
        years = len(rets) / 252
        out["attribution"][lab] = {"years": years,
                                   "total_ret": float((1 + rets).prod() - 1),
                                   "annualized_pp": {b: float(c / years * 100) for b, c in contrib.items()},
                                   "cumulative_pp": {b: float(c * 100) for b, c in contrib.items()}}

    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved:", OUT)
    # 打印关键对照
    for wlab in ("real", "full"):
        b = out["backtests"][wlab]
        print(f"\n== {wlab}")
        for lab in ("nav_gate_on", "nav_gate_off", "px_gate_on", "px_gate_off"):
            d = b[lab]
            print(f"  {lab}: CAGR={d['cagr']:.2%} MDD={d['max_dd']:.2%} Sharpe={d['sharpe']:.2f} Cal={d['calmar']:.2f} 现金={d['avg_cash']:.1%} 换手={d['turnover']:.1f}")


if __name__ == "__main__":
    main()
