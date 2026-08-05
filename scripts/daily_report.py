# -*- coding: utf-8 -*-
"""每日行情日报: 最新信号 / 实际仓位vs目标 / 今日动作 / 组合表现
用法: python3 scripts/daily_report.py [输出md路径]
数据目录: 默认 skill 的 assets/data, 可用环境变量 ETF_DATA_DIR 覆盖"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel, read_table, rets_from
from data_prep import DATA_DIR
from engine import run_backtest
from strategy import DynamicStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = (os.path.join(HERE, "..", "references", "final_cfg_v28.json") if os.path.exists(os.path.join(HERE, "..", "references", "final_cfg_v28.json")) else os.path.join(HERE, "references", "final_cfg_v28.json"))
NAMES = {"159232": "自由现金流", "515100": "红利低波100", "159941": "纳指100",
         "513500": "标普500", "159952": "创业板"}
SLOTS = ["159232", "515100", "159941", "513500", "159952"]
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

def pct(x, d=1):
    return f"{x * 100:.{d}f}%"

def nav_last_date():
    import pandas as _pd
    mx = None
    for f in ["159232_nav.csv", "515100_nav.csv", "159941_nav.csv", "513500_nav.csv",
              "159952_nav.csv", "270042_nav.csv", "050025_nav.csv"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            d = str(_pd.read_csv(p, usecols=[0]).iloc[-1, 0])
            if mx is None or d > mx:
                mx = d
    return mx

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = json.load(open(CFG_FILE))
    R, W = build_panel("proxy")
    ds = DynamicStrategy(R, cfg=cfg)
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav") if cfg.get("cash_bond_pct") else None
    res = run_backtest(R, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(), start="2014-06-23", min_delta=cfg.get("min_delta", 0.02), repo=cfg.get("repo_rate", 0.022),
                       tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 2), rebal_freq=cfg.get("rebal_freq", "weekly"), strict=True)
    wdf, rets = res["weights"], res["rets"]
    wealth = (1 + rets).cumprod()
    last = wdf.index[-1]
    nav_last = nav_last_date()
    today = pd.Timestamp.now().normalize()
    w_act = wdf.iloc[-1]
    # 实盘目标: 最新数据日=T-1收盘; 用signal_lag=0实例在T-1数据上计算"下一个周三决策目标"
    # (等价于周三早盘用前一日数据决策; 历史统计段仍用cfg内signal_lag=1的严格口径)
    ds_live = DynamicStrategy(R, cfg=dict(cfg, signal_lag=0))
    tgt = ds_live.regular_target(last, {"pf_rets": rets})
    sc = ds.state_log[-1][1] if ds.state_log else 0

    lines = [f"# ETF动态配置日报 {today:%Y-%m-%d}（{WEEKDAY_CN[today.weekday()]}）", ""]
    lines.append(f"- **数据截至**: {nav_last or last.date()}（净值口径，最新收盘；决策信号使用截至前一日数据，无未来函数）")
    lines.append(f"- **有效打分**: {sc}/9 ｜ CN深熊锁={'有' if ds._lock['CN'] else '无'} ｜ US深熊锁={'有' if ds._lock['US'] else '无'}")
    for mkt in ["CN", "US"]:
        dd, ok20, ok20_60, rec, ok120 = ds.sig.mkt_info(last, mkt, ds.gate_win)
        lines.append(f"- **{mkt}市场**: 3年窗回撤 {pct(dd)} ｜ 站上SMA20={'是' if ok20 else '否'} ｜ "
                     f"双均线多头={'是' if ok20_60 else '否'} ｜ SMA{ds.gate_win}={'是' if ok120 else '否'} ｜ 恢复度{rec * 100:.0f}%")
    lines.append("")
    lines.append("## 仓位现状（实际 vs 目标）")
    lines.append("| 标的 | 实际 | 目标 | 偏差 |")
    for s in SLOTS:
        lines.append(f"| {s} {NAMES[s]} | {pct(w_act[s])} | {pct(tgt[s])} | {pct(w_act[s] - tgt[s])} |")
    lines.append(f"| 现金(逆回购) | {pct(w_act['cash'])} | {pct(tgt['cash'])} | {pct(w_act['cash'] - tgt['cash'])} |")
    lines.append("")
    rebal_wd = int(cfg.get("rebal_weekday", 2))
    is_rebal_day = today.weekday() == rebal_wd
    gap = max(abs(w_act[s] - tgt[s]) for s in SLOTS + ["cash"])
    if is_rebal_day:
        action = f"今日是周{WEEKDAY_CN[rebal_wd][1]}调仓日：按 v28 规则用前一日(T-1)收盘信号决策，目标缺口 1 笔当日收盘成交；QDII 溢价用 T-2 口径（>3% 注意、>5% 买入暂缓），下单前查 IOPV"
    elif gap > 0.02:
        action = f"非调仓日，仓位与目标基本一致（最大偏差 {pct(gap)}），等待下一个周{WEEKDAY_CN[rebal_wd][1]}检查"
    else:
        action = "非调仓日，仓位与目标基本一致，仅观察"
    # ---- QDII 溢价监控 (场内价/单位净值-1) ----
    prem_info = []
    for code, nm in [("159941", "纳指100"), ("513500", "标普500")]:
        try:
            px = pd.read_csv(os.path.join(DATA_DIR, f"qdii_price_{code}.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
            nav = pd.read_csv(os.path.join(DATA_DIR, f"{code}_nav.csv"), parse_dates=["date"]).set_index("date")
            nav = nav[~nav.index.duplicated(keep="last")].sort_index()["unit_nav"]
            df = pd.concat([px.rename("px"), nav.rename("nav")], axis=1).dropna()
            if code == "159941" and "2022-07-04" in df.index:
                df = df.drop("2022-07-04")
            prem = float(df["px"].iloc[-1] / df["nav"].iloc[-1] - 1)
            flag = "⚠️高溢价,买入暂缓" if prem > 0.05 else ("注意" if prem > 0.03 else "正常")
            prem_info.append(f"{nm}({code}) {pct(prem, 2)} {flag}")
        except Exception:
            pass
    if prem_info:
        lines.append("## QDII溢价监控（场内买入成本）")
        lines.append("")
        for t in prem_info:
            lines.append(f"- {t}")
        lines.append("")
    lines.append("## 今日动作")
    lines.append("")
    lines.append(action)
    lines.append("")
    last_ret = rets.iloc[-1]
    r5 = float((1 + rets.iloc[-5:]).prod() - 1)
    r20 = float((1 + rets.iloc[-20:]).prod() - 1)
    y0 = rets.index.max().year
    ytd_mask = rets.index >= f"{y0}-01-01"
    ytd = float((1 + rets[ytd_mask]).prod() - 1) if ytd_mask.any() else np.nan
    cur_dd = float(wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1)
    lines.append("## 组合表现")
    lines.append("")
    lines.append(f"- 上一交易日: {pct(last_ret, 2)} ｜ 近5日: {pct(r5, 2)} ｜ 近20日: {pct(r20, 2)}")
    lines.append(f"- 今年以来: {pct(ytd, 2) if ytd == ytd else 'n/a'} ｜ 当前距历史高点: {pct(cur_dd, 2)}")
    lines.append("")
    lines.append("> 生成: ETF动态配置skill v28（周五前日信号决策+1笔当日成交+溢价T-2门控+相对溢价倾斜+溢价门控削减增强+相关性风控+速度刹车(恢复期12日)+逆回购2.2%+国债ETF 511010七成半+前提驱动v28定稿）｜ 无未来函数口径｜ 仅供参考，非投资建议")
    text = "\n".join(lines)
    print(text)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[ok] 已保存: {out_path}")

if __name__ == "__main__":
    main()
