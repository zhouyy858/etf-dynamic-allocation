# -*- coding: utf-8 -*-
"""执行方案网格搜索: 决策日(周几) × 频率(周/双周/月) × 分笔数(1-4) × 每笔比例
策略参数固定 v15, 只改执行纪律; 双窗口评估(proxy全史 + real 2025-04起)
理论依据: 分笔=过渡期DCA(摊薄入场/延迟降仓), 频率=信号新鲜度vs换手,
紧急风控(快刹车/深熊锁/速度刹车)仍逐日触发并分N笔在后续决策日完成
"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import build_panel
from engine import TRADING_DAYS
from strategy import DynamicStrategy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = json.load(open("out/final_cfg_v15.json"))
SLOTS = ["159232", "515100", "159941", "513500", "159952"]
FEE, REPO = 0.0005, 0.022

def decision_mask(dates, cadence, weekday=None, month_day=None):
    n = len(dates); mask = np.zeros(n, bool)
    if cadence == "weekly":
        for i, dt in enumerate(dates):
            mask[i] = (dt.weekday() == weekday)
    elif cadence == "biweekly":
        for i, dt in enumerate(dates):
            mask[i] = (dt.weekday() == weekday) and (dt.isocalendar()[1] % 2 == 0)
    elif cadence == "monthly_first":
        prev = None
        for i, dt in enumerate(dates):
            if prev is None or dt.month != prev:
                mask[i] = True
            prev = dt.month
    elif cadence == "monthly_mid":
        # 每月离15日最近的交易日
        cur_m = None; best = None
        for i, dt in enumerate(dates):
            if dt.month != cur_m:
                cur_m = dt.month; best = i
            else:
                if abs(dt.day - 15) < abs(dates[best].day - 15):
                    best = i
            mask[best] = True
        # 清掉上月的残留(粗处理: 重新构建)
        for i in range(n):
            if dates[i].month != dates[best_of_month(dates, i)].month:
                pass
    return mask

def best_of_month(dates, i):
    m = dates[i].month; b = i
    for j in range(i, -1, -1):
        if dates[j].month != m: break
        b = j
    return b

def decision_mask_mid(dates):
    n = len(dates); mask = np.zeros(n, bool)
    cur_m = None; best = None
    for i, dt in enumerate(dates):
        if dt.month != cur_m:
            if best is not None: mask[best] = True
            cur_m = dt.month; best = i
        else:
            if abs(dt.day - 15) < abs(dates[best].day - 15):
                best = i
    if best is not None: mask[best] = True
    return mask

def run_schedule(R, cfg, start, dmask, tweights):
    """泛化执行: dmask=决策日掩码, tweights=每笔比例(list, 和为1)"""
    R_full = R.copy()                       # 策略用全历史面板(2006起), 与现行框架一致
    R = R_full.copy()
    if start: R = R[R.index >= start]       # 引擎循环用切片
    dmask = np.asarray(dmask, bool)[R_full.index >= start] if start else np.asarray(dmask, bool)
    R = R.ffill().fillna(0.0)
    dates = R.index; n = len(dates)
    dmask = dmask.copy()
    if n: dmask[0] = True  # 首日建仓(与现行引擎一致)
    tw = np.array(tweights, float); tw = tw / tw.sum()
    ds = DynamicStrategy(R_full, cfg=cfg)   # 策略看原始全历史面板(NaN保留)
    tf, df = ds.target_fn(), ds.daily_fn()
    w = np.zeros(len(SLOTS)); pf = []
    w_hist = np.zeros((n, len(SLOTS) + 1)); rets = np.zeros(n); to_day = np.zeros(n)
    pending = {}
    repo_d = (1 + REPO) ** (1 / TRADING_DAYS) - 1
    for i in range(n):
        dt = dates[i]
        for delta in pending.pop(i, []):
            w = w + delta; to_day[i] += np.abs(delta).sum()
        scheduled = sum(len(v) for v in pending.values())
        target = None; is_emergency = False
        ctx = ({"pf_rets": pd.Series(pf, index=dates[:len(pf)])} if pf else {"pf_rets": pd.Series(dtype=float)})
        ctx["equity"] = float(w.sum()); ctx["weights"] = w.copy()
        if dmask[i]:
            t0 = tf(dt, R.iloc[:i], ctx)
            if t0 is not None: target = np.array([t0[s] for s in SLOTS], float)
        e0 = df(dt, R.iloc[:i], ctx)
        if e0 is not None:
            target = np.array([e0[s] for s in SLOTS], float); is_emergency = True
        if target is not None:
            if not is_emergency and scheduled > 0:
                target = None
        if target is not None:
            total = target.sum()
            if total > 1.0 + 1e-9: target = target / total
            delta = target - w
            if i == 0 or np.abs(delta).max() >= 0.02:
                if is_emergency: pending = {}
                d0 = delta * tw[0]
                w = w + d0; to_day[i] += np.abs(d0).sum()
                j = i
                for t in range(1, len(tw)):
                    j = j + 1
                    while j < n and not dmask[j]: j += 1
                    if j >= n: break
                    pending.setdefault(j, []).append(delta * tw[t])
        cash = 1.0 - w.sum()
        r = R.iloc[i].values
        g = w * (1 + r); c = cash * (1 + repo_d)
        pf_ret = float(g.sum() + c - 1 - to_day[i] * FEE)
        factor = 1 + pf_ret
        w = g / factor; cash = c / factor
        w_hist[i] = np.concatenate([w / (w.sum() + cash), [cash / (w.sum() + cash)]])
        rets[i] = pf_ret; pf.append(pf_ret)
    rs = pd.Series(rets, index=dates); W = (1 + rs).cumprod()
    wdf = pd.DataFrame(w_hist, index=dates, columns=SLOTS + ["cash"])
    out = {"name": "", "rets": rs, "wealth": W, "weights": wdf,
           "turnover": float(to_day.sum()), "avg_cash": float(wdf["cash"].mean())}
    return out, ds

def metrics(res):
    r, wv = res["rets"], res["wealth"]
    mdd = float((wv / wv.cummax() - 1).min())
    cagr = float(wv.iloc[-1] ** (TRADING_DAYS / max(len(r), 1)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0
    return dict(cagr=cagr * 100, mdd=mdd * 100, sharpe=sharpe, calmar=calmar,
                to=res["turnover"], cash=res["avg_cash"] * 100)

def main():
    R_P, _ = build_panel("proxy"); R_R, _ = build_panel("real")
    rows = []
    cadences = []
    for wd, nm in [(0, "周一"), (1, "周二"), (2, "周三"), (3, "周四"), (4, "周五")]:
        cadences.append((f"周频-{nm}", lambda d, wd=wd: decision_mask(d, "weekly", weekday=wd)))
    for wd, nm in [(0, "周一"), (2, "周三"), (4, "周五")]:
        cadences.append((f"双周-{nm}", lambda d, wd=wd: decision_mask(d, "biweekly", weekday=wd)))
    cadences.append(("月频-月初", lambda d: decision_mask(d, "monthly_first")))
    cadences.append(("月频-月中", decision_mask_mid))
    for tw, twn in [([1.0], "1笔"), ([0.5, 0.5], "2笔"), ([1/3]*3, "3笔"), ([0.25]*4, "4笔")]:
        for cname, fn in cadences:
            dm = fn(R_P.index)
            p = run_schedule(R_P, BASE, "2014-06-23", dm, tw)[0]
            dmr = fn(R_R.index)
            rr = run_schedule(R_R, BASE, "2025-04-23", dmr, tw)[0]
            mp, mr = metrics(p), metrics(rr)
            rows.append({"variant": f"{cname}-{twn}", "p": mp, "r": mr})
    rows.append({"variant": "现版周三-3笔", "p": {"cagr": 13.49, "mdd": -11.91, "sharpe": 1.28, "calmar": 1.13, "to": 59.7, "cash": 38.8},
                 "r": {"cagr": 27.24, "mdd": -5.61, "sharpe": 2.28, "calmar": 4.86, "to": 7.9, "cash": 21.5}})
    # 按 proxy Calmar 排序
    rows.sort(key=lambda x: x["p"]["calmar"], reverse=True)
    print(f"{'方案':16s} | {'proxy CAGR':>10s}{'MDD':>8s}{'Cal':>7s}{'TO':>6s} | {'real CAGR':>10s}{'MDD':>8s}{'Cal':>7s}{'TO':>5s}")
    for r in rows[:16]:
        p, rr = r["p"], r["r"]
        print(f"{r['variant']:16s} | {p['cagr']:9.2f}%{p['mdd']:8.2f}{p['calmar']:7.2f}{p['to']:6.1f} | {rr['cagr']:9.2f}%{rr['mdd']:8.2f}{rr['calmar']:7.2f}{rr['to']:5.1f}")
    json.dump(rows, open(os.path.join(OUT, "exp_schedule_search.json"), "w"), ensure_ascii=False, indent=1)
    print("[ok] out/exp_schedule_search.json")

if __name__ == "__main__":
    main()
