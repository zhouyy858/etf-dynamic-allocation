# -*- coding: utf-8 -*-
"""回测引擎: 静态基准 + 动态策略
统一约束: 每周三执行调仓; 分笔由tranche_weights配置(默认等分3笔: 决策当周及随后2个周三各1/3;
v17用[1.0]=决策当日1笔全部成交); 紧急风控任意交易日触发并取代未完成计划; 单边费万5; 闲置现金享逆回购收益
动态策略: 15%国内底仓 + 15%海外底仓固定不动; 弹性仓按牛熊信号切换; 波动率+回撤刹车"""
import numpy as np, pandas as pd
from metrics import TRADING_DAYS, annualized_ret, annualized_vol, max_drawdown, sharpe

FEE = 0.0005
REPO = 0.018
TRANCHES = 3
SLOTS = ["159232", "515100", "159941", "513500", "159952"]

def run_backtest(R, target_weights_fn=None, daily_override_fn=None, fixed_weights=None,
                 start=None, end=None, tranches=TRANCHES, fee=FEE, repo=REPO,
                 min_delta=0.002, name="", tranche_weights=None):
    """tranche_weights: 分笔比例列表(如[1.0]立即执行、[0.5,0.5]、[1/3]*3), 默认None=等分tranches笔"""
    if tranche_weights is None:
        tw = np.array([1.0 / tranches] * tranches)
    else:
        tw = np.array(tranche_weights, float); tw = tw / tw.sum()
    """顺序模拟回测。
    fixed_weights: 静态权重, 每周三调回目标(分3周三笔)。
    target_weights_fn(dt, R_up_to_prev, ctx): 动态目标权重(dict, 含cash)或None。
    分笔规则: 每次决策的目标缺口等分tranches笔, 第1笔决策当日成交, 其余在随后(tranches-1)个
    周三各成交1笔 -> 每周三调仓、三周调整完; 若上一计划仍有未执行分笔, 常规决策顺延(计划不重叠),
    紧急风控可取代未完成计划并立即重启新的3周三笔计划。
    """
    R = R.copy()
    if start: R = R[R.index >= start]
    if end: R = R[R.index <= end]
    R = R.ffill().fillna(0.0)
    dates = R.index
    n = len(dates)
    k = len(SLOTS)
    repo_d = (1 + repo) ** (1 / TRADING_DAYS) - 1

    def fixed_target():
        t = np.array([fixed_weights[s] for s in SLOTS], dtype=float)
        return t / t.sum()

    w = np.zeros(k)
    pf_rets_ctx = []
    weights_history = np.zeros((n, k + 1))
    rets = np.zeros(n)
    turnover_day = np.zeros(n)
    pending = {}

    def next_wed(i):
        j = i + 1
        while j < n and dates[j].weekday() != 2:
            j += 1
        return j if j < n else None

    for i in range(n):
        dt = dates[i]
        # 1) 执行本日到期的分笔
        for delta in pending.pop(i, []):
            w = w + delta
            turnover_day[i] += np.abs(delta).sum()
        scheduled = sum(len(v) for v in pending.values())
        # 2) 决策: 首日建仓/每周三常规调仓/日度紧急刹车 (决策用截至前一日数据, 无未来函数)
        is_rebal = (i == 0) or (dt.weekday() == 2)
        target = None
        if is_rebal:
            if fixed_weights is not None:
                target = fixed_target()
            elif target_weights_fn is not None:
                ctx = ({"pf_rets": pd.Series(pf_rets_ctx, index=dates[:len(pf_rets_ctx)])}
                       if pf_rets_ctx else {"pf_rets": pd.Series(dtype=float)})
                ctx["equity"] = float(w.sum())
                ctx["weights"] = w.copy()
                tgt = target_weights_fn(dt, R.iloc[:i], ctx)
                if tgt is not None:
                    target = np.array([tgt[s] for s in SLOTS], dtype=float)
        is_emergency = False
        if daily_override_fn is not None:
            ctx = ({"pf_rets": pd.Series(pf_rets_ctx, index=dates[:len(pf_rets_ctx)])}
                   if pf_rets_ctx else {"pf_rets": pd.Series(dtype=float)})
            ctx["equity"] = float(w.sum())
            ctx["weights"] = w.copy()
            tgt2 = daily_override_fn(dt, R.iloc[:i], ctx)
            if tgt2 is not None:
                target = np.array([tgt2[s] for s in SLOTS], dtype=float)
                is_emergency = True
        if target is not None:
            # 常规决策: 上一计划未执行完毕则顺延, 保证同一时点只有一份计划、三周内调整完
            if not is_emergency and scheduled > 0:
                target = None
        if target is not None:
            total = target.sum()
            if total > 1.0 + 1e-9:
                target = target / total
            delta = target - w
            if i == 0 or np.abs(delta).max() >= min_delta:
                if is_emergency:
                    pending = {}  # 紧急风控取代未完成计划
                # 分笔: 第1笔当日执行(按当日净值), 其余在随后各决策周三分笔(比例按tranche_weights)
                d0 = delta * tw[0]
                w = w + d0
                turnover_day[i] += np.abs(d0).sum()
                j = i
                for t in range(1, len(tw)):
                    j = next_wed(j)
                    if j is None:
                        break
                    pending.setdefault(j, []).append(delta * tw[t])
        # 3) 组合收益(含费用), 并按当日收益更新权重漂移
        cash = 1.0 - w.sum()
        r = R.iloc[i].values
        fee_today = turnover_day[i] * fee
        g = w * (1.0 + r)          # 权益增长
        c = cash * (1.0 + repo_d)  # 现金增长
        pf_ret = float(g.sum() + c - 1.0 - fee_today)
        factor = 1.0 + pf_ret
        w = g / factor
        cash = c / factor
        weights_history[i] = np.concatenate([w, [cash]])
        rets[i] = pf_ret
        pf_rets_ctx.append(pf_ret)

    rs = pd.Series(rets, index=dates)
    W = (1 + rs).cumprod()
    wdf = pd.DataFrame(weights_history, index=dates, columns=SLOTS + ["cash"])
    return {"rets": rs, "wealth": W, "weights": wdf, "name": name,
            "turnover": float(turnover_day.sum()), "schedule_n": int(sum(len(v) for v in pending.values()))}

def evaluate(result, rf=REPO, periods=None):
    r, w = result["rets"], result["wealth"]
    mdd, pk, tr = max_drawdown(w)
    out = {"name": result.get("name", ""), "cagr": annualized_ret(r), "vol": annualized_vol(r),
           "max_dd": mdd, "mdd_peak": pk, "mdd_trough": tr,
           "sharpe": sharpe(r, rf),
           "calmar": (annualized_ret(r) / abs(mdd) if mdd == mdd and mdd < 0 else np.nan),
           "total_ret": w.iloc[-1] - 1, "final_wealth": w.iloc[-1],
           "turnover": result["turnover"], "avg_cash": float(result["weights"]["cash"].mean()),
           "avg_equity": float((1 - result["weights"]["cash"]).mean())}
    if periods:
        out["periods"] = {}
        for pn, (ps, pe) in periods.items():
            sub = r[(r.index >= ps) & (r.index <= pe)]
            if len(sub) > 5:
                sw = (1 + sub).cumprod()
                out["periods"][pn] = {"cagr": annualized_ret(sub), "vol": annualized_vol(sub),
                                      "max_dd": max_drawdown(sw)[0], "total": sw.iloc[-1] - 1}
    return out

def fmt_eval(e):
    return (f"{e['name']:<18} CAGR={e['cagr']*100:6.2f}%  Vol={e['vol']*100:5.2f}%  "
            f"MDD={e['max_dd']*100:6.2f}%  Sharpe={e['sharpe']:5.2f}  Calmar={e['calmar']:5.2f}  "
            f"Cash={e['avg_cash']*100:4.1f}%  换手={e['turnover']:.2f}")
