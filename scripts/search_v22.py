# -*- coding: utf-8 -*-
"""v22 全参数重训搜索 (无未来函数强制: signal_lag>=1 硬编码 + accrual=pre + 溢价T-2)
阶段:
  S 调仓时间: rebal_weekday(周一~五) x rebal_freq(每周/隔周/每月) x 分笔数(1/2/3)
  F 底仓比例: cn x us 网格 (0~15%)
  P 策略参数: 坐标下降 + 随机探索 (打分骨架/滞回/波动率/回撤上限/刹车阈值/动量/相关性/溢价/估值)
  G 风控框架: 各风控开关 + 标的剔除 (标的选择)
评分: proxy 全历史 Calmar 优先; 约束 proxy MDD>=-10% 且 real CAGR>=18% 且 real Calmar>=3.0
用法: python3 scripts/search_v22.py <stage> [rounds]
输出: out/search_v22_<stage>.json
"""
import sys, os, json, copy, itertools, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_REF = os.path.join(HERE, "..", "references")
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
CFG0 = json.load(open(f"{SKILL_REF}/final_cfg_v21.json"))
CFG0["signal_lag"] = 1; CFG0["premium_shift"] = 2

Rg = Rrg = BONDg = None
def _init():
    global Rg, Rrg, BONDg
    from data_prep import build_panel, read_table, rets_from
    Rg, _ = build_panel("proxy"); Rrg, _ = build_panel("real")
    BONDg = rets_from(read_table("511010_nav.csv"), "cum_nav")

def eval_one(args):
    cfg, weekday, freq, tw = args
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    c = dict(cfg)
    ds = DynamicStrategy(Rg, cfg=c)
    res = run_backtest(Rg, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start="2014-06-23", min_delta=c.get("min_delta", 0.02), repo=0.022,
                       tranche_weights=tw, cash_bond_rets=BONDg, cash_bond_pct=c.get("cash_bond_pct", 0.0),
                       rebal_weekday=weekday, rebal_freq=freq, strict=True)
    e = evaluate(res)
    ds2 = DynamicStrategy(Rrg, cfg=c)
    res2 = run_backtest(Rrg, target_weights_fn=ds2.target_fn(), daily_override_fn=ds2.daily_fn(),
                        start="2025-04-23", min_delta=c.get("min_delta", 0.02), repo=0.022,
                        tranche_weights=tw, cash_bond_rets=BONDg, cash_bond_pct=c.get("cash_bond_pct", 0.0),
                        rebal_weekday=weekday, rebal_freq=freq, strict=True)
    er = evaluate(res2)
    return {"cfg": c, "weekday": weekday, "freq": freq, "tw": tw,
            "p_cagr": e["cagr"], "p_mdd": e["max_dd"], "p_sharpe": e["sharpe"], "p_calmar": e["calmar"],
            "p_to": e["turnover"], "p_cash": e["avg_cash"],
            "r_cagr": er["cagr"], "r_mdd": er["max_dd"], "r_sharpe": er["sharpe"], "r_calmar": er["calmar"],
            "valid": (e["max_dd"] >= -0.10 and er["cagr"] >= 0.18 and er["calmar"] >= 3.0)}

def run_pool(tasks, workers=None):
    workers = workers or int(os.environ.get("SEARCH_WORKERS", "8"))
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init) as pool:
        return pool.map(eval_one, tasks)

def fmt(r):
    return (f"CAGR {r['p_cagr']*100:6.2f}% MDD {r['p_mdd']*100:6.2f}% Calmar {r['p_calmar']:.2f} "
            f"| real {r['r_cagr']*100:6.2f}%/{r['r_mdd']*100:6.2f}%/Cal{r['r_calmar']:.2f}")

def save(name, results):
    path = f"{OUT}/search_v22_{name}.json"
    json.dump(results, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"[ok] {path}  ({len(results)} 组)")

# ---------------- 阶段 S: 调仓时间 ----------------
def stage_schedule():
    tasks = []
    for wd in range(5):
        for freq in ("weekly", "biweekly", "monthly"):
            for tr in (1, 2, 3):
                tw = [1.0 / tr] * tr
                tasks.append((CFG0, wd, freq, tw))
    res = run_pool(tasks)
    best = sorted([r for r in res if r["valid"]], key=lambda r: (-r["p_calmar"], r["p_mdd"]))
    print(f"== S 调仓时间: {len(res)}组, 达标{len(best)}组 ==")
    for r in best[:12]:
        print(f"  周{'一二三四五'[r['weekday']]}-{r['freq']:<8} {len(r['tw'])}笔 {fmt(r)}")
    save("S_schedule", res)
    return best[0] if best else sorted(res, key=lambda r: -r["p_calmar"])[0]

# ---------------- 阶段 F: 底仓比例 ----------------
def stage_floor(base, tag="F_floor"):
    tasks = []
    for cn in (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0):
        for us in (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0):
            c = copy.deepcopy(base["cfg"])
            c["floor_pct"] = {"cn": cn, "us": us}
            tasks.append((c, base["weekday"], base["freq"], base["tw"]))
    res = run_pool(tasks)
    best = sorted([r for r in res if r["valid"]], key=lambda r: (-r["p_calmar"], r["p_mdd"]))
    print(f"== F 底仓[{tag}]: {len(res)}组, 达标{len(best)}组 ==")
    for r in best[:12]:
        fp = r["cfg"]["floor_pct"]
        print(f"  cn={fp['cn']:5.1f} us={fp['us']:5.1f} (总{fp['cn']+fp['us']:4.1f}) {fmt(r)}")
    save(tag, res)
    return res

def stage_floor_multi(indices, tag="F_floor_all"):
    """对多个S候选分别跑底仓网格, 合并后按 proxy Calmar 排序"""
    S = json.load(open(f"{OUT}/search_v22_S_schedule.json"))
    allres = []
    for idx in indices:
        base = S[idx]
        wd, freq, n = base["weekday"], base["freq"], len(base["tw"])
        res = stage_floor(base, tag=f"F_floor_{wd}{freq}{n}")
        for r in res:
            r["_s_idx"] = idx
        allres += res
    best = sorted([r for r in allres if r["valid"]], key=lambda r: (-r["p_calmar"], r["p_mdd"]))
    print(f"== F 全部候选合并: {len(allres)}组, 达标{len(best)}组 ==")
    for r in best[:15]:
        fp = r["cfg"]["floor_pct"]
        print(f"  S#{r['_s_idx']} 周{'一二三四五'[r['weekday']]}-{r['freq']:<8}{len(r['tw'])}笔 cn={fp['cn']:4.1f} us={fp['us']:4.1f} {fmt(r)}")
    save(tag, allres)
    return best[0] if best else sorted(allres, key=lambda r: -r["p_calmar"])[0]

# ---------------- 阶段 P: 策略参数 ----------------
PERTURB_GROUPS = [
    ("state_g9", lambda c, v: _mut_state_map(c, "g9", v), [78, 84, 88, 92]),
    ("state_d9", lambda c, v: _mut_state_map(c, "d9", v), [2, 3, 4, 6]),
    ("state_g0", lambda c, v: _mut_state_map(c, "g0", v), [6, 8, 11, 15]),
    ("state_d0", lambda c, v: _mut_state_map(c, "d0", v), [24, 27, 30, 34]),
    ("hyst_up", lambda c, v: _mut(c, "hyst_up", v), [0.50, 0.58, 0.66]),
    ("hyst_down", lambda c, v: _mut(c, "hyst_down", v), [0.12, 0.18, 0.24]),
    ("vol_target", lambda c, v: _mut(c, "vol_target", v), [0.15, 0.17, 0.19, 0.22]),
    ("vol_buf", lambda c, v: _mut(c, "vol_buf", v), [1.0, 1.15, 1.3]),
    ("min_cash", lambda c, v: _mut(c, "min_cash", v), [0.02, 0.03, 0.05]),
    ("max_eq", lambda c, v: _mut(c, "max_eq", v), [0.95, 0.97, 0.98, 0.99]),
    ("dd_cap_scale", lambda c, v: _mut_dd(c, v), [0.8, 1.0, 1.2]),
    ("dd_caps", lambda c, v: _mut_ddcaps(c, v), [[85, 75, 65, 55], [80, 65, 50, 35], [90, 80, 70, 60]]),
    ("mdd_cn_fb", lambda c, v: _mut_mdd(c, "CN", 0, v), [0.05, 0.07, 0.10]),
    ("mdd_cn_cut", lambda c, v: _mut_mdd(c, "CN", 1, v), [0.05, 0.08, 0.12]),
    ("mdd_cn_deep", lambda c, v: _mut_mdd(c, "CN", 2, v), [0.18, 0.22, 0.26]),
    ("mdd_cn_rec", lambda c, v: _mut_mdd(c, "CN", 3, v), [0.08, 0.10, 0.14]),
    ("mdd_us_fb", lambda c, v: _mut_mdd(c, "US", 0, v), [0.08, 0.10, 0.13]),
    ("mdd_us_cut", lambda c, v: _mut_mdd(c, "US", 1, v), [0.08, 0.12, 0.18]),
    ("mdd_us_deep", lambda c, v: _mut_mdd(c, "US", 2, v), [0.20, 0.24, 0.28]),
    ("mdd_us_rec", lambda c, v: _mut_mdd(c, "US", 3, v), [0.10, 0.15, 0.20]),
    ("gate_win", lambda c, v: _mut(c, "gate_win", int(v)), [100, 120, 140]),
    ("df_win", lambda c, v: _mut(c, "defense_momentum_win", int(v)), [40, 60, 80]),
    ("df_t", lambda c, v: _mut(c, "defense_momentum_t", v), [2.0, 3.0, 4.0]),
    ("df_clamp", lambda c, v: _mut_list(c, "defense_clamp", v), [[0.30, 0.70], [0.35, 0.65], [0.40, 0.60]]),
    ("corr_win", lambda c, v: _mut(c, "corr_risk_win", int(v)), [30, 40, 60]),
    ("corr_thr", lambda c, v: _mut_list(c, "corr_risk_thr", v), [[0.25, 0.35], [0.30, 0.42], [0.40, 0.50]]),
    ("corr_cut", lambda c, v: _mut_list(c, "corr_risk_cut", v), [[0.85, 0.65], [0.90, 0.75], [0.80, 0.60]]),
    ("prem_thr", lambda c, v: _mut_list(c, "premium_thr", v), [[0.03, 0.05, 0.08], [0.05, 0.08, 0.12]]),
    ("prem_cut", lambda c, v: _mut_list(c, "premium_cut", v), [[0.6, 0.3, 0.15], [0.7, 0.4, 0.2]]),
    ("spd_thr", lambda c, v: _mut(c, "speed_brake_thr", v), [-0.05, -0.04, -0.03]),
    ("spd_cut", lambda c, v: _mut(c, "speed_brake_cut", v), [0.5, 0.6, 0.7]),
    ("spd_rec", lambda c, v: _mut(c, "speed_brake_recover", int(v)), [5, 8, 12]),
    ("confirm", lambda c, v: _mut(c, "score_confirm_weeks", int(v)), [0, 1, 2]),
]

def _mut(c, k, v):
    c = copy.deepcopy(c); c[k] = v; return c
def _mut_list(c, k, v):
    c = copy.deepcopy(c); c[k] = list(v); return c
def _mut_state_map(c, key, v):
    c = copy.deepcopy(c); sm = {int(k): list(x) for k, x in c["state_map"].items()}
    if key == "g9": sm[9][0] = v
    if key == "d9": sm[9][1] = v
    if key == "g0": sm[0][0] = v
    if key == "d0": sm[0][1] = v
    c["state_map"] = sm; return c
def _mut_dd(c, v):
    c = copy.deepcopy(c); c["dd_eq_cap"] = [[round(t * v, 4), cap] for t, cap in c["dd_eq_cap"]]; return c
def _mut_ddcaps(c, v):
    c = copy.deepcopy(c); c["dd_eq_cap"] = [[t, int(cap)] for (t, _), cap in zip(c["dd_eq_cap"], v)]; return c
def _mut_mdd(c, mkt, idx, v):
    c = copy.deepcopy(c); md = dict(c["market_dd"]); row = list(md[mkt]); row[idx] = v; md[mkt] = row
    c["market_dd"] = md; return c

def stage_params(base, rounds=3):
    cur = copy.deepcopy(base)
    best = cur
    for rnd in range(rounds):
        improved = False
        for name, fn, vals in PERTURB_GROUPS:
            tasks = []
            for v in vals:
                cc = fn(cur["cfg"], v)
                tasks.append((cc, cur["weekday"], cur["freq"], cur["tw"]))
            res = run_pool(tasks)
            for r in res:
                if r["valid"] and (r["p_calmar"] > best["p_calmar"] + 1e-6 or
                                   (abs(r["p_calmar"] - best["p_calmar"]) <= 1e-6 and r["p_mdd"] > best["p_mdd"])):
                    best = r; improved = True
            if improved:
                cur = best
                print(f"  [r{rnd}] {name} -> {fmt(cur)}")
                improved = False
        print(f"== P 第{rnd+1}轮完成, 当前最优 {fmt(cur)}")
    return cur

def stage_random(base, n=150, seed=42):
    rng = random.Random(seed)
    tasks = []
    sm0 = {int(k): v for k, v in base["cfg"]["state_map"].items()}
    for _ in range(n):
        c = copy.deepcopy(base["cfg"])
        c["state_map"] = {str(k): [_interp(0, 9, k, sm0[0][0], sm0[9][0], rng),
                                    _interp(0, 9, k, sm0[0][1], sm0[9][1], rng)]
                           for k in range(10)}
        tasks.append((c, base["weekday"], base["freq"], base["tw"]))
    res = run_pool(tasks)
    best = sorted([r for r in res if r["valid"]], key=lambda r: (-r["p_calmar"], r["p_mdd"]))
    print(f"== P 随机探索: {len(res)}组, 达标{len(best)}组 ==")
    for r in best[:8]:
        print(f"  {fmt(r)}")
    save("P_random", res)
    return best[0] if best else sorted(res, key=lambda r: -r["p_calmar"])[0]

def _interp(lo, hi, k, v0, v9, rng):
    t = (k - lo) / (hi - lo)
    v = v0 + (v9 - v0) * t
    return int(round(v * (0.85 + 0.3 * rng.random())))

# ---------------- 阶段 G: 风控框架 + 标的选择 ----------------
def stage_framework(base):
    c = copy.deepcopy(base["cfg"])
    variants = []
    for name, fn in [
        ("去相关性风控", lambda cc: _mut(cc, "corr_risk", False)),
        ("去溢价门", lambda cc: _mut(cc, "premium_gate", False)),
        ("去速度刹车", lambda cc: _mut(cc, "speed_brake", False)),
        ("去防御动量", lambda cc: _mut(cc, "defense_momentum", False)),
        ("去估值门", lambda cc: _mut(cc, "valuation_gate", False)),
        ("回撤上限无条件", lambda cc: _mut(cc, "dd_cap_unconditional", True)),
        ("回撤上限需弱趋势", lambda cc: _mut(cc, "dd_cap_unconditional", False)),
        ("成长轮动开", lambda cc: _mut(cc, "growth_rotation", True)),
        ("恢复渐进开", lambda cc: _mut(cc, "recovery_ramp", True)),
        ("下行波动率", lambda cc: _mut(cc, "downside_vol", True)),
        ("剔除159952创业板", lambda cc: _mut(cc, "exclude", ["159952"])),
        ("剔除513500标普", lambda cc: _mut(cc, "exclude", ["513500"])),
        ("剔除159941纳指", lambda cc: _mut(cc, "exclude", ["159941"])),
        ("剔除159232现金流", lambda cc: _mut(cc, "exclude", ["159232"])),
        ("剔除515100红利", lambda cc: _mut(cc, "exclude", ["515100"])),
    ]:
        try:
            variants.append((name, fn(c)))
        except Exception as ex:
            print(f"  [skip] {name}: {ex}")
    tasks = [(vc, base["weekday"], base["freq"], base["tw"]) for _, vc in variants]
    res = run_pool(tasks)
    print(f"== G 框架变体: {len(res)}组 ==")
    for (name, _), r in zip(variants, res):
        mark = "  <= 最优" if r == max(res, key=lambda x: x["p_calmar"] if x["valid"] else -9) else ""
        print(f"  {name:<14} {fmt(r)}{mark}")
    save("G_framework", res)
    return res

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    t0 = time.time()
    base = None
    if stage in ("S", "all"):
        base = stage_schedule()
    if stage.startswith("Fm"):
        idxs = [int(x) for x in sys.argv[1].split(":")[1].split(",")] if ":" in stage else [0, 24, 31, 39, 36, 19]
        base = stage_floor_multi(idxs)
        print("[Fm] best ->", base.get("_s_idx"), fmt(base))
        return
    if stage.startswith("F:"):
        idx = int(stage.split(":")[1])
        base = json.load(open(f"{OUT}/search_v22_S_schedule.json"))[idx]
        stage_floor(base, tag=f"F_floor_S{idx}")
        return
    if stage in ("F", "all"):
        base = base or json.load(open(f"{OUT}/search_v22_S_schedule.json"))[0] if os.path.exists(f"{OUT}/search_v22_S_schedule.json") else stage_schedule()
        base = stage_floor(base)
    if stage.startswith("P:"):
        sel = stage.split(":")[1]
        sort_real = sel.startswith("r")
        idx = int(sel[1:]) if (sort_real or sel.startswith("p")) else int(sel)
        allf = json.load(open(f"{OUT}/search_v22_F_floor_all.json"))
        valid = [r for r in allf if r["valid"]]
        base = sorted(valid, key=lambda r: (-r["r_calmar"], -r["p_calmar"]) if sort_real else (-r["p_calmar"], r["p_mdd"]))[idx]
        fp = base["cfg"]["floor_pct"]
        print(f"[P] 起始: S#{base.get('_s_idx')} 周{'一二三四五'[base['weekday']]}-{base['freq']}{len(base['tw'])}笔 cn={fp['cn']} us={fp['us']} {fmt(base)}")
        base = stage_params(base, rounds)
        base = stage_random(base)
        save("P_best_" + ("real" if sort_real else "proxy"), base)
        return
        return
    if stage in ("P", "all"):
        base = base or json.load(open(f"{OUT}/search_v22_F_floor.json"))[0]
        base = stage_params(base, rounds)
        base = stage_random(base)
    if stage.startswith("G:"):
        tag = stage.split(":")[1]
        base = json.load(open(f"{OUT}/search_v22_P_best_{tag}.json"))
        stage_framework(base)
        return
    if stage in ("G", "all"):
        base = base or json.load(open(f"{OUT}/search_v22_P_random.json"))[0]
        stage_framework(base)
    print(f"[done] {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
