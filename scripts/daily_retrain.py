# -*- coding: utf-8 -*-
"""收盘后全链路重训(由 Codex 定时任务「每周训练计划」每周五 16:00 触发):
  ① 拉取最新数据(场内收盘价/指数当日值) ② 重建面板 ③ 全参数复检(8轴×邻域, 双窗口)
  ④ 三关验证(双窗口同向>0.15Cal + 平台平坦 + OOS不劣化) → 通过则升级新版本配置并记录,
     否则维持 v26 ⑤ 生成日报+更新README POSITIONS ⑥ git commit+push ⑦ 通知
防过拟合硬纪律: 拒绝孤峰/尖峰/单窗口改善, 只接受平台+双窗口+OOS 三关全过。
用法: python3 scripts/daily_retrain.py [--no-fetch] [--no-push]
"""
import os, sys, json, shutil, copy, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import daily_automation as da

PY = sys.executable
WORK = os.environ.get("ETF_REPORT_DIR", os.path.expanduser("~/ETF策略日报"))
DATA_DIR = os.path.join(WORK, "data")
LOG_DIR = os.path.join(WORK, "logs")
CFG_PATH = os.path.join(SKILL, "references", "final_cfg_v27.json")
OOS_START = "2022-01-04"   # WFO 样本外窗口起点

def _run_backtest(Rs, ps, cfg):
    import numpy as np, pandas as pd
    from data_prep import build_panel, read_table, rets_from
    from engine import run_backtest, evaluate
    from strategy import DynamicStrategy
    bond = rets_from(read_table("511010_nav.csv"), "cum_nav")
    ds = DynamicStrategy(Rs, cfg=cfg)
    res = run_backtest(Rs, target_weights_fn=ds.target_fn(), daily_override_fn=ds.daily_fn(),
                       start=ps, end=None, name="DYN", min_delta=cfg.get("min_delta", 0.02),
                       repo=cfg.get("repo_rate", 0.022), tranche_weights=cfg.get("tranche_weights"),
                       cash_bond_rets=bond, cash_bond_pct=cfg.get("cash_bond_pct", 0.0),
                       rebal_weekday=cfg.get("rebal_weekday", 4), rebal_freq=cfg.get("rebal_freq", "weekly"),
                       strict=True)
    e = evaluate(res)
    return e["calmar"], e["cagr"] * 100, e["max_dd"] * 100

def param_check(logmsg):
    """全参数复检: 8轴 × 邻域, 双窗口 + OOS; 返回候选或 None"""
    sys.path.insert(0, HERE)
    import numpy as np, pandas as pd
    from data_prep import build_panel
    R, _ = build_panel("proxy"); Rr, _ = build_panel("real")
    CFG = json.load(open(CFG_PATH))

    def run(cfg, Rs, ps): return _run_backtest(Rs, ps, cfg)

    base_p = run(CFG, R, "2014-06-23"); base_r = run(CFG, Rr, "2025-04-23")
    base_o = run(CFG, R, OOS_START)
    logmsg(f"[base] v26 proxyCal {base_p[0]:.2f}/realCal {base_r[0]:.2f}/OOS Cal {base_o[0]:.2f}")

    def set_cfg(c, ax, v):
        c = copy.deepcopy(c)
        if ax == "premium_cut": c["premium_cut"] = v
        elif ax == "min_delta": c["min_delta"] = v
        elif ax == "CN_fb": c["market_dd"] = {"CN": [v, 0.05, 0.18, 0.14], "US": [0.13, 0.12, 0.24, 0.15]}
        elif ax == "US_fb": c["market_dd"] = {"CN": [0.07, 0.05, 0.18, 0.14], "US": [v, 0.12, 0.24, 0.15]}
        elif ax == "gate_win": c["gate_win"] = v
        elif ax == "hyst_up": c["hyst_up"] = v
        elif ax == "corr_thr": c["corr_risk_thr"] = v
        elif ax == "speed_brake_thr": c["speed_brake_thr"] = v
        return c

    AXES = {
        "premium_cut": ([0.40, 0.20, 0.07], [0.50, 0.25, 0.10]),
        "min_delta": (0.030, 0.040),
        "CN_fb": (0.06, 0.08),
        "US_fb": (0.12, 0.14),
        "gate_win": (90, 110),
        "hyst_up": (0.60, 0.72),
        "corr_thr": ([0.28, 0.40], [0.35, 0.47]),
        "speed_brake_thr": (-0.035, -0.045),
    }
    cand = []
    for ax, (v1, v2) in AXES.items():
        for v in (v1, v2):
            try:
                cfg = set_cfg(CFG, ax, v)
                p = run(cfg, R, "2014-06-23"); r = run(cfg, Rr, "2025-04-23")
                dp, dr = p[0] - base_p[0], r[0] - base_r[0]
                logmsg(f"[check] {ax}={v} proxyCal {p[0]:.2f}({dp:+.2f}) realCal {r[0]:.2f}({dr:+.2f})")
                if dp > 0.15 and dr > 0.15:
                    cand.append((ax, v, cfg, dp, dr))
            except Exception as e:
                logmsg(f"[check] {ax}={v} ERR {str(e)[:100]}")
    if not cand:
        logmsg("[cand] 无双窗口同向改善候选, 维持 v26")
        return None
    cand.sort(key=lambda x: x[3] + x[4], reverse=True)
    ax, v, cfg, dp, dr = cand[0]
    logmsg(f"[cand] {ax}={v} 进入平台验证")
    # 平台验证: 候选轴扫5点(候选居中), 至少3点双窗口同向>0.10 且相邻落差<0.5Cal
    vals = [v]
    step = 0.01 if ax in ("min_delta",) else (0.005 if ax in ("CN_fb", "US_fb", "speed_brake_thr") else
             (5 if ax == "gate_win" else (0.02 if ax == "hyst_up" else 0.05)))
    if not isinstance(v, list):
        for k in (1, 2):
            vals.append(v + step * k); vals.append(v - step * k)
        vals = sorted(vals)
        ok = 0; prev = None
        for vv in vals:
            cc = set_cfg(CFG, ax, vv)
            pp = run(cc, R, "2014-06-23"); rr = run(cc, Rr, "2025-04-23")
            dpp, drr = pp[0] - base_p[0], rr[0] - base_r[0]
            logmsg(f"[plat] {ax}={vv} proxyCal {pp[0]:.2f}({dpp:+.2f}) realCal {rr[0]:.2f}({drr:+.2f})")
            if prev is not None and abs(pp[0] - prev) > 0.5:
                logmsg(f"[plat] {ax}={vv} 与相邻点落差{abs(pp[0]-prev):.2f}>0.5 悬崖, 平台失败")
                return None
            prev = pp[0]
            if dpp > 0.10 and drr > 0.10: ok += 1
        if ok < 3:
            logmsg(f"[plat] 仅{ok}点同向>0.10(<3), 平台失败")
            return None
        # OOS 复检
        oo = run(cfg, R, OOS_START)
        logmsg(f"[oos] {ax}={v} OOS Cal {oo[0]:.2f} vs 基线 {base_o[0]:.2f}")
        if oo[0] < base_o[0] - 0.05:
            logmsg("[oos] OOS 劣化, 拒绝")
            return None
        return ax, v, cfg
    logmsg(f"[plat] {ax} 列表参数跳过平台扫描(手工验证)")
    return None

def main():
    do_fetch = "--no-fetch" not in sys.argv
    do_push = "--no-push" not in sys.argv
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, f"retrain_{dt.date.today():%Y%m%d}.log"), "a", encoding="utf-8")

    def logmsg(m):
        print(m)
        log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {m}\n"); log.flush()

    env = dict(os.environ); env["ETF_DATA_DIR"] = DATA_DIR
    for f in os.listdir(os.path.join(SKILL, "assets", "data")):
        if f.endswith(".csv") and not os.path.exists(os.path.join(DATA_DIR, f)):
            shutil.copy(os.path.join(SKILL, "assets", "data", f), os.path.join(DATA_DIR, f))
            logmsg(f"[seed] 补齐 {f}")

    if do_fetch:
        rc, out, err = da.run([PY, "-W", "ignore", os.path.join(HERE, "daily_fetch.py")], cwd=WORK, env=env)
        logmsg(f"[fetch] rc={rc}" + (f" ERR={err[:200]}" if err else ""))
    rc, out, err = da.run([PY, "-W", "ignore", "-c",
                           "import sys; sys.path.insert(0, r'" + HERE + "'); from data_prep import save_cache; save_cache()"],
                          cwd=WORK, env=env)
    logmsg(f"[panel] rc={rc}" + (f" ERR={err[:200]}" if err else ""))

    before = None
    if os.path.exists(da.STATE):
        before = json.load(open(da.STATE)).get("nav_last_date")
    last_date = da.nav_last_date()
    json.dump({"nav_last_date": last_date, "updated_at": dt.datetime.now().isoformat()},
              open(da.STATE, "w"), ensure_ascii=False, indent=2)
    data_new = last_date is not None and (before != last_date or not do_fetch)

    # ③ 全参数复检(每次运行都做, 数据无更新也复检参数, 保证参数始终处于验证状态)
    hit = param_check(logmsg)

    # ④ 三关通过 → 升级版本; 否则维持 v26
    version_note = ""
    if hit is not None:
        ax, v, cfg = hit
        import json as _j
        new_ver = "v" + str(int(CFG_PATH.split("v")[-1].split(".")[0]) + 1)
        new_path = os.path.join(SKILL, "references", f"final_cfg_{new_ver}.json")
        cfg["version"] = new_ver
        cfg["_auto_upgrade"] = f"{ax}={v} 每日复检三关通过"
        _j.dump(cfg, open(new_path, "w"), ensure_ascii=False, indent=1)
        logmsg(f"[upgrade] 升级为 {new_ver} ({ax}={v}), 配置写入 {new_path}")
        version_note = f"自动升级 {new_ver}({ax}={v})"

    if not data_new and hit is None:
        logmsg("[skip] 数据无更新且参数无变化, 仅复检完成")
        return

    # ⑤ 日报 + README POSITIONS
    today_str = dt.date.today().isoformat()
    out_md = os.path.join(WORK, "reports", f"{today_str}.md")
    rc, out, err = da.run([PY, "-W", "ignore", os.path.join(HERE, "daily_report.py"), out_md], cwd=WORK, env=env)
    logmsg(f"[report] rc={rc}" + (f" ERR={err[:200]}" if err else ""))
    if rc == 0 and os.path.exists(out_md):
        sec = da.parse_report_section(out_md)
        da.update_readme_positions(sec, logmsg)

    # ⑥ 推送
    if do_push:
        da.git_sync_skill(logmsg)
    else:
        logmsg("[push] --no-push, 跳过推送")

    # ⑦ 通知
    da.notify("ETF策略每日重训", f"{version_note or '参数复检: 维持v26'} ｜ 数据 {last_date}",
              f"详见 {LOG_DIR}/retrain_{today_str}.log")
    logmsg("[notify] 已发送通知")

if __name__ == "__main__":
    main()
