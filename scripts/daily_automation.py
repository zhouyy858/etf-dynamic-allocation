# -*- coding: utf-8 -*-
"""每日自动化入口: 拉取最新数据 → 重建面板 → 生成日报 → 发送通知
由 launchd 每个交易日 09:00 触发（周一至五）；节假日/无新数据时仅记日志不发通知。
用法: python3 scripts/daily_automation.py [--no-fetch]
环境变量: ETF_REPORT_DIR 覆盖报告目录(默认 ~/ETF策略日报)"""
import os, sys, json, subprocess, datetime as dt, shutil

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
WORK = os.environ.get("ETF_REPORT_DIR", os.path.expanduser("~/ETF策略日报"))
DATA_DIR = os.path.join(WORK, "data")
REPORT_DIR = os.path.join(WORK, "reports")
LOG_DIR = os.path.join(WORK, "logs")
STATE = os.path.join(WORK, "state.json")
for d in (DATA_DIR, REPORT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    return r.returncode, (r.stdout or "")[-1500:], (r.stderr or "")[-1500:]

def notify(title, subtitle, body=""):
    sc = "tell application \"System Events\" to display notification \"" + \
         body.replace('"', "'") + "\" with title \"" + title.replace('"', "'") + \
         "\" subtitle \"" + subtitle.replace('"', "'") + "\""
    subprocess.run(["osascript", "-e", sc], capture_output=True, timeout=30)
    # 可选推送: Server酱(微信) / Bark(iOS) — 配置 ~/.config/etf_skill/notify.json
    cfg_path = os.path.expanduser("~/.config/etf_skill/notify.json")
    if os.path.exists(cfg_path):
        try:
            ncfg = json.load(open(cfg_path))
            key = ncfg.get("serverchan_key")
            if key:
                import urllib.parse, urllib.request
                data = urllib.parse.urlencode({"title": title, "desp": f"{subtitle}\n{body}"}).encode()
                urllib.request.urlopen(f"https://sctapi.ftqq.com/{key}.send", data=data, timeout=30)
            bark = ncfg.get("bark_url")
            if bark:
                import urllib.parse, urllib.request
                urllib.request.urlopen(f"{bark}/{urllib.parse.quote(title)}/{urllib.parse.quote(subtitle + '\\n' + body)}", timeout=30)
        except Exception as e:
            print("[notify-extra]", e)

def nav_last_date():
    """ETF净值/联接净值(真正约束数据)的最后日期; 指数当日就有值, 净值T日晚才公布"""
    import pandas as pd
    mx = None
    for f in ["159232_nav.csv", "515100_nav.csv", "159941_nav.csv", "513500_nav.csv",
              "159952_nav.csv", "270042_nav.csv", "050025_nav.csv"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            d = str(pd.read_csv(p, usecols=[0]).iloc[-1, 0])
            if mx is None or d > mx:
                mx = d
    return mx

def main():
    do_fetch = "--no-fetch" not in sys.argv
    log = open(os.path.join(LOG_DIR, f"run_{dt.date.today():%Y%m%d}.log"), "a", encoding="utf-8")

    def logmsg(m):
        print(m)
        log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {m}\n")
        log.flush()

    env = dict(os.environ)
    env["ETF_DATA_DIR"] = DATA_DIR

    # 种子/补齐: 每次运行把 skill 自带数据中缺失的 csv 复制到工作目录(幂等)
    # 修复: 2026-08-04 日报失败——工作目录缺 qdii_price_*.csv/511010 等, 原逻辑仅首次运行复制
    for f in os.listdir(os.path.join(SKILL, "assets", "data")):
        if f.endswith(".csv") and not os.path.exists(os.path.join(DATA_DIR, f)):
            shutil.copy(os.path.join(SKILL, "assets", "data", f), os.path.join(DATA_DIR, f))
            logmsg(f"[seed] 补齐缺失数据文件 {f}")

    before = None
    if os.path.exists(STATE):
        before = json.load(open(STATE)).get("nav_last_date")

    # 1) 拉取最新数据
    if do_fetch:
        rc, out, err = run([PY, "-W", "ignore", os.path.join(HERE, "daily_fetch.py")], cwd=WORK, env=env)
        logmsg(f"[fetch] daily_fetch.py rc={rc}" + (f" ERR={err[:200]}" if err else ""))

    # 2) 重建数据面板(写回 DATA_DIR)
    rc, out, err = run([PY, "-W", "ignore", "-c",
                        "import sys; sys.path.insert(0, r'" + HERE + "'); from data_prep import save_cache; save_cache()"],
                       cwd=WORK, env=env)
    logmsg(f"[panel] rc={rc}" + (f" ERR={err[:200]}" if err else ""))

    # 3) 交易日判定: 净值日期无更新 → 节假日/停牌
    last_date = nav_last_date()
    state = {"nav_last_date": last_date, "updated_at": dt.datetime.now().isoformat()}
    json.dump(state, open(STATE, "w"), ensure_ascii=False, indent=2)
    if last_date is None or (before == last_date and do_fetch):
        logmsg(f"[skip] 数据无更新(last={last_date})，非交易日，不发送")
        return
    logmsg(f"[data] 最新数据日期: {last_date}")

    # 4) 生成日报
    today_str = dt.date.today().isoformat()
    out_md = os.path.join(REPORT_DIR, f"{today_str}.md")
    rc, out, err = run([PY, "-W", "ignore", os.path.join(HERE, "daily_report.py"), out_md], cwd=WORK, env=env)
    logmsg(f"[report] rc={rc} 已保存 {out_md}" + (f" ERR={err[:300]}" if err else ""))
    if rc != 0:
        logmsg("[report] 生成失败，仍发通知提示")
        notify("ETF策略日报", "生成失败", err[:200])
        return

    # 5) 发送通知
    head = out.strip().splitlines()
    score_line = next((l for l in head if "有效打分" in l), "")
    cash_line = next((l for l in head if l.startswith("| 现金")), "")
    act_line = next((l for l in head if l.startswith("非调仓") or l.startswith("今日是周三")), "")
    sub = f"{score_line.split('｜')[0] if '｜' in score_line else score_line}"
    sub += f" ｜ {cash_line.split('|')[2] if len(cash_line.split('|'))>2 else ''}现金"
    notify("ETF策略日报", sub.strip(" ｜"), act_line[:120])
    logmsg("[notify] 已发送 macOS 通知")

if __name__ == "__main__":
    main()
