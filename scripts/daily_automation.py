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

def parse_report_section(report_md):
    """从日报解析: 仓位表(实际/目标) + 有效打分 + 市场状态 + QDII溢价"""
    lines = open(report_md, encoding="utf-8").read().splitlines()
    out = {"rows": [], "score": "", "cn": "", "us": "", "premium": [], "date": ""}
    for l in lines:
        if l.startswith("| 标的"):
            continue
        if l.startswith("| ---"):
            continue
        if l.startswith("| 159232") or l.startswith("| 515100") or l.startswith("| 159941") or \
           l.startswith("| 513500") or l.startswith("| 159952") or l.startswith("| 现金"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            out["rows"].append(cells[:3])  # [名称, 实际, 目标]
        if "有效打分" in l:
            out["score"] = l.strip()
        if "数据截至" in l:
            import re as _re
            m = _re.search(r"\d{4}-\d{2}-\d{2}", l)
            if m: out["date"] = m.group(0)
        if l.startswith("- **CN市场**"):
            out["cn"] = l.strip().lstrip("- ")
        if l.startswith("- **US市场**"):
            out["us"] = l.strip().lstrip("- ")
        if l.startswith("- 纳指100") or l.startswith("- 标普500"):
            out["premium"].append(l.strip().lstrip("- "))
    return out

def update_readme_positions(sec, logmsg):
    """更新 SKILL README 的 POSITIONS 区块(实际持仓 vs 目标 + 信号状态)"""
    readme = os.path.join(SKILL, "README.md")
    if not os.path.exists(readme) or not sec["rows"]:
        return False
    start_marker, end_marker = "<!-- POSITIONS-START -->", "<!-- POSITIONS-END -->"
    s = open(readme, encoding="utf-8").read()
    if start_marker not in s or end_marker not in s:
        logmsg("[readme] README 缺少 POSITIONS 标记, 跳过自动更新")
        return False
    date_str = sec.get("date", "") or ""
    t = [f"> 数据截至 **{date_str}**（净值口径）｜ v25 定稿｜ 本段由每日自动化在数据刷新后更新并推送", "",
         "| 标的 | 实际持仓 | 目标（下周五决策） |", "|---|---|---|"]
    for name, act, tgt in sec["rows"]:
        t.append(f"| {name} | {act} | {tgt} |")
    t += ["", sec["score"] if sec["score"] else "-",
          "- " + sec["cn"] if sec["cn"] else "", "- " + sec["us"] if sec["us"] else ""]
    if sec["premium"]:
        t += ["- **QDII 溢价**：" + "；".join(sec["premium"])]
    blk = "\n".join(t)
    new = s[:s.index(start_marker) + len(start_marker)] + "\n" + blk + "\n" + s[s.index(end_marker):]
    open(readme, "w", encoding="utf-8").write(new)
    logmsg(f"[readme] 已更新 README 持仓段({len(sec['rows'])}行)")
    return True

def git_sync_skill(logmsg):
    """同步数据→SKILL仓库并 commit+push(本地 token 配置 ~/.config/etf_skill/git_token 或环境变量 ETF_GIT_TOKEN)"""
    # 先把工作目录最新数据复制回 skill assets/data, 保证 README/回测可复现
    src = os.path.join(WORK, "data")
    dst = os.path.join(SKILL, "assets", "data")
    n = 0
    if os.path.isdir(src):
        for f in os.listdir(src):
            if f.endswith(".csv"):
                sp, dp = os.path.join(src, f), os.path.join(dst, f)
                if not os.path.exists(dp) or os.path.getsize(sp) != os.path.getsize(dp):
                    shutil.copy(sp, dp); n += 1
    token = os.environ.get("ETF_GIT_TOKEN")
    if not token:
        tp = os.path.expanduser("~/.config/etf_skill/git_token")
        if os.path.exists(tp):
            token = open(tp).read().strip()
    if not token:
        logmsg(f"[git] 无 token(环境变量 ETF_GIT_TOKEN 或 ~/.config/etf_skill/git_token), 仅本地更新数据{n}个文件, 未推送")
        return
    url = f"https://{token}@github.com/zhouyy858/etf-dynamic-allocation.git"
    run(["git", "-C", SKILL, "add", "-A"], timeout=300)
    rc, out, err = run(["git", "-C", SKILL, "commit", "-m", f"每日数据+持仓更新 {dt.date.today().isoformat()}"], timeout=300)
    if rc != 0 and "nothing to commit" not in out + err:
        logmsg(f"[git] commit 失败: {err[:200]}")
        return
    rc, out, err = run(["git", "-C", SKILL, "push", url, "main"], timeout=600)
    if rc == 0:
        logmsg("[git] 已推送到 GitHub")
    else:
        logmsg(f"[git] push 失败(网络可能抖动, 稍后手动重试): {err[:150]}")

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

    # 4.5) 更新 README 持仓段 + 同步数据 + 推送到 GitHub
    sec = parse_report_section(out_md)
    update_readme_positions(sec, logmsg)
    git_sync_skill(logmsg)

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
