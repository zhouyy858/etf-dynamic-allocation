# -*- coding: utf-8 -*-
"""增量拉取: 只更新最近约20个交易日并合并进已有CSV(保留全历史), 适合每日定时
来源: 东方财富基金净值lsjz / 腾讯指数fqkline / 中证指数csindex
数据目录: 默认 skill 的 assets/data, 可用环境变量 ETF_DATA_DIR 覆盖"""
import os, time
import requests, pandas as pd

H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
     "Referer": "https://fundf10.eastmoney.com/"}
DATA = os.environ.get("ETF_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "data")
NAV_FUNDS = ["159232", "515100", "159941", "513500", "159952", "270042", "050025"]
TENCENT = ["sz399006", "sh000300", "sh000905", "sh000016"]
CSINDEX = [("932365", "中证全指自由现金流"), ("930955", "中证红利低波100"), ("000922", "中证红利")]

def merge_save(path, df, cols):
    df = df.copy()
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["date"])
        df = pd.concat([old, df.reset_index()], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    keep = ["date"] + [c for c in cols if c in df.columns]
    df[keep].to_csv(path, index=False)

def fetch_nav_latest(fund):
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    params = {"fundCode": fund, "pageIndex": 1, "pageSize": 20, "startDate": "", "endDate": ""}
    j = requests.get(url, params=params, headers=H, timeout=30).json()
    rows = [{"date": d["FSRQ"], "unit_nav": float(d["DWJZ"]), "cum_nav": float(d["LJJZ"]),
             "ret_pct": float(d.get("JZZZL") or 0)} for d in ((j.get("Data") or {}).get("LSJZList") or [])]
    return pd.DataFrame(rows)

def fetch_tencent_latest(sym):
    start = (pd.Timestamp.now() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    p = f"{sym},day,{start},{pd.Timestamp.now():%Y-%m-%d},100,qfq"
    j = requests.get(url, params={"param": p}, headers=H, timeout=30).json()
    day = ((j.get("data") or {}).get(sym) or {}).get("day") or []
    df = pd.DataFrame(day, columns=["date", "open", "close", "high", "low", "vol"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df

def fetch_csindex_latest(code):
    start = (pd.Timestamp.now() - pd.Timedelta(days=40)).strftime("%Y%m%d")
    url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
    j = requests.get(url, params={"indexCode": code, "startDate": start, "endDate": "20261231"}, headers=H, timeout=30).json()
    rows = [{"date": d["tradeDate"], "close": d["close"], "pct": d["changePct"]} for d in (j.get("data") or [])]
    return pd.DataFrame(rows)

def main():
    os.makedirs(DATA, exist_ok=True)
    ok = fail = 0
    for fund in NAV_FUNDS:
        try:
            df = fetch_nav_latest(fund); df["date"] = pd.to_datetime(df["date"])
            merge_save(f"{DATA}/{fund}_nav.csv", df.set_index("date"), ["unit_nav", "cum_nav", "ret_pct"])
            ok += 1
        except Exception as e:
            print(f"[FAIL nav] {fund}: {str(e)[:120]}"); fail += 1
        time.sleep(0.4)
    for sym in TENCENT:
        try:
            df = fetch_tencent_latest(sym); df["date"] = pd.to_datetime(df["date"])
            merge_save(f"{DATA}/index_{sym}.csv", df.set_index("date"), ["open", "close", "high", "low", "vol"])
            ok += 1
        except Exception as e:
            print(f"[FAIL tencent] {sym}: {str(e)[:120]}"); fail += 1
        time.sleep(0.4)
    for code, name in CSINDEX:
        try:
            df = fetch_csindex_latest(code); df["date"] = pd.to_datetime(df["date"])
            merge_save(f"{DATA}/index_{code}.csv", df.set_index("date"), ["close", "pct"])
            ok += 1
        except Exception as e:
            print(f"[FAIL csindex] {code}: {str(e)[:120]}"); fail += 1
        time.sleep(0.4)
    print(f"增量更新完成: 成功{ok} 失败{fail} ｜ 目录: {DATA}")

if __name__ == "__main__":
    main()
