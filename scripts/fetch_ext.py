# -*- coding: utf-8 -*-
"""补充长历史代理数据: 中证指数(csindex) + 腾讯指数 + QDII联接基金净值"""
import requests, json, time, pandas as pd

H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
     "Referer": "https://fundf10.eastmoney.com/"}

def fetch_csindex(code, start="19900101", end="20261231"):
    url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
    params = {"indexCode": code, "startDate": start, "endDate": end}
    j = requests.get(url, params=params, headers=H, timeout=30).json()
    if j.get("code") != "200":
        print(f"[FAIL csindex] {code}: {j}")
        return None
    rows = [{"date": d["tradeDate"], "close": d["close"], "pct": d["changePct"]} for d in j["data"]]
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

def fetch_tencent(symbol, start="19900101", end="20261231"):
    """腾讯fqkline分窗拉全量"""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    chunks, cur = [], pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=400), end_dt)
        p = f"{symbol},day,{cur:%Y-%m-%d},{nxt:%Y-%m-%d},100000,qfq"
        j = requests.get(url, params={"param": p}, headers=H, timeout=30).json()
        data = (j.get("data") or {}).get(symbol) or {}
        day = data.get("day") or data.get("qfqday") or []
        if not day:
            break
        chunks.append(pd.DataFrame(day, columns=["date","open","close","high","low","vol"]))
        cur = nxt
        time.sleep(0.4)
        if len(day) < 5:
            break
    if not chunks:
        print(f"[FAIL tencent] {symbol}: empty")
        return None
    df = pd.concat(chunks); df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.drop_duplicates("date").set_index("date").sort_index()

def fetch_lsjz(fund):
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    out, page, total = [], 1, None
    while True:
        params = {"fundCode": fund, "pageIndex": page, "pageSize": 20, "startDate": "", "endDate": ""}
        j = requests.get(url, params=params, headers=H, timeout=30).json()
        if total is None:
            total = int(j.get("TotalCount", 0) or 0)
        data = (j.get("Data") or {}).get("LSJZList") or []
        if not data: break
        for d in data:
            out.append({"date": d["FSRQ"], "unit_nav": float(d["DWJZ"]), "cum_nav": float(d["LJJZ"]), "ret_pct": float(d.get("JZZZL") or 0)})
        if page * 20 >= total or len(data) < 20: break
        page += 1; time.sleep(0.3)
    df = pd.DataFrame(out); df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()

# 1) 中证指数
for code, name in [("932365", "中证全指自由现金流"), ("930955", "中证红利低波100"), ("000922", "中证红利")]:
    df = fetch_csindex(code)
    if df is not None:
        df.to_csv(f"data/index_{code}.csv")
        print(f"[CSINDEX] {code} {name}: {df.index.min().date()}~{df.index.max().date()} n={len(df)}")
    time.sleep(0.5)

# 2) 腾讯指数
for sym, name in [("sz399006", "创业板指"), ("sh000300", "沪深300"), ("sh000905", "中证500"), ("sh000016", "上证50")]:
    df = fetch_tencent(sym)
    if df is not None:
        df.to_csv(f"data/index_{sym}.csv")
        print(f"[TENCENT] {sym} {name}: {df.index.min().date()}~{df.index.max().date()} n={len(df)}")
    time.sleep(0.5)

# 3) QDII联接基金
for fund, name in [("270042", "广发纳指100联接A"), ("050025", "博时标普500联接A")]:
    try:
        df = fetch_lsjz(fund)
        df.to_csv(f"data/{fund}_nav.csv")
        print(f"[LSJZ] {fund} {name}: {df.index.min().date()}~{df.index.max().date()} n={len(df)}")
    except Exception as e:
        print(f"[FAIL lsjz] {fund}: {e}")
    time.sleep(0.5)
print("DONE")
