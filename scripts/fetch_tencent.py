# -*- coding: utf-8 -*-
import requests, time, pandas as pd
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
STARTS = {"sz399006": "2010-01-01", "sh000300": "2004-01-01", "sh000905": "2006-01-01", "sh000016": "2003-01-01"}

def fetch_tencent(symbol, start, end="2026-12-31"):
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    chunks, cur = [], pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    empty_streak = 0
    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=360), end_dt)
        p = f"{symbol},day,{cur:%Y-%m-%d},{nxt:%Y-%m-%d},400,qfq"
        try:
            j = requests.get(url, params={"param": p}, headers=H, timeout=30).json()
        except Exception as e:
            print(f"  [err] {symbol} {cur:%Y-%m-%d}: {str(e)[:60]}"); time.sleep(1); continue
        data = (j.get("data") or {}).get(symbol) or {}
        day = data.get("day") or data.get("qfqday") or []
        if not day:
            empty_streak += 1
            if empty_streak > 2:  # 连续3个空窗则停止
                break
            cur = nxt; time.sleep(0.35); continue
        empty_streak = 0
        chunks.append(pd.DataFrame(day, columns=["date","open","close","high","low","vol"]))
        cur = nxt; time.sleep(0.3)
    if not chunks:
        print(f"[FAIL] {symbol} empty"); return None
    df = pd.concat(chunks); df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df

for sym, name in [("sz399006","创业板指"),("sh000300","沪深300"),("sh000905","中证500"),("sh000016","上证50")]:
    df = fetch_tencent(sym, STARTS[sym])
    if df is not None:
        df.to_csv(f"data/index_{sym}.csv")
        print(f"[OK] {sym} {name}: {df.index.min().date()}~{df.index.max().date()} n={len(df)}")
print("DONE")
