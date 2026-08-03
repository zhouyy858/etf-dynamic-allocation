# -*- coding: utf-8 -*-
"""拉取5只ETF最长历史日度净值(累计净值复权) - 东方财富lsjz接口 pageSize上限20"""
import requests, json, time, pandas as pd

ETFS = {
    "159232": "自由现金流ETF", "515100": "红利低波100ETF",
    "159941": "纳指100ETF", "513500": "标普500ETF", "159952": "创业板ETF",
}
HDRS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://fundf10.eastmoney.com/"}
URL = "https://api.fund.eastmoney.com/f10/lsjz"

def fetch_nav(fund):
    out, page, total = [], 1, None
    while True:
        params = {"fundCode": fund, "pageIndex": page, "pageSize": 20, "startDate": "", "endDate": ""}
        j = None
        for attempt in range(5):
            try:
                r = requests.get(URL, params=params, headers=HDRS, timeout=30)
                j = r.json(); break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(1.5 + attempt)
        if total is None:
            total = int(j.get("TotalCount", 0) or 0)
        data = (j.get("Data") or {}).get("LSJZList") or []
        if not data:
            break
        for d in data:
            out.append({"date": d["FSRQ"], "unit_nav": float(d["DWJZ"]),
                        "cum_nav": float(d["LJJZ"]), "ret_pct": float(d.get("JZZZL") or 0)})
        if page * 20 >= total or len(data) < 20:
            break
        page += 1; time.sleep(0.3)
    df = pd.DataFrame(out); df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").sort_index()

summary = {}
for fund, name in ETFS.items():
    try:
        n = fetch_nav(fund)
        n.to_csv(f"data/{fund}_nav.csv")
        print(f"[NAV] {fund} {name}: {n.index.min().date()}~{n.index.max().date()} n={len(n)}")
        summary[fund] = {"name": name, "start": str(n.index.min().date()), "end": str(n.index.max().date()), "n": len(n)}
    except Exception as e:
        print(f"[FAIL nav] {fund} {name}: {e}")
    time.sleep(0.6)

with open("data/summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("DONE")
