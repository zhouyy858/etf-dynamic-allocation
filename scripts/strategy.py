# -*- coding: utf-8 -*-
"""动态策略v10: 结构闸门 + 快刹车(成长仓) + 深熊锁定(全部CN/US弹性清零, 仅留15%底仓)
- 打分9维 -> 状态(成长g/防御d) -> 权重骨架
- CN弹性(159952成长 + 515100/159232防御弹性) × CN市场乘数; US弹性(159941/513500额外) × US市场乘数
- 快刹车: 市场回撤>8%(CN)/10%(US)且跌破SMA20 -> 成长弹性×0.10/0.15
- 深熊锁定: CN回撤>20%/US回撤>25% -> 该市场全部弹性归零(仅15%底仓), 恢复条件: 回撤收窄或SMA120+双均线确认
- 组合深度回撤 -> 权益上限(趋势弱时); 15%国内+15%海外底仓固定; 每周三调仓、每次目标分3周三笔执行;
- 日度风控触发(组合回撤上限/市场快刹车/深熊锁/闸门导致目标权益显著降低)同样分3周三笔执行"""
import os, numpy as np, pandas as pd
from data_prep import DATA_DIR

SLOTS = ["159232", "515100", "159941", "513500", "159952"]
FLOOR = {"159232": 7.5, "515100": 7.5, "159941": 7.5, "513500": 7.5, "159952": 0.0}
FLOOR_EQ = 30.0

STATE_MAP = {
    9: (78, 4), 8: (74, 7), 7: (69, 10), 6: (62, 13), 5: (55, 16),
    4: (47, 18), 3: (39, 21), 2: (31, 24), 1: (23, 27), 0: (16, 30),
}
GROWTH_SPLIT_BULL = {"159952": 0.34, "159941": 0.48, "513500": 0.20}
GROWTH_SPLIT_BEAR = {"159952": 0.44, "159941": 0.36, "513500": 0.20}
DEFENSE_SPLIT = {"159232": 0.45, "515100": 0.55}
DEFENSE_MOMENTUM_WIN = 60
DEFENSE_MOMENTUM_T = 3.0
DEFENSE_CLAMP = (0.35, 0.65)

VOL_TARGET = 0.18
MIN_CASH = 0.05
MAX_EQ = 0.97
# 市场刹车: (快刹车触发DD%, 快刹车成长削减, 深熊锁定DD%, 解锁DD%)
MARKET_DD = {"CN": (0.07, 0.08, 0.20, 0.10), "US": (0.12, 0.18, 0.26, 0.12)}
DD_EQ_CAP = [(-0.12, 80), (-0.18, 65), (-0.25, 50)]
HYST_UP, HYST_DOWN = 0.54, 0.17

class SignalSet:
    def __init__(self, R):
        hs300 = pd.read_csv(os.path.join(DATA_DIR, "index_sh000300.csv"), parse_dates=["date"]).set_index("date")["close"].sort_index()
        lvl = {}
        for s in SLOTS:
            lvl[s] = (1 + R[s].dropna()).cumprod()
        self.a_mkt = hs300.reindex(R.index).ffill()
        self.a_g = lvl["159952"].reindex(R.index).ffill()
        self.u_g = lvl["159941"].reindex(R.index).ffill()
        self.u_m = lvl["513500"].reindex(R.index).ffill()
        self.R = R
        self.sma = {}
        for nm, x in [("a_mkt", self.a_mkt), ("a_g", self.a_g), ("u_g", self.u_g), ("u_m", self.u_m)]:
            self.sma[nm] = {w: x.rolling(w, min_periods=10).mean() for w in (20, 60, 120)}

    def score(self, dt):
        i = self.R.index.get_indexer([dt], method="ffill")[0]
        def gt(x, nm, win):
            return 1.0 if (i < len(x) and x.iloc[i] > self.sma[nm][win].iloc[i]) else 0.0
        comps = [gt(self.a_mkt, "a_mkt", 120), gt(self.a_mkt, "a_mkt", 60), gt(self.a_mkt, "a_mkt", 20),
                 gt(self.a_g, "a_g", 120), gt(self.a_g, "a_g", 60), gt(self.a_g, "a_g", 20),
                 gt(self.u_g, "u_g", 120), gt(self.u_m, "u_m", 120), gt(self.u_g, "u_g", 60)]
        return int(sum(comps))

    def mkt_info(self, dt, market, gate_win=120):
        i = self.R.index.get_indexer([dt], method="ffill")[0]
        x = self.a_mkt if market == "CN" else self.u_m
        if i >= len(x):
            return 0.0, True, True, 1.0, True
        px = x.iloc[i]
        j0 = max(0, i - 755)  # 市场回撤用3年滚动窗口, 避免长历史早期峰值造成失真
        window = x.iloc[j0:i + 1]
        peak = window.max()
        trough = window.min()
        dd = float(px / peak - 1.0) if peak > 0 else 0.0
        nm = "a_mkt" if market == "CN" else "u_m"
        s20 = self.sma[nm][20].iloc[i]; s60 = self.sma[nm][60].iloc[i]; sg = self.sma[nm][gate_win].iloc[i]
        rec = float((px - trough) / (peak - trough)) if peak > trough else 1.0
        return dd, bool(px > s20), bool(s20 > s60), rec, bool(px > sg)

class DynamicStrategy:
    def __init__(self, R_full, cfg=None):
        self.sig = SignalSet(R_full)
        self.R = R_full
        cfg = cfg or {}
        sm = cfg.get("state_map", STATE_MAP)
        self.state_map = {int(k): tuple(v) for k, v in sm.items()}
        self.vol_target = cfg.get("vol_target", VOL_TARGET)
        self.min_cash = cfg.get("min_cash", MIN_CASH)
        self.max_eq = cfg.get("max_eq", MAX_EQ)
        self.dd_eq_cap = cfg.get("dd_eq_cap", DD_EQ_CAP)
        self.market_dd = cfg.get("market_dd", MARKET_DD)
        self.hyst_up = cfg.get("hyst_up", HYST_UP)
        self.hyst_down = cfg.get("hyst_down", HYST_DOWN)
        self.gate_win = cfg.get("gate_win", 120)
        self.growth_split_bull = dict(cfg.get("growth_split_bull", GROWTH_SPLIT_BULL))
        self.growth_split_bear = dict(cfg.get("growth_split_bear", GROWTH_SPLIT_BEAR))
        self.defense_momentum = bool(cfg.get("defense_momentum", False))
        self.defense_momentum_win = int(cfg.get("defense_momentum_win", DEFENSE_MOMENTUM_WIN))
        self.defense_momentum_t = float(cfg.get("defense_momentum_t", DEFENSE_MOMENTUM_T))
        self.defense_clamp = tuple(cfg.get("defense_clamp", DEFENSE_CLAMP))
        self.dd_cap_unconditional = bool(cfg.get("dd_cap_unconditional", False))
        self._prev_eff = None
        self._lock = {"CN": False, "US": False}
        self.state_log = []
        self.risk_log = []

    def _defense_split(self, dt):
        """国内防御双持轮动: 159232(自由现金流) vs 515100(红利低波100) 按相对动量分配防御弹性"""
        if not self.defense_momentum:
            return dict(DEFENSE_SPLIT)
        i = self.R.index.get_indexer([dt], method="ffill")[0]
        if i < self.defense_momentum_win or i >= len(self.R):
            return dict(DEFENSE_SPLIT)
        seg = self.R.iloc[i - self.defense_momentum_win + 1: i + 1]
        g232 = float((1 + seg["159232"].fillna(0.0)).prod())
        g100 = float((1 + seg["515100"].fillna(0.0)).prod())
        if g232 <= 0 or g100 <= 0:
            return dict(DEFENSE_SPLIT)
        w232 = g232 ** self.defense_momentum_t / (g232 ** self.defense_momentum_t + g100 ** self.defense_momentum_t)
        lo, hi = self.defense_clamp
        w232 = min(max(float(w232), lo), hi)
        return {"159232": w232, "515100": 1.0 - w232}

    def target_fn(self):
        def fn(dt, R_prev, ctx):
            return self.regular_target(dt, ctx)
        return fn

    def daily_fn(self):
        def fn(dt, R_prev, ctx):
            pf = ctx.get("pf_rets", pd.Series(dtype=float))
            if len(pf) < 40:
                return None
            wv = (1 + pf).cumprod()
            dd = wv.iloc[-1] / wv.cummax().iloc[-1] - 1.0 if wv.cummax().iloc[-1] > 0 else 0.0
            sc = self.sig.score(dt)
            cap = 1.0
            for thr, c in self.dd_eq_cap:
                if dd < thr and (self.dd_cap_unconditional or sc < 6):
                    cap = min(cap, c)
            # 市场级风控: 目标权益比当前显著低(深熊锁/快刹车/结构闸门生效) -> 日度触发, 仍分3周三笔
            base = self._base_target(dt, sc)
            eq_tgt = sum(base.values()) / 100.0
            cur = float(ctx.get("equity", 1.0))
            if eq_tgt <= cur - 0.04:
                return self._finalize(base)
            if cap < 1.0:
                return self.target_with_eq_cap(dt, ctx, cap)
            return None
        return fn

    def _eff_score(self, dt):
        sc = self.sig.score(dt)
        if self._prev_eff is None:
            self._prev_eff = float(sc)
        else:
            w = self.hyst_up if sc > self._prev_eff else self.hyst_down
            self._prev_eff = w * sc + (1 - w) * self._prev_eff
        return int(round(self._prev_eff))

    def _market_mult(self, dt, market):
        """返回 (成长弹性乘数, 全弹性乘数)"""
        dd, ok20, ok20_60, rec, ok120 = self.sig.mkt_info(dt, market, self.gate_win)
        fb_thr, fb_cut, deep_thr, recov = self.market_dd[market]
        m_g, m_all = 1.0, 1.0
        # 深熊锁定: 该市场全部弹性清零
        if self._lock[market]:
            if dd > -recov or (ok120 and ok20_60):
                self._lock[market] = False
            else:
                return 0.0, 0.0
        if dd < -deep_thr:
            self._lock[market] = True
            return 0.0, 0.0
        # 结构闸门
        if not ok120:
            m_g = 0.0
        # 快刹车
        if dd < -fb_thr and not (ok20 and ok20_60):
            m_g = min(m_g, fb_cut)
        return m_g, m_all

    def _base_target(self, dt, sc):
        g, d = self.state_map.get(sc, self.state_map[min(max(sc, 0), 9)])
        split_g = self.growth_split_bull if g >= 30 else self.growth_split_bear
        split_d = self._defense_split(dt)
        out = {
            "159232": FLOOR["159232"] + split_d["159232"] * d,
            "515100": FLOOR["515100"] + split_d["515100"] * d,
            "159941": FLOOR["159941"] + split_g["159941"] * g,
            "513500": FLOOR["513500"] + split_g["513500"] * g,
            "159952": FLOOR["159952"] + split_g["159952"] * g,
        }
        m_cn_g, m_cn_all = self._market_mult(dt, "CN")
        m_us_g, m_us_all = self._market_mult(dt, "US")
        # CN弹性: 成长仓受快刹车+闸门; 防御弹性只受深熊锁定
        out["159952"] = FLOOR["159952"] + (out["159952"] - FLOOR["159952"]) * m_cn_g * m_cn_all
        out["159232"] = FLOOR["159232"] + (out["159232"] - FLOOR["159232"]) * m_cn_all
        out["515100"] = FLOOR["515100"] + (out["515100"] - FLOOR["515100"]) * m_cn_all
        out["159941"] = FLOOR["159941"] + (out["159941"] - FLOOR["159941"]) * m_us_g * m_us_all
        out["513500"] = FLOOR["513500"] + (out["513500"] - FLOOR["513500"]) * m_us_g * m_us_all
        return out

    def regular_target(self, dt, ctx):
        sc = self._eff_score(dt)
        self.state_log.append((dt, sc))
        out = self._base_target(dt, sc)
        pf = ctx.get("pf_rets", pd.Series(dtype=float))
        scale = 1.0
        if len(pf) > 30:
            ew = pf.ewm(halflife=20).std().iloc[-1] * np.sqrt(252)
            if ew > 0 and ew > self.vol_target * 1.15:
                scale = min(scale, (self.vol_target * 1.15) / ew)
            wv = (1 + pf).cumprod()
            dd = wv.iloc[-1] / wv.cummax().iloc[-1] - 1.0 if wv.cummax().iloc[-1] > 0 else 0.0
            cap = 1.0
            for thr, c in self.dd_eq_cap:
                if dd < thr and (self.dd_cap_unconditional or sc < 6):
                    cap = min(cap, c)
            if cap < 1.0:
                total_flex = sum(out[s] - FLOOR[s] for s in SLOTS)
                scale = min(scale, max(0.0, (cap - FLOOR_EQ) / max(total_flex, 1e-9)))
        if scale < 1.0:
            for s in SLOTS:
                out[s] = FLOOR[s] + (out[s] - FLOOR[s]) * scale
        self.risk_log.append((dt, sc, scale))
        return self._finalize(out)

    def target_with_eq_cap(self, dt, ctx, cap_pct):
        sc = self._eff_score(dt)
        out = self._base_target(dt, sc)
        total_flex = sum(out[s] - FLOOR[s] for s in SLOTS)
        scale = max(0.0, (cap_pct - FLOOR_EQ) / max(total_flex, 1e-9))
        if scale < 1.0:
            for s in SLOTS:
                out[s] = FLOOR[s] + (out[s] - FLOOR[s]) * scale
        return self._finalize(out)

    def _finalize(self, out):
        total = sum(out.values())
        cash = 100.0 - total
        cash = max(cash, self.min_cash * 100)
        if 100 - cash > self.max_eq * 100:
            cash = (1 - self.max_eq) * 100
        eq = 100.0 - cash
        for s in SLOTS:
            out[s] = out[s] * (eq / total) / 100.0
        out["cash"] = cash / 100.0
        return out
