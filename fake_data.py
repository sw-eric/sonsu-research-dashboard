"""
fake_data.py — Template data that mirrors the shape of IBKR Flex Report output.

When ready to connect to IBKR, replace FakeDataSource with IBKRFlexSource
(see ibkr_client.py). All downstream code consumes the same DataFrame schema.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

INITIAL_NAV = 1_000.0
START_DATE  = datetime(2025, 1, 6)

# ── Instrument universe ───────────────────────────────────────────────────────
INSTRUMENTS = [
    # Korean equities
    dict(symbol="005930.KS", name="Samsung Electronics",  sector="Semiconductors",    region="Korea", currency="KRW", fx=0.00073),
    dict(symbol="000660.KS", name="SK Hynix",             sector="Semiconductors",    region="Korea", currency="KRW", fx=0.00073),
    dict(symbol="035420.KS", name="NAVER Corp",           sector="Internet",          region="Korea", currency="KRW", fx=0.00073),
    dict(symbol="373220.KS", name="LG Energy Solution",   sector="Energy Storage",    region="Korea", currency="KRW", fx=0.00073),
    dict(symbol="009830.KS", name="Hanwha Solutions",     sector="Renewables",        region="Korea", currency="KRW", fx=0.00073),
    dict(symbol="051910.KS", name="LG Chem",              sector="Chemicals",         region="Korea", currency="KRW", fx=0.00073),
    dict(symbol="207940.KS", name="Samsung Biologics",    sector="Biotech",           region="Korea", currency="KRW", fx=0.00073),
    # US equities
    dict(symbol="NVDA",      name="NVIDIA",               sector="Semiconductors",    region="US",    currency="USD", fx=1.0),
    dict(symbol="AMAT",      name="Applied Materials",    sector="Semiconductor Equip", region="US",  currency="USD", fx=1.0),
    dict(symbol="ASML",      name="ASML Holding",         sector="Semiconductor Equip", region="US",  currency="USD", fx=1.0),
    dict(symbol="CEG",       name="Constellation Energy", sector="Nuclear/Utilities", region="US",    currency="USD", fx=1.0),
    dict(symbol="VST",       name="Vistra Corp",          sector="Nuclear/Utilities", region="US",    currency="USD", fx=1.0),
    # Taiwan
    dict(symbol="TSM",       name="Taiwan Semiconductor", sector="Semiconductors",    region="Taiwan", currency="USD", fx=1.0),
]

THEMES = [
    "AI Infrastructure",
    "Korean Macro",
    "Energy Transition",
    "Semiconductor Supply Chain",
    "Memory Cycle",
    "Nuclear Baseload",
]

THEME_MAP = {
    "Semiconductors":       ["AI Infrastructure", "Semiconductor Supply Chain"],
    "Semiconductor Equip":  ["AI Infrastructure", "Semiconductor Supply Chain"],
    "Internet":             ["Korean Macro"],
    "Energy Storage":       ["Energy Transition", "Korean Macro"],
    "Renewables":           ["Energy Transition", "Korean Macro"],
    "Chemicals":            ["Korean Macro", "Energy Transition"],
    "Biotech":              ["Korean Macro"],
    "Nuclear/Utilities":    ["AI Infrastructure", "Nuclear Baseload"],
}


def _random_seed_trades(rng: np.random.Generator) -> pd.DataFrame:
    """Generate a realistic closed-trade history."""
    records = []
    current_date = START_DATE
    trade_id = 1

    # Simulate ~45 trades over ~16 weeks
    for _ in range(45):
        instr = rng.choice(INSTRUMENTS)
        # Advance time by 1-5 trading days between trades
        current_date += timedelta(days=int(rng.integers(1, 5)))
        open_date = current_date
        duration  = int(rng.integers(1, 14))
        close_date = open_date + timedelta(days=duration)

        direction = "Long" if rng.random() < 0.80 else "Short"

        # Notional per trade ~3-8% of NAV
        notional = INITIAL_NAV * rng.uniform(0.03, 0.08)
        price    = rng.uniform(10, 800)
        qty      = max(1, int(notional / price))

        # P&L: 58% win rate, fat tails
        win = rng.random() < 0.58
        if win:
            ret = rng.uniform(0.008, 0.055)   # +0.8% to +5.5%
        else:
            ret = -rng.uniform(0.005, 0.035)  # -0.5% to -3.5%

        gross_pnl  = qty * price * ret * (1 if direction == "Long" else -1)
        commission = qty * 0.005 + 0.35        # ~$0.005/share + $0.35 fixed
        net_pnl    = gross_pnl - commission

        sector = instr["sector"]
        themes = THEME_MAP.get(sector, ["Korean Macro"])

        records.append(dict(
            trade_id    = trade_id,
            symbol      = instr["symbol"],
            name        = instr["name"],
            sector      = sector,
            region      = instr["region"],
            currency    = instr["currency"],
            direction   = direction,
            open_date   = open_date,
            close_date  = close_date,
            duration    = duration,
            quantity    = qty,
            open_price  = round(price, 2),
            close_price = round(price * (1 + ret), 2),
            gross_pnl   = round(gross_pnl, 2),
            commission  = round(commission, 2),
            net_pnl     = round(net_pnl, 2),
            theme       = themes[0],
        ))
        trade_id += 1

    df = pd.DataFrame(records)
    df["open_date"]  = pd.to_datetime(df["open_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    return df.sort_values("close_date").reset_index(drop=True)


def _build_equity_curve(trades: pd.DataFrame, nav_start: float) -> pd.DataFrame:
    """Daily NAV from trade close dates."""
    if trades.empty:
        return pd.DataFrame(columns=["date", "nav", "daily_return", "drawdown"])

    date_range = pd.date_range(
        start=trades["close_date"].min(),
        end=trades["close_date"].max(),
        freq="B",
    )
    daily_pnl = trades.groupby("close_date")["net_pnl"].sum().reindex(date_range, fill_value=0)

    nav = nav_start + daily_pnl.cumsum()
    daily_return = nav.pct_change().fillna(0)
    rolling_max  = nav.cummax()
    drawdown     = (nav - rolling_max) / rolling_max * 100

    return pd.DataFrame({
        "date":         date_range,
        "nav":          nav.values,
        "daily_return": daily_return.values,
        "drawdown":     drawdown.values,
    })


def _build_open_positions(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate 3 current open positions."""
    positions = []
    for instr in rng.choice(INSTRUMENTS, size=3, replace=False):
        price    = rng.uniform(10, 800)
        qty      = max(1, int(INITIAL_NAV * rng.uniform(0.04, 0.07) / price))
        cost     = price * rng.uniform(0.96, 1.02)
        cur_px   = price
        unreal   = (cur_px - cost) * qty
        positions.append(dict(
            symbol        = instr["symbol"],
            name          = instr["name"],
            sector        = instr["sector"],
            region        = instr["region"],
            direction     = "Long",
            quantity      = qty,
            avg_cost      = round(cost, 2),
            current_price = round(cur_px, 2),
            market_value  = round(cur_px * qty, 2),
            unrealized_pnl= round(unreal, 2),
            open_date     = datetime.today() - timedelta(days=int(rng.integers(1, 10))),
            theme         = THEME_MAP.get(instr["sector"], ["Korean Macro"])[0],
        ))
    return pd.DataFrame(positions)


class FakeDataSource:
    """
    Drop-in replacement data source.  To wire IBKR, replace this class
    with IBKRFlexSource (ibkr_client.py) — the rest of the app is unchanged.
    """

    def __init__(self, seed: int = 42):
        self._rng    = np.random.default_rng(seed)
        self._trades = _random_seed_trades(self._rng)
        self._equity = _build_equity_curve(self._trades, INITIAL_NAV)
        self._pos    = _build_open_positions(self._rng)

    # ── Public API (same interface as IBKRFlexSource) ─────────────────────────
    def get_closed_trades(self) -> pd.DataFrame:
        return self._trades.copy()

    def get_equity_curve(self) -> pd.DataFrame:
        return self._equity.copy()

    def get_open_positions(self) -> pd.DataFrame:
        return self._pos.copy()

    def get_initial_nav(self) -> float:
        return INITIAL_NAV
