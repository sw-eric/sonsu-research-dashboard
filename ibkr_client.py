"""
ibkr_client.py — IBKR Flex Web Service integration (stub).

When you're ready to connect live data:
  1. Generate a Flex Query in IBKR Account Management:
       Reports → Flex Queries → Create → pick TradesExecuted + PortfolioAnalyst fields
  2. Add to .streamlit/secrets.toml:
       IBKR_FLEX_TOKEN   = "your_token"
       IBKR_FLEX_QUERY_ID = "your_query_id"
  3. Swap FakeDataSource for IBKRFlexSource in app.py.

IBKR Flex REST endpoint:
  https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest
"""

from __future__ import annotations
import pandas as pd
import streamlit as st
import requests
import xml.etree.ElementTree as ET
from datetime import datetime


FLEX_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
GET_URL  = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"


class IBKRFlexSource:
    """
    Reads closed trades and portfolio data from an IBKR Flex Report.
    Exposes the same interface as FakeDataSource.
    """

    def __init__(self):
        self._token    = st.secrets["IBKR_FLEX_TOKEN"]
        self._query_id = st.secrets["IBKR_FLEX_QUERY_ID"]
        self._xml      = self._fetch_flex_statement()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _fetch_flex_statement(self) -> ET.Element:
        import time

        # Step 1: request the statement — retry up to 4 times for transient errors
        ref_code = None
        for attempt in range(4):
            r = requests.get(FLEX_URL, params={"t": self._token, "q": self._query_id, "v": 3}, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            status = root.findtext("Status") or root.findtext(".//Status")
            if status == "Success":
                ref_code = root.findtext("ReferenceCode") or root.findtext(".//ReferenceCode")
                break
            err = root.findtext("ErrorMessage") or root.findtext(".//ErrorMessage") or ""
            # Transient errors — wait and retry
            if "try again" in err.lower() or "too many" in err.lower() or "could not be generated" in err.lower():
                if attempt < 3:
                    time.sleep(15)
                    continue
            raise RuntimeError(f"IBKR Flex Step 1 failed — {err}")

        if not ref_code:
            raise RuntimeError(f"IBKR Flex Step 1: no ReferenceCode in response — {r.text[:400]}")

        ref_code = root.findtext("ReferenceCode") or root.findtext(".//ReferenceCode")
        if not ref_code:
            raise RuntimeError(f"IBKR Flex Step 1: no ReferenceCode in response — {r.text[:400]}")

        # Step 2: poll until the report is ready (up to ~60s)
        for attempt in range(12):
            time.sleep(5)
            r2 = requests.get(GET_URL, params={"t": self._token, "q": ref_code, "v": 3}, timeout=30)
            r2.raise_for_status()
            root2 = ET.fromstring(r2.text)
            # Still generating — IBKR returns another FlexStatementResponse
            if root2.tag == "FlexStatementResponse":
                status2 = root2.findtext("Status") or ""
                if "not available" in status2.lower() or "try again" in status2.lower():
                    continue
                err2 = root2.findtext("ErrorMessage") or r2.text[:400]
                raise RuntimeError(f"IBKR Flex Step 2 failed — {status2!r} | {err2}")
            return root2

        raise RuntimeError("IBKR Flex report not ready after 60 seconds — reboot the app to retry.")

    def _parse_trades(self) -> pd.DataFrame:
        rows = []
        for t in self._xml.iter("Trade"):
            a = t.attrib
            if a.get("openCloseIndicator") != "C":
                continue
            rows.append(dict(
                trade_id    = a.get("tradeID"),
                symbol      = a.get("symbol"),
                name        = a.get("description"),
                sector      = "Unknown",          # enrich via yfinance/Refinitiv
                region      = "Unknown",
                currency    = a.get("currency"),
                direction   = "Long" if float(a.get("quantity", 0)) > 0 else "Short",
                open_date   = pd.to_datetime(a.get("openDateTime")),
                close_date  = pd.to_datetime(a.get("dateTime")),
                duration    = None,               # compute after parsing
                quantity    = abs(float(a.get("quantity", 0))),
                open_price  = float(a.get("cost", 0)) / abs(float(a.get("quantity", 1))),
                close_price = float(a.get("tradePrice", 0)),
                gross_pnl   = float(a.get("fifoPnlRealized", 0)),
                commission  = abs(float(a.get("ibCommission", 0))),
                net_pnl     = float(a.get("fifoPnlRealized", 0)) + float(a.get("ibCommission", 0)),
                theme       = "Untagged",
            ))
        if not rows:
            return pd.DataFrame(columns=[
                "trade_id","symbol","name","sector","region","currency","direction",
                "open_date","close_date","duration","quantity","open_price","close_price",
                "gross_pnl","commission","net_pnl","theme",
            ])
        df = pd.DataFrame(rows)
        df["duration"] = (df["close_date"] - df["open_date"]).dt.days
        return df

    # ── Public API ────────────────────────────────────────────────────────────
    def get_closed_trades(self) -> pd.DataFrame:
        return self._parse_trades()

    def get_equity_curve(self) -> pd.DataFrame:
        # Try EquitySummaryByReportDateInBase first
        rows = []
        for node in self._xml.iter("EquitySummaryByReportDateInBase"):
            rows.append(dict(
                date = pd.to_datetime(node.attrib.get("reportDate")),
                nav  = float(node.attrib.get("total", 0)),
            ))
        if rows:
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            df["daily_return"] = df["nav"].pct_change().fillna(0)
            roll_max = df["nav"].cummax()
            df["drawdown"] = (df["nav"] - roll_max) / roll_max * 100
            return df

        # Fall back: build curve from closed trades
        trades = self._parse_trades()
        if trades.empty:
            return pd.DataFrame(columns=["date", "nav", "daily_return", "drawdown"])
        nav0 = self.get_initial_nav()
        date_range = pd.date_range(start=trades["close_date"].min(), end=trades["close_date"].max(), freq="B")
        daily_pnl  = trades.groupby("close_date")["net_pnl"].sum().reindex(date_range, fill_value=0)
        nav        = nav0 + daily_pnl.cumsum()
        roll_max   = nav.cummax()
        return pd.DataFrame({
            "date":         date_range,
            "nav":          nav.values,
            "daily_return": nav.pct_change().fillna(0).values,
            "drawdown":     ((nav - roll_max) / roll_max * 100).values,
        })

    def get_open_positions(self) -> pd.DataFrame:
        rows = []
        for p in self._xml.iter("OpenPosition"):
            a = p.attrib
            cost    = float(a.get("costBasisMoney", 0))
            mktval  = float(a.get("positionValue", 0))
            rows.append(dict(
                symbol         = a.get("symbol"),
                name           = a.get("description"),
                sector         = "Unknown",
                region         = "Unknown",
                direction      = "Long" if float(a.get("position", 0)) > 0 else "Short",
                quantity       = abs(float(a.get("position", 0))),
                avg_cost       = float(a.get("costBasisPrice", 0)),
                current_price  = float(a.get("markPrice", 0)),
                market_value   = mktval,
                unrealized_pnl = mktval - cost,
                open_date      = pd.NaT,
                theme          = "Untagged",
            ))
        return pd.DataFrame(rows)

    def get_initial_nav(self) -> float:
        return 1_000.0
