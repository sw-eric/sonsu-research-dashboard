"""
Sonsu Research — Performance Dashboard
Streamlit app  ·  fake_data.py drives template data
Swap FakeDataSource → IBKRFlexSource in ibkr_client.py to go live.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ibkr_client import IBKRFlexSource

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sonsu Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Color palette (mirrors quartz.config.ts lightMode)
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    bg        = "#eae9e7",
    bg_card   = "#f0eeec",
    border    = "#d4d2ce",
    gray      = "#8a877f",
    dim       = "#3d3a35",
    text      = "#1a1816",
    primary   = "#456079",
    primary_l = "#5a7a8f",
    primary_a = "rgba(69,96,121,0.12)",
    profit    = "#3d6b4f",
    profit_a  = "rgba(61,107,79,0.13)",
    loss      = "#8b3a3a",
    loss_a    = "rgba(139,58,58,0.13)",
    neutral   = "#b5a98e",
)


def chart_layout(**overrides) -> dict:
    """
    Base Plotly layout dict. Caller overrides win — no duplicate-key errors.
    Usage: fig.update_layout(**chart_layout(height=300, margin=dict(l=0,r=0,t=8,b=0)))
    """
    base = dict(
        paper_bgcolor = C["bg"],
        plot_bgcolor  = C["bg"],
        font          = dict(family="Inter, sans-serif", color=C["text"], size=12),
        legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin        = dict(l=16, r=16, t=32, b=16),
    )
    base.update(overrides)
    return base


def style_axes(fig):
    fig.update_xaxes(gridcolor=C["border"], linecolor=C["border"], zeroline=False)
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border"], zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C["bg"]};
    color: {C["text"]};
  }}
  [data-testid="stSidebar"] {{
    background-color: {C["bg_card"]};
    border-right: 1px solid {C["border"]};
  }}
  [data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}

  .metric-card {{
    background: {C["bg_card"]};
    border: 1px solid {C["border"]};
    border-radius: 6px;
    padding: 1rem 1.2rem;
    min-height: 90px;
  }}
  .metric-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {C["gray"]};
    margin-bottom: 0.35rem;
  }}
  .metric-value {{
    font-family: 'EB Garamond', serif;
    font-size: 1.9rem;
    font-weight: 500;
    color: {C["text"]};
    line-height: 1.1;
  }}
  .metric-value.profit {{ color: {C["profit"]}; }}
  .metric-value.loss   {{ color: {C["loss"]};   }}
  .metric-sub {{
    font-size: 0.75rem;
    color: {C["gray"]};
    margin-top: 0.2rem;
  }}
  .section-header {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C["gray"]};
    border-bottom: 1px solid {C["border"]};
    padding-bottom: 0.4rem;
    margin: 1.6rem 0 0.9rem 0;
  }}
  .page-title {{
    font-family: 'EB Garamond', serif;
    font-size: 1.8rem;
    font-weight: 500;
    color: {C["text"]};
    margin-bottom: 0;
  }}
  .page-sub {{
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {C["gray"]};
    margin-top: 0.15rem;
    margin-bottom: 1.2rem;
  }}
  .accent-bar {{
    width: 2.5rem; height: 2px;
    background: {C["primary"]};
    margin: 0.5rem 0 1.2rem 0;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; }}
  [data-testid="stTabs"] [role="tab"] {{
    font-size: 0.8rem; font-weight: 500; letter-spacing: 0.03em;
    color: {C["gray"]}; border-bottom: 2px solid transparent;
  }}
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {C["primary"]}; border-bottom-color: {C["primary"]};
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_data():
    try:
        src    = IBKRFlexSource()
        trades = src.get_closed_trades()
        equity = src.get_equity_curve()
        pos    = src.get_open_positions()
        nav0   = src.get_initial_nav()
        return trades, equity, pos, nav0, None
    except Exception as e:
        empty_trades = __import__("pandas").DataFrame(columns=[
            "trade_id","symbol","name","sector","region","currency","direction",
            "open_date","close_date","duration","quantity","open_price","close_price",
            "gross_pnl","commission","net_pnl","theme",
        ])
        empty_equity = __import__("pandas").DataFrame(columns=["date","nav","daily_return","drawdown"])
        empty_pos    = __import__("pandas").DataFrame(columns=[
            "symbol","name","sector","region","direction","quantity",
            "avg_cost","current_price","market_value","unrealized_pnl","theme",
        ])
        return empty_trades, empty_equity, empty_pos, 1_000.0, str(e)

trades_raw, equity_raw, positions, NAV0, _ibkr_error = load_data()
if _ibkr_error:
    st.warning(f"⚠️ Could not load IBKR data — {_ibkr_error}", icon="⚠️")


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(trades: pd.DataFrame, equity: pd.DataFrame, nav0: float) -> dict:
    if trades.empty:
        return {}
    wins         = trades[trades["net_pnl"] > 0]
    losses       = trades[trades["net_pnl"] < 0]
    net_pnl      = trades["net_pnl"].sum()
    net_return   = net_pnl / nav0 * 100
    win_rate     = len(wins) / len(trades) * 100
    gross_profit = wins["net_pnl"].sum()
    gross_loss   = abs(losses["net_pnl"].sum())
    profit_factor= gross_profit / gross_loss if gross_loss else float("inf")
    avg_win      = wins["net_pnl"].mean()  if len(wins)   else 0
    avg_loss     = losses["net_pnl"].mean() if len(losses) else 0
    expectancy   = (win_rate/100)*avg_win + (1-win_rate/100)*avg_loss
    rr_ratio     = avg_win / abs(avg_loss) if avg_loss else float("inf")

    dr      = equity["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    rf      = 0.053 / 252
    excess  = dr - rf
    sharpe  = (excess.mean() / dr.std() * np.sqrt(252)) if dr.std() else 0
    neg_dr  = dr[dr < 0]
    sortino = (excess.mean() / neg_dr.std() * np.sqrt(252)) if len(neg_dr) and neg_dr.std() else 0
    max_dd  = equity["drawdown"].min()
    ann_ret = ((equity["nav"].iloc[-1] / nav0) ** (252 / max(len(equity), 1)) - 1) * 100
    calmar  = ann_ret / abs(max_dd) if max_dd else 0

    results = (trades["net_pnl"] > 0).astype(int).tolist()
    max_w = max_l = cur_w = cur_l = 0
    for r in results:
        if r: cur_w += 1; cur_l = 0
        else: cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w); max_l = max(max_l, cur_l)

    return dict(
        net_pnl=net_pnl, net_return=net_return, win_rate=win_rate,
        profit_factor=profit_factor, sharpe=sharpe, sortino=sortino,
        calmar=calmar, max_drawdown=max_dd, avg_win=avg_win, avg_loss=avg_loss,
        rr_ratio=rr_ratio, expectancy=expectancy, total_trades=len(trades),
        n_wins=len(wins), n_losses=len(losses),
        best_trade=trades["net_pnl"].max(), worst_trade=trades["net_pnl"].min(),
        avg_duration=trades["duration"].mean(),
        max_cons_wins=max_w, max_cons_loss=max_l,
        gross_profit=gross_profit, gross_loss=gross_loss,
    )


def metric(label, value, sub="", cls=""):
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def pnl_color(v): return "profit" if v > 0 else ("loss" if v < 0 else "")
def fmt_pnl(v):   return f"{'+'if v>0 else ''}${v:,.2f}"
def fmt_pct(v, d=1): return f"{'+'if v>0 else ''}{v:.{d}f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.4rem">
      <div style="font-family:'EB Garamond',serif;font-size:1.3rem;font-weight:500;color:{C['text']}">Sonsu Research</div>
      <div style="font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;color:{C['gray']}">Performance Dashboard</div>
      <div style="width:2rem;height:2px;background:{C['primary']};margin-top:0.5rem"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;color:{C['gray']};margin-bottom:0.4rem'>Date Range</div>", unsafe_allow_html=True)
    if not trades_raw.empty:
        all_dates  = pd.to_datetime(trades_raw["close_date"])
        min_date, max_date = all_dates.min().date(), all_dates.max().date()
    else:
        from datetime import date, timedelta
        max_date = date.today()
        min_date = max_date - timedelta(days=365)
    date_range = st.date_input("", value=(min_date, max_date), min_value=min_date, max_value=max_date, label_visibility="collapsed", key="date_range")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;color:{C['gray']};margin-bottom:0.4rem'>Region</div>", unsafe_allow_html=True)
    region_opts = sorted(trades_raw["region"].unique()) if not trades_raw.empty else []
    sel_region = st.selectbox("", ["All"] + region_opts, label_visibility="collapsed", key="sel_region")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;color:{C['gray']};margin-bottom:0.4rem'>Theme</div>", unsafe_allow_html=True)
    theme_opts = sorted(trades_raw["theme"].unique()) if not trades_raw.empty else []
    sel_theme = st.selectbox("", ["All"] + theme_opts, label_visibility="collapsed", key="sel_theme")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;color:{C['gray']};margin-bottom:0.4rem'>Direction</div>", unsafe_allow_html=True)
    sel_dir = st.selectbox("", ["All", "Long", "Short"], label_visibility="collapsed", key="sel_dir")

    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.72rem;color:{C['gray']}">
      <b>Data:</b> Template (demo)<br>
      <b>Account:</b> $1,000 personal<br>
      <b>Mandate:</b> APAC · AI · Energy
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────
d0 = pd.Timestamp(date_range[0] if len(date_range) > 0 else min_date)
d1 = pd.Timestamp(date_range[1] if len(date_range) > 1 else max_date)

trades = trades_raw.copy()
trades = trades[(trades["close_date"] >= d0) & (trades["close_date"] <= d1)]
if sel_region != "All": trades = trades[trades["region"] == sel_region]
if sel_theme  != "All": trades = trades[trades["theme"]  == sel_theme]
if sel_dir    != "All": trades = trades[trades["direction"] == sel_dir]

equity = equity_raw[(equity_raw["date"] >= d0) & (equity_raw["date"] <= d1)].copy()
stats  = compute_stats(trades, equity, NAV0)


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-title">Performance Dashboard</div>
<div class="page-sub">Sonsu Research · Asia-Pacific Research · Personal Account · $1,000</div>
<div class="accent-bar"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_trades, tab_analytics, tab_positions = st.tabs([
    "Overview", "Trade Log", "Analytics", "Open Positions"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    if stats:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.markdown(metric("Net P&L",       fmt_pnl(stats["net_pnl"]),    f"{stats['total_trades']} closed trades", pnl_color(stats["net_pnl"])),    unsafe_allow_html=True)
        c2.markdown(metric("Net Return",    fmt_pct(stats["net_return"]),  f"on ${NAV0:,.0f} capital",              pnl_color(stats["net_return"])),  unsafe_allow_html=True)
        c3.markdown(metric("Win Rate",      f"{stats['win_rate']:.1f}%",   f"{stats['n_wins']}W · {stats['n_losses']}L"),                            unsafe_allow_html=True)
        c4.markdown(metric("Profit Factor", f"{stats['profit_factor']:.2f}", f"Expectancy {fmt_pnl(stats['expectancy'])}"),                          unsafe_allow_html=True)
        c5.markdown(metric("Sharpe Ratio",  f"{stats['sharpe']:.2f}",      f"Sortino {stats['sortino']:.2f}"),                                       unsafe_allow_html=True)
        c6.markdown(metric("Max Drawdown",  f"{stats['max_drawdown']:.1f}%", f"Calmar {stats['calmar']:.2f}", "loss"),                               unsafe_allow_html=True)

    # Equity + drawdown
    st.markdown("<div class='section-header'>Equity Curve</div>", unsafe_allow_html=True)
    if not equity.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(
            x=equity["date"], y=equity["nav"], mode="lines", name="NAV",
            line=dict(color=C["primary"], width=2),
            fill="tozeroy", fillcolor=C["primary_a"],
            hovertemplate="<b>%{x|%b %d}</b><br>NAV: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=equity["date"], y=equity["drawdown"], mode="lines", name="Drawdown",
            line=dict(color=C["loss"], width=1.5),
            fill="tozeroy", fillcolor=C["loss_a"],
            hovertemplate="<b>%{x|%b %d}</b><br>DD: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)
        fig.update_layout(**chart_layout(height=420, showlegend=False, margin=dict(l=0,r=0,t=8,b=0)))
        style_axes(fig)
        fig.update_yaxes(title_text="NAV ($)", row=1, tickprefix="$", tickfont_size=11)
        fig.update_yaxes(title_text="DD (%)",  row=2, ticksuffix="%",  tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    # Secondary stats
    if stats:
        st.markdown("<div class='section-header'>Trade Statistics</div>", unsafe_allow_html=True)
        s1,s2,s3,s4,s5,s6 = st.columns(6)
        s1.markdown(metric("Avg Win",      fmt_pnl(stats["avg_win"]),     "", "profit"), unsafe_allow_html=True)
        s2.markdown(metric("Avg Loss",     fmt_pnl(stats["avg_loss"]),    "", "loss"),   unsafe_allow_html=True)
        s3.markdown(metric("Win/Loss R",   f"{stats['rr_ratio']:.2f}x",   "avg win ÷ avg loss"), unsafe_allow_html=True)
        s4.markdown(metric("Best Trade",   fmt_pnl(stats["best_trade"]),  "", "profit"), unsafe_allow_html=True)
        s5.markdown(metric("Worst Trade",  fmt_pnl(stats["worst_trade"]), "", "loss"),   unsafe_allow_html=True)
        s6.markdown(metric("Avg Duration", f"{stats['avg_duration']:.1f}d", "calendar days"), unsafe_allow_html=True)

    # Monthly heatmap
    st.markdown("<div class='section-header'>Monthly Returns</div>", unsafe_allow_html=True)
    if not trades.empty:
        trades_m = trades.copy()
        trades_m["month"] = trades_m["close_date"].dt.to_period("M")
        monthly = trades_m.groupby("month")["net_pnl"].sum().reset_index()
        monthly["return_pct"] = monthly["net_pnl"] / NAV0 * 100
        monthly["year"] = monthly["month"].dt.year
        monthly["mon"]  = monthly["month"].dt.month
        pivot = monthly.pivot(index="year", columns="mon", values="return_pct")
        mon_abbr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [mon_abbr[m-1] for m in pivot.columns]

        fig2 = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=[str(y) for y in pivot.index],
            colorscale=[[0.0, C["loss"]], [0.5, C["bg_card"]], [1.0, C["profit"]]],
            zmid=0,
            text=[[f"{v:+.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}", textfont=dict(size=11, color=C["text"]),
            hovertemplate="<b>%{x} %{y}</b><br>%{z:+.2f}%<extra></extra>",
            showscale=True,
            colorbar=dict(ticksuffix="%", len=0.8, thickness=12, tickfont_size=10),
        ))
        fig2.update_layout(**chart_layout(height=160, margin=dict(l=0,r=60,t=8,b=0)))
        style_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # P&L by region + theme
    st.markdown("<div class='section-header'>P&L Breakdown</div>", unsafe_allow_html=True)
    if not trades.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            by_region = trades.groupby("region")["net_pnl"].sum().sort_values()
            fig3 = go.Figure(go.Bar(
                x=by_region.values, y=by_region.index, orientation="h",
                marker_color=[C["profit"] if v >= 0 else C["loss"] for v in by_region.values],
                text=[fmt_pnl(v) for v in by_region.values],
                textposition="outside", textfont=dict(size=11),
                hovertemplate="%{y}: <b>%{x:+.2f}</b><extra></extra>",
            ))
            fig3.update_layout(**chart_layout(height=220, margin=dict(l=0,r=60,t=30,b=0),
                title=dict(text="by Region", font_size=11, font_color=C["gray"], x=0),
                xaxis_tickprefix="$"))
            style_axes(fig3)
            st.plotly_chart(fig3, use_container_width=True)

        with col_r:
            by_theme = trades.groupby("theme")["net_pnl"].sum().sort_values()
            fig4 = go.Figure(go.Bar(
                x=by_theme.values, y=by_theme.index, orientation="h",
                marker_color=[C["profit"] if v >= 0 else C["loss"] for v in by_theme.values],
                text=[fmt_pnl(v) for v in by_theme.values],
                textposition="outside", textfont=dict(size=11),
                hovertemplate="%{y}: <b>%{x:+.2f}</b><extra></extra>",
            ))
            fig4.update_layout(**chart_layout(height=220, margin=dict(l=0,r=60,t=30,b=0),
                title=dict(text="by Theme", font_size=11, font_color=C["gray"], x=0),
                xaxis_tickprefix="$"))
            style_axes(fig4)
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_trades:
    if trades.empty:
        st.info("No trades match the current filters.")
    else:
        st.markdown("<div class='section-header'>P&L Distribution</div>", unsafe_allow_html=True)
        fig_dist = go.Figure(go.Histogram(
            x=trades["net_pnl"], nbinsx=25,
            marker_color=C["primary"], marker_line_color=C["border"],
            marker_line_width=0.8, opacity=0.85,
            hovertemplate="P&L: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig_dist.add_vline(x=0, line_color=C["gray"], line_dash="dash", line_width=1)
        fig_dist.update_layout(**chart_layout(height=200, margin=dict(l=0,r=0,t=8,b=0),
            xaxis_tickprefix="$"))
        style_axes(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("<div class='section-header'>Closed Trades</div>", unsafe_allow_html=True)
        display = trades[["close_date","symbol","name","sector","region","direction",
                           "quantity","open_price","close_price","gross_pnl","commission","net_pnl","duration","theme"]].copy()
        display["close_date"] = display["close_date"].dt.strftime("%Y-%m-%d")
        display = display.rename(columns={
            "close_date":"Close","symbol":"Symbol","name":"Name","sector":"Sector",
            "region":"Region","direction":"Dir","quantity":"Qty",
            "open_price":"Open $","close_price":"Close $",
            "gross_pnl":"Gross P&L","commission":"Comm","net_pnl":"Net P&L",
            "duration":"Days","theme":"Theme",
        })
        def _style(val):
            if isinstance(val,(int,float)):
                if val>0: return f"color:{C['profit']};font-weight:500"
                if val<0: return f"color:{C['loss']};font-weight:500"
            return ""
        styled = display.style.map(_style, subset=["Net P&L","Gross P&L"]) \
            .format({"Open $":"${:,.2f}","Close $":"${:,.2f}",
                     "Gross P&L":"${:+,.2f}","Comm":"${:.2f}","Net P&L":"${:+,.2f}"}) \
            .set_properties(**{"font-size":"0.8rem"})
        st.dataframe(styled, use_container_width=True, height=480)

        st.markdown("<div class='section-header'>Performance by Instrument</div>", unsafe_allow_html=True)
        sym_pnl = trades.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False)
        fig_sym = go.Figure(go.Bar(
            x=sym_pnl.index, y=sym_pnl.values,
            marker_color=[C["profit"] if v>=0 else C["loss"] for v in sym_pnl.values],
            text=[fmt_pnl(v) for v in sym_pnl.values],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>%{y:+.2f}<extra></extra>",
        ))
        fig_sym.update_layout(**chart_layout(height=280, margin=dict(l=0,r=0,t=8,b=0),
            yaxis_tickprefix="$"))
        style_axes(fig_sym)
        st.plotly_chart(fig_sym, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if not stats:
        st.info("No data for the selected filters.")
    else:
        st.markdown("<div class='section-header'>Risk-Adjusted Performance</div>", unsafe_allow_html=True)
        a1,a2,a3,a4 = st.columns(4)
        a1.markdown(metric("Sharpe Ratio",  f"{stats['sharpe']:.2f}",        "annualised, rf=5.3%"),           unsafe_allow_html=True)
        a2.markdown(metric("Sortino Ratio", f"{stats['sortino']:.2f}",       "downside deviation"),            unsafe_allow_html=True)
        a3.markdown(metric("Calmar Ratio",  f"{stats['calmar']:.2f}",        "ann. return / max DD"),          unsafe_allow_html=True)
        a4.markdown(metric("Profit Factor", f"{stats['profit_factor']:.2f}", f"${stats['gross_profit']:.0f} gross profit"), unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Win / Loss Profile</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)

        with b1:
            fig_wr = go.Figure(go.Pie(
                values=[stats["n_wins"], stats["n_losses"]], labels=["Wins","Losses"],
                hole=0.62, marker_colors=[C["profit"], C["loss"]],
                textinfo="label+percent", textfont=dict(size=12),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig_wr.add_annotation(
                text=f"<b>{stats['win_rate']:.1f}%</b><br><span style='font-size:10px'>win rate</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=C["text"]),
            )
            fig_wr.update_layout(**chart_layout(height=280, margin=dict(l=0,r=0,t=30,b=0),
                title=dict(text="Win Rate", font_size=11, font_color=C["gray"], x=0)))
            st.plotly_chart(fig_wr, use_container_width=True)

        with b2:
            fig_wl = go.Figure(go.Bar(
                x=["Avg Win","Avg Loss"], y=[stats["avg_win"], abs(stats["avg_loss"])],
                marker_color=[C["profit"], C["loss"]],
                text=[fmt_pnl(stats["avg_win"]), fmt_pnl(stats["avg_loss"])],
                textposition="outside", textfont=dict(size=12),
            ))
            fig_wl.update_layout(**chart_layout(height=280, margin=dict(l=0,r=0,t=30,b=0),
                title=dict(text="Avg Win vs Avg Loss", font_size=11, font_color=C["gray"], x=0),
                yaxis_tickprefix="$", showlegend=False))
            style_axes(fig_wl)
            st.plotly_chart(fig_wl, use_container_width=True)

        # Rolling metrics
        st.markdown("<div class='section-header'>Rolling 20-Trade Metrics</div>", unsafe_allow_html=True)
        if len(trades) >= 5:
            rt = trades.sort_values("close_date").reset_index(drop=True)
            w  = min(20, len(rt))
            roll_wr  = rt["net_pnl"].gt(0).rolling(w).mean() * 100
            roll_g   = rt["net_pnl"].clip(lower=0).rolling(w).sum()
            roll_l   = rt["net_pnl"].clip(upper=0).abs().rolling(w).sum()
            roll_pf  = (roll_g / roll_l.replace(0, np.nan)).fillna(0)

            fig_roll = make_subplots(rows=1, cols=2,
                subplot_titles=["Rolling Win Rate", "Rolling Profit Factor"])
            fig_roll.add_trace(go.Scatter(x=rt["close_date"], y=roll_wr,
                mode="lines", line=dict(color=C["primary"], width=2),
                hovertemplate="%{x|%b %d}: %{y:.1f}%<extra></extra>"), row=1, col=1)
            fig_roll.add_hline(y=50, line_dash="dash", line_color=C["gray"], line_width=1, row=1, col=1)
            fig_roll.add_trace(go.Scatter(x=rt["close_date"], y=roll_pf,
                mode="lines", line=dict(color=C["primary_l"], width=2),
                hovertemplate="%{x|%b %d}: %{y:.2f}x<extra></extra>"), row=1, col=2)
            fig_roll.add_hline(y=1, line_dash="dash", line_color=C["gray"], line_width=1, row=1, col=2)
            fig_roll.update_layout(**chart_layout(height=260, margin=dict(l=0,r=0,t=30,b=0), showlegend=False))
            style_axes(fig_roll)
            fig_roll.update_yaxes(ticksuffix="%", row=1, col=1)
            fig_roll.update_yaxes(ticksuffix="x", row=1, col=2)
            st.plotly_chart(fig_roll, use_container_width=True)

        # Duration scatter
        st.markdown("<div class='section-header'>Duration vs P&L</div>", unsafe_allow_html=True)
        fig_dur = go.Figure(go.Scatter(
            x=trades["duration"], y=trades["net_pnl"], mode="markers",
            marker=dict(color=[C["profit"] if v>=0 else C["loss"] for v in trades["net_pnl"]],
                        size=8, opacity=0.75, line=dict(color=C["border"], width=0.5)),
            text=trades["symbol"],
            hovertemplate="<b>%{text}</b><br>%{x}d · %{y:+.2f}<extra></extra>",
        ))
        fig_dur.add_hline(y=0, line_color=C["gray"], line_dash="dash", line_width=1)
        fig_dur.update_layout(**chart_layout(height=260, margin=dict(l=0,r=0,t=8,b=0),
            xaxis_title="Days Held", yaxis_title="Net P&L ($)", yaxis_tickprefix="$"))
        style_axes(fig_dur)
        st.plotly_chart(fig_dur, use_container_width=True)

        # Streaks
        st.markdown("<div class='section-header'>Streak Summary</div>", unsafe_allow_html=True)
        st1,st2,st3,st4 = st.columns(4)
        st1.markdown(metric("Max Cons. Wins",   str(stats["max_cons_wins"]),   "consecutive wins"),      unsafe_allow_html=True)
        st2.markdown(metric("Max Cons. Losses",  str(stats["max_cons_loss"]),  "consecutive losses","loss"), unsafe_allow_html=True)
        st3.markdown(metric("Expectancy",        fmt_pnl(stats["expectancy"]), "avg $ per trade"),       unsafe_allow_html=True)
        st4.markdown(metric("Gross Profit",      fmt_pnl(stats["gross_profit"]), f"vs ${stats['gross_loss']:.2f} gross loss","profit"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.markdown("<div class='section-header'>Current Positions</div>", unsafe_allow_html=True)
    if positions.empty:
        st.info("No open positions.")
    else:
        total_unreal = positions["unrealized_pnl"].sum()
        total_mktval = positions["market_value"].sum()
        p1,p2,p3 = st.columns(3)
        p1.markdown(metric("Open Positions", str(len(positions)),        "active trades"),    unsafe_allow_html=True)
        p2.markdown(metric("Market Value",   f"${total_mktval:,.2f}",   "total exposure"),   unsafe_allow_html=True)
        p3.markdown(metric("Unrealized P&L", fmt_pnl(total_unreal),     "mark-to-market",    pnl_color(total_unreal)), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pos_disp = positions[["symbol","name","sector","region","direction","quantity",
                               "avg_cost","current_price","market_value","unrealized_pnl","theme"]].copy()
        pos_disp = pos_disp.rename(columns={"symbol":"Symbol","name":"Name","sector":"Sector",
            "region":"Region","direction":"Dir","quantity":"Qty","avg_cost":"Avg Cost",
            "current_price":"Last $","market_value":"Mkt Val","unrealized_pnl":"Unreal P&L","theme":"Theme"})
        styled_pos = pos_disp.style \
            .map(lambda v: f"color:{C['profit']};font-weight:500" if isinstance(v,(int,float)) and v>0
                       else (f"color:{C['loss']};font-weight:500" if isinstance(v,(int,float)) and v<0 else ""),
                       subset=["Unreal P&L"]) \
            .format({"Avg Cost":"${:,.2f}","Last $":"${:,.2f}","Mkt Val":"${:,.2f}","Unreal P&L":"${:+,.2f}"}) \
            .set_properties(**{"font-size":"0.82rem"})
        st.dataframe(styled_pos, use_container_width=True, height=220)

        by_sec = positions.groupby("sector")["market_value"].sum().reset_index()
        st.markdown("<div class='section-header'>Exposure by Sector</div>", unsafe_allow_html=True)
        fig_exp = go.Figure(go.Pie(
            labels=by_sec["sector"], values=by_sec["market_value"],
            hole=0.55, marker_colors=[C["primary"], C["primary_l"], C["neutral"], C["dim"]],
            textinfo="label+percent", textfont_size=12,
            hovertemplate="%{label}: $%{value:,.2f} (%{percent})<extra></extra>",
        ))
        fig_exp.update_layout(**chart_layout(height=280, margin=dict(l=0,r=0,t=8,b=0)))
        st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:1.5rem;padding:0.9rem 1rem;background:{C['bg_card']};
                border:1px solid {C['border']};border-radius:6px;font-size:0.75rem;color:{C['gray']}">
      <b style="color:{C['text']}">Live data note:</b> Positions show template data.
      Connect live IBKR positions via <code>IBKRFlexSource</code> in <code>ibkr_client.py</code>.
    </div>""", unsafe_allow_html=True)
