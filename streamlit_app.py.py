# Strategy Builder Agent (Forex & Crypto) – Streamlit MVP
# -------------------------------------------------------
# Features
# - Simple UI to pick market, style, indicators, risk, timeframe
# - Generates human‑readable rules from your selections
# - Fetches crypto OHLCV via CCXT or lets you upload a CSV
# - Computes indicators (SMA/EMA/RSI/MACD/Bollinger)
# - Backtests rules with a lightweight engine
# - Shows metrics + equity curve + trade list
# - Exports rules to text and a basic Pine Script v5 (for simple templates)
#
# How to run locally:
#   pip install streamlit pandas numpy ccxt yfinance ta matplotlib
#   streamlit run app.py
#
# CSV format (if you upload): columns with headers
#   timestamp, open, high, low, close, volume  (timestamp in ms or ISO8601)

import io
import json
import math
import time
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports guarded for environments without these packages
try:
    import ccxt  # for crypto data
except Exception:  # pragma: no cover
    ccxt = None

try:
    import yfinance as yf  # optional for forex via Yahoo (e.g., "EURUSD=X")
except Exception:  # pragma: no cover
    yf = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# ------------------------
# Utility & Indicator funcs
# ------------------------

def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        # Support ms (int) or ISO strings
        if np.issubdtype(df["timestamp"].dtype, np.number):
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("datetime").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        raise ValueError("Provide a 'timestamp' column (ms or ISO8601) or a DatetimeIndex.")
    return df


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(length).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(method="bfill")


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, length: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    basis = sma(series, length)
    dev = series.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    return upper, basis, lower


# ------------------------
# Data acquisition
# ------------------------

def fetch_crypto_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 1000,
                        exchange_id: str = "binance") -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Run: pip install ccxt")
    ex = getattr(ccxt, exchange_id)()
    ex.load_markets()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return to_datetime_index(df)


def fetch_forex_yfinance(ticker: str = "EURUSD=X", interval: str = "1h", lookback_days: int = 180) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    period_map = {
        "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
        "60m": "730d", "1h": "730d", "1d": "max"
    }
    period = period_map.get(interval, "730d")
    hist = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if hist.empty:
        raise RuntimeError("No data returned from yfinance.")
    hist = hist.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    hist.index = pd.to_datetime(hist.index, utc=True)
    hist["timestamp"] = (hist.index.view(np.int64) // 1_000_000)
    return hist


# ------------------------
# Strategy templates & rules
# ------------------------

@dataclass
class StrategyRules:
    name: str
    entry: List[str]
    exit: List[str]
    stop_loss_pct: float
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None


RISK_PRESETS = {
    "Conservative": {"rsi_buy": 30, "rsi_sell": 70, "sl": 0.02, "tp": 0.04},
    "Balanced": {"rsi_buy": 35, "rsi_sell": 70, "sl": 0.03, "tp": 0.06},
    "Aggressive": {"rsi_buy": 40, "rsi_sell": 80, "sl": 0.05, "tp": 0.10},
}


def generate_rules(template: str, indicators: List[str], risk: str, timeframe: str) -> StrategyRules:
    p = RISK_PRESETS.get(risk, RISK_PRESETS["Balanced"])
    entry, exit = [], []
    name = f"{template} | {risk} | {timeframe}"

    if template == "Trend-Following":
        entry.append("EMA50 > EMA200 (uptrend filter)")
        if "RSI" in indicators:
            entry.append(f"RSI < {p['rsi_buy']}")
        else:
            entry.append("Close crosses above EMA20")
        exit.append(f"RSI > {p['rsi_sell']} OR Close crosses below EMA20")

    elif template == "Mean-Reversion":
        if "Bollinger" in indicators:
            entry.append("Close touches below Lower Bollinger Band")
            exit.append("Close back to Basis (middle band)")
        else:
            entry.append(f"RSI < {p['rsi_buy']}")
            exit.append(f"RSI > {p['rsi_sell']}")

    elif template == "Breakout":
        entry.append("Close breaks above 20-bar High (confirm with Volume if available)")
        exit.append("Close falls back below 20-bar High OR time-based exit 10 bars")
        if "RSI" in indicators:
            entry.append(f"RSI between {p['rsi_buy']} and {p['rsi_sell']} (avoid overbought/oversold extremes)")

    else:
        entry.append("Close crosses above EMA20")
        exit.append("Close crosses below EMA20")

    return StrategyRules(
        name=name,
        entry=entry,
        exit=exit,
        stop_loss_pct=p["sl"],
        take_profit_pct=p["tp"],
        trailing_stop_pct=None,
    )


# ------------------------
# Backtester (long-only, single position)
# ------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["EMA20"] = ema(d["close"], 20)
    d["EMA50"] = ema(d["close"], 50)
    d["EMA200"] = ema(d["close"], 200)
    d["SMA20"] = sma(d["close"], 20)
    d["RSI14"] = rsi(d["close"], 14)
    macd_line, signal_line, hist = macd(d["close"]) 
    d["MACD"], d["MACD_signal"], d["MACD_hist"] = macd_line, signal_line, hist
    bb_u, bb_b, bb_l = bollinger(d["close"], 20, 2.0)
    d["BB_upper"], d["BB_basis"], d["BB_lower"] = bb_u, bb_b, bb_l
    return d


def rule_signals(df: pd.DataFrame, rules: StrategyRules, template: str) -> pd.DataFrame:
    d = df.copy()
    d["buy_sig"] = False
    d["sell_sig"] = False

    if template == "Trend-Following":
        d.loc[(d["EMA50"] > d["EMA200"]) & (d["close"].shift(1) < d["EMA20"].shift(1)) & (d["close"] > d["EMA20"]), "buy_sig"] = True
        d.loc[(d["close"].shift(1) > d["EMA20"].shift(1)) & (d["close"] < d["EMA20"]), "sell_sig"] = True
        if any("RSI" in s for s in rules.entry):
            d.loc[d["RSI14"] >= float(RISK_PRESETS['Conservative']['rsi_buy']), "buy_sig"] &= d["RSI14"] < float(RISK_PRESETS['Aggressive']['rsi_buy'])
    elif template == "Mean-Reversion":
        if any("Bollinger" in s for s in rules.entry):
            d.loc[(d["close"] < d["BB_lower"]) , "buy_sig"] = True
            d.loc[(d["close"] >= d["BB_basis"]) , "sell_sig"] = True
        else:
            p = None
            for rk, rv in RISK_PRESETS.items():
                if rk in rules.name:
                    p = rv
            p = p or RISK_PRESETS["Balanced"]
            d.loc[d["RSI14"] < p["rsi_buy"], "buy_sig"] = True
            d.loc[d["RSI14"] > p["rsi_sell"], "sell_sig"] = True
    elif template == "Breakout":
        rolling_high = d["high"].rolling(20, min_periods=20).max()
        d.loc[(d["close"].shift(1) <= rolling_high.shift(1)) & (d["close"] > rolling_high), "buy_sig"] = True
        d.loc[(d["close"] < rolling_high), "sell_sig"] = True
    else:
        d.loc[(d["close"].shift(1) < d["EMA20"].shift(1)) & (d["close"] > d["EMA20"]), "buy_sig"] = True
        d.loc[(d["close"].shift(1) > d["EMA20"].shift(1)) & (d["close"] < d["EMA20"]), "sell_sig"] = True

    return d


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: Dict[str, float]


def backtest(df: pd.DataFrame, rules: StrategyRules, template: str, initial_capital: float = 10_000.0,
             fee_pct: float = 0.0004) -> BacktestResult:
    d = df.copy()
    d = compute_indicators(d)
    d = rule_signals(d, rules, template)

    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    equity = initial_capital
    qty = 0.0

    eq_curve = []
    trade_records = []

    for ts, row in d.iterrows():
        price = float(row["close"]) if not math.isnan(row["close"]) else None
        if price is None:
            eq_curve.append((ts, equity))
            continue

        # Check exit first
        if position == 1:
            exit_now = False
            reason = None
            # Stop-loss / Take-profit
            if rules.stop_loss_pct:
                if price <= entry_price * (1 - rules.stop_loss_pct):
                    exit_now = True
                    reason = f"SL {rules.stop_loss_pct*100:.1f}%"
            if not exit_now and rules.take_profit_pct:
                if price >= entry_price * (1 + rules.take_profit_pct):
                    exit_now = True
                    reason = f"TP {rules.take_profit_pct*100:.1f}%"
            # Signal-based exit
            if not exit_now and row.get("sell_sig", False):
                exit_now = True
                reason = "Signal"
            if exit_now:
                proceeds = qty * price * (1 - fee_pct)
                pnl = proceeds - (qty * entry_price)
                equity += pnl
                trade_records.append({
                    "entry_time": entry_time,
                    "entry": entry_price,
                    "exit_time": ts,
                    "exit": price,
                    "pnl": pnl,
                    "ret_pct": pnl / (qty * entry_price) if qty > 0 else 0.0,
                    "reason": reason,
                })
                position = 0
                qty = 0.0
                entry_price = 0.0

        # Entry after exit logic
        if position == 0 and row.get("buy_sig", False):
            qty = (equity * 0.99) / price  # use 99% of equity
            cost = qty * price * (1 + fee_pct)
            if cost <= equity:
                equity -= cost - (qty * price)  # deduct only fees effectively
                entry_price = price
                entry_time = ts
                position = 1

        # Mark to market
        if position == 1:
            eq_curve.append((ts, equity + qty * price))
        else:
            eq_curve.append((ts, equity))

    # Close at end if still open
    if position == 1:
        price = float(d.iloc[-1]["close"])
        proceeds = qty * price
        pnl = proceeds - (qty * entry_price)
        equity += pnl
        trade_records.append({
            "entry_time": entry_time,
            "entry": entry_price,
            "exit_time": d.index[-1],
            "exit": price,
            "pnl": pnl,
            "ret_pct": pnl / (qty * entry_price) if qty > 0 else 0.0,
            "reason": "EOD",
        })

    eq_series = pd.Series({ts: val for ts, val in eq_curve})
    trades_df = pd.DataFrame(trade_records)

    # Metrics
    if not eq_series.empty:
        ret_pct_total = (eq_series.iloc[-1] / eq_series.iloc[0]) - 1.0
    else:
        ret_pct_total = 0.0

    # Max drawdown
    if not eq_series.empty:
        roll_max = eq_series.cummax()
        dd = eq_series / roll_max - 1.0
        max_dd = dd.min()
    else:
        max_dd = 0.0

    win_rate = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0
    num_trades = int(len(trades_df))

    metrics = {
        "Total Return %": round(ret_pct_total * 100, 2),
        "Max Drawdown %": round(max_dd * 100, 2),
        "Win Rate %": round(win_rate * 100, 2),
        "Trades": num_trades,
        "Final Equity": round(float(eq_series.iloc[-1]) if not eq_series.empty else initial_capital, 2),
    }

    return BacktestResult(trades=trades_df, equity=eq_series, metrics=metrics)


# ------------------------
# Pine Script generator (basic)
# ------------------------

def to_pinescript(rules: StrategyRules, template: str, timeframe: str) -> str:
    lines = [
        "//@version=5",
        f"strategy(\"{rules.name}\", overlay=true, initial_capital=10000)",
        "src = close",
        "ema20 = ta.ema(src, 20)",
        "ema50 = ta.ema(src, 50)",
        "ema200 = ta.ema(src, 200)",
        "rsi14 = ta.rsi(src, 14)",
        "bb_basis = ta.sma(src, 20)",
        "bb_dev = ta.stdev(src, 20)",
        "bb_upper = bb_basis + 2.0*bb_dev",
        "bb_lower = bb_basis - 2.0*bb_dev",
    ]

    buy_cond = ""
    sell_cond = ""

    if template == "Trend-Following":
        buy_cond = "ema50 > ema200 and ta.crossover(src, ema20)"
        sell_cond = "ta.crossunder(src, ema20)"
        if any("RSI <" in s for s in rules.entry):
            # extract number
            th = [int(s.split("<")[-1].strip()) for s in rules.entry if "RSI <" in s]
            if th:
                buy_cond += f" and rsi14 < {th[0]}"
    elif template == "Mean-Reversion":
        if any("Bollinger" in s for s in rules.entry):
            buy_cond = "src < bb_lower"
            sell_cond = "src >= bb_basis"
        else:
            # default RSI MR
            p = RISK_PRESETS.get("Balanced")
            buy_cond = f"rsi14 < {p['rsi_buy']}"
            sell_cond = f"rsi14 > {p['rsi_sell']}"
    elif template == "Breakout":
        buy_cond = "ta.crossover(src, ta.highest(high, 20))"
        sell_cond = "src < ta.highest(high, 20)"
    else:
        buy_cond = "ta.crossover(src, ema20)"
        sell_cond = "ta.crossunder(src, ema20)"

    lines += [
        f"longEntry = {buy_cond}",
        f"longExit = {sell_cond}",
        "if (longEntry)",
        "    strategy.entry(\"Long\", strategy.long)",
    ]

    # Stop-loss / Take-profit
    sl = rules.stop_loss_pct or 0.03
    tp = rules.take_profit_pct or 0.06
    lines += [
        f"strategy.exit(\"Exit\", \"Long\", stop=strategy.position_avg_price*(1-{sl:.4f}), limit=strategy.position_avg_price*(1+{tp:.4f}))",
    ]

    lines += [
        "plot(ema20, title=\"EMA20\")",
        "plot(ema50, title=\"EMA50\")",
        "plot(ema200, title=\"EMA200\")",
    ]

    return "\n".join(lines)


# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="Strategy Builder Agent", layout="wide")
st.title("🧠 Strategy Builder Agent – Forex & Crypto (MVP)")

with st.sidebar:
    st.header("Inputs")
    market = st.selectbox("Market", ["Crypto", "Forex", "CSV Upload"])  
    template = st.selectbox("Strategy Template", ["Trend-Following", "Mean-Reversion", "Breakout"])  
    indicators = st.multiselect("Indicators", ["EMA", "SMA", "RSI", "MACD", "Bollinger"], default=["EMA", "RSI"])  
    risk = st.selectbox("Risk", list(RISK_PRESETS.keys()), index=1)

    timeframe_map_crypto = {"15m":"15m", "1h":"1h", "4h":"4h", "1d":"1d"}
    timeframe_map_fx = {"15m":"15m", "1h":"60m", "4h":"60m", "1d":"1d"}

    if market == "Crypto":
        exchange_id = st.text_input("Exchange (ccxt id)", value="binance")
        symbol = st.text_input("Symbol (e.g., BTC/USDT)", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", list(timeframe_map_crypto.keys()), index=1)
        lookback = st.slider("Bars to fetch", min_value=300, max_value=5000, value=1500, step=100)
    elif market == "Forex":
        ticker = st.text_input("Yahoo Ticker (e.g., EURUSD=X)", value="EURUSD=X")
        timeframe = st.selectbox("Timeframe", list(timeframe_map_fx.keys()), index=1)
        lookback = st.slider("Lookback days (approx)", min_value=30, max_value=720, value=180, step=30)
    else:
        timeframe = st.selectbox("Assumed Timeframe (for labeling)", ["15m","1h","4h","1d"], index=1)
        uploaded = st.file_uploader("Upload CSV (timestamp,open,high,low,close,volume)", type=["csv"])  

    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=10_000.0, step=500.0)
    fee_pct = st.number_input("Fee % (per side)", min_value=0.0, value=0.04, step=0.01, help="In percent, e.g., 0.04 = 0.04%") / 100.0

st.subheader("1) Generated Strategy Rules")
rules = generate_rules(template, indicators, risk, timeframe)
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Name:** {rules.name}")
    st.markdown("**Entry:**\n- " + "\n- ".join(rules.entry))
with col2:
    st.markdown("**Exit:**\n- " + "\n- ".join(rules.exit))
    st.markdown(f"**Stop-Loss:** {rules.stop_loss_pct*100:.2f}%  |  **Take-Profit:** { (rules.take_profit_pct or 0)*100:.2f}%")

# Data section
st.subheader("2) Data")
load_ok = False
err = None
df: Optional[pd.DataFrame] = None

try:
    if market == "Crypto":
        with st.spinner("Fetching crypto data via CCXT..."):
            df = fetch_crypto_ohlcv(symbol=symbol, timeframe=timeframe_map_crypto[timeframe], limit=int(lookback), exchange_id=exchange_id)
            load_ok = True
    elif market == "Forex":
        with st.spinner("Fetching forex data via yfinance..."):
            df = fetch_forex_yfinance(ticker=ticker, interval=timeframe_map_fx[timeframe], lookback_days=int(lookback))
            load_ok = True
    else:
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df = to_datetime_index(df)
            load_ok = True
        else:
            st.info("Upload a CSV to proceed or switch to Crypto/Forex data.")
except Exception as e:
    err = str(e)

if err:
    st.error(f"Data error: {err}")

if load_ok and df is not None and not df.empty:
    st.write(df.tail())

    st.subheader("3) Backtest")
    with st.spinner("Running backtest..."):
        df_bt = df.copy()
        result = backtest(df_bt, rules, template, initial_capital=initial_capital, fee_pct=fee_pct)

    m = result.metrics
    m_cols = st.columns(len(m))
    for (k, v), c in zip(m.items(), m_cols):
        c.metric(k, v)

    st.subheader("Equity Curve")
    st.line_chart(result.equity.rename("Equity"))

    st.subheader("Trades")
    if not result.trades.empty:
        st.dataframe(result.trades)
        csv_buf = io.StringIO()
        result.trades.to_csv(csv_buf, index=False)
        st.download_button("Download Trades CSV", csv_buf.getvalue(), file_name="trades.csv", mime="text/csv")
    else:
        st.info("No trades generated by the rules in this period.")

    st.subheader("4) Export Strategy")
    rules_text = f"""
    Strategy: {rules.name}
    Timeframe: {timeframe}

    ENTRY RULES:\n- {"\n- ".join(rules.entry)}

    EXIT RULES:\n- {"\n- ".join(rules.exit)}

    Risk Management:\n- Stop-Loss: {rules.stop_loss_pct*100:.2f}%\n- Take-Profit: {(rules.take_profit_pct or 0)*100:.2f}%
    """.strip()

    st.code(rules_text)

    pine = to_pinescript(rules, template, timeframe)
    st.download_button("Download Pine Script (v5)", data=pine, file_name="strategy.pine", mime="text/plain")

else:
    st.stop()

st.caption("This is an educational MVP. Markets are risky; do your own research.")
