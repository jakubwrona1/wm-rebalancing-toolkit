from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class RebalanceConfig:
    """
    Config options (these map to your CLI flags in run_rebalance.py)

    cash:
        Extra cash you want to add to the portfolio (CHF).
        Example: --cash 1000 means "assume client adds CHF 1000".

    cash_buffer:
        Fraction of TOTAL portfolio value kept as cash (not invested).
        Example: 0.02 means keep 2% cash, invest 98%.

    min_weight / max_weight:
        Optional constraints on target weights.
        Example: max_weight=0.40 prevents concentration > 40%.

    allow_sells:
        If False, only buy trades are allowed (no sells).

    min_trade_chf:
        Ignore small trades below this CHF threshold.

    round_shares:
        If True, trade shares are rounded to whole integers (more realistic).
        This affects the trade list output, and can introduce small tracking error.
    """
    cash: float = 0.0
    cash_buffer: float = 0.0

    min_weight: float = 0.0
    max_weight: float = 1.0

    allow_sells: bool = True
    min_trade_chf: float = 0.0

    round_shares: bool = False


def load_holdings(path: str) -> pd.DataFrame:
    """
    Reads holdings CSV with columns: Ticker, Shares
    Example:
        Ticker,Shares
        SPY,10
        AGG,25
    """
    df = pd.read_csv(path)

    required = {"Ticker", "Shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"holdings file missing columns: {missing}")

    # Normalize tickers (e.g. ' spy ' -> 'SPY')
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Shares"] = pd.to_numeric(df["Shares"], errors="raise")

    # If user listed the same ticker multiple times, sum it
    return df.groupby("Ticker", as_index=False)["Shares"].sum()


def load_targets(path: str) -> pd.DataFrame:
    """
    Reads target weights CSV with columns: Ticker, TargetWeight.
    Validates weights sum to 1.0.

    Example:
        Ticker,TargetWeight
        SPY,0.60
        AGG,0.40
    """
    df = pd.read_csv(path)

    required = {"Ticker", "TargetWeight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"targets file missing columns: {missing}")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["TargetWeight"] = pd.to_numeric(df["TargetWeight"], errors="raise")

    total = float(df["TargetWeight"].sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Target weights must sum to 1.0 (currently {total:.6f})")

    return df


def fetch_latest_prices(tickers: list[str]) -> pd.Series:
    """
    Fetches latest adjusted close prices from Yahoo Finance via yfinance.
    Returns a pandas Series indexed by ticker.
    """
    data = yf.download(tickers, period="7d", auto_adjust=False, progress=False)["Adj Close"]

    # If only one ticker, yfinance returns a Series, so normalize to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Use the last available row of prices
    last = data.dropna(how="all").iloc[-1]
    last.index = last.index.astype(str).str.upper()

    missing = set(tickers) - set(last.index)
    if missing:
        raise ValueError(f"Missing prices for tickers: {sorted(missing)}")

    return last


def compute_rebalance(
    holdings: pd.DataFrame,
    targets: pd.DataFrame,
    prices: pd.Series,
    cfg: RebalanceConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core logic:
    1) Merge tickers from holdings + targets into a universe
    2) Compute current CHF value of each holding
    3) Apply target weights (after constraints) to investable value
    4) TradeValueCHF = TargetValue - CurrentValue
    5) Apply constraints like no-sell and min-trade
    6) Compute shares to trade (optionally rounded)
    7) Output trade list + summary
    """

    # --- 1) Build the universe of tickers ---
    universe = sorted(set(holdings["Ticker"]).union(set(targets["Ticker"])))
    px = prices.reindex(universe)

    # --- 2) Merge holdings + targets ---
    df = pd.DataFrame({"Ticker": universe}).merge(holdings, on="Ticker", how="left")
    df["Shares"] = df["Shares"].fillna(0.0)

    df = df.merge(targets, on="Ticker", how="left")
    df["TargetWeight"] = df["TargetWeight"].fillna(0.0)

    # --- 3) Apply min/max weight constraints to the target weights ---
    # This is a "simple and stable" approach:
    # - clamp weights into [min_weight, max_weight]
    # - renormalize so weights sum to 1 again
    if cfg.min_weight < 0 or cfg.max_weight > 1 or cfg.min_weight > cfg.max_weight:
        raise ValueError("Invalid min/max weights. Example: --min-weight 0.00 --max-weight 0.40")

    w = df["TargetWeight"].astype(float).copy()
    w = w.clip(lower=cfg.min_weight, upper=cfg.max_weight)

    if w.sum() <= 0:
        raise ValueError("Target weights sum to 0 after constraints. Check target_weights.csv and constraints.")

    df["TargetWeight"] = w / w.sum()

    # --- 4) Compute current portfolio values ---
    df["Price"] = px.values
    df["CurrentValue"] = df["Shares"] * df["Price"]

    current_portfolio_value = float(df["CurrentValue"].sum())
    total_value = float(current_portfolio_value + cfg.cash)

    if total_value <= 0:
        raise ValueError("Total portfolio value must be > 0")

    # --- 5) Apply cash buffer ---
    # Keep some % as cash, invest the rest according to target weights
    if cfg.cash_buffer < 0 or cfg.cash_buffer >= 1:
        raise ValueError("cash_buffer must be between 0 and 1 (e.g., 0.02 for 2 percent)")

    investable_value = total_value * (1 - cfg.cash_buffer)
    expected_cash_left = total_value - investable_value

    # --- 6) Compute target CHF values and required trades ---
    df["TargetValue"] = df["TargetWeight"] * investable_value
    df["TradeValueCHF"] = df["TargetValue"] - df["CurrentValue"]

    # Optional: disallow sells (only buys)
    if not cfg.allow_sells:
        df["TradeValueCHF"] = df["TradeValueCHF"].clip(lower=0.0)

    # Ignore tiny trades (reduce noise)
    if cfg.min_trade_chf > 0:
        df.loc[df["TradeValueCHF"].abs() < cfg.min_trade_chf, "TradeValueCHF"] = 0.0

    # Convert CHF trade value to shares (approx)
    df["TradeShares"] = df["TradeValueCHF"] / df["Price"]

    # --- 7) Optional rounding to whole shares (more realistic trading) ---
    # NOTE: rounding introduces small tracking error vs exact target weights.
    if cfg.round_shares:
        def round_shares_safe(x: float) -> int:
            if x > 0:
                # BUY: round DOWN so we don't overspend cash
                return int(np.floor(x))
            elif x < 0:
            # SELL: round UP toward zero so we don't oversell
                return int(np.ceil(x))
            return 0

        df["TradeSharesRounded"] = df["TradeShares"].apply(round_shares_safe)
        df["TradeValueCHFRounded"] = df["TradeSharesRounded"] * df["Price"]
    else:
        df["TradeSharesRounded"] = df["TradeShares"]
        df["TradeValueCHFRounded"] = df["TradeValueCHF"]

    # Side should be based on the final (rounded) trade amount
    def side(x: float) -> str:
        if x > 0:
            return "BUY"
        if x < 0:
            return "SELL"
        return "HOLD"

    df["Side"] = df["TradeValueCHFRounded"].apply(side)

    # --- 8) Output tables ---
    trades = df[[
        "Ticker",
        "Price",
        "Shares",
        "CurrentValue",
        "TargetWeight",
        "TargetValue",
        "Side",
        "TradeValueCHF",
        "TradeShares",
        "TradeValueCHFRounded",
        "TradeSharesRounded"
    ]].copy()

    # Helpful summary: proves the rules were applied
    buys = float(trades.loc[trades["TradeValueCHFRounded"] > 0, "TradeValueCHFRounded"].sum())
    sells = float(-trades.loc[trades["TradeValueCHFRounded"] < 0, "TradeValueCHFRounded"].sum())

    summary = pd.DataFrame([{
        "CurrentPortfolioValueCHF": current_portfolio_value,
        "CashAddedCHF": float(cfg.cash),
        "TargetPortfolioValueCHF": total_value,

        "CashBufferFraction": float(cfg.cash_buffer),
        "InvestableValueCHF": float(investable_value),
        "ExpectedCashLeftCHF": float(expected_cash_left),

        "MinWeight": float(cfg.min_weight),
        "MaxWeight": float(cfg.max_weight),

        "RoundShares": bool(cfg.round_shares),
        "AllowSells": bool(cfg.allow_sells),
        "MinTradeCHF": float(cfg.min_trade_chf),

        "TotalBuyCHF_Rounded": buys,
        "TotalSellCHF_Rounded": sells
    }])

    return trades, summary