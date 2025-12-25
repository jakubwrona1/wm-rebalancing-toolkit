# WM Rebalancing Toolkit

A Python tool that generates a trade list (BUY/SELL/HOLD) to rebalance a portfolio to target weights.
Built for Wealth Management / model-portfolio style workflows.

## Why this project
Wealth managers often manage model portfolios (e.g., 60/40, balanced, growth) and need to generate a clear trade list when:
- new cash is added,
- allocations drift away from target weights,
- constraints require limiting concentration.

## Features
- Fetches latest prices via `yfinance`
- Rebalances to target weights
- Supports cash additions (`--cash`)
- Supports a cash buffer (keep part of the portfolio in cash) (`--cash-buffer`)
- Minimum trade filter to ignore small trades (`--min-trade`)
- Max/min target weight constraints (`--min-weight`, `--max-weight`)
- Optional whole-share “cash-safe” rounding (`--round-shares`)
- Exports CSV + Excel output

## Inputs
- `data/holdings.csv` with columns: `Ticker,Shares`
- `data/target_weights.csv` with columns: `Ticker,TargetWeight` (must sum to 1.0)

Default sample tickers:
- SPY (US equities), AGG (bonds), VXUS (intl equities), GLD (gold)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


```
## Example Output

The tool generates:
- A trade list with BUY / SELL / HOLD decisions
- A portfolio summary
- An Excel file for client-facing review

Example output files:
- `outputs/trades.csv`
- `outputs/summary.csv`
- `outputs/rebalance_output.xlsx`

This mirrors how portfolio managers prepare rebalancing instructions in practice.
