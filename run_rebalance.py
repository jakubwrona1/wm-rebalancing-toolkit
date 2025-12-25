import argparse
import pandas as pd

from src.rebalance import (
    load_holdings,
    load_targets,
    fetch_latest_prices,
    compute_rebalance,
    RebalanceConfig
)


def main():
    parser = argparse.ArgumentParser(
        description="WM Rebalancing Toolkit",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--holdings", default="data/holdings.csv")
    parser.add_argument("--targets", default="data/target_weights.csv")
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--cash-buffer", type=float, default=0.0,
                    help="Cash buffer as a fraction (e.g., 0.02 for 2 percent)")
    parser.add_argument("--no-sells", action="store_true")
    parser.add_argument("--min-trade", type=float, default=0.0)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Min allowed weight (e.g., 0.00)")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Max allowed weight (e.g., 0.40)")
    parser.add_argument("--round-shares", action="store_true", help="Round trade shares to whole numbers")
    args = parser.parse_args()

    holdings = load_holdings(args.holdings)
    targets = load_targets(args.targets)

    universe = sorted(set(holdings["Ticker"]).union(set(targets["Ticker"])))
    prices = fetch_latest_prices(universe)

    cfg = RebalanceConfig(
        cash=args.cash,
        allow_sells=(not args.no_sells),
        min_trade_chf=args.min_trade,
        cash_buffer=args.cash_buffer,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        round_shares=args.round_shares
    )

    trades, summary = compute_rebalance(holdings, targets, prices, cfg)

    out_dir = args.output.rstrip("/")
    trades_path_csv = f"{out_dir}/trades.csv"
    summary_path_csv = f"{out_dir}/summary.csv"
    xlsx_path = f"{out_dir}/rebalance_output.xlsx"

    trades.to_csv(trades_path_csv, index=False)
    summary.to_csv(summary_path_csv, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        trades.to_excel(writer, sheet_name="Trades", index=False)

    print("Saved:")
    print(f"- {trades_path_csv}")
    print(f"- {summary_path_csv}")
    print(f"- {xlsx_path}")


if __name__ == "__main__":
    main()