#!/usr/bin/env python3
"""
Clean Bloomberg commodity panel into aligned prices and standardized log-returns.

Run from the data/ directory (or anywhere; paths are resolved relative to this file):
    python clean_commodities.py

Outputs (written next to this script):
    prices.csv        — date-intersection aligned price levels
    returns.csv       — per-commodity z-scored log-returns (autoencoder input)
    CLEANING_LOG.md   — provenance for every transformation
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and workbook configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
RAW_XLSX = DATA_DIR / "Raw_Data_Final_Universe.xlsx"
PRICES_CSV = DATA_DIR / "prices.csv"
RETURNS_CSV = DATA_DIR / "returns.csv"
CLEANING_LOG = DATA_DIR / "CLEANING_LOG.md"

SHEET = "Sheet1"
BLOCK_WIDTH = 3  # date, price, spacer
EXCEL_DATE_ORIGIN = "1899-12-30"
HEADER_PATTERN = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")

# Canonical CSV column names keyed by Bloomberg ticker (when present in headers).
TICKER_TO_NAME: dict[str, str] = {
    "CO1": "Brent",
    "CL1": "WTI",
    "NG1": "NaturalGas",
    "XB1": "Gasoline",
    "QS1": "Diesel",
    "HO1": "HeatingOil",
    "GC1": "Gold",
    "SI1": "Silver",
    "HG1": "Copper",
    "LA1": "Aluminium",
    "LN1": "Nickel",
    "LX1": "Zinc",
    "PL1": "Platinum",
    "KC1": "Coffee",
    "C 1": "Corn",
    "CT1": "Cotton",
    "LH1": "LeanHogs",
    "LC1": "LiveCattle",
    "SB1": "Sugar",
    "S1": "Soybeans",
    "BO1": "SoybeanOil",
    "KC": "HRWWheat",
    "XW1": "ThermalCoal",
    "CKCA": "CokingCoal",
    "ZME1": "Methanol",
    "BAPA": "Propane",
}

# Headers without tickers in parentheses.
DISPLAY_TO_NAME: dict[str, str] = {
    "HRC Steel": "HRCSteel",
    "SGX Iron Ore": "SGXIronOre",
    "Lithium": "Lithium",
}

SECTOR_BY_NAME: dict[str, str] = {
    "Brent": "Energy",
    "WTI": "Energy",
    "NaturalGas": "Energy",
    "Gasoline": "Energy",
    "Diesel": "Energy",
    "HeatingOil": "Energy",
    "ThermalCoal": "Energy",
    "CokingCoal": "Energy",
    "Methanol": "Energy",
    "Propane": "Energy",
    "Gold": "Metals",
    "Silver": "Metals",
    "Copper": "Metals",
    "Aluminium": "Metals",
    "Nickel": "Metals",
    "Zinc": "Metals",
    "Platinum": "Metals",
    "HRCSteel": "Metals",
    "SGXIronOre": "Metals",
    "Lithium": "Metals",
    "Coffee": "Agri",
    "Corn": "Agri",
    "Cotton": "Agri",
    "LeanHogs": "Agri",
    "LiveCattle": "Agri",
    "Sugar": "Agri",
    "Soybeans": "Agri",
    "SoybeanOil": "Agri",
    "HRWWheat": "Agri",
}

# Sparse or short-history series parsed from the workbook but omitted from the
# shared calendar so the aligned panel keeps a usable history length.
ALIGNMENT_EXCLUSIONS: set[str] = {"CokingCoal", "Propane"}


def _md_cell(value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(_md_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
    return "\n".join(lines)


class ProvenanceLog:
    """Append-only provenance log rendered as Markdown; mirrors key lines to stdout."""

    def __init__(self) -> None:
        self._blocks: list[str] = []

    def _emit(self, block: str, console: str | None = None) -> None:
        self._blocks.append(block)
        if console is not None:
            print(console)

    def section(self, title: str, level: int = 2) -> None:
        hashes = "#" * level
        self._emit(f"\n{hashes} {title}\n", console=f"\n=== {title} ===")

    def paragraph(self, text: str) -> None:
        self._emit(f"{text}\n", console=text)

    def rule(self, text: str) -> None:
        self._emit(f"> **Rule:** {text}\n", console=f"Rule: {text}")

    def bullet(self, text: str) -> None:
        self._emit(f"- {text}\n", console=f"  {text}")

    def code(self, text: str) -> None:
        self._emit(f"`{text}`\n", console=text)

    def table(self, headers: list[str], rows: list[list[Any]]) -> None:
        self._emit(_md_table(headers, rows) + "\n")

    def write(self, path: Path) -> None:
        generated = datetime.now().isoformat(timespec="seconds")
        meta = _md_table(
            ["Field", "Value"],
            [
                ["Generated", generated],
                ["Source workbook", f"`{RAW_XLSX.name}`"],
                ["Sheet", f"`{SHEET}`"],
                ["Outputs", f"`{PRICES_CSV.name}`, `{RETURNS_CSV.name}`"],
            ],
        )
        body = (
            "# Commodity cleaning provenance log\n\n"
            "Automated record of every parsing, alignment, guard, and transform step "
            "applied by `clean_commodities.py`.\n\n"
            "## Run metadata\n\n"
            f"{meta}\n\n"
            "---\n\n"
            + "\n".join(self._blocks)
            + "\n"
        )
        path.write_text(body, encoding="utf-8")


def parse_excel_date(value) -> pd.Timestamp | pd.NaT:
    """Parse Bloomberg dates: Timestamps, strings, or Excel serial integers."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > 10_000:
            return pd.to_datetime(value, unit="D", origin=EXCEL_DATE_ORIGIN)
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def parse_header_label(raw: Any) -> tuple[str, str]:
    """Return (display label, ticker) from a Bloomberg header cell."""
    text = str(raw).strip() if pd.notna(raw) else ""
    if not text:
        return "", ""
    match = HEADER_PATTERN.match(text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text, ""


def canonical_name(display: str, ticker: str) -> str:
    if ticker and ticker in TICKER_TO_NAME:
        return TICKER_TO_NAME[ticker]
    if display in DISPLAY_TO_NAME:
        return DISPLAY_TO_NAME[display]
    return re.sub(r"[^A-Za-z0-9]+", "", display)


def discover_commodity_blocks(header_row: pd.Series) -> list[dict[str, Any]]:
    """Infer (date, price) column pairs from the header row."""
    blocks: list[dict[str, Any]] = []
    for start_col in range(0, len(header_row), BLOCK_WIDTH):
        price_col = start_col + 1
        if price_col >= len(header_row):
            break
        display, ticker = parse_header_label(header_row.iloc[price_col])
        if not display:
            continue
        name = canonical_name(display, ticker)
        blocks.append(
            {
                "date_col": start_col,
                "price_col": price_col,
                "display": display,
                "ticker": ticker or "—",
                "name": name,
                "sector": SECTOR_BY_NAME.get(name, "Other"),
            }
        )
    return blocks


def parse_price_blocks(log: ProvenanceLog) -> dict[str, pd.Series]:
    """Read Sheet1 horizontal blocks and return named price series."""
    log.section("Input and block layout")

    raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET, header=None)
    log.table(
        ["Metric", "Value"],
        [
            ["Sheet", SHEET],
            ["Raw shape (rows × cols)", f"{raw.shape[0]} × {raw.shape[1]}"],
        ],
    )

    header_row = raw.iloc[0]
    blocks = discover_commodity_blocks(header_row)
    if not blocks:
        log.paragraph("**FATAL:** No commodity blocks found in header row; stopping.")
        log.write(CLEANING_LOG)
        sys.exit(1)

    log.paragraph(
        f"Detected **{len(blocks)}** `(date, price)` blocks from the header row "
        f"(stride = `{BLOCK_WIDTH}` columns)."
    )
    log.table(
        ["#", "Name", "Ticker", "Sector", "Date col", "Price col", "Header"],
        [
            [
                i + 1,
                block["name"],
                block["ticker"],
                block["sector"],
                block["date_col"],
                block["price_col"],
                f"{block['display']} ({block['ticker']})"
                if block["ticker"] != "—"
                else block["display"],
            ]
            for i, block in enumerate(blocks)
        ],
    )

    data = raw.iloc[1:].reset_index(drop=True)
    series_by_name: dict[str, pd.Series] = {}
    parse_rows: list[list[Any]] = []

    log.section("Per-commodity parse summary", level=3)
    for i, block in enumerate(blocks):
        name = block["name"]
        if name in series_by_name:
            log.paragraph(
                f"**FATAL:** Duplicate canonical name `{name}` at block {i + 1} "
                f"(ticker `{block['ticker']}`)."
            )
            log.write(CLEANING_LOG)
            sys.exit(1)

        frame = data[[block["date_col"], block["price_col"]]].copy()
        frame.columns = ["date", "price"]
        frame["date"] = frame["date"].map(parse_excel_date)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna(subset=["date", "price"])
        n_before_dedup = len(frame)
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        n_dupes = n_before_dedup - len(frame)
        frame = frame.sort_values("date")
        series = frame.set_index("date")["price"]
        series.name = name
        series_by_name[name] = series

        dmin, dmax = series.index.min(), series.index.max()
        notes = f"deduped {n_dupes} date(s)" if n_dupes else "—"
        parse_rows.append(
            [
                i + 1,
                name,
                block["ticker"],
                block["sector"],
                len(series),
                dmin.date(),
                dmax.date(),
                notes,
            ]
        )

    log.table(
        ["#", "Name", "Ticker", "Sector", "Obs", "Start", "End", "Notes"],
        parse_rows,
    )
    log.paragraph(f"Parsed **{len(series_by_name)}** commodity series.")
    return series_by_name


def align_on_master_calendar(
    series_by_name: dict[str, pd.Series], log: ProvenanceLog
) -> pd.DataFrame:
    log.section("Calendar alignment")
    log.rule(
        "Outer-join all series on date, then keep only dates where every commodity "
        "has a non-missing price (complete-case intersection)."
    )

    excluded = sorted(name for name in series_by_name if name in ALIGNMENT_EXCLUSIONS)
    included = {
        name: series
        for name, series in series_by_name.items()
        if name not in ALIGNMENT_EXCLUSIONS
    }
    if excluded:
        log.paragraph(
            "The following parsed series are **excluded from alignment** because "
            "their sparse or short calendars would collapse the shared panel: "
            + ", ".join(f"`{name}`" for name in excluded)
            + "."
        )
    if not included:
        log.paragraph("**FATAL:** No series remain for calendar alignment.")
        log.write(CLEANING_LOG)
        sys.exit(1)

    prices = pd.DataFrame(included)
    n_outer = len(prices)
    complete = prices.dropna(how="any")
    n_complete = len(complete)
    dropped = n_outer - n_complete

    align_rows: list[list[Any]] = [
        ["After outer join", f"{n_outer} rows × {prices.shape[1]} cols"],
        ["Rows dropped (any missing price)", dropped],
        ["Rows retained (intersection)", n_complete],
        ["Aligned matrix shape", f"{complete.shape[0]} × {complete.shape[1]}"],
    ]
    if n_complete > 0:
        align_rows.append(
            [
                "Date range",
                f"{complete.index.min().date()} → {complete.index.max().date()}",
            ]
        )
    log.table(["Step", "Result"], align_rows)

    return complete


def guard_non_positive_prices(prices: pd.DataFrame, log: ProvenanceLog) -> pd.DataFrame:
    log.section("Non-positive price guard")
    log.rule("Any price ≤ 0 is replaced with `NaN`; each replacement is logged below.")
    log.paragraph("Applied **before** log-returns so `ln(price)` is never taken on non-positive values.")

    guarded = prices.copy()
    mask = guarded <= 0
    violations = mask.stack()
    violations = violations[violations]

    if violations.empty:
        log.paragraph("No non-positive prices found.")
        return guarded

    guard_rows: list[list[Any]] = []
    for (date, commodity), _ in violations.items():
        value = prices.loc[date, commodity]
        guard_rows.append([commodity, pd.Timestamp(date).date(), value, "`NaN`"])
        guarded.loc[date, commodity] = np.nan

    log.table(["Commodity", "Date", "Original value", "Replacement"], guard_rows)
    log.paragraph(f"**Total guard actions:** {len(violations)}")
    return guarded


def compute_log_returns(prices: pd.DataFrame, log: ProvenanceLog) -> pd.DataFrame:
    log.section("Log-returns")
    log.paragraph("Formula per commodity: `r_t = ln(p_t / p_{t-1})`")

    log_rets = np.log(prices / prices.shift(1))
    n_before = len(log_rets)

    if n_before == 0:
        log.paragraph("**FATAL:** Price panel is empty after alignment; cannot compute returns.")
        log.write(CLEANING_LOG)
        sys.exit(1)

    first_date = log_rets.index[0]
    drop_rows: list[list[Any]] = [
        [
            str(first_date.date()),
            "First calendar row",
            "All commodities",
            "No prior price for returns",
        ],
    ]

    after_first = log_rets.iloc[1:]
    nan_rows = after_first.index[after_first.isna().any(axis=1)]
    for dt in nan_rows:
        bad_cols = after_first.columns[after_first.loc[dt].isna()].tolist()
        drop_rows.append(
            [
                str(pd.Timestamp(dt).date()),
                "NaN log-return",
                ", ".join(bad_cols),
                "Row removed from return matrix",
            ]
        )

    returns = after_first.dropna(how="any")
    n_after = len(returns)

    log.table(["Date", "Reason", "Affected commodities", "Action"], drop_rows)
    log.table(
        ["Metric", "Value"],
        [
            ["Rows before return drop", n_before - 1],
            ["Rows dropped (NaN returns)", len(nan_rows)],
            ["Final return rows", n_after],
            ["Shape (before z-score)", f"{returns.shape[0]} × {returns.shape[1]}"],
        ],
    )
    if n_after > 0:
        log.paragraph(
            f"Return date range: **{returns.index.min().date()}** → "
            f"**{returns.index.max().date()}**"
        )

    return returns


def zscore_returns(returns: pd.DataFrame, log: ProvenanceLog) -> pd.DataFrame:
    log.section("Per-commodity z-score standardization")
    log.rule(
        "Each column is transformed as `(r - mean) / std` using the final return sample "
        "(population std, `ddof=0`)."
    )
    log.paragraph(
        "Full-sample scaling is used here for **representation learning** only. "
        "For out-of-sample forecasting, refit the scaler inside each rolling window "
        "to avoid look-ahead bias."
    )

    standardized = returns.apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    zscore_rows = [
        [name, f"{returns[name].mean():.6f}", f"{returns[name].std(ddof=0):.6f}"]
        for name in standardized.columns
    ]
    log.table(["Commodity", "Mean (raw returns)", "Std (raw returns)"], zscore_rows)
    log.paragraph(f"Standardized return matrix shape: **{standardized.shape[0]} × {standardized.shape[1]}**")
    return standardized


def write_matrix_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index.name = "date"
    out.reset_index(inplace=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def log_final_summary(prices: pd.DataFrame, returns: pd.DataFrame, log: ProvenanceLog) -> None:
    log.section("Final outputs")
    log.table(
        ["File", "Shape (rows × cols)", "Path"],
        [
            [PRICES_CSV.name, f"{prices.shape[0]} × {prices.shape[1]}", f"`{PRICES_CSV}`"],
            [RETURNS_CSV.name, f"{returns.shape[0]} × {returns.shape[1]}", f"`{RETURNS_CSV}`"],
        ],
    )

    log.section("Aligned price panel (post-guard)", level=3)
    price_rows = []
    for name in prices.columns:
        valid = prices[name].notna()
        if valid.any():
            idx = prices.index[valid]
            price_rows.append(
                [
                    name,
                    int(valid.sum()),
                    idx.min().date(),
                    idx.max().date(),
                    int((~valid).sum()) if (~valid).any() else 0,
                ]
            )
        else:
            price_rows.append([name, 0, "—", "—", len(prices)])
    log.table(
        ["Commodity", "Non-null obs", "Start", "End", "NaN count"],
        price_rows,
    )

    log.section("Return sample (post row-drop)", level=3)
    ret_start = returns.index.min().date()
    ret_end = returns.index.max().date()
    log.paragraph(
        f"All **{len(returns.columns)}** commodities share **{len(returns)}** return "
        f"observations from **{ret_start}** → **{ret_end}**."
    )


def main() -> None:
    log = ProvenanceLog()

    if not RAW_XLSX.is_file():
        print(f"ERROR: Raw workbook not found: {RAW_XLSX}", file=sys.stderr)
        sys.exit(1)

    series_by_name = parse_price_blocks(log)
    prices_aligned = align_on_master_calendar(series_by_name, log)
    prices_guarded = guard_non_positive_prices(prices_aligned, log)
    returns_raw = compute_log_returns(prices_guarded, log)

    # For representation learning only: full-sample z-score.
    # Out-of-sample forecasting must refit the scaler inside each rolling window
    # to avoid look-ahead bias.
    returns_standardized = zscore_returns(returns_raw, log)

    write_matrix_csv(prices_guarded, PRICES_CSV)
    write_matrix_csv(returns_standardized, RETURNS_CSV)
    log_final_summary(prices_guarded, returns_standardized, log)
    log.write(CLEANING_LOG)
    print(f"Wrote provenance log: {CLEANING_LOG}")


if __name__ == "__main__":
    main()
