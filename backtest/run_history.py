"""
Run history — SQLite database of every backtest's config + outcome.

After every backtest, the headline metrics + full config get saved. Lets you:
  - Compare 50 runs over weeks ("which config did I run that had Sharpe 2?")
  - Search by strategy / ticker / interval / metric thresholds
  - Diff configs side-by-side
  - Spot config drift (your "best" results from last month vs today)

Schema is flat for the most-queried fields (sharpe, return, profit_factor,
etc.) but also keeps a full metrics JSON for completeness. Pure stdlib, no
new dependency.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import PROJECT_ROOT


DEFAULT_DB_PATH = PROJECT_ROOT / "run_history.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy_key TEXT NOT NULL,
    strategy_label TEXT,
    params_json TEXT NOT NULL,
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    source TEXT NOT NULL,
    mode TEXT,
    n_bars INTEGER,
    period_start TEXT,
    period_end TEXT,
    final_balance REAL,
    total_return_pct REAL,
    sharpe REAL,
    sortino REAL,
    profit_factor REAL,
    max_drawdown_pct REAL,
    num_trades INTEGER,
    win_rate_pct REAL,
    expectancy_r REAL,
    cagr_pct REAL,
    total_gross_pnl REAL,
    total_spread_cost REAL,
    total_slippage_cost REAL,
    total_financing_cost REAL,
    metrics_json TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_key);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_sharpe ON runs(sharpe);
"""


@contextmanager
def _conn(db_path: Path = DEFAULT_DB_PATH):
    """Context-managed connection. Creates the DB file if absent."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Ensure the schema is in place. Called automatically by save_run."""
    with _conn(db_path):
        pass


def save_run(
    strategy_key: str,
    strategy_label: str,
    params: dict,
    ticker: str,
    interval: str,
    source: str,
    mode: str,
    result,        # BacktestResult
    metrics,       # Metrics
    notes: str = "",
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Persist one run. Returns the new row's id."""
    m = asdict(metrics) if is_dataclass(metrics) else dict(metrics)
    period_start = (str(result.equity_curve.index[0])
                    if len(result.equity_curve) else "")
    period_end = (str(result.equity_curve.index[-1])
                  if len(result.equity_curve) else "")

    row = (
        datetime.now().isoformat(timespec="seconds"),
        strategy_key, strategy_label,
        json.dumps(params, default=str),
        ticker, interval, source, mode,
        result.bars_processed, period_start, period_end,
        m.get("final_balance"),
        m.get("total_return_pct"),
        m.get("sharpe"),
        m.get("sortino"),
        m.get("profit_factor"),
        m.get("max_drawdown_pct"),
        m.get("num_trades"),
        m.get("win_rate_pct"),
        m.get("expectancy_r"),
        m.get("cagr_pct"),
        m.get("total_gross_pnl"),
        m.get("total_spread_cost"),
        m.get("total_slippage_cost"),
        m.get("total_financing_cost"),
        json.dumps(m, default=str),
        notes,
    )
    with _conn(db_path) as conn:
        cur = conn.execute("""
            INSERT INTO runs (
                timestamp, strategy_key, strategy_label, params_json,
                ticker, interval, source, mode, n_bars, period_start, period_end,
                final_balance, total_return_pct, sharpe, sortino, profit_factor,
                max_drawdown_pct, num_trades, win_rate_pct, expectancy_r, cagr_pct,
                total_gross_pnl, total_spread_cost, total_slippage_cost, total_financing_cost,
                metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?)
        """, row)
        return cur.lastrowid


def list_runs(
    limit: int = 50,
    strategy_key: str | None = None,
    min_sharpe: float | None = None,
    min_trades: int | None = None,
    db_path: Path = DEFAULT_DB_PATH,
):
    """List recent runs, most-recent first. Filterable by basic criteria."""
    clauses, params = [], []
    if strategy_key:
        clauses.append("strategy_key = ?")
        params.append(strategy_key)
    if min_sharpe is not None:
        clauses.append("sharpe >= ?")
        params.append(min_sharpe)
    if min_trades is not None:
        clauses.append("num_trades >= ?")
        params.append(min_trades)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT id, timestamp, strategy_key, strategy_label, ticker, interval,
               source, mode, num_trades, total_return_pct, sharpe, profit_factor,
               max_drawdown_pct, params_json, notes
        FROM runs {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params.append(limit)
    with _conn(db_path) as conn:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def get_run(run_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict | None:
    with _conn(db_path) as conn:
        cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_run(run_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    with _conn(db_path) as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


def delete_all_runs(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Wipe history. Returns count deleted."""
    with _conn(db_path) as conn:
        cur = conn.execute("DELETE FROM runs")
        return cur.rowcount


def count_runs(db_path: Path = DEFAULT_DB_PATH) -> int:
    with _conn(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM runs")
        return cur.fetchone()[0]
