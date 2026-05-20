# Balanced Price Range (BPR) on BTC_60d_1h

_BTC-USD via yfinance, 24/7, 60d at 1h_

## Summary

- Bars processed: **600**
- Trades: **3**
- Final balance: £10,223.38 (start £10,000)
- Expired orders: 0, cancelled: 0, dropped(leverage/maxpos): 1, dropped(geometry): 0

## Red flags

- **[ERROR] same_bar_inout** — 1/3 (33%) trades opened and closed on the same bar — geometry / fill-order issue suspected
- **[WARN] immediate_stops** — 1/3 (33%) trades stopped out on entry bar — stop probably too tight relative to fill

## Trades

| entry_time          | exit_time           | side   |   entry_price |   exit_price |   planned_stop_loss |   planned_take_profit |   bars_held | exit_reason   |   net_pnl_gbp |
|:--------------------|:--------------------|:-------|--------------:|-------------:|--------------------:|----------------------:|------------:|:--------------|--------------:|
| 2026-04-28 03:00:00 | 2026-04-28 07:00:00 | short  |       77103.6 |      76641.4 |             77294   |               76641.4 |           4 | target        |        194.18 |
| 2026-05-03 12:00:00 | 2026-05-03 12:00:00 | short  |       78551.7 |      78650.5 |             78650.5 |               78354.1 |           0 | stop          |       -163.58 |
| 2026-05-11 18:00:00 | 2026-05-12 02:00:00 | short  |       81727.5 |      80890.4 |             82146.1 |               80890.4 |           8 | target        |        192.78 |