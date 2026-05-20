# Balanced Price Range (BPR) on AAPL_60d_15m

_AAPL via EODHD, US cash hours in London time, 60d at 15m_

## Summary

- Bars processed: **600**
- Trades: **6**
- Final balance: £9,739.49 (start £10,000)
- Expired orders: 0, cancelled: 0, dropped(leverage/maxpos): 7, dropped(geometry): 0

## Red flags

- **[WARN] same_bar_inout** — 1/6 (17%) trades opened and closed on the same bar — geometry / fill-order issue suspected

## Trades

| entry_time          | exit_time           | side   |   entry_price |   exit_price |   planned_stop_loss |   planned_take_profit |   bars_held | exit_reason   |   net_pnl_gbp |
|:--------------------|:--------------------|:-------|--------------:|-------------:|--------------------:|----------------------:|------------:|:--------------|--------------:|
| 2026-04-29 14:45:00 | 2026-04-29 17:45:00 | long   |        268.51 |      270.329 |             267.826 |               270.329 |          12 | target        |        185.57 |
| 2026-04-29 19:15:00 | 2026-04-30 14:30:00 | long   |        269.75 |      268.966 |             268.966 |               271.318 |           8 | stop          |       -150.78 |
| 2026-05-04 15:45:00 | 2026-05-04 15:45:00 | long   |        277.73 |      277.42  |             277.42  |               280.271 |           0 | stop          |        -73.78 |
| 2026-05-04 16:00:00 | 2026-05-05 14:30:00 | short  |        277.67 |      276.9   |             280.39  |               272.23  |          21 | session_end   |         18.06 |
| 2026-05-11 15:15:00 | 2026-05-11 17:30:00 | long   |        292.25 |      291.586 |             291.586 |               296.367 |           9 | stop          |        -67.12 |
| 2026-05-11 20:45:00 | 2026-05-12 14:30:00 | short  |        292.7  |      293.255 |             293.255 |               291.59  |           2 | stop          |       -172.46 |