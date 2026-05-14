"""
IG demo connection test.

Run this FIRST after setting up your .env file to verify credentials work
and you can pull data. Doesn't write or trade anything — read-only.

Usage:
    cd ~/Developer/UK-100-Backtest/uk100-backtest
    source .venv/bin/activate
    pip install -r requirements.txt   # if not already
    python scripts/ig_test.py

Expected output: account ID confirmation, then a small sample of FTSE 100
daily bars from IG. If anything errors out, the message tells you which
piece of the .env is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("=" * 60)
    print("IG demo connection test")
    print("=" * 60)

    # 1. Check .env exists
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        print(f"\n❌ No .env file found at {env_path}")
        print("   Create it with: IG_USERNAME, IG_PASSWORD, IG_API_KEY, "
              "IG_ACCOUNT_ID, IG_ENV=demo")
        sys.exit(1)
    print(f"\n✓ Found .env at {env_path}")

    # 2. Try loading credentials (without printing them)
    try:
        from data.ig_fetcher import _load_env
        creds = _load_env()
        print(f"✓ Credentials loaded (username starts with '{creds['username'][:2]}…', "
              f"environment = {creds['env']})")
    except Exception as e:
        print(f"\n❌ Credential loading failed: {e}")
        sys.exit(1)

    # 3. Try logging in + fetching account info
    print("\nAttempting to log in...")
    try:
        from data.ig_fetcher import check_connection
        info = check_connection()
        print(f"✓ Logged in. Found {info['n_accounts']} account(s):")
        for aid, atype in zip(info["account_ids"], info["account_types"]):
            print(f"    - {aid} ({atype})")
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
        print("\nMost common fixes:")
        print("  1. IG_USERNAME should be your LOGIN username, not your email.")
        print("     Find it on demo.ig.com → top-right account dropdown.")
        print("  2. IG_API_KEY must be a DEMO key (generated in My IG → "
              "Settings → API keys → Demo).")
        print("  3. IG_ENV must match: =demo for demo keys, =live for live keys.")
        sys.exit(1)

    # 4. Try fetching a tiny historical sample (uses ~10 of weekly allowance)
    print("\nFetching 10 daily bars of UK 100...")
    try:
        from data.ig_fetcher import fetch_ig
        df = fetch_ig(ticker="^FTSE", interval="1d", num_points=10, use_cache=False)
        print(f"✓ Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        print()
        print(df.tail(5).to_string())
    except Exception as e:
        print(f"\n❌ Data fetch failed: {e}")
        print("\nMost common fixes:")
        print("  1. Wrong epic. UK 100 spread bet = IX.D.FTSE.DAILY.IP")
        print("     If your demo account is CFD-only, try IX.D.FTSE.CFD.IP")
        print("  2. Weekly allowance exceeded — wait or use yfinance for now.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✓ All checks passed. IG data integration is ready to use.")
    print("  In the Streamlit UI, switch the 'Data source' radio to 'IG demo'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
