"""
EODHD connection test.

Run this FIRST after putting your EODHD_API_KEY in .env. Read-only, no money
involved. Verifies the API key works and shows what your current plan can pull.

Usage:
    python scripts/eodhd_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("=" * 60)
    print("EODHD connection test")
    print("=" * 60)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        print(f"\n❌ No .env file found at {env_path}")
        sys.exit(1)
    print(f"\n✓ Found .env at {env_path}")

    # 1. Try loading the key
    try:
        from data.eodhd_fetcher import _load_api_key
        key = _load_api_key()
        print(f"✓ API key loaded (starts with '{key[:4]}…', ends with '…{key[-4:]}')")
    except Exception as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # 2. Test AAPL.US daily fetch (free tier supports this)
    print("\nFetching 5 daily bars of AAPL.US (free-tier symbol)...")
    try:
        from data.eodhd_fetcher import check_connection
        info = check_connection()
        print(f"✓ Got {info['bars_returned']} bars; latest = {info['latest_date']}")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        sys.exit(1)

    # 3a. Ask EODHD's search endpoint what FTSE-related symbols are available
    print("\nSearching EODHD for FTSE-related symbols (so we know the right one)...")
    try:
        import requests
        from data.eodhd_fetcher import _load_api_key
        url = "https://eodhd.com/api/search/FTSE%20100"
        resp = requests.get(url, params={"api_token": _load_api_key(), "fmt": "json"},
                            timeout=15)
        if resp.status_code == 200:
            results = resp.json()
            indices = [r for r in results if r.get("Type") == "Index"]
            others = [r for r in results if r.get("Type") != "Index"][:5]
            if indices:
                print(f"✓ EODHD has {len(indices)} index match(es) for 'FTSE 100':")
                for r in indices[:10]:
                    print(f"    {r.get('Code')}.{r.get('Exchange')}  —  {r.get('Name')}")
            if others:
                print(f"  Other matches (ETFs / stocks):")
                for r in others:
                    print(f"    {r.get('Code')}.{r.get('Exchange')}  —  {r.get('Name')}  ({r.get('Type')})")
        else:
            print(f"⚠️  Search returned HTTP {resp.status_code}")
    except Exception as e:
        print(f"⚠️  Search failed: {e}")

    # 3b. Try FTSE 100 via ISF.LSE ETF (works on All-World plans)
    print("\nAttempting FTSE 100 daily via ETF proxy ISF.LSE...")
    try:
        from data.eodhd_fetcher import fetch_eodhd
        df = fetch_eodhd(ticker="^FTSE", interval="1d", num_points=10, use_cache=False)
        print(f"✓ Got {len(df)} ISF.LSE daily bars (FTSE 100 proxy)")
        print()
        print(df.tail(5).to_string())
    except Exception as e:
        print(f"⚠️  ETF daily failed: {e}")

    # 4. Try FTSE 100 intraday via ETF (requires EOD+Intraday plan)
    print("\nAttempting FTSE 100 15m intraday via ETF proxy...")
    try:
        from data.eodhd_fetcher import fetch_eodhd
        df = fetch_eodhd(ticker="^FTSE", interval="15m", num_points=20, use_cache=False)
        print(f"✓ Got {len(df)} ISF.LSE 15m bars")
        print()
        print(df.tail(5).to_string())
    except Exception as e:
        print(f"⚠️  ETF intraday failed: {e}")
        print("    Confirm 'EOD+Intraday All World Extended' on your account.")

    print()
    print("=" * 60)
    print("Done. If the FTSE 100 daily + intraday tests above succeeded, you're")
    print("ready to switch the Streamlit 'Data source' radio to 'EODHD'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
