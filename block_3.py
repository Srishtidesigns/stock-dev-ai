# ===== Block 3 (Fundamentals fetcher: interactive single OR bulk from latest combined CSV) =====
import os
import sys
import re
import glob
import json
import time
import random
import pandas as pd
import asyncio
from datetime import datetime
from contextlib import redirect_stderr

# === Setup for custom scraper function ===
script_dir = "/home/sudipaiml/nixflakes/stock-prediction/scripts"
if script_dir not in sys.path:
    sys.path.append(script_dir)

from playwright_bridge3 import init_browser, fetch_fundamentals, shutdown

# ====== Configs ======
BASE_RESULTS_DIR = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/screener-scripts-results"
CACHE_BASE_DIR  = "/home/sudipaiml/stock_analysis/agcodes/scrape_results/cache"

MIN_WAIT_S = 3
MAX_WAIT_S = 6

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ====== Helpers ======
def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def find_latest_combined_csv() -> str | None:
    pattern = os.path.join(BASE_RESULTS_DIR, "combined-scan-results_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_ts(fn: str) -> str:
        m = re.search(r"combined-scan-results_(\d{8}_\d{6})\.csv$", os.path.basename(fn))
        return m.group(1) if m else "00000000_000000"
    files.sort(key=lambda p: (extract_ts(p), os.path.getmtime(p)))
    return files[-1]

def sanitize_symbol(ticker: str) -> str:
    core = ticker.strip().upper()
    if "." in core:
        core = core.split(".", 1)[0]
    core = re.sub(r"[^A-Z0-9]", "", core)
    return core

def polite_wait():
    delay = random.uniform(MIN_WAIT_S, MAX_WAIT_S)
    print(f"{CYAN}⏳ Cooling down ~{delay:.1f}s before next fetch...{RESET}")
    time.sleep(delay)

def load_bulk_universe(latest_csv_path: str) -> list[tuple[str, str]]:
    df = pd.read_csv(latest_csv_path)
    if "Ticker" not in df.columns:
        raise RuntimeError("Combined CSV missing 'Ticker' column.")
    if "Name" not in df.columns:
        raise RuntimeError("Combined CSV missing 'Name' column.")
    df = df[df["Ticker"].astype(str).str.upper() != "UNKNOWN"]
    df["SymbolCore"] = df["Ticker"].astype(str).map(sanitize_symbol)
    df = df.drop_duplicates(subset=["SymbolCore"])
    return [(row["SymbolCore"], str(row["Name"])) for _, row in df.iterrows()]

async def run_bulk(jobs):
    """Run all scraping jobs in one persistent Chromium session."""
    p, browser, context = await init_browser()
    ok_cnt, fail_cnt = 0, 0

    for idx, (sym, name) in enumerate(jobs, start=1):
        print(f"\n{CYAN}[{idx}/{len(jobs)}] → {sym}{' | ' + name if name else ''}{RESET}")
        try:
            ok = await fetch_fundamentals(context, sym)
        except Exception as e:
            print(f"{YELLOW}⚠ Fetch raised for {sym}: {e}{RESET}")
            ok = False

        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1

        polite_wait()

    await shutdown(p, browser)
    return ok_cnt, fail_cnt

# ====== Main flow ======
def main():
    os.makedirs(CACHE_BASE_DIR, exist_ok=True)

    user_inp = input(
        "📈 Enter stock ticker(s) (comma-separated) or press Enter for bulk from latest CSV: "
    ).strip()

    jobs: list[tuple[str, str | None]] = []
    if user_inp:
        raw_parts = [p for p in re.split(r"[,\s]+", user_inp) if p]
        uniq, seen = [], set()
        for p in raw_parts:
            sym = sanitize_symbol(p)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            uniq.append(sym)
        jobs = [(sym, None) for sym in uniq]
        print(f"{CYAN}🗂️ Manual mode: {len(jobs)} ticker(s) queued.{RESET}")
    else:
        latest_csv = find_latest_combined_csv()
        if not latest_csv or not os.path.exists(latest_csv):
            print(f"{RED}❌ No combined CSV found in {BASE_RESULTS_DIR}. Run Block 2 first.{RESET}")
            sys.exit(1)
        pairs = load_bulk_universe(latest_csv)
        if not pairs:
            print(f"{RED}❌ Combined CSV has no valid tickers. Nothing to do.{RESET}")
            sys.exit(1)
        jobs = [(sym, name) for sym, name in pairs]
        print(f"{CYAN}🗃️ Bulk mode: {len(jobs)} unique ticker(s) from: {latest_csv}{RESET}")

    print(f"{CYAN}🚀 Starting fundamentals fetch. Respectful pacing {MIN_WAIT_S}-{MAX_WAIT_S}s between calls.{RESET}")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            ok_cnt, fail_cnt = loop.run_until_complete(run_bulk(jobs))
        else:
            ok_cnt, fail_cnt = loop.run_until_complete(run_bulk(jobs))
    except Exception as e:
        print(f"{RED}❌ Asyncio error: {e}{RESET}")
        ok_cnt, fail_cnt = 0, len(jobs)

    print(f"\n{GREEN}Done.{RESET} {GREEN}OK={ok_cnt}{RESET}, {RED}FAIL={fail_cnt}{RESET}, TOTAL={len(jobs)}")
    print(f"{CYAN}Cache root:{RESET} {CACHE_BASE_DIR}")

    # === Mission manifest ===
    manifest_stocks = [
        sym for sym, _ in jobs
        if os.path.exists(os.path.join(CACHE_BASE_DIR, sym, f"debug_{sym}.html"))
    ]
    manifest_path = os.path.join(BASE_RESULTS_DIR, f"mission_manifest_{ts_now()}.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_stocks, mf, indent=2)
        print(f"{CYAN}📜 Mission manifest saved:{RESET} {manifest_path}")
        print(f"{CYAN}📦 Stocks in this mission:{RESET} {len(manifest_stocks)}")
    except Exception as e:
        print(f"{YELLOW}⚠ Failed to write mission manifest: {e}{RESET}")

if __name__ == "__main__":
    main()
