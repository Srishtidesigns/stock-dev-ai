# ===== Block 2 (Rewired + Screener auto-fixer for LOW/NONE matches) =====
SYMBOL_ALIASES = {
    'SSPOWER': 'S&SPOWER',
    'M&M': 'M&MFIN',
}

import html  # added to decode HTML entities

def safe_filename(name: str) -> str:
    name = html.unescape(name)
    return re.sub(r'[<>:\"/\\|?* ]', '_', name)

import os
import re
import sys
import json
import shutil
import pandas as pd
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
import html as html_lib
from urllib.parse import quote

# ===== Config =====
SYSTEM_CHROMIUM_PATH = "/run/current-system/sw/bin/chromium"
BASE_QUERY_DIR = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/screener-scripts"
BASE_RESULTS_DIR = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/screener-scripts-results"

# Screener credentials
SCREENER_USER = "sudip.nixos.flakes@gmail.com"
SCREENER_PASS = "nantum@1983"

# Alias JSON path (external)
ALIAS_JSON_PATH = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/NSE-BSE-equity-list/nse_match.json"

# Screener base
SCREENER_BASE = "https://www.screener.in"

# Human delay (seconds). Tweak to taste or set via environment var
HUMAN_DELAY = float(os.environ.get("HUMAN_DELAY", "1.0"))

# ===== ANSI color codes =====
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ===== Prompt: single scan or all scans =====
user_scan = input("📄 Enter scan filename WITHOUT .txt (leave blank to run ALL scans in folder): ").strip()
if user_scan:
    scan_files = [os.path.join(BASE_QUERY_DIR, f"{user_scan}.txt")]
else:
    scan_files = [
        os.path.join(BASE_QUERY_DIR, f)
        for f in sorted(os.listdir(BASE_QUERY_DIR))
        if f.lower().endswith(".txt")
    ]
    scan_files = [p for p in scan_files if os.path.isfile(p)]

if not scan_files:
    raise FileNotFoundError(f"❌ No scan files found. Folder checked: {BASE_QUERY_DIR}")

print(f"{CYAN}🗃️ Will run {len(scan_files)} scan(s).{RESET}")

# Ask whether to enable automatic screener page fixer for low/unknown matches
_auto_fix_input = input("🔧 Attempt auto-fix for LOW/NONE matches by visiting screener company pages? (Y/n): ").strip().lower()
AUTO_FIX_ENABLED = (not _auto_fix_input) or (_auto_fix_input == "y")

# ===== Timestamp and combined outputs =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
combined_rows = []        # accumulate all per-scan rows into one DataFrame
global_match_log = []     # full match log across scans

# ===== Normalization (safer) =====
def normalize_name(name: str) -> str:
    """
    Normalize a company name: lowercase, remove common suffixes, punctuation,
    multiple spaces collapsed. Returns normalized string for matching.
    """
    if not isinstance(name, str):
        return ""
    s = name.lower()
    # remove common legal/company tokens but keep meaningful words
    s = re.sub(r"\b(limited|ltd|private|pvt|pvtltd|pvt\.ltd|india|inc|corporation|co|company)\b", " ", s)
    # expand a few common shorthand safely if present (non-destructive)
    s = re.sub(r"\btech\b", "technology", s)
    s = re.sub(r"\bsys\b", "systems", s)
    # remove punctuation, keep alphanumerics and spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ===== NSE/BSE mappings loader (keep original paths) =====
def get_mappings():
    nse_csv_path = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/NSE-BSE-equity-list/NSE.csv"
    bse_csv_path = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/NSE-BSE-equity-list/BhavCopyBSE.csv"

    nse_mapping = {}   # normalized_name -> SYMBOL(.NS)
    bse_mapping = {}   # normalized_name -> SCRIPCODE(.BO)
    nse_symbols = set()
    bse_symbols = set()

    # Load NSE
    if not os.path.exists(nse_csv_path):
        raise FileNotFoundError(f"❌ NSE equity list not found: {nse_csv_path}")
    df_nse = pd.read_csv(nse_csv_path, dtype=str)
    df_nse = df_nse.fillna("")  # guard
    for _, row in df_nse.iterrows():
        raw_name = str(row.get("NAME OF COMPANY", "")).strip()
        name = normalize_name(raw_name)
        symbol_raw = str(row.get("SYMBOL", "")).strip().upper()
        # append .NS to keep parity with earlier code's output
        symbol = symbol_raw + ".NS" if symbol_raw else ""
        if name:
            nse_mapping[name] = symbol
        if symbol:
            nse_symbols.add(symbol)
    # Keep original df for lookup later
    df_nse["norm_name"] = df_nse["NAME OF COMPANY"].apply(lambda x: normalize_name(str(x)))

    # Load BSE
    if not os.path.exists(bse_csv_path):
        raise FileNotFoundError(f"❌ BSE equity list not found: {bse_csv_path}")
    df_bse = pd.read_csv(bse_csv_path, dtype=str)
    df_bse = df_bse.fillna("")
    for _, row in df_bse.iterrows():
        raw_name = str(row.get("Security Name", "")).strip()
        name = normalize_name(raw_name)
        scrip = str(row.get("Scrip code", "")).strip().upper()
        symbol = scrip + ".BO" if scrip else ""
        if name:
            bse_mapping[name] = symbol
        if symbol:
            bse_symbols.add(symbol)
    df_bse["norm_name"] = df_bse["Security Name"].apply(lambda x: normalize_name(str(x)))

    return (nse_mapping, bse_mapping, df_nse, df_bse, nse_symbols, bse_symbols)

# load mappings
nse_mapping, bse_mapping, df_nse_ref, df_bse_ref, nse_symbols, bse_symbols = get_mappings()
print(f"✅ Loaded {len(nse_mapping)} NSE, {len(bse_mapping)} BSE tickers into memory.")

# ===== Load alias overrides from JSON (normalized keys) =====
def load_aliases(path=ALIAS_JSON_PATH):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # normalize keys
            return {normalize_name(k): v for k, v in raw.items()}
        else:
            print(f"{YELLOW}⚠ Alias JSON not found at {path}. Continuing without aliases.{RESET}")
            return {}
    except Exception as e:
        print(f"{RED}❌ Failed to load alias JSON: {e}{RESET}")
        return {}

ALIAS_MAP = load_aliases()

# ===== Matching (NSE priority, BSE fallback) =====
def get_ticker_from_name(company_name, cutoff=0.8):
    """
    Returns (ticker, match_type, source)
    - ticker: like 'INFY.NS' or '500XXX.BO' or 'UNKNOWN'
    - match_type: 'HIGH' / 'LOW' / 'NONE'
    - source: 'NSE' or 'BSE' or 'NONE'
    """
    name_norm = normalize_name(company_name)

    # Quick: if input looks like a symbol (.NS or .BO) or plain symbol
    cand = str(company_name).strip().upper()
    if cand.endswith(".NS") and cand in nse_symbols:
        global_match_log.append([company_name, name_norm, cand, 1.0, cand, "HIGH", "NSE"])
        print(f"{GREEN}✅ High: {company_name} → {cand} (direct symbol) → {cand}{RESET}")
        return cand, "HIGH", "NSE"
    if cand.endswith(".BO") and cand in bse_symbols:
        global_match_log.append([company_name, name_norm, cand, 1.0, cand, "HIGH", "BSE"])
        print(f"{GREEN}✅ High: {company_name} → {cand} (direct symbol) → {cand}{RESET}")
        return cand, "HIGH", "BSE"
    # plain symbol without extension
    if cand in {s.replace(".NS", "") for s in nse_symbols}:
        sym = cand + ".NS"
        global_match_log.append([company_name, name_norm, sym, 1.0, sym, "HIGH", "NSE"])
        print(f"{GREEN}✅ High: {company_name} → {sym} (direct symbol) → {sym}{RESET}")
        return sym, "HIGH", "NSE"
    if cand in {s.replace(".BO", "") for s in bse_symbols}:
        sym = cand + ".BO"
        global_match_log.append([company_name, name_norm, sym, 1.0, sym, "HIGH", "BSE"])
        print(f"{GREEN}✅ High: {company_name} → {sym} (direct symbol) → {sym}{RESET}")
        return sym, "HIGH", "BSE"

    # === NSE attempts ===
    # 1) Alias map (NSE-side)
    if name_norm in ALIAS_MAP:
        alias_company = ALIAS_MAP[name_norm]
        # find symbol in NSE ref by exact company name
        row = df_nse_ref[df_nse_ref["NAME OF COMPANY"].str.strip().str.lower() == alias_company.strip().lower()]
        if not row.empty:
            symbol = row.iloc[0]["SYMBOL"].strip().upper() + ".NS"
            score = 0.95
            global_match_log.append([company_name, name_norm, alias_company, round(score, 3), symbol, "HIGH", "NSE"])
            print(f"{GREEN}✅ High: {company_name} → {alias_company} ({score:.2f}) → {symbol}{RESET}")
            return symbol, "HIGH", "NSE"

    # 2) Exact normalized name match (NSE)
    if name_norm in nse_mapping:
        ticker = nse_mapping[name_norm]
        score = 0.90
        global_match_log.append([company_name, name_norm, name_norm, round(score, 3), ticker, "HIGH", "NSE"])
        print(f"{GREEN}✅ High: {company_name} → {name_norm} ({score:.2f}) → {ticker}{RESET}")
        return ticker, "HIGH", "NSE"

    # 3) Fuzzy match in NSE
    nse_choices = list(nse_mapping.keys())
    if nse_choices:
        best = process.extractOne(name_norm, nse_choices, scorer=fuzz.token_sort_ratio)
        if best:
            best_match_name, score, _ = best
            ticker = nse_mapping.get(best_match_name)
            # thresholds: >=85 high, 70-84 low, <70 none
            if score >= 85:
                match_type = "HIGH"
            elif score >= 70:
                match_type = "LOW"
            else:
                match_type = "NONE"
            if match_type != "NONE":
                global_match_log.append([company_name, name_norm, best_match_name, round(score, 3), ticker, match_type, "NSE"])
                print(f"{GREEN if match_type=='HIGH' else YELLOW}{'✅ High' if match_type=='HIGH' else '⚠ Low'}: {company_name} → {best_match_name} ({score:.2f}) → {ticker}{RESET}")
                return ticker, match_type, "NSE"
            # else fallthrough to BSE
    # === BSE fallback (same steps) ===
    # alias map attempt on BSE side - if alias mapped company exists in BSE we'll match
    if name_norm in ALIAS_MAP:
        alias_company = ALIAS_MAP[name_norm]
        row = df_bse_ref[df_bse_ref["Security Name"].str.strip().str.lower() == alias_company.strip().lower()]
        if not row.empty:
            symbol = str(row.iloc[0]["Scrip code"]).strip().upper() + ".BO"
            score = 0.85
            global_match_log.append([company_name, name_norm, alias_company, round(score, 3), symbol, "LOW", "BSE"])
            print(f"{YELLOW}⚠ Low: {company_name} → {alias_company} ({score:.2f}) → {symbol}{RESET}")
            return symbol, "LOW", "BSE"

    # exact normalized name match for BSE
    if name_norm in bse_mapping:
        ticker = bse_mapping[name_norm]
        score = 0.90
        global_match_log.append([company_name, name_norm, name_norm, round(score, 3), ticker, "HIGH", "BSE"])
        print(f"{GREEN}✅ High: {company_name} → {name_norm} ({score:.2f}) → {ticker}{RESET}")
        return ticker, "HIGH", "BSE"

    # fuzzy match in BSE
    bse_choices = list(bse_mapping.keys())
    if bse_choices:
        best = process.extractOne(name_norm, bse_choices, scorer=fuzz.token_sort_ratio)
        if best:
            best_match_name, score, _ = best
            ticker = bse_mapping.get(best_match_name)
            if score >= 85:
                match_type = "HIGH"
            elif score >= 70:
                match_type = "LOW"
            else:
                match_type = "NONE"
            if match_type != "NONE":
                global_match_log.append([company_name, name_norm, best_match_name, round(score, 3), ticker, match_type, "BSE"])
                print(f"{GREEN if match_type=='HIGH' else YELLOW}{'✅ High' if match_type=='HIGH' else '⚠ Low'}: {company_name} → {best_match_name} ({score:.2f}) → {ticker}{RESET}")
                return ticker, match_type, "BSE"

    # No match
    global_match_log.append([company_name, name_norm, "NO MATCH", 0.0, "UNKNOWN", "NONE", "NONE"])
    print(f"{RED}❌ None: {company_name} → UNKNOWN{RESET}")
    return "UNKNOWN", "NONE", "NONE"

# ===== match_with_fallback (applies to DataFrame rows) =====
def match_with_fallback(df):
    tickers, qualities, sources = [], [], []
    for name in df["Name"]:
        t, q, s = get_ticker_from_name(name, cutoff=0.6)
        tickers.append(t); qualities.append(q); sources.append(s)

    df["Ticker"] = tickers
    df["Match_Quality"] = qualities
    df["Match_Source"] = sources

    unknown_mask = df["Ticker"] == "UNKNOWN"
    if unknown_mask.sum() > 0:
        print(f"\n🔄 Retrying {unknown_mask.sum()} UNKNOWN matches with relaxed cutoff...\n")
        for idx in df[unknown_mask].index:
            name = df.at[idx, "Name"]
            t, q, s = get_ticker_from_name(name, cutoff=0.4)
            df.at[idx, "Ticker"] = t
            df.at[idx, "Match_Quality"] = q
            df.at[idx, "Match_Source"] = s

    return df

# ===== Utility: extract NSE/BSE codes from a company screener page =====
async def extract_codes_from_company_page(context, company_path, timeout=7000):
    """
    Open the company page and parse text for NSE/BSE codes.
    Returns (nse_code_or_None, bse_code_or_None, company_full_name_or_None)
    """
    if not company_path:
        return None, None, None

    # Build absolute URL carefully: unescape HTML entities and percent-encode path segments
    cp = html_lib.unescape(company_path)
    if cp.startswith("http"):
        url = cp
    else:
        # preserve leading/trailing slashes while quoting segments
        if cp.startswith("/"):
            path = "/".join(quote(p, safe='') for p in cp.split('/'))
            url = SCREENER_BASE.rstrip("/") + path
        else:
            path = "/".join(quote(p, safe='') for p in cp.split('/'))
            url = SCREENER_BASE.rstrip("/") + "/" + path

    page = await context.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=timeout)
        await asyncio.sleep(HUMAN_DELAY)  # human delay after navigation
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

        # attempt to find NSE/BSE tokens - common pattern: "NSE: SYMBOL" and "BSE: 543767"
        # allow ampersand and dots in ticker tokens
        found = re.findall(r'\b(NSE|BSE):\s*([A-Z0-9&\-\.]+)', text)
        nse_code = None
        bse_code = None
        for exch, code in found:
            exch = exch.upper()
            # unescape and sanitize code
            code_clean = html_lib.unescape(code).strip().replace('\u00a0', '').replace(' ', '')
            if exch == "NSE":
                nse_code = code_clean
            elif exch == "BSE":
                bse_code = code_clean

        # try to get canonical company name from page (h1 or title)
        company_name = None
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            company_name = h1.get_text(strip=True)
        else:
            title = soup.title.string if soup.title else None
            if title:
                company_name = title.split("|")[0].strip()

        await asyncio.sleep(HUMAN_DELAY * 0.5)
        await page.close()
        return nse_code, bse_code, company_name
    except Exception:
        try:
            await page.close()
        except Exception:
            pass
        return None, None, None

# ===== Auto-fix low/none matches by visiting screener pages =====
async def auto_fix_low_matches(context, df, max_retries=2):
    """
    For rows in df with Match_Quality != 'HIGH', attempt to fetch company page (Company_Path)
    and extract NSE/BSE codes. Update df in-place for successful finds and record proposed
    mappings (alias -> canonical company name).
    Returns dict of proposed alias fixes: {original_name: canonical_name}
    """
    proposed_fixes = {}
    # Only act on rows with LOW or NONE
    mask = df["Match_Quality"].isin(["LOW", "NONE"])
    candidates = df[mask].copy()

    if candidates.empty:
        return proposed_fixes

    # We prefer rows that have a Company_Path (extracted from the original table)
    # If Company_Path is empty, we will attempt a search on explore page as fallback.
    for retry in range(max_retries):
        any_found_this_round = False
        for idx, row in candidates.iterrows():
            orig = row["Name"]
            cur_quality = row["Match_Quality"]
            cur_ticker = row.get("Ticker", "UNKNOWN")
            company_path = row.get("Company_Path", "") if "Company_Path" in row else ""

            # Skip if already got fixed by previous retry
            if df.at[idx, "Match_Quality"] == "HIGH":
                continue

            # Try company page if we have path
            nse_code = None
            bse_code = None
            canonical_name = None

            if company_path:
                nse_code, bse_code, canonical_name = await extract_codes_from_company_page(context, company_path)
            else:
                # fallback: attempt to search from explore page (open separate page)
                try:
                    page = await context.new_page()
                    await page.goto(f"{SCREENER_BASE}/explore/", wait_until="networkidle", timeout=8000)
                    await asyncio.sleep(HUMAN_DELAY)
                    # try to find the company search input
                    try:
                        await page.wait_for_selector('input[data-company-search="true"]', timeout=3500)
                        await asyncio.sleep(HUMAN_DELAY * 0.5)
                        await page.fill('input[data-company-search="true"]', orig)
                        await asyncio.sleep(HUMAN_DELAY)
                        # press Enter to trigger search
                        await page.keyboard.press("Enter")
                    except Exception:
                        # If the search input is not present/available, just try to find company anchor in page
                        pass

                    # wait briefly and then try to locate a company link
                    try:
                        await asyncio.sleep(HUMAN_DELAY * 0.5)
                        await page.wait_for_selector("a[href^='/company/']", timeout=4000)
                        a = await page.query_selector("a[href^='/company/']")
                        href = await a.get_attribute("href") if a else None
                        if href:
                            # small human delay before following found link
                            await asyncio.sleep(HUMAN_DELAY * 0.5)
                            nse_code, bse_code, canonical_name = await extract_codes_from_company_page(context, href)
                    except Exception:
                        # nothing found
                        pass
                    try:
                        await page.close()
                    except Exception:
                        pass
                except Exception:
                    # network/search fallback failed; continue
                    pass

            # If we found NSE/BSE code, update df and global_match_log, propose mapping
            if nse_code or bse_code:
                # prefer NSE if present, else BSE
                if nse_code:
                    ticker = nse_code.strip().upper() + ".NS"
                    source = "NSE"
                else:
                    ticker = bse_code.strip().upper() + ".BO"
                    source = "BSE"

                # canonical_name resolution: if we can map ticker back to the official name from NSE/BSE files
                canonical = None
                try:
                    if source == "NSE":
                        sym = ticker.replace(".NS", "")
                        rown = df_nse_ref[df_nse_ref["SYMBOL"].str.strip().str.upper() == sym]
                        if not rown.empty:
                            canonical = rown.iloc[0]["NAME OF COMPANY"].strip()
                    else:
                        scrip = ticker.replace(".BO", "")
                        rowb = df_bse_ref[df_bse_ref["Scrip code"].astype(str).str.strip().str.upper() == scrip]
                        if not rowb.empty:
                            canonical = rowb.iloc[0]["Security Name"].strip()
                except Exception:
                    canonical = canonical_name or canonical

                # fallback canonical_name if still None
                if not canonical:
                    canonical = canonical_name or (nse_code or bse_code)

                # update df
                df.at[idx, "Ticker"] = ticker
                df.at[idx, "Match_Quality"] = "HIGH"
                df.at[idx, "Match_Source"] = source

                # Add to global_match_log with a high score (100) and matched name as canonical
                global_match_log.append([orig, normalize_name(orig), canonical, 100.0, ticker, "HIGH", source])
                print(f"{GREEN}✅ Auto-fixed: {orig} → {canonical} → {ticker}{RESET}")

                # propose alias mapping: orig -> canonical (full company name)
                proposed_fixes[orig] = canonical

                any_found_this_round = True

            # short human pause between candidate checks to avoid hammering the site
            await asyncio.sleep(HUMAN_DELAY * 0.25)

        # refresh candidate list (only those still not HIGH)
        mask = df["Match_Quality"].isin(["LOW", "NONE"])
        candidates = df[mask].copy()
        if not any_found_this_round:
            break  # nothing new found this retry, stop early

    return proposed_fixes

# ===== Summary printing (keeps your original output style) =====
def print_match_summary_for(df, scan_name):
    total = len(df)
    high = (df["Match_Quality"] == "HIGH").sum()
    low = (df["Match_Quality"] == "LOW").sum()
    none = (df["Match_Quality"] == "NONE").sum()

    def bar(pct):
        term_width = shutil.get_terminal_size((80, 20)).columns
        bar_len = max(10, min(40, term_width - 40))
        filled = int(round(bar_len * pct / 100))
        try:
            return "█" * filled + "░" * (bar_len - filled)
        except UnicodeEncodeError:
            return "#" * filled + "-" * (bar_len - filled)

    hp = round(100 * high / total, 1) if total else 0
    lp = round(100 * low / total, 1) if total else 0
    np = round(100 * none / total, 1) if total else 0
    print(f"\n📊 [{scan_name}] Match Quality Summary:")
    print(f"{GREEN}✅ HIGH: {high} ({hp}%) {bar(hp)} {hp}%{RESET}")
    print(f"{YELLOW}⚠ LOW: {low} ({lp}%) {bar(lp)} {lp}%{RESET}")
    print(f"{RED}❌ NONE: {none} ({np}%) {bar(np)} {np}%{RESET}")

# ===== Core scrape for one scan text (with pagination) =====
async def scrape_one_scan(page, scan_path):
    scan_name = os.path.splitext(os.path.basename(scan_path))[0]

    scan_out_dir = os.path.join(BASE_RESULTS_DIR, scan_name)
    os.makedirs(scan_out_dir, exist_ok=True)

    with open(scan_path, "r", encoding="utf-8") as f:
        query_text = f.read().strip()

    # run query
    await page.goto(f"{SCREENER_BASE}/explore/")
    await asyncio.sleep(HUMAN_DELAY)
    await page.click('a.button.button-primary[href="/screen/new/"]')
    await page.wait_for_selector('textarea#query-builder', timeout=20000)
    await asyncio.sleep(HUMAN_DELAY * 0.5)
    await page.fill('textarea#query-builder', query_text)
    print(f"✅ [{scan_name}] Query filled.")
    await asyncio.sleep(HUMAN_DELAY * 0.5)
    await page.click('button.button-primary:has-text("Run this Query")')
    await page.wait_for_load_state("networkidle")
    await page.wait_for_selector("table", timeout=20000)
    print(f"✅ [{scan_name}] Query executed.")
    await asyncio.sleep(HUMAN_DELAY)

    # collect rows from all pages, but capture link href in Name column (Company_Path)
    all_rows = []
    headers = []
    page_num = 1
    while True:
        html = await page.content()
        html_path = os.path.join(scan_out_dir, f"{scan_name}_results_{timestamp}_page{page_num}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            break

        rows = table.find_all("tr")
        if rows:
            if not headers:
                headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            # find index of Name column if present
            name_idx = None
            try:
                name_idx = headers.index("Name")
            except ValueError:
                name_idx = None

            for r in rows[1:]:
                cells = r.find_all("td")
                if not cells:
                    continue
                row_data = {}
                for i, h in enumerate(headers):
                    if i < len(cells):
                        row_data[h] = cells[i].get_text(strip=True)
                    else:
                        row_data[h] = ""
                # capture Company_Path (href) from Name cell if present
                company_path = ""
                if (name_idx is not None) and (name_idx < len(cells)):
                    a = cells[name_idx].find("a")
                    if a and a.has_attr("href"):
                        company_path = a["href"]
                row_data["Company_Path"] = company_path
                all_rows.append(row_data)

        # find "Next" link robustly
        next_link = None
        pag_div = soup.find("div", class_="pagination")
        if pag_div:
            for a in pag_div.find_all("a"):
                txt = a.get_text(strip=True)
                if txt and txt.lower().startswith("next"):
                    next_link = a
                    break

        if not next_link:
            break

        try:
            await page.click("div.pagination a:has-text('Next')")
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(HUMAN_DELAY)
            page_num += 1
            print(f"➡️  [{scan_name}] Navigated to page {page_num}")
        except Exception:
            break

    if not all_rows:
        print(f"{RED}❌ [{scan_name}] No rows found across pages.{RESET}")
        return pd.DataFrame()

    # build dataframe
    df = pd.DataFrame(all_rows)

    if "Name" not in df.columns:
        print(f"{RED}❌ [{scan_name}] No 'Name' column; skipping.{RESET}")
        return pd.DataFrame()

    # run matching
    df = match_with_fallback(df)

    # If enabled, attempt auto-fix from screener company pages for low/none matches
    proposed_fixes = {}
    if AUTO_FIX_ENABLED:
        fixes = await auto_fix_low_matches(page.context, df, max_retries=2)
        if fixes:
            proposed_fixes.update(fixes)

    # sort & write csv
    order = {"HIGH": 0, "LOW": 1, "NONE": 2}
    df["__key"] = df["Match_Quality"].map(order)
    df = df.sort_values(by=["__key", "Name"]).drop(columns="__key")
    df.insert(0, "Scan_Name", scan_name)

    csv_path = os.path.join(scan_out_dir, f"{scan_name}_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📄 [{scan_name}] Saved CSV → {csv_path}")

    print_match_summary_for(df, scan_name)

    # Return df and proposed fixes for possible later merging into nse_match.json
    return df, proposed_fixes

# ===== Main runner =====
async def run_all_scans():
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=SYSTEM_CHROMIUM_PATH, headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(f"{SCREENER_BASE}/login/")
        await asyncio.sleep(HUMAN_DELAY)
        await page.fill('input[name="username"]', SCREENER_USER)
        await asyncio.sleep(HUMAN_DELAY * 0.5)
        await page.fill('input[name="password"]', SCREENER_PASS)
        await asyncio.sleep(HUMAN_DELAY * 0.5)
        await page.click('button[type="submit"]')
        await page.wait_for_url(f"{SCREENER_BASE}/dash/")
        await asyncio.sleep(HUMAN_DELAY)
        print("🔐 Logged in.")

        combined = []
        overall_proposed_fixes = {}

        for scan_path in scan_files:
            try:
                df_scan, fixes = await scrape_one_scan(page, scan_path)
                if df_scan is not None and not df_scan.empty:
                    combined.append(df_scan)
                if fixes:
                    # merge fixes (later we will show and ask permission to write)
                    overall_proposed_fixes.update(fixes)
                # human pause between scans
                await asyncio.sleep(HUMAN_DELAY)
            except Exception as e:
                print(f"{RED}❌ Error in scan {scan_path}: {e}{RESET}")

        await browser.close()

    if not combined:
        print(f"{RED}⚠ No successful scans; nothing to combine.{RESET}")
        return

    combined_df = pd.concat(combined, ignore_index=True)
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    combined_csv = os.path.join(BASE_RESULTS_DIR, f"combined-scan-results_{timestamp}.csv")
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Combined CSV saved → {combined_csv}")

    match_log_cols = ["Original Name", "Normalized Name", "Matched Name", "Score", "Ticker", "Match_Quality", "Match_Source"]
    log_df = pd.DataFrame(global_match_log, columns=match_log_cols)
    log_csv = os.path.join(BASE_RESULTS_DIR, f"combined-ticker-match-log_{timestamp}.csv")
    log_df.to_csv(log_csv, index=False, encoding="utf-8-sig")
    print(f"📝 Combined match log saved → {log_csv}")

    # If we discovered proposed fixes, show and optionally write to nse_match.json (backup first)
    if overall_proposed_fixes:
        print("\n⚠️ Proposed fixes gathered by auto-fixer:")
        for k, v in overall_proposed_fixes.items():
            print(f"  - {k} → {v}")

        confirm = input("\nCommit these fixes to nse_match.json? (y/n): ").strip().lower()
        if confirm == "y":
            mapping_path = ALIAS_JSON_PATH
            # load raw mapping (preserve existing raw keys)
            if os.path.exists(mapping_path):
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mapping_raw = json.load(f)
            else:
                mapping_raw = {}

            # backup existing mapping
            if mapping_raw:
                backup_path = os.path.join(os.path.dirname(mapping_path),
                                           f"nse_match_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                shutil.copy(mapping_path, backup_path)
                print(f"🗂 Backup created: {backup_path}")

            # merge proposed fixes into mapping_raw with original alias keys (as they appear)
            # (we intentionally keep user-friendly original key strings)
            mapping_raw.update(overall_proposed_fixes)

            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping_raw, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Updated nse_match.json with {len(overall_proposed_fixes)} new fixes.")
        else:
            print("\n❌ No changes written to nse_match.json.")
    else:
        print("\n✅ No proposed fixes were discovered by the auto-fixer.")

# ---- Runner ----
if "ipykernel" in sys.modules:
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(run_all_scans())
else:
    asyncio.run(run_all_scans())
