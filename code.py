# block 0
import pandas as pd

bse_csv_path = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/NSE-BSE-equity-list/BhavCopyBSE.csv"
df_bse = pd.read_csv(bse_csv_path)
print(df_bse.columns.tolist())


# must be run only once to create the DB schema and only repeat if schema changes
# it creates a new SQLite DB file market_data.db in the current directory
# run: python3 create_db_schema.py
# then check: sqlite3 market_data.db
# ===== Block 0.1 (Expanded ORM Schema with SQLAlchemy + Cascade Deletes) =====
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

# ---------------- Core ----------------
class Symbol(Base):
    __tablename__ = "symbols"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, nullable=False)
    name = Column(String)
    exchange = Column(String)

    # Fundamentals with cascade deletes
    fundamentals_quarterly = relationship("QuarterlyResult", back_populates="symbol", cascade="all, delete-orphan")
    fundamentals_balance = relationship("BalanceSheet", back_populates="symbol", cascade="all, delete-orphan")
    fundamentals_cashflow = relationship("CashFlow", back_populates="symbol", cascade="all, delete-orphan")
    fundamentals_ratios = relationship("Ratio", back_populates="symbol", cascade="all, delete-orphan")
    fundamentals_shareholding = relationship("Shareholding", back_populates="symbol", cascade="all, delete-orphan")

    # CMP, Sentiment, Trades with cascade
    cmp_prices = relationship("CMPPrice", back_populates="symbol", cascade="all, delete-orphan")
    sentiments = relationship("Sentiment", back_populates="symbol", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="symbol", cascade="all, delete-orphan")


# ---------------- Fundamentals ----------------
class QuarterlyResult(Base):
    __tablename__ = "quarterly_results"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    quarter = Column(String, nullable=False)  # e.g. 'Mar 2025'
    sales = Column(Float)
    expenses = Column(Float)
    operating_profit = Column(Float)
    net_profit = Column(Float)
    eps = Column(Float)

    symbol = relationship("Symbol", back_populates="fundamentals_quarterly")
    __table_args__ = (UniqueConstraint("symbol_id", "quarter", name="uq_quarterly_symbol_quarter"),)


class BalanceSheet(Base):
    __tablename__ = "balance_sheets"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    report_date = Column(String, nullable=False)
    share_capital = Column(Float)
    reserves = Column(Float)
    borrowings = Column(Float)
    assets = Column(Float)
    fixed_assets = Column(Float)
    investments = Column(Float)

    symbol = relationship("Symbol", back_populates="fundamentals_balance")
    __table_args__ = (UniqueConstraint("symbol_id", "report_date", name="uq_balance_symbol_date"),)


class CashFlow(Base):
    __tablename__ = "cash_flows"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    report_date = Column(String, nullable=False)
    operating_cash = Column(Float)
    investing_cash = Column(Float)
    financing_cash = Column(Float)
    net_cash_flow = Column(Float)
    opening_cash = Column(Float)
    closing_cash = Column(Float)

    symbol = relationship("Symbol", back_populates="fundamentals_cashflow")
    __table_args__ = (UniqueConstraint("symbol_id", "report_date", name="uq_cashflow_symbol_date"),)


class Ratio(Base):
    __tablename__ = "ratios"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    report_date = Column(String, nullable=False)
    roce = Column(Float)  # decimal fraction (0.41 for 41%)
    debt_to_equity = Column(Float)
    inventory_days = Column(Float)
    net_profit = Column(Float)
    fii_holding = Column(Float)  # decimal fraction (0.122 for 12.2%)

    symbol = relationship("Symbol", back_populates="fundamentals_ratios")
    __table_args__ = (UniqueConstraint("symbol_id", "report_date", name="uq_ratios_symbol_date"),)


class Shareholding(Base):
    __tablename__ = "shareholding"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    quarter = Column(String, nullable=False)  # e.g. 'Mar 2025'
    promoters = Column(Float)
    fii = Column(Float)
    dii = Column(Float)
    public = Column(Float)
    mutual_funds = Column(Float)

    symbol = relationship("Symbol", back_populates="fundamentals_shareholding")
    __table_args__ = (UniqueConstraint("symbol_id", "quarter", name="uq_shareholding_symbol_quarter"),)


# ---------------- CMP Prices ----------------
class CMPPrice(Base):
    __tablename__ = "cmp_prices"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cmp_close = Column(Float, nullable=True)
    cmp_live = Column(Float, nullable=True)
    note = Column(String, nullable=True)

    symbol = relationship("Symbol", back_populates="cmp_prices")


# ---------------- Sentiment ----------------
class Sentiment(Base):
    __tablename__ = "sentiment"
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    source = Column(String)
    headline = Column(String)
    sentiment_score = Column(Float)  # -1.0 .. +1.0
    timestamp = Column(DateTime)

    symbol = relationship("Symbol", back_populates="sentiments")


# ---------------- Trades ----------------
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    trade_id = Column(String, unique=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    timestamp = Column(DateTime)
    price = Column(Float)
    volume = Column(Integer)
    order_type = Column(String)  # Buy/Sell

    symbol = relationship("Symbol", back_populates="trades")


# ---------------- Engine/Session ----------------
engine = create_engine("sqlite:///market_data.db")
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

print("✅ Expanded ORM schema with cascade rules created in market_data.db")


#!/usr/bin/env python3
# ===== Block 0.2 (Schema for query results metrics, extended with ScanMetric) =====

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

RESULTS_DB_PATH = "sqlite:///market_data_results.db"

results_engine = create_engine(RESULTS_DB_PATH, echo=False, future=True)
Base = declarative_base()

class QueryResult(Base):
    __tablename__ = "query_results"
    id = Column(Integer, primary_key=True)
    scan_id = Column(String, nullable=False)  # timestamp or batch ID
    ticker = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    cmp = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    pe = Column(Float, nullable=True)
    roe = Column(Float, nullable=True)
    roce = Column(Float, nullable=True)

    market_cap = Column(Float, nullable=True)
    div_yld = Column(Float, nullable=True)
    np_qtr = Column(Float, nullable=True)
    qtr_profit_var = Column(Float, nullable=True)
    sales_qtr = Column(Float, nullable=True)
    qtr_sales_var = Column(Float, nullable=True)
    avg_pat_10yrs = Column(Float, nullable=True)
    avg_div_payout_3yrs = Column(Float, nullable=True)

    scan_metrics = relationship("ScanMetric", back_populates="query_result", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("scan_id", "ticker", name="uq_scan_ticker"),
    )

class ScanMetric(Base):
    __tablename__ = "scan_metrics"
    id = Column(Integer, primary_key=True)
    query_result_id = Column(Integer, ForeignKey("query_results.id"), nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=True)

    query_result = relationship("QueryResult", back_populates="scan_metrics")

results_sessionmaker = sessionmaker(bind=results_engine)

if __name__ == "__main__":
    Base.metadata.create_all(results_engine)
    print("✅ Extended schema with ScanMetric created in market_data_results.db")


# Block 1 load the list of NSE BSE stocks into memory
import os
import pandas as pd
import difflib
from datetime import datetime
import re

# === Function to normalize company names ===
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.upper()

    # Common abbreviation expansions
    replacements = {
        r"\bLTD\b": "LIMITED",
        r"\bPVT\b": "PRIVATE",
        r"\bPVTLTD\b": "PRIVATE LIMITED",
        r"\bTECH\b": "TECHNOLOGIES",
        r"\bIND\b": "INDUSTRIES",
        r"\bSURREND\b": "SURRENDRA",
        r"\bCO\b": "COMPANY",
        r"\bCORP\b": "CORPORATION",
        r"\bSYS\b": "SYSTEMS",
        r"\bSYS\.\b": "SYSTEMS",
        r"\bEXCHAN\b": "EXCHANGERS",
        r"\bSEATIN\b": "SEATING",
        r"\b&\b": "AND",
        r" +": " ",           # collapse multiple spaces
        r"[^\w\s]": "",       # remove punctuation
    }
    for pattern, repl in replacements.items():
        name = re.sub(pattern, repl, name)
    return name.strip()

# === Load NSE + BSE mapping ===
def get_combined_mapping():
    nse_csv_path = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/NSE-BSE-equity-list/NSE.csv"
    bse_csv_path = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/NSE-BSE-equity-list/BhavCopyBSE.csv"

    mapping = {}

    # --- NSE ---
    if not os.path.exists(nse_csv_path):
        raise FileNotFoundError(f"❌ NSE equity list not found: {nse_csv_path}")
    df_nse = pd.read_csv(nse_csv_path)
    for _, row in df_nse.iterrows():
        name = normalize_name(str(row["NAME OF COMPANY"]))
        symbol = str(row["SYMBOL"]).strip().upper() + ".NS"
        mapping[name] = symbol

    # --- BSE ---
    if not os.path.exists(bse_csv_path):
        raise FileNotFoundError(f"❌ BSE equity list not found: {bse_csv_path}")
    df_bse = pd.read_csv(bse_csv_path)
    for _, row in df_bse.iterrows():
        name = normalize_name(str(row["Security Name"]))
        symbol = str(row["Scrip code"]).strip().upper() + ".BO"
        mapping[name] = symbol

    return mapping

# === Fuzzy match function ===
def get_ticker_from_name(company_name, mapping, cutoff=0.8):
    name_norm = normalize_name(company_name)
    possible_matches = difflib.get_close_matches(name_norm, mapping.keys(), n=1, cutoff=cutoff)
    if possible_matches:
        return mapping[possible_matches[0]]
    return "UNKNOWN"

# === Two-pass matching ===
def match_with_fallback(df, mapping):
    # First pass (strict)
    df["Ticker"] = df["Name"].apply(lambda x: get_ticker_from_name(x, mapping, cutoff=0.6))

    # Identify UNKNOWNs
    unknown_mask = df["Ticker"] == "UNKNOWN"
    if unknown_mask.sum() > 0:
        print(f"🔄 Retrying {unknown_mask.sum()} UNKNOWN matches with relaxed cutoff...")
        df.loc[unknown_mask, "Ticker"] = df.loc[unknown_mask, "Name"].apply(
            lambda x: get_ticker_from_name(x, mapping, cutoff=0.4)
        )
    return df

# === Load mapping ===
combined_mapping = get_combined_mapping()
print(f"✅ Loaded {len(combined_mapping)} combined NSE + BSE tickers into memory.")

# === Example integration after scraping ===
# df = ...  # your scraped dataframe
# if "Name" in df.columns:
#     df = match_with_fallback(df, combined_mapping)


#!/usr/bin/env python3
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

# Add imports for Block 0.1 and Block 0.2
from block_0_1_schema import Symbol  # We don't need engine here, but Symbol for potential use
from block_0_2_schema import results_engine, results_sessionmaker, QueryResult

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
    order = {"HIGH": 1, "LOW": 2, "NONE": 3}
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


#!/usr/bin/env python3
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

from playwright_bridge5 import init_browser, fetch_fundamentals, shutdown

# ====== Configs ======
BASE_RESULTS_DIR = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/screener-scripts-results"
CACHE_BASE_DIR = "/home/sudipaiml/stock_analysis/agcodes/scrape_results/cache"

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
    # Filter out numeric-only tickers (e.g., ISIN codes like '505710')
    if core.isdigit():
        return ""
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
    df = df[df["SymbolCore"] != ""]  # Remove invalid tickers
    df = df.drop_duplicates(subset=["SymbolCore"])
    return [(row["SymbolCore"], str(row["Name"])) for _, row in df.iterrows()]

async def run_bulk(jobs):
    """Run all scraping jobs in one persistent Chromium session."""
    p, browser, context = await init_browser()
    ok_cnt, fail_cnt = 0, 0
    failed_tickers = []
    try:
        for idx, (sym, name) in enumerate(jobs, start=1):
            print(f"\n{CYAN}[{idx}/{len(jobs)}] → {sym}{' | ' + name if name else ''}{RESET}")
            try:
                ok = await fetch_fundamentals(context, sym, retries=2)
                if ok:
                    ok_cnt += 1
                else:
                    fail_cnt += 1
                    failed_tickers.append(sym)
                polite_wait()
            except Exception as e:
                print(f"{RED}❌ Error scraping {sym}: {e}{RESET}")
                fail_cnt += 1
                failed_tickers.append(sym)
                polite_wait()
    except KeyboardInterrupt:
        print(f"{YELLOW}⚠ Interrupted by user. Shutting down gracefully...{RESET}")
    except Exception as e:
        print(f"{RED}❌ Unexpected error: {e}{RESET}")
    finally:
        try:
            await shutdown(p, browser)  # Ensure cleanup
        except Exception as e:
            print(f"{RED}❌ Error during shutdown: {e}{RESET}")
    if failed_tickers:
        fail_path = os.path.join(BASE_RESULTS_DIR, f"failed_tickers_{ts_now()}.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(failed_tickers))
        print(f"{RED}📜 Failed tickers saved to: {fail_path}{RESET}")
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

    # Prompt for all or first 50 tickers
    if jobs:
        test_choice = input(
            f"🧪 Scrape all tickers ({len(jobs)}) or first 50 for testing? (all/50): "
        ).strip().lower()
        if test_choice == "50":
            jobs = jobs[:50]
            print(f"{CYAN}🧪 Test mode: Limited to first {len(jobs)} ticker(s).{RESET}")

    print(f"{CYAN}🚀 Starting fundamentals fetch. Respectful pacing {MIN_WAIT_S}-{MAX_WAIT_S}s between calls.{RESET}")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            ok_cnt, fail_cnt = loop.run_until_complete(run_bulk(jobs))
        else:
            ok_cnt, fail_cnt = loop.run_until_complete(run_bulk(jobs))
    except KeyboardInterrupt:
        print(f"{YELLOW}⚠ Main loop interrupted. Ensuring cleanup...{RESET}")
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



#!/usr/bin/env python3
# ===== Block 4 (Bridge Loader from QueryResults & Fundamentals Cache → Fundamentals DB) =====
import sys
import os
import json
import glob
import re
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc, inspect
from bs4 import BeautifulSoup

# Import both DB schemas
from block_0_1_schema import engine as fundamentals_engine, Symbol, QuarterlyResult, CMPPrice, Ratio, Shareholding
from block_0_2_schema import results_engine, QueryResult, ScanMetric

# ====== Configs ======
BASE_RESULTS_DIR = "/home/sudipaiml/stock_analysis/agcodes/scraper-run/screener-scripts-results"
CACHE_BASE_DIR = "/home/sudipaiml/stock_analysis/agcodes/scrape_results/cache"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ====== Sessions ======
FundSession = sessionmaker(bind=fundamentals_engine)
fund_session = FundSession()

ResultSession = sessionmaker(bind=results_engine)
result_session = ResultSession()

# ====== Helpers ======
def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def find_latest_manifest() -> str | None:
    pattern = os.path.join(BASE_RESULTS_DIR, "mission_manifest_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

def load_manifest(manifest_path: str) -> list[str]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{RED}❌ Failed to load manifest {manifest_path}: {e}{RESET}")
        return []

def parse_fundamentals_html(html_path: str) -> dict | None:
    """Parse fundamentals from HTML file using BeautifulSoup."""
    if not os.path.exists(html_path):
        print(f"{YELLOW}⚠ HTML file not found: {html_path}{RESET}")
        return None

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        data = {
            "company_name": None,
            "exchange": "NSE",  # Default
            "cmp": None,
            "quarterly_results": [],
            "ratios": {},
            "shareholding": {}
        }

        # TODO: Update selectors based on actual Screener.in HTML structure
        # Extract company name
        name_tag = soup.select_one("h1")  # Placeholder
        if name_tag:
            data["company_name"] = name_tag.get_text(strip=True)

        # Extract exchange
        ticker_tag = soup.select_one(".company-info .company-status")  # Placeholder
        if ticker_tag and ("NSE" in ticker_tag.get_text() or "BSE" in ticker_tag.get_text()):
            data["exchange"] = "NSE" if "NSE" in ticker_tag.get_text() else "BSE"

        # Extract CMP
        cmp_tag = soup.select_one(".company-ratios li[name='current-price'] .value")  # Placeholder
        if cmp_tag:
            data["cmp"] = cmp_tag.get_text(strip=True).replace(",", "")

        # Extract quarterly results (latest quarter)
        qr_table = soup.select_one(".data-table.responsive-text-nowrap")  # Placeholder
        if qr_table:
            rows = qr_table.select("tbody tr")[:1]  # Latest quarter
            for row in rows:
                cols = row.select("td")
                if len(cols) >= 5:  # Quarter, Sales, Expenses, Net Profit, EPS
                    qr_data = {
                        "quarter": cols[0].get_text(strip=True),
                        "sales": cols[1].get_text(strip=True).replace(",", ""),
                        "expenses": cols[2].get_text(strip=True).replace(",", ""),
                        "net_profit": cols[3].get_text(strip=True).replace(",", ""),
                        "eps": cols[4].get_text(strip=True).replace(",", "")
                    }
                    data["quarterly_results"].append(qr_data)

        # Extract ratios
        ratios_list = soup.select(".company-ratios li")  # Placeholder
        for li in ratios_list:
            name = li.get("name")
            value = li.select_one(".value").get_text(strip=True).replace(",", "") if li.select_one(".value") else None
            if name and value:
                if name == "roce": data["ratios"]["roce"] = value
                elif name == "debt-equity": data["ratios"]["debt_to_equity"] = value
                elif name == "inventory-days": data["ratios"]["inventory_days"] = value
                elif name == "net-profit": data["ratios"]["net_profit"] = value
                elif name == "fii-holding": data["ratios"]["fii_holding"] = value

        # Extract shareholding
        sh_table = soup.select_one(".shareholding-table")  # Placeholder
        if sh_table:
            sh_row = sh_table.select("tbody tr")[:1]  # Latest quarter
            for row in sh_row:
                cols = row.select("td")
                if len(cols) >= 5:  # Quarter, Promoters, FII, DII, Public
                    sh_data = {
                        "quarter": cols[0].get_text(strip=True),
                        "promoters": cols[1].get_text(strip=True).replace(",", ""),
                        "fii": cols[2].get_text(strip=True).replace(",", ""),
                        "dii": cols[3].get_text(strip=True).replace(",", ""),
                        "public": cols[4].get_text(strip=True).replace(",", "")
                    }
                    data["shareholding"] = sh_data

        return data
    except Exception as e:
        print(f"{RED}❌ Error parsing {html_path}: {e}{RESET}")
        return None

def get_latest_scan_id():
    row = result_session.query(QueryResult.scan_id).order_by(desc(QueryResult.scan_id)).first()
    return row[0] if row else None

def safe_float(v):
    try:
        return float(v) if v is not None and v != "" else None
    except:
        return None

def verify_database_schema():
    """Verify that the fundamentals database schema matches expectations."""
    inspector = inspect(fundamentals_engine)
    if not inspector.has_table("symbols"):
        print(f"{RED}❌ Table 'symbols' not found in market_data.db. Run block_0_1_schema.py first.{RESET}")
        sys.exit(1)
    columns = [col["name"] for col in inspector.get_columns("symbols")]
    expected = ["id", "ticker", "name", "exchange"]
    if not all(col in columns for col in expected):
        print(f"{RED}❌ 'symbols' table missing expected columns: {expected}. Found: {columns}{RESET}")
        print(f"{YELLOW}⚠ Run block_0_1_schema.py to recreate the schema.{RESET}")
        sys.exit(1)
    print(f"{GREEN}✅ Fundamentals database schema verified.{RESET}")

def bridge_scan_to_fundamentals(scan_id: str | None):
    """Bridge QueryResult data to fundamentals DB (if scan_id provided)."""
    if not scan_id:
        print(f"{YELLOW}⚠ No scan_id provided; skipping QueryResult bridging.{RESET}")
        return 0, 0

    print(f"{CYAN}🔗 Bridging QueryResult scan {scan_id} into fundamentals DB{RESET}")
    rows = result_session.query(QueryResult).filter_by(scan_id=scan_id).all()
    print(f"{CYAN}📊 Found {len(rows)} rows in results DB{RESET}")

    ok_cnt, fail_cnt = 0, 0
    for qr in rows:
        try:
            # Ensure symbol exists
            sym = fund_session.query(Symbol).filter_by(ticker=qr.ticker).first()
            if not sym:
                sym = Symbol(ticker=qr.ticker, name=qr.ticker, exchange="NSE")
                fund_session.add(sym)
                fund_session.commit()

            # CMP → CMPPrice
            if qr.cmp:
                cmp = CMPPrice(symbol_id=sym.id, cmp_close=safe_float(qr.cmp))
                fund_session.add(cmp)

            # EPS, Sales, Net Profit → QuarterlyResult
            if any([qr.eps, qr.np_qtr, qr.sales_qtr]):
                qr_entry = QuarterlyResult(
                    symbol_id=sym.id,
                    quarter=f"Scan_{scan_id}",
                    eps=safe_float(qr.eps),
                    net_profit=safe_float(qr.np_qtr),
                    sales=safe_float(qr.sales_qtr)
                )
                fund_session.merge(qr_entry)

            # Ratios → Ratio
            if any([qr.roce, qr.roe, qr.pe, qr.div_yld, qr.market_cap]):
                rt = Ratio(
                    symbol_id=sym.id,
                    report_date=f"Scan_{scan_id}",
                    roce=safe_float(qr.roce),
                    net_profit=safe_float(qr.np_qtr),
                    debt_to_equity=None,
                    inventory_days=None,
                    fii_holding=None
                )
                fund_session.merge(rt)

            # Miscellaneous → ScanMetric
            for metric_name, metric_val in {
                "qtr_profit_var": qr.qtr_profit_var,
                "qtr_sales_var": qr.qtr_sales_var,
                "avg_pat_10yrs": qr.avg_pat_10yrs,
                "avg_div_payout_3yrs": qr.avg_div_payout_3yrs,
            }.items():
                if metric_val is not None:
                    sm = ScanMetric(query_result_id=qr.id, metric_name=metric_name, metric_value=safe_float(metric_val))
                    result_session.merge(sm)

            ok_cnt += 1
        except Exception as e:
            print(f"{RED}❌ Error processing QueryResult for {qr.ticker}: {e}{RESET}")
            fail_cnt += 1

    fund_session.commit()
    result_session.commit()
    print(f"{GREEN}✅ QueryResult bridge complete: OK={ok_cnt}, FAIL={fail_cnt}{RESET}")
    return ok_cnt, fail_cnt

def bridge_fundamentals_from_cache(tickers: list[str], test_mode: bool = False):
    """Bridge fundamentals from HTML cache to fundamentals DB."""
    if test_mode:
        tickers = tickers[:50]
        print(f"{CYAN}🧪 Test mode: Limited to first {len(tickers)} tickers{RESET}")

    print(f"{CYAN}🔗 Bridging {len(tickers)} tickers from cache to fundamentals DB{RESET}")
    ok_cnt, fail_cnt = 0, 0
    failed_tickers = []

    for idx, ticker in enumerate(tickers, start=1):
        print(f"{CYAN}[{idx}/{len(tickers)}] Processing {ticker}{RESET}")
        try:
            # Load and parse HTML
            html_path = os.path.join(CACHE_BASE_DIR, ticker, f"debug_{ticker}.html")
            data = parse_fundamentals_html(html_path)
            if not data:
                failed_tickers.append(ticker)
                fail_cnt += 1
                continue

            # Ensure symbol exists
            sym = fund_session.query(Symbol).filter_by(ticker=ticker).first()
            if not sym:
                name = data.get("company_name", ticker)
                exchange = data.get("exchange", "NSE")
                sym = Symbol(ticker=ticker, name=name, exchange=exchange)
                fund_session.add(sym)
                fund_session.commit()

            # CMP → CMPPrice
            if data.get("cmp"):
                cmp = CMPPrice(symbol_id=sym.id, cmp_close=safe_float(data["cmp"]))
                fund_session.add(cmp)

            # Quarterly Results
            for qr in data.get("quarterly_results", []):
                qr_entry = QuarterlyResult(
                    symbol_id=sym.id,
                    quarter=qr.get("quarter", "Unknown"),
                    sales=safe_float(qr.get("sales")),
                    expenses=safe_float(qr.get("expenses")),
                    net_profit=safe_float(qr.get("net_profit")),
                    eps=safe_float(qr.get("eps"))
                )
                fund_session.merge(qr_entry)

            # Ratios
            if data.get("ratios"):
                rt = Ratio(
                    symbol_id=sym.id,
                    report_date=f"Cache_{ts_now()}",
                    roce=safe_float(data["ratios"].get("roce")),
                    debt_to_equity=safe_float(data["ratios"].get("debt_to_equity")),
                    inventory_days=safe_float(data["ratios"].get("inventory_days")),
                    net_profit=safe_float(data["ratios"].get("net_profit")),
                    fii_holding=safe_float(data["ratios"].get("fii_holding"))
                )
                fund_session.merge(rt)

            # Shareholding
            if data.get("shareholding"):
                sh = Shareholding(
                    symbol_id=sym.id,
                    quarter=data["shareholding"].get("quarter", "Unknown"),
                    promoters=safe_float(data["shareholding"].get("promoters")),
                    fii=safe_float(data["shareholding"].get("fii")),
                    dii=safe_float(data["shareholding"].get("dii")),
                    public=safe_float(data["shareholding"].get("public"))
                )
                fund_session.merge(sh)

            ok_cnt += 1
        except Exception as e:
            print(f"{RED}❌ Error processing {ticker}: {e}{RESET}")
            failed_tickers.append(ticker)
            fail_cnt += 1

    fund_session.commit()
    if failed_tickers:
        fail_path = os.path.join(BASE_RESULTS_DIR, f"failed_cache_tickers_{ts_now()}.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(failed_tickers))
        print(f"{RED}📜 Failed tickers saved to: {fail_path}{RESET}")

    print(f"{GREEN}✅ Cache bridge complete: OK={ok_cnt}, FAIL={fail_cnt}{RESET}")
    return ok_cnt, fail_cnt

def main():
    # Verify database schema
    verify_database_schema()

    # Bridge QueryResult data (if available)
    scan_id = get_latest_scan_id()
    qr_ok, qr_fail = bridge_scan_to_fundamentals(scan_id)

    # Bridge fundamentals from cache
    manifest_path = find_latest_manifest()
    if not manifest_path:
        print(f"{RED}❌ No mission manifest found in {BASE_RESULTS_DIR}. Run Block 3 first.{RESET}")
        if qr_ok == 0 and qr_fail == 0:
            sys.exit(1)
        return

    tickers = load_manifest(manifest_path)
    if not tickers:
        print(f"{RED}❌ No tickers in manifest {manifest_path}. Nothing to do.{RESET}")
        if qr_ok == 0 and qr_fail == 0:
            sys.exit(1)
        return

    print(f"{CYAN}🗂️ Loaded {len(tickers)} tickers from manifest: {manifest_path}{RESET}")

    # Prompt for test mode
    test_choice = input(
        f"🧪 Bridge all tickers ({len(tickers)}) or first 50 for testing? (all/50): "
    ).strip().lower()
    test_mode = test_choice == "50"

    cache_ok, cache_fail = bridge_fundamentals_from_cache(tickers, test_mode)

    print(f"\n{GREEN}Done.{RESET} QueryResult: {GREEN}OK={qr_ok}{RESET}, {RED}FAIL={qr_fail}{RESET} | Cache: {GREEN}OK={cache_ok}{RESET}, {RED}FAIL={cache_fail}{RESET}")

if __name__ == "__main__":
    main()
