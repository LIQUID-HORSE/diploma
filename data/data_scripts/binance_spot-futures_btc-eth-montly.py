import os
import io
import zipfile
import time
import random
from datetime import datetime
from typing import Optional, Iterable

import pandas as pd
import requests


# =======================
# НАСТРОЙКИ
# =======================
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1d"
START_MONTH = "2020-01"
END_MONTH = "2025-12"

# Сеть/ретраи (чтобы не зависать "навечно")
REQUEST_TIMEOUT = (10, 45)          # (connect, read)
MAX_RETRIES = 6
MAX_TOTAL_PER_FILE_SEC = 180        # максимум времени на 1 zip
SLEEP_BETWEEN_REQ = (0.2, 0.8)      # джиттер между успешными запросами
MAX_CONSECUTIVE_FAILS = 30          # аварийный стоп если сеть/ДНС умерли

OUT_ROOT = "binance_monthly_klines"


# =======================
# КОНСТАНТЫ
# =======================
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (binance-vision-klines-downloader)"
})


# =======================
# ВСПОМОГАТЕЛЬНОЕ
# =======================
def month_range(start_ym: str, end_ym: str) -> Iterable[str]:
    s = datetime.strptime(start_ym, "%Y-%m")
    e = datetime.strptime(end_ym, "%Y-%m")
    y, m = s.year, s.month
    while (y < e.year) or (y == e.year and m <= e.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            m = 1
            y += 1

def _is_dns_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("getaddrinfo failed" in msg) or ("failed to resolve" in msg) or ("name or service not known" in msg)

def download_zip(url: str) -> Optional[bytes]:
    """
    Возвращает bytes; None если 404.
    Ретраит таймауты/обрывы/429/5xx с backoff + ограничение времени на файл.
    """
    start = time.time()
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)

            if r.status_code == 404:
                return None

            if r.status_code in (429, 418):
                sleep_s = min(60, 3 ** attempt) + random.random()
                print(f"[THROTTLE {r.status_code}] sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            if 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} server error", response=r)

            r.raise_for_status()

            if not r.content or len(r.content) < 20:
                raise IOError("Empty/too small response")

            return r.content

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.HTTPError,
                IOError) as e:

            last_err = e

            if time.time() - start > MAX_TOTAL_PER_FILE_SEC:
                raise last_err

            if _is_dns_error(e):
                sleep_s = 10 + random.random() * 5
                print(f"[DNS ISSUE] {e} | sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            sleep_s = min(30, 1.7 ** attempt) + random.random()
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e} | sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise last_err

def read_zip_csv(zip_bytes: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    csv_candidates = [n for n in names if n.lower().endswith(".csv")]
    if not csv_candidates:
        raise ValueError(f"No CSV inside ZIP. Names: {names[:10]}")
    with zf.open(csv_candidates[0]) as f:
        df = pd.read_csv(f, header=None)

    if df.shape[1] != 12:
        raise ValueError(f"Unexpected column count: {df.shape[1]} (expected 12)")

    df.columns = KLINE_COLS
    return df

def normalize_ts_to_ms(series: pd.Series) -> pd.Series:
    """
    На Binance Vision у spot (с 2025-01-01) встречаются микросекунды.
    На всякий случай нормализуем и futures тоже.
    """
    s = pd.to_numeric(series, errors="coerce")
    # если > 1e14 — считаем микросекундами и делим на 1000
    s = s.where(s <= 1e14, s / 1000.0)
    return s

def parse_klines_market_symbol(
    market: str, symbol: str, interval: str,
    start_month: str, end_month: str
) -> pd.DataFrame:
    """
    market: "spot" | "futures_um"
    """
    if market == "spot":
        base_url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/"
    elif market == "futures_um":
        base_url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/"
    else:
        raise ValueError("market must be 'spot' or 'futures_um'")

    out_dir = os.path.join(OUT_ROOT, market, f"{symbol}_{interval}_monthly_zips")
    os.makedirs(out_dir, exist_ok=True)

    dfs = []
    ok = 0
    missing = 0
    failed = 0
    consecutive_fails = 0

    for ym in month_range(start_month, end_month):
        fname_zip = f"{symbol}-{interval}-{ym}.zip"
        url = base_url + fname_zip
        zip_path = os.path.join(out_dir, fname_zip)

        try:
            # кэш
            if os.path.exists(zip_path):
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()
            else:
                zip_bytes = download_zip(url)
                if zip_bytes is None:
                    print(f"[{market.upper()}][{symbol}] [SKIP 404] {fname_zip}")
                    missing += 1
                    consecutive_fails = 0
                    continue
                with open(zip_path, "wb") as f:
                    f.write(zip_bytes)
                time.sleep(random.uniform(*SLEEP_BETWEEN_REQ))

            df = read_zip_csv(zip_bytes)
            dfs.append(df)
            ok += 1
            consecutive_fails = 0

            if ok % 12 == 0:
                print(f"[{market.upper()}][{symbol}] parsed: {ok} months...")

        except Exception as e:
            failed += 1
            consecutive_fails += 1
            print(f"[{market.upper()}][{symbol}] [FAIL] {fname_zip}: {e} (consecutive={consecutive_fails})")
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                raise RuntimeError(f"Too many consecutive failures for {market}/{symbol} — likely DNS/ISP/VPN issue.") from e

    if not dfs:
        raise RuntimeError(f"No data collected for {market}/{symbol}. Check URL/dates/connectivity.")

    full = pd.concat(dfs, ignore_index=True)

    # нормализация времени
    full["open_time_ms"] = normalize_ts_to_ms(full["open_time"])
    full["close_time_ms"] = normalize_ts_to_ms(full["close_time"])
    full["datetime_utc"] = pd.to_datetime(full["open_time_ms"], unit="ms", utc=True)

    # числовые колонки
    num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    full["trades"] = pd.to_numeric(full["trades"], errors="coerce").astype("Int64")

    # чистка
    full = full.dropna(subset=["open_time_ms"])
    full = full.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")

    out = full[[
        "datetime_utc",
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
    ]].reset_index(drop=True)

    print(f"\n[{market.upper()}][{symbol}] Done ✅ months_ok={ok}, missing_404={missing}, failed={failed}, rows={len(out)}")
    return out

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    jobs = [
        ("futures_um", "BTCUSDT"),
        ("futures_um", "ETHUSDT"),
        ("spot", "BTCUSDT"),
        ("spot", "ETHUSDT"),
    ]

    for market, symbol in jobs:
        df = parse_klines_market_symbol(
            market=market,
            symbol=symbol,
            interval=INTERVAL,
            start_month=START_MONTH,
            end_month=END_MONTH
        )
        out_csv = os.path.join(
            OUT_ROOT,
            f"{symbol}_{market}_{INTERVAL}_monthly_{START_MONTH}_{END_MONTH}.csv"
        )
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}\n")

if __name__ == "__main__":
    main()