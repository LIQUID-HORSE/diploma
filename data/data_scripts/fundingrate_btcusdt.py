import os
import io
import zipfile
import time
import random
from datetime import datetime
from typing import Optional

import pandas as pd
import requests


# =======================
# НАСТРОЙКИ
# =======================
SYMBOL = "BTCUSDT"
BASE_URL = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYMBOL}/"

START_MONTH = "2020-01"
END_MONTH   = "2025-12"

OUT_DIR = f"binance_{SYMBOL.lower()}_fundingrate_monthly_zips"
OUT_CSV = f"{SYMBOL}_fundingRate_monthly_{START_MONTH}_{END_MONTH}.csv"

# сеть/ретраи
REQUEST_TIMEOUT = (10, 45)          # connect, read
MAX_RETRIES = 6
MAX_TOTAL_PER_FILE_SEC = 180        # максимум 3 минуты на один файл
SLEEP_BETWEEN_REQ = (0.2, 0.8)      # джиттер между успешными запросами
MAX_CONSECUTIVE_FAILS = 30          # аварийный стоп если сеть умерла


# =======================
# ВСПОМОГАТЕЛЬНОЕ
# =======================
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (binance-vision-fundingrate-downloader)"
})

COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

def month_range(start_ym: str, end_ym: str):
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
    Ретраит таймауты/429/5xx/обрывы.
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

    if df.shape[1] != 3:
        raise ValueError(f"Unexpected column count: {df.shape[1]} (expected 3)")

    df.columns = COLS
    return df

def normalize_ts_to_ms(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # если вдруг встретятся микросекунды (~1e15), приводим к ms
    s = s.where(s <= 1e14, s / 1000.0)
    return s


# =======================
# ОСНОВНАЯ ЛОГИКА
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = []
    ok = 0
    missing = 0
    bad = 0
    consecutive_fails = 0

    for ym in month_range(START_MONTH, END_MONTH):
        fname_zip = f"{SYMBOL}-fundingRate-{ym}.zip"
        url = BASE_URL + fname_zip
        zip_path = os.path.join(OUT_DIR, fname_zip)

        try:
            # кэш
            if os.path.exists(zip_path):
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()
            else:
                zip_bytes = download_zip(url)
                if zip_bytes is None:
                    print(f"[SKIP 404] {fname_zip}")
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
                print(f"Downloaded/parsed: {ok} months...")

        except Exception as e:
            consecutive_fails += 1
            bad += 1
            print(f"[FAIL] {fname_zip}: {e} (consecutive={consecutive_fails})")

            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                raise RuntimeError("Too many consecutive failures — likely DNS/ISP/VPN issue.") from e

            continue

    if not dfs:
        raise RuntimeError("No data collected. Check dates / URL / connectivity.")

    full = pd.concat(dfs, ignore_index=True)

    # типы
    full["calc_time_ms"] = normalize_ts_to_ms(full["calc_time"])
    full["datetime_utc"] = pd.to_datetime(full["calc_time_ms"], unit="ms", utc=True)

    full["funding_interval_hours"] = pd.to_numeric(full["funding_interval_hours"], errors="coerce").astype("Int64")
    full["last_funding_rate"] = pd.to_numeric(full["last_funding_rate"], errors="coerce")

    # чистка/сортировка
    full = full.dropna(subset=["calc_time_ms"])
    full = full.drop_duplicates(subset=["calc_time_ms"]).sort_values("calc_time_ms")

    out = full[["datetime_utc", "funding_interval_hours", "last_funding_rate"]].reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print("\nDone ✅")
    print(f"Months parsed: {ok}, missing(404): {missing}, failed: {bad}")
    print(f"Rows in combined CSV: {len(out)}")
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()