from __future__ import annotations

import argparse
import io
import re
import time
import zipfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import pandas as pd
import requests


BINANCE_VISION_BASE = "https://data.binance.vision"

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


@dataclass(frozen=True)
class DownloadConfig:
    timeout_connect: int = 10
    timeout_read: int = 120
    max_retries: int = 6
    backoff_base_sec: float = 1.5
    pause_between_requests_sec: float = 0.05
    user_agent: str = "binance-monthly-downloader/1.0"


def build_monthly_prefix(
    *,
    symbol: str,
    interval: str,
    market: str = "spot",
) -> str:
    """
    Build a Binance Vision prefix for monthly klines.

    Example:
      data/spot/monthly/klines/BTCUSDT/1d/
    """
    symbol = symbol.strip().upper()
    interval = interval.strip()
    market = market.strip().lower()
    return f"data/{market}/monthly/klines/{symbol}/{interval}/"


def _extract_xml_namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return ""


def _iter_keys_from_xml(xml_payload: bytes) -> tuple[list[str], Optional[str]]:
    root = ET.fromstring(xml_payload)
    ns = _extract_xml_namespace(root.tag)
    tag_key = f"{{{ns}}}Key" if ns else "Key"
    tag_truncated = f"{{{ns}}}IsTruncated" if ns else "IsTruncated"
    tag_next = f"{{{ns}}}NextContinuationToken" if ns else "NextContinuationToken"

    keys = [el.text for el in root.findall(f".//{tag_key}") if el.text]
    truncated_text = root.findtext(tag_truncated, default="false")
    is_truncated = truncated_text.strip().lower() == "true"
    next_token = root.findtext(tag_next, default=None) if is_truncated else None
    return keys, next_token


def _iter_keys_from_html(html_payload: str) -> list[str]:
    # Fallback parser for HTML listings.
    # Match keys like: data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2020-01.zip
    return sorted(set(re.findall(r"(data/[^\"'<>]+?\.zip)", html_payload)))


def list_monthly_zip_keys(
    *,
    symbol: str,
    interval: str,
    market: str = "spot",
    max_keys_per_page: int = 1000,
) -> list[str]:
    """
    List all available monthly ZIP object keys for a symbol/interval.

    The function first tries S3 XML listing (`list-type=2`) and falls back to
    HTML parsing if needed.
    """
    prefix = build_monthly_prefix(symbol=symbol, interval=interval, market=market)
    keys: list[str] = []
    token: Optional[str] = None

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys_per_page)}
        if token:
            query["continuation-token"] = token
        url = f"{BINANCE_VISION_BASE}/?{urlencode(query)}"

        resp = session.get(url, timeout=(10, 60))
        resp.raise_for_status()
        payload = resp.content

        if payload.lstrip().startswith(b"<?xml"):
            page_keys, token = _iter_keys_from_xml(payload)
            keys.extend(page_keys)
            if not token:
                break
        else:
            # HTML response usually already includes all links for this prefix.
            html = payload.decode("utf-8", errors="replace")
            keys = _iter_keys_from_html(html)
            break

    zip_keys = [k for k in keys if k.endswith(".zip")]
    zip_keys = [k for k in zip_keys if f"/{symbol.upper()}-{interval}-" in k]
    return sorted(set(zip_keys))


def _iter_months(start_month: str, end_month: str) -> list[str]:
    sy, sm = map(int, start_month.split("-"))
    ey, em = map(int, end_month.split("-"))
    cur_y, cur_m = sy, sm
    out: list[str] = []
    while (cur_y, cur_m) <= (ey, em):
        out.append(f"{cur_y:04d}-{cur_m:02d}")
        cur_m += 1
        if cur_m > 12:
            cur_m = 1
            cur_y += 1
    return out


def probe_monthly_zip_keys(
    *,
    symbol: str,
    interval: str,
    market: str = "spot",
    start_month: str = "2017-01",
    end_month: Optional[str] = None,
    timeout: tuple[int, int] = (10, 60),
) -> list[str]:
    """
    Fallback discovery: probe expected monthly ZIP names by month.
    """
    symbol_u = symbol.strip().upper()
    end_month = end_month or datetime.utcnow().strftime("%Y-%m")
    months = _iter_months(start_month, end_month)
    prefix = build_monthly_prefix(symbol=symbol_u, interval=interval, market=market)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    keys: list[str] = []
    for ym in months:
        filename = f"{symbol_u}-{interval}-{ym}.zip"
        key = f"{prefix}{filename}"
        url = f"{BINANCE_VISION_BASE}/{key}"

        # HEAD is often enough and cheaper, but some CDNs are picky;
        # fallback to GET on non-200/exception.
        ok = False
        try:
            r = session.head(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                ok = True
            elif r.status_code not in (404, 403, 405):
                # unknown status, try GET below
                pass
        except Exception:
            pass

        if not ok:
            try:
                r = session.get(url, timeout=timeout, stream=True)
                if r.status_code == 200:
                    ok = True
                r.close()
            except Exception:
                pass

        if ok:
            keys.append(key)

    return keys


def _extract_ym_from_key(key: str) -> Optional[str]:
    m = re.search(r"-(\d{4}-\d{2})\.zip$", key)
    return m.group(1) if m else None


def filter_monthly_zip_keys(
    keys: Iterable[str],
    *,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> list[str]:
    """
    Filter object keys by YYYY-MM inclusive bounds.
    """
    out: list[str] = []
    for key in keys:
        ym = _extract_ym_from_key(key)
        if ym is None:
            continue
        if start_month and ym < start_month:
            continue
        if end_month and ym > end_month:
            continue
        out.append(key)
    return sorted(out)


def _download_bytes_with_retries(
    session: requests.Session,
    url: str,
    *,
    cfg: DownloadConfig,
) -> Optional[bytes]:
    timeout = (cfg.timeout_connect, cfg.timeout_read)

    for attempt in range(1, cfg.max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return None
            if response.status_code in (429, 418) or 500 <= response.status_code < 600:
                raise requests.HTTPError(f"Retryable status: {response.status_code}", response=response)
            response.raise_for_status()
            return response.content
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt >= cfg.max_retries:
                raise
            delay = min(60.0, cfg.backoff_base_sec ** attempt)
            time.sleep(delay)
    return None


def read_kline_zip_to_df(zip_bytes: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = zf.namelist()
    csv_candidates = [n for n in names if n.lower().endswith(".csv")]
    if not csv_candidates:
        raise ValueError(f"ZIP does not contain CSV: {names[:5]}")
    with zf.open(csv_candidates[0]) as f:
        df = pd.read_csv(f, header=None)

    if df.shape[1] != 12:
        raise ValueError(f"Expected 12 columns in kline CSV, got {df.shape[1]}")
    df.columns = KLINE_COLUMNS
    return df


def normalize_binance_ts_to_ms(ts: pd.Series) -> pd.Series:
    """
    Spot data can be in ms (~1e12) or us (~1e15).
    Normalize everything to milliseconds.
    """
    s = pd.to_numeric(ts, errors="coerce")
    return s.where(s <= 1e14, s / 1000.0)


def infer_active_name(symbol: str) -> str:
    """
    Infer human-friendly active name from trading pair symbol.

    Examples:
      BTCUSDT -> btc
      ETHUSDC -> eth
      SOLBTC  -> sol
      BTCUSDT_PERP -> btcusdt_perp (fallback)
    """
    s = symbol.strip().upper()
    common_quotes = (
        "USDT",
        "USDC",
        "BUSD",
        "FDUSD",
        "TUSD",
        "BTC",
        "ETH",
        "BNB",
        "TRY",
        "EUR",
        "BRL",
        "RUB",
    )
    for q in common_quotes:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)].lower()
    return s.lower()


def make_raw_output_path(
    *,
    symbol: str,
    interval: str,
    raw_dir: Path,
    active_name: Optional[str] = None,
) -> Path:
    """
    Build output path with format:
      raw/ACTIVE-TIMEFRAME-RAW.csv
    """
    active = (active_name or infer_active_name(symbol)).lower()
    filename = f"{active}-{interval}-RAW.csv"
    return raw_dir / filename


def postprocess_klines(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = df_raw.copy()
    out["open_time_ms"] = normalize_binance_ts_to_ms(out["open_time"])
    out["close_time_ms"] = normalize_binance_ts_to_ms(out["close_time"])
    out["datetime_utc"] = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").astype("Int64")

    out = out.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")
    out = out[
        [
            "datetime_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
    ]
    return out.reset_index(drop=True)


def download_monthly_klines(
    *,
    symbol: str,
    interval: str,
    market: str = "spot",
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    cfg: DownloadConfig = DownloadConfig(),
) -> pd.DataFrame:
    """
    Download all available Binance monthly klines for symbol/interval.

    Parameters
    ----------
    symbol, interval, market
        Asset and timeframe selection (e.g. BTCUSDT + 1d or 4h).
    start_month, end_month
        Optional bounds in YYYY-MM (inclusive).
    cache_dir
        Optional directory to store/read downloaded ZIP files.
    cfg
        Download/retry settings.
    """
    symbol_u = symbol.strip().upper()
    keys: list[str] = []
    list_error: Optional[Exception] = None
    try:
        keys = list_monthly_zip_keys(symbol=symbol_u, interval=interval, market=market)
    except Exception as e:
        list_error = e

    keys = filter_monthly_zip_keys(keys, start_month=start_month, end_month=end_month)

    if not keys:
        # Robust fallback for environments where prefix listing is blocked.
        probe_start = start_month or "2017-01"
        probe_end = end_month
        keys = probe_monthly_zip_keys(
            symbol=symbol_u,
            interval=interval,
            market=market,
            start_month=probe_start,
            end_month=probe_end,
        )

    if not keys:
        hint = f" Listing error: {list_error}" if list_error is not None else ""
        raise RuntimeError(
            f"No monthly ZIP keys found for symbol={symbol_u}, interval={interval}, market={market}."
            f"{hint}"
        )

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})

    frames: list[pd.DataFrame] = []
    for i, key in enumerate(keys, start=1):
        url = f"{BINANCE_VISION_BASE}/{key}"
        local_zip = cache_dir / Path(key).name if cache_dir is not None else None

        if local_zip is not None and local_zip.exists():
            zip_bytes = local_zip.read_bytes()
        else:
            zip_bytes = _download_bytes_with_retries(session, url, cfg=cfg)
            if zip_bytes is None:
                continue
            if local_zip is not None:
                local_zip.write_bytes(zip_bytes)

        df_part = read_kline_zip_to_df(zip_bytes)
        frames.append(df_part)

        if cfg.pause_between_requests_sec > 0:
            time.sleep(cfg.pause_between_requests_sec)

        if i % 50 == 0:
            print(f"Processed {i}/{len(keys)} monthly files...")

    if not frames:
        raise RuntimeError("No monthly files were successfully downloaded/parsing failed for all files.")

    raw = pd.concat(frames, ignore_index=True)
    return postprocess_klines(raw)


def download_and_save_monthly_klines_csv(
    *,
    symbol: str,
    interval: str,
    output_csv: Path,
    market: str = "spot",
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    cfg: DownloadConfig = DownloadConfig(),
) -> pd.DataFrame:
    """
    Convenience wrapper: download monthly klines and write final CSV.
    """
    df = download_monthly_klines(
        symbol=symbol,
        interval=interval,
        market=market,
        start_month=start_month,
        end_month=end_month,
        cache_dir=cache_dir,
        cfg=cfg,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def download_and_save_monthly_klines_raw(
    *,
    symbol: str,
    interval: str,
    market: str = "spot",
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    raw_dir: Path = Path(__file__).resolve().parent / "raw",
    cache_dir: Optional[Path] = None,
    active_name: Optional[str] = None,
    cfg: DownloadConfig = DownloadConfig(),
) -> Path:
    """
    Download monthly klines and save raw CSV in:
      raw/ACTIVE-TIMEFRAME-RAW.csv
    """
    output_csv = make_raw_output_path(
        symbol=symbol,
        interval=interval,
        raw_dir=raw_dir,
        active_name=active_name,
    )
    download_and_save_monthly_klines_csv(
        symbol=symbol,
        interval=interval,
        output_csv=output_csv,
        market=market,
        start_month=start_month,
        end_month=end_month,
        cache_dir=cache_dir,
        cfg=cfg,
    )
    return output_csv


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Binance monthly kline history to raw CSV.")
    p.add_argument("--symbol", required=True, help="Trading pair, e.g. BTCUSDT")
    p.add_argument("--interval", required=True, help="Kline interval, e.g. 1d or 4h")
    p.add_argument("--market", default="spot", help="Market section on Binance Vision (default: spot)")
    p.add_argument("--start-month", default=None, help="Optional YYYY-MM lower bound")
    p.add_argument("--end-month", default=None, help="Optional YYYY-MM upper bound")
    p.add_argument("--raw-dir", default=str(Path(__file__).resolve().parent / "raw"), help="Raw output directory")
    p.add_argument("--cache-dir", default=None, help="Optional ZIP cache directory")
    p.add_argument("--active-name", default=None, help="Optional ACTIVE label override in filename")
    return p


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    out = download_and_save_monthly_klines_raw(
        symbol=args.symbol,
        interval=args.interval,
        market=args.market,
        start_month=args.start_month,
        end_month=args.end_month,
        raw_dir=Path(args.raw_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        active_name=args.active_name,
    )
    print(f"Saved raw CSV: {out}")


if __name__ == "__main__":
    main()
