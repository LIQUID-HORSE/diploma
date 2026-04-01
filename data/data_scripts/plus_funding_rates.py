import os
import pandas as pd


# =======================
# НАСТРОЙКИ
# =======================
START_MONTH = "2020-01"
END_MONTH   = "2025-12"

# где лежат futures daily klines (из прошлого скрипта)
FUTURES_DIR = "binance_monthly_klines"

# где лежат уже собранные funding CSV
FUNDING_DIR = "."  # текущая папка, поменяй если нужно

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# имена входных файлов (ожидаемые)
FUTURES_TMPL = os.path.join(FUTURES_DIR, "{sym}_futures_um_1d_monthly_{start}_{end}.csv")
FUNDING_TMPL = os.path.join(FUNDING_DIR, "{sym}_fundingRate_monthly_{start}_{end}.csv")

# имена выходных файлов
OUT_TMPL = os.path.join(FUTURES_DIR, "{sym}_futures_um_1d_with_funding_{start}_{end}.csv")


# =======================
# ВСПОМОГАТЕЛЬНОЕ
# =======================
def ensure_datetime_utc_mixed(series: pd.Series) -> pd.Series:
    """
    В funding CSV часто смешаны форматы:
    - 2020-06-03 08:00:00+00:00
    - 2020-06-04 08:00:00.004000+00:00
    Чтобы pandas не превращал часть строк в NaT, используем format="mixed".
    """
    # series может быть уже datetime, но безопасно привести к str
    return pd.to_datetime(series.astype(str), utc=True, errors="coerce", format="mixed")

def normalize_ts_to_ms(series: pd.Series) -> pd.Series:
    """
    На всякий случай: если вдруг попались микросекунды (~1e15), приводим к ms.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s <= 1e14, s / 1000.0)

def build_daily_funding(funding_csv_path: str) -> pd.DataFrame:
    """
    Делает дневной funding:
    date_utc, funding_rate_day_sum, funding_points_in_day
    """
    f = pd.read_csv(funding_csv_path)

    if "datetime_utc" in f.columns:
        f["datetime_utc"] = ensure_datetime_utc_mixed(f["datetime_utc"])
    else:
        # поддержка "сырых" выгрузок, если есть только calc_time/calc_time_ms
        ts_col = "calc_time_ms" if "calc_time_ms" in f.columns else "calc_time"
        f[ts_col] = normalize_ts_to_ms(f[ts_col])
        f["datetime_utc"] = pd.to_datetime(f[ts_col], unit="ms", utc=True, errors="coerce")

    f["last_funding_rate"] = pd.to_numeric(f["last_funding_rate"], errors="coerce")
    f = f.dropna(subset=["datetime_utc", "last_funding_rate"]).copy()

    f["date_utc"] = f["datetime_utc"].dt.date

    daily = (f.groupby("date_utc", as_index=False)
              .agg(
                  funding_rate_day_sum=("last_funding_rate", "sum"),
                  funding_points_in_day=("last_funding_rate", "size"),
              ))

    return daily

def merge_funding_into_futures(futures_csv_path: str, daily_funding: pd.DataFrame) -> pd.DataFrame:
    k = pd.read_csv(futures_csv_path)

    if "datetime_utc" not in k.columns:
        raise ValueError(f"Expected 'datetime_utc' column in futures file: {futures_csv_path}")

    k["datetime_utc"] = pd.to_datetime(k["datetime_utc"], utc=True, errors="coerce")
    k = k.dropna(subset=["datetime_utc"]).copy()

    k["date_utc"] = k["datetime_utc"].dt.date

    merged = k.merge(daily_funding, on="date_utc", how="left")

    # ВАЖНО:
    # NaN здесь означает, что funding для дня не найден (дыра),
    # а не "нулевой funding". Поэтому:
    # - оставим NaN как есть
    # - отдельно выведем отчёт
    return merged

def report_coverage(symbol: str, merged: pd.DataFrame):
    total_days = len(merged)
    missing_days = merged["funding_rate_day_sum"].isna().sum()

    # Сколько записей funding в день (обычно 3 при interval=8)
    # Считаем только там, где funding найден
    pts = merged["funding_points_in_day"].dropna()
    pts_dist = pts.value_counts().sort_index()

    print(f"\n[{symbol}] Coverage report")
    print(f"  Futures days: {total_days}")
    print(f"  Missing funding days (NaN): {missing_days}")
    if len(pts_dist):
        print("  funding_points_in_day distribution (found days):")
        for k, v in pts_dist.items():
            print(f"    {int(k)} -> {int(v)} days")
    else:
        print("  No funding matched at all (check paths / parsing).")

    if missing_days:
        # покажем первые 20 дат с пропусками
        miss_dates = merged.loc[merged["funding_rate_day_sum"].isna(), "date_utc"].astype(str).head(20).tolist()
        print("  First missing dates:", ", ".join(miss_dates))

def process_symbol(symbol: str):
    futures_path = FUTURES_TMPL.format(sym=symbol, start=START_MONTH, end=END_MONTH)
    funding_path = FUNDING_TMPL.format(sym=symbol, start=START_MONTH, end=END_MONTH)
    out_path = OUT_TMPL.format(sym=symbol, start=START_MONTH, end=END_MONTH)

    if not os.path.exists(futures_path):
        raise FileNotFoundError(f"Futures file not found: {futures_path}")
    if not os.path.exists(funding_path):
        raise FileNotFoundError(f"Funding file not found: {funding_path}")

    daily_funding = build_daily_funding(funding_path)
    merged = merge_funding_into_futures(futures_path, daily_funding)

    report_coverage(symbol, merged)

    # Если ты ХОЧЕШЬ именно нули (как раньше) — раскомментируй:
    # merged["funding_rate_day_sum"] = merged["funding_rate_day_sum"].fillna(0.0)
    # merged["funding_points_in_day"] = merged["funding_points_in_day"].fillna(0).astype(int)

    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def main():
    os.makedirs(FUTURES_DIR, exist_ok=True)
    for sym in SYMBOLS:
        process_symbol(sym)

if __name__ == "__main__":
    main()