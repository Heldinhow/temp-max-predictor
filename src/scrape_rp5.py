#!/usr/bin/env python3
import math
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

TODAY = datetime.now().strftime('%Y-%m-%d')

MONTH_MAP = {
    'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
    'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
    'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
}

CSV_COLUMNS = [
    'Hora local em Guarulhos (aeroporto)',
    'T', 'Po', 'P', 'U', 'DD', 'Ff', 'ff10', 'WW', "W'W'", 'c', 'VV', 'Td'
]


def fetch_rp5_page() -> str:
    url = 'https://rp5.lv/Arquivo_de_tempo_em_Guarulhos_(aeroporto),_METAR'
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8', errors='replace')


def parse_date_td(td_text: str) -> int | None:
    match = re.search(r'(\d+)\s*de\s+(\w+)', td_text)
    if not match:
        return None
    day = int(match.group(1))
    month_name = match.group(2).lower()
    month = MONTH_MAP.get(month_name)
    if month is None:
        return None
    year_match = re.search(r'(\d{4})', td_text)
    year = int(year_match.group(1)) if year_match else datetime.now().year
    return year * 10000 + month * 100 + day


def extract_wind_speed(td_element) -> float | None:
    wv = td_element.find('div', class_='wv_0')
    if wv:
        m = re.search(r'([\d.]+)\s*m/s', wv.get_text())
        if m:
            return float(m.group(1))
    return None


def extract_cloud_cover(td_element) -> float | None:
    cc = td_element.find('div', class_='cc_0')
    if cc:
        text = cc.get_text()
        m = re.search(r'\((\d+)(?:[-\s]*(\d+))?%\)', text)
        if m:
            low = int(m.group(1))
            high = int(m.group(2)) if m.group(2) else low
            avg = (low + high) / 2.0
            return round(avg / 100 * 8, 1)
        if 'Sem nuvens' in text or '0%' in text:
            return 0.0
    return None


def extract_visibility(td_element) -> float | None:
    vv = td_element.find('div', class_='vv_0')
    if vv:
        text = vv.get_text(strip=True)
        m = re.search(r'^([\d.]+)', text)
        if m:
            return float(m.group(1))
    return None


def extract_value(td_element, div_class: str) -> float | None:
    div = td_element.find('div', class_=div_class)
    if div:
        text = div.get_text(strip=True)
        try:
            return float(text)
        except ValueError:
            pass
    return None


def extract_temperature(td_element) -> float | None:
    return extract_value(td_element, 't_0')


def extract_pressure(td_element) -> float | None:
    return extract_value(td_element, 'p_0')


def extract_humidity_text(td_element) -> str:
    div = td_element.find('div', class_='dfs')
    if div:
        return div.get_text(strip=True)
    return ''


def parse_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', id='archiveTable')
    if not table:
        raise ValueError("Could not find archive table")

    rows = table.find_all('tr')
    observations = []
    current_date = None

    for tr in rows:
        tds = tr.find_all('td')
        if len(tds) < 12:
            continue

        first_td = tds[0]
        is_date_row = first_td.get('colspan') == '2' or 'cl_dt' in (first_td.get('class') or [])

        if is_date_row:
            date_val = parse_date_td(first_td.get_text())
            if date_val:
                current_date = date_val
            time_td = tds[1]
            data_offset = 2
        else:
            time_td = tds[0]
            data_offset = 1

        time_div = time_td.find('div', class_='dfs')
        if not time_div:
            continue
        time_str = time_div.get_text(strip=True)
        if ':' not in time_str:
            continue

        if current_date is None:
            continue

        year = current_date // 10000
        month = (current_date // 100) % 100
        day = current_date % 100
        ts = f'{year:04d}-{month:02d}-{day:02d} {time_str}:00'

        if data_offset + 12 > len(tds):
            continue

        def _td(idx):
            return tds[data_offset + idx]

        temp = extract_temperature(_td(0))
        po = extract_pressure(_td(1))
        p = extract_pressure(_td(2))
        hum_raw = extract_humidity_text(_td(3))
        dd = _td(4).get_text(strip=True) if len(_td(4).get_text(strip=True)) > 0 else None
        ff = extract_wind_speed(_td(5))
        ff10_text = _td(6).get_text(strip=True).replace('\xa0', '').strip()
        ff10 = float(ff10_text) if ff10_text and ff10_text != '-' else None
        ww = _td(7).get_text(strip=True) or None
        w1 = _td(8).get_text(strip=True) or None
        cloud = extract_cloud_cover(_td(9))
        vv = extract_visibility(_td(10))
        td_n = extract_temperature(_td(11))

        try:
            hum = float(hum_raw) if hum_raw else None
        except ValueError:
            hum = None

        observations.append({
            'datetime': ts,
            'T': temp,
            'Po': po,
            'P': p,
            'U': hum,
            'DD': dd,
            'Ff': ff,
            'ff10': ff10,
            'WW': ww,
            "W'W'": w1 if w1 else '',
            'c': cloud,
            'VV': vv,
            'Td': td_n,
        })

    if not observations:
        raise ValueError("Could not parse any observations from archive table")

    df = pd.DataFrame(observations)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    return df


def write_raw_csv(df: pd.DataFrame, path: str):
    df = df.copy()
    for col in df.columns:
        if col == 'datetime':
            continue
        df[col] = df[col].apply(_fmt)
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    with open(path, 'w', encoding='latin-1') as f:
        f.write('# Estacao meteorologica Guarulhos (aeroporto), Brasil, METAR=SBGR\n')
        f.write('# Formato: ANSI\n')
        f.write(f'# Dados raspados em {now}\n')
        f.write('#\n')
        f.write(';'.join(f'"{c}"' for c in CSV_COLUMNS) + '\n')
        for _, row in df.iterrows():
            vals = [
                row['datetime'].strftime('%d.%m.%Y %H:%M'),
                row['T'], row['Po'], row['P'], row['U'],
                row['DD'], row['Ff'], row['ff10'],
                row['WW'], row["W'W'"], row['c'], row['VV'], row['Td'],
            ]
            f.write(';'.join(f'"{v}"' for v in vals) + '\n')
    print(f"Saved {len(df)} observations to {path}")


def _fmt(val):
    if val is None:
        return ''
    if isinstance(val, float):
        if pd.isna(val) or not math.isfinite(val):
            return ''
        return f'{val:.1f}'
    s = str(val)
    if s in ('', '<NA>', 'nan', 'nat', 'none'):
        return ''
    return s


def run_prediction(daily_csv: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_weather_data, prepare_features, get_feature_columns
    from src.analog_days import AnalogDaysFinder
    from src.model import TempPredictor
    import numpy as np

    df = prepare_features(load_weather_data(daily_csv))
    df = df.dropna(subset=['temp_max_real'])

    features = get_feature_columns()
    available = [f for f in features if f in df.columns]
    target = 'temp_max_real'
    cutoff = '2024-01-01'

    train_df = df[df['date'] < pd.Timestamp(cutoff)].copy()
    val_df = df[df['date'] >= pd.Timestamp(cutoff)].copy()
    today = pd.Timestamp(TODAY)
    today_rows = df[df['date'] == today]

    if len(today_rows) == 0:
        print(f"No data for {TODAY} in daily CSV")
        return {'date': TODAY, 'error': 'no data for today'}

    today_row = today_rows.iloc[0]

    for col in available:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        val_df[col] = val_df[col].fillna(med)

    X_train = train_df[available].astype(np.float64)
    y_train = train_df[target].astype(np.float64)
    X_val = val_df[available].astype(np.float64)
    y_val = val_df[target].astype(np.float64)

    af = AnalogDaysFinder(n_neighbors=20, month_window=1)
    af.fit(train_df, available)
    cond = {f: float(today_row[f]) for f in available}
    cond['date'] = today
    analog_pred = af.predict_max_temp(cond, target_col=target)

    lgb = TempPredictor()
    lgb.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

    # Optimal ensemble weight from validation
    lgb_val_preds = lgb.predict(X_val)
    analog_val_preds = []
    for _, row in val_df.iterrows():
        c = {f: row[f] for f in available}
        c['date'] = row['date']
        analog_val_preds.append(af.predict_max_temp(c, target_col=target)['prediction'])
    analog_val_preds = np.array(analog_val_preds)
    actuals = y_val.values

    best_w = 0.6
    best_mae = float('inf')
    for w in np.arange(0, 1.05, 0.05):
        mae = np.mean(np.abs(w * lgb_val_preds + (1 - w) * analog_val_preds - actuals))
        if mae < best_mae:
            best_mae = mae
            best_w = w

    X_today = pd.DataFrame([[float(today_row[f]) for f in available]], columns=available)
    lgb_pred = lgb.predict(X_today)[0]
    ensemble = best_w * lgb_pred + (1 - best_w) * analog_pred['prediction']
    actual = float(today_row[target]) if pd.notna(today_row[target]) else None

    observed_max = max(
        float(today_row.get('temp_06h', -999)),
        float(today_row.get('temp_09h', -999)),
        float(today_row.get('temp_morning_mean', -999)),
    )
    if observed_max > 0 and ensemble < observed_max:
        print(f"  ⚠ Previsão corrigida: {ensemble:.1f}°C → {observed_max:.1f}°C "
              f"(mínimo = máxima já observada hoje)")
        ensemble = round(observed_max, 1)

    print(f"\n{'='*55}")
    print(f"  PREVISÃO PARA {TODAY}")
    print(f"{'='*55}")
    print(f"  Peso adaptativo: LGB={best_w:.0%} / Analog={1-best_w:.0%}")
    print(f"\n  Condições matinais:")
    for f in ['temp_06h', 'temp_09h', 'humidity_06h', 'pressure_06h',
              'temp_morning_mean', 'wind_speed', 'cloud_cover']:
        if f in today_row and pd.notna(today_row[f]):
            print(f"    {f}: {today_row[f]:.1f}")
    lh = today_row.get('latest_hour', None)
    lt = today_row.get('latest_temp', None)
    if pd.notna(lh):
        print(f"\n  Última obs: {int(lh):02d}h — {lt:.1f}°C")
    print(f"\n  LightGBM:     {lgb_pred:.1f}°C")
    print(f"  Analog Days:  {analog_pred['prediction']:.1f}°C")
    print(f"  ───────────────────────")
    print(f"  ENSEMBLE:     {ensemble:.1f}°C")
    if actual:
        print(f"  Real max:     {actual:.1f}°C (erro: {ensemble - actual:+.1f}°C)")

    return {
        'date': str(TODAY),
        'lgb': round(lgb_pred, 1),
        'analog': round(analog_pred['prediction'], 1),
        'ensemble': round(ensemble, 1),
        'actual': actual,
        'lgb_weight': round(best_w, 2),
    }


def merge_with_existing(scraped_csv: str, existing_csv: str) -> str:
    """Merge scraped data into existing raw CSV, deduplicating by timestamp.
    Updates the existing file in-place so data accumulates between runs."""
    from src.process_rp5 import parse_rp5_csv as read_raw

    existing = read_raw(existing_csv)
    scraped = read_raw(scraped_csv)

    scraped_ts = set(scraped['Hora local em Guarulhos (aeroporto)'].dropna())
    new_rows = existing[~existing['Hora local em Guarulhos (aeroporto)'].isin(scraped_ts)].copy()

    merged = pd.concat([new_rows, scraped], ignore_index=True)
    merged = merged.drop_duplicates(subset=['Hora local em Guarulhos (aeroporto)']).reset_index(drop=True)

    with open(existing_csv, 'r', encoding='latin-1') as f:
        header = []
        for line in f:
            header.append(line)
            if not line.startswith('#'):
                break

    with open(existing_csv, 'w', encoding='latin-1') as f:
        f.writelines(header)
        merged.to_csv(f, sep=';', index=False, header=False, quoting=1, encoding='latin-1')

    print(f"Merged: {len(existing)} existing + {len(scraped)} scraped → {len(merged)} unique rows")
    return existing_csv


def process_and_predict(raw_csv: str, daily_csv: str = 'data/daily_latest.csv',
                        existing_raw: str | None = None, cutoff_hour: int = 23):
    from src.process_rp5 import parse_rp5_csv, process_to_daily

    if existing_raw:
        csv_path = merge_with_existing(raw_csv, existing_raw)
    else:
        csv_path = raw_csv

    df = parse_rp5_csv(csv_path)
    daily = process_to_daily(df, cutoff_hour=cutoff_hour)
    daily.to_csv(daily_csv, index=False)
    print(f"Daily: {len(daily)} days, {daily['date'].min()} → {daily['date'].max()}, "
          f"cutoff_hour={cutoff_hour}")
    return run_prediction(daily_csv)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Scrape rp5.lv live data + predict max temp for Guarulhos')
    parser.add_argument('--raw-output', '-o', default='/tmp/sbgr_latest.csv',
                        help='Output for scraped raw CSV')
    parser.add_argument('--daily-output', default='data/daily_latest.csv',
                        help='Output for processed daily CSV')
    local = 'data/sbgr_raw.csv'
    parser.add_argument('--existing', default=local if os.path.exists(local) else None,
                        help='Path to existing full raw rp5.lv CSV to merge with')
    parser.add_argument('--scrape-only', action='store_true',
                        help='Only scrape, skip prediction')
    parser.add_argument('--cutoff-hour', type=int, default=23,
                        help='Use only data up to this hour for features (default: 23 = all)')
    args = parser.parse_args()

    print("Fetching rp5.lv archive page...")
    html = fetch_rp5_page()
    df = parse_table(html)
    print(f"Parsed {len(df)} observations from {df['datetime'].min()} to {df['datetime'].max()}")

    write_raw_csv(df, args.raw_output)

    if args.scrape_only:
        sys.exit(0)

    result = process_and_predict(args.raw_output, args.daily_output,
                                 args.existing, args.cutoff_hour)
