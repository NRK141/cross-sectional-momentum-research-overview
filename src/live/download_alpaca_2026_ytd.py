from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

YEAR = 2026
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
ASSET_URLS = (
    "https://paper-api.alpaca.markets/v2/assets",
    "https://api.alpaca.markets/v2/assets",
)
FINAL_REL = Path("data/alpaca/2026/alpaca_us_equities_2026_ytd.parquet")
MANIFEST_REL = Path("data/manifests/alpaca_2026_ytd.json")
SYMBOLS_REL = Path("data/alpaca/2026/alpaca_symbols_2026.csv")
REJECTED_SYMBOLS_REL = Path("data/alpaca/2026/alpaca_rejected_symbols_2026.csv")
LOG_REL = Path("logs/alpaca_2026_download.log")
PREFILTERED_SYMBOLS_REL = Path("data/alpaca/2026/alpaca_prefiltered_symbols_2026.csv")
STRATEGY_TZ = ZoneInfo("America/New_York")
SIP_BUFFER_MINUTES = 20
LISTED_EXCHANGES = {"AMEX", "ARCA", "BATS", "NYSE", "NASDAQ", "NYSEARCA"}
TICKER_LIKE_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")

KEY_NAMES = (
    "APCA_API_KEY_ID",
    "APCA-API-KEY-ID",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_KEY",
    "API_KEY",
)
SECRET_NAMES = (
    "APCA_API_SECRET_KEY",
    "APCA-API-SECRET-KEY",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_SECRET_KEY",
    "SECRET_KEY",
    "API_SECRET",
)


class DownloadError(RuntimeError):
    pass


class AlpacaHTTPError(DownloadError):
    def __init__(self, status_code: int, body: str):
        self.status_code = int(status_code)
        self.body = body
        super().__init__(f"Alpaca HTTP {self.status_code}. Response: {body[:300]}")


@dataclass(frozen=True)
class Credentials:
    key_id: str
    secret_key: str
    source: str


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _norm_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _lookup(mapping: dict[str, Any], names: Iterable[str]) -> str | None:
    normalized = {_norm_key(str(k)): v for k, v in mapping.items()}
    for name in names:
        v = normalized.get(_norm_key(name))
        if v is not None and str(v).strip():
            return str(v).strip().strip('"').strip("'")
    return None


def _parse_kv_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {}
        # Flatten one level for common {alpaca:{...}} files.
        flat = dict(obj)
        for v in obj.values():
            if isinstance(v, dict):
                flat.update(v)
        return flat
    if path.suffix.lower() in {".yaml", ".yml"}:
        # Avoid making PyYAML a hard dependency for credential discovery.
        return _parse_kv_text(text)
    return _parse_kv_text(text)


def _candidate_secret_files(root: Path) -> list[Path]:
    preferred = [
        root / ".env",
        root / "alpaca.env",
        root / "alpaca_keys.env",
        root / "alpaca_keys.json",
        root / "alpaca_credentials.json",
        root / "alpaca_credentials.txt",
        root / "api_keys.json",
        root / "api_keys.txt",
        root / "credentials.json",
        root / "credentials.txt",
        root / "secrets.json",
        root / "secrets.txt",
        root / "config" / ".env",
        root / "config" / "alpaca.env",
        root / "config" / "alpaca_keys.json",
        root / "secrets" / "alpaca.env",
        root / "secrets" / "alpaca_keys.json",
    ]
    result: list[Path] = [p for p in preferred if p.is_file()]
    seen = {p.resolve() for p in result}
    for base in (root, root / "config", root / "secrets"):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if not p.is_file() or p.resolve() in seen:
                continue
            name = p.name.lower()
            if p.stat().st_size > 128_000:
                continue
            if p.suffix.lower() not in {".txt", ".env", ".json", ".yaml", ".yml", ""}:
                continue
            if (
                any(token in name for token in ("alpaca", "credential", "secret", "api_key", "apikey"))
                or ("api" in name and "key" in name)
            ):
                result.append(p)
                seen.add(p.resolve())
    return result


def load_credentials(root: Path, explicit: str | None = None) -> Credentials:
    env = dict(os.environ)
    key = _lookup(env, KEY_NAMES)
    secret = _lookup(env, SECRET_NAMES)
    if key and secret:
        return Credentials(key, secret, "environment")

    files: list[Path] = []
    if explicit:
        p = (root / explicit).resolve() if not Path(explicit).is_absolute() else Path(explicit)
        if not p.is_file():
            raise DownloadError(f"Credential file not found: {p}")
        files.append(p)
    files.extend(p for p in _candidate_secret_files(root) if p not in files)

    partial_key: tuple[str, str] | None = None
    partial_secret: tuple[str, str] | None = None
    for path in files:
        try:
            mapping = _load_mapping(path)
        except Exception:
            continue
        k = _lookup(mapping, KEY_NAMES)
        s = _lookup(mapping, SECRET_NAMES)
        if k and s:
            return Credentials(k, s, str(path))
        if k:
            partial_key = (k, str(path))
        if s:
            partial_secret = (s, str(path))

    # Support two separate one-value files with obvious names.
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="ignore").strip()
        except Exception:
            continue
        if not raw or "\n" in raw or "=" in raw or ":" in raw:
            continue
        name = path.name.lower()
        if "secret" in name and len(raw) >= 16:
            partial_secret = (raw, str(path))
        elif ("api" in name or "key" in name) and len(raw) >= 12:
            partial_key = (raw, str(path))
    if partial_key and partial_secret:
        return Credentials(partial_key[0], partial_secret[0], f"{partial_key[1]} + {partial_secret[1]}")

    raise DownloadError(
        "Alpaca credentials were not found. Put the existing key/secret in environment variables "
        "APCA_API_KEY_ID and APCA_API_SECRET_KEY, or in a root/config/secrets credential file. "
        "Credential values are never printed."
    )


def _headers(creds: Credentials) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": creds.key_id,
        "APCA-API-SECRET-KEY": creds.secret_key,
        "Accept": "application/json",
        "User-Agent": "CrossSectionalMomentumPullback/1.19",
    }


class StdlibResponse:
    def __init__(self, status_code: int, headers: Any, body: bytes):
        self.status_code = int(status_code)
        self.headers = headers
        self._body = body
        self.text = body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if not 200 <= self.status_code <= 299:
            raise AlpacaHTTPError(self.status_code, self.text)


class StdlibSession:
    """Minimal requests-like GET client implemented only with Python's standard library."""

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> StdlibResponse:
        if params:
            query = urlencode({k: str(v) for k, v in params.items() if v is not None})
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{query}"
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                return StdlibResponse(response.getcode(), response.headers, response.read())
        except HTTPError as exc:
            # HTTPError is also a readable response. Return it so request_json can
            # apply the same 401/403/429/5xx policy used by Patch 1.16.
            body = exc.read() if exc.fp is not None else b""
            return StdlibResponse(exc.code, exc.headers, body)


def request_json(
    session: StdlibSession,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    attempts: int = 7,
    timeout: int = 60,
) -> Any:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code <= 599:
                retry_after = r.headers.get("Retry-After") if r.headers else None
                try:
                    wait = float(retry_after) if retry_after else delay
                except (TypeError, ValueError):
                    wait = delay
                print(f"HTTP {r.status_code}; retrying in {wait:.1f}s (attempt {attempt}/{attempts})...")
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            if r.status_code in {401, 403}:
                msg = "authentication failed" if r.status_code == 401 else "access forbidden / insufficient market-data entitlement"
                raise DownloadError(f"Alpaca HTTP {r.status_code}: {msg}. Response: {r.text[:300]}")
            if r.status_code == 400:
                raise AlpacaHTTPError(r.status_code, r.text)
            r.raise_for_status()
            return r.json()
        except DownloadError:
            raise
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            if attempt == attempts:
                break
            print(f"Request error; retrying in {delay:.1f}s (attempt {attempt}/{attempts}): {exc}")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise DownloadError(f"Alpaca request failed after {attempts} attempts: {last}")


def fetch_assets(session: StdlibSession, headers: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    errors = []
    for url in ASSET_URLS:
        try:
            obj = request_json(
                session,
                url,
                headers=headers,
                params={"asset_class": "us_equity"},
                attempts=2,
            )
            if not isinstance(obj, list):
                raise DownloadError(f"Unexpected assets payload from {url}")
            assets = [x for x in obj if isinstance(x, dict) and x.get("symbol")]
            if not assets:
                raise DownloadError(f"Assets endpoint returned no US-equity symbols: {url}")
            return assets, url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise DownloadError("Could not retrieve Alpaca assets. " + " | ".join(errors))


def _ticker_like_symbol(symbol: str) -> bool:
    """Conservative prefilter for actual exchange ticker syntax.

    Alpaca's all-status asset master can contain CUSIP/corporate-action-style identifiers that
    are accepted by the asset endpoint but rejected by the stock-bars endpoint as symbols.
    Real US exchange tickers begin with a letter; dots/hyphens and later digits are retained.
    """
    return bool(TICKER_LIKE_RE.fullmatch(symbol.strip().upper()))


def select_assets(
    assets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    # Keep active + inactive *listed* US-equity records so a name that traded earlier in 2026
    # and later became inactive does not disappear from a subsequent monthly YTD refresh.
    # However, the all-status Alpaca asset master also contains legacy security identifiers
    # (CUSIP/corporate-action style strings) that are not valid stock-bar symbols.  Prefilter
    # those deterministically before making thousands of doomed API requests; all exclusions
    # are written to an audit CSV.  Exact API invalid-symbol quarantine remains as a fallback.
    grouped: dict[str, list[dict[str, Any]]] = {}
    excluded_otc = 0
    excluded_non_equity = 0
    excluded_nonlisted_exchange = 0
    excluded_non_ticker_like = 0
    prefiltered: list[dict[str, Any]] = []
    for a in assets:
        symbol = str(a.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        asset_class = str(a.get("class", a.get("asset_class", "us_equity"))).lower()
        if asset_class not in {"us_equity", "usequity"}:
            excluded_non_equity += 1
            continue
        exchange = str(a.get("exchange", "")).strip().upper()
        if exchange == "OTC":
            excluded_otc += 1
            continue
        if exchange not in LISTED_EXCHANGES:
            excluded_nonlisted_exchange += 1
            prefiltered.append({
                "symbol": symbol,
                "reason": "nonlisted_or_unknown_exchange",
                "exchange": exchange,
                "asset_status": a.get("status"),
                "asset_id": a.get("id"),
                "asset_name": a.get("name"),
            })
            continue
        if not _ticker_like_symbol(symbol):
            excluded_non_ticker_like += 1
            prefiltered.append({
                "symbol": symbol,
                "reason": "non_ticker_like_asset_master_identifier",
                "exchange": exchange,
                "asset_status": a.get("status"),
                "asset_id": a.get("id"),
                "asset_name": a.get("name"),
            })
            continue
        grouped.setdefault(symbol, []).append(a)

    selected: list[dict[str, Any]] = []
    for symbol, records in grouped.items():
        records = sorted(
            records,
            key=lambda x: (str(x.get("status", "")).lower() == "active", x.get("tradable") is True),
            reverse=True,
        )
        chosen = dict(records[0])
        chosen["symbol"] = symbol
        chosen["asset_metadata_records"] = len(records)
        selected.append(chosen)
    selected.sort(key=lambda x: str(x.get("symbol", "")))
    if not selected:
        raise DownloadError("No listed ticker-like US-equity symbols remained after asset-master filtering.")
    return selected, {
        "asset_master_records": len(assets),
        "excluded_otc_records": excluded_otc,
        "excluded_non_equity_records": excluded_non_equity,
        "excluded_nonlisted_exchange_records": excluded_nonlisted_exchange,
        "excluded_non_ticker_like_records": excluded_non_ticker_like,
        "selected_unique_symbols": len(selected),
    }, prefiltered


def _recent_sip_entitlement_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "subscription does not permit querying recent sip data" in text


def _previous_weekday(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def _default_safe_end(now: datetime | None = None, buffer_minutes: int = SIP_BUFFER_MINUTES) -> date:
    """Return the latest completed weekday daily-bar date with an SIP entitlement buffer.

    Today's weekday is included only after 16:00 ET plus the buffer. Otherwise the prior
    weekday is used. This deliberately avoids current/partial daily bars without requiring a
    market-calendar dependency; exchange holidays simply produce no bar for that date.
    """
    local = (now or datetime.now(STRATEGY_TZ)).astimezone(STRATEGY_TZ)
    cutoff = dt_time(16, 0)
    session_safe = datetime.combine(local.date(), cutoff, tzinfo=STRATEGY_TZ) + timedelta(minutes=buffer_minutes)
    if local.weekday() < 5 and local >= session_safe:
        return local.date()
    return _previous_weekday(local.date())


def resolve_end_date(value: str | None, *, feed: str, now: datetime | None = None) -> tuple[str, str]:
    """Resolve a reproducible completed-daily-bar end date.

    Default monthly refreshes deliberately avoid a partial/recent daily bar.  An explicit
    --end is honored, but SIP users receive a targeted entitlement error later if it is too
    recent for their subscription.
    """
    if value:
        d = date.fromisoformat(value)
        source = "explicit"
    else:
        d = _default_safe_end(now=now)
        source = f"auto_completed_session_buffer_{SIP_BUFFER_MINUTES}m"
    if d.year != YEAR:
        if d.year < YEAR:
            raise DownloadError(f"Current/requested date {d} is before {YEAR}; nothing to download.")
        d = date(YEAR, 12, 31)
    return d.isoformat(), source


def probe_historical_feed(
    session: StdlibSession,
    headers: dict[str, str],
    *,
    start: str,
    end: str,
    feed_request: str,
    adjustment: str,
) -> str:
    """Resolve `auto` to SIP and verify entitlement before the expensive universe download.

    Historical SIP is preferred because it is consolidated across US exchanges.  There is no
    silent IEX fallback: IEX is a single-exchange dataset and would change the research data
    contract.  Users can explicitly request --feed iex if they intentionally want that source.
    """
    actual = "sip" if feed_request == "auto" else feed_request
    params = {
        "symbols": "SPY",
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "limit": 1,
        "adjustment": adjustment,
        "sort": "asc",
        "asof": "-",
        "feed": actual,
    }
    try:
        obj = request_json(session, DATA_URL, headers=headers, params=params, attempts=2)
    except DownloadError as exc:
        if actual == "sip" and _recent_sip_entitlement_error(exc):
            raise DownloadError(
                f"Historical SIP entitlement probe failed for end={end}: Alpaca says the query is too recent. "
                f"The monthly default keeps a {SIP_BUFFER_MINUTES}-minute post-close buffer; rerun later or "
                "use an older --end date. To intentionally use the single-exchange IEX dataset instead, "
                "rerun with --feed iex. Patch 1.19 never silently substitutes IEX for SIP."
            ) from exc
        raise
    if not isinstance(obj, dict):
        raise DownloadError("Unexpected Alpaca historical-bars entitlement probe response.")
    return actual

def _invalid_symbol_from_http_error(exc: AlpacaHTTPError) -> str | None:
    if exc.status_code != 400:
        return None
    message = exc.body
    try:
        obj = json.loads(exc.body)
        if isinstance(obj, dict) and obj.get("message"):
            message = str(obj["message"])
    except Exception:
        pass
    match = re.search(r"invalid\s+symbol\s*:\s*([^\s,;\"'}]+)", message, flags=re.IGNORECASE)
    return match.group(1).strip().upper() if match else None


def chunks(seq: list[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def bars_to_frame(payload: dict[str, Any], asset_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    bars = payload.get("bars") or {}
    rows: list[dict[str, Any]] = []
    if isinstance(bars, dict):
        items = bars.items()
    else:
        raise DownloadError("Unexpected Alpaca bars payload: 'bars' was not a symbol mapping.")
    for symbol, values in items:
        meta = asset_map.get(str(symbol).upper(), {})
        for b in values or []:
            ts = b.get("t")
            if not ts:
                continue
            rows.append(
                {
                    "date": str(ts)[:10],
                    "timestamp": ts,
                    "symbol": str(symbol).upper(),
                    "open": b.get("o"),
                    "high": b.get("h"),
                    "low": b.get("l"),
                    "close": b.get("c"),
                    "volume": b.get("v"),
                    "trade_count": b.get("n"),
                    "vwap": b.get("vw"),
                    "asset_id": meta.get("id"),
                    "asset_name": meta.get("name"),
                    "exchange": meta.get("exchange"),
                    "asset_class": meta.get("class", meta.get("asset_class")),
                    "asset_status": meta.get("status"),
                    "asset_metadata_records": meta.get("asset_metadata_records", 1),
                    "tradable": meta.get("tradable"),
                    "fractionable": meta.get("fractionable"),
                    "shortable": meta.get("shortable"),
                    "marginable": meta.get("marginable"),
                    "easy_to_borrow": meta.get("easy_to_borrow"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=[
            "date", "timestamp", "symbol", "open", "high", "low", "close", "volume",
            "trade_count", "vwap", "asset_id", "asset_name", "exchange", "asset_class",
            "asset_status", "asset_metadata_records", "tradable", "fractionable", "shortable",
            "marginable", "easy_to_borrow"
        ])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def download_batch(
    session: StdlibSession,
    headers: dict[str, str],
    symbols: list[str],
    asset_map: dict[str, dict[str, Any]],
    start: str,
    end: str,
    feed: str,
    adjustment: str,
    limit: int,
) -> pd.DataFrame:
    token: str | None = None
    frames = []
    pages = 0
    while True:
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "limit": limit,
            "adjustment": adjustment,
            "sort": "asc",
            "asof": "-",
        }
        if feed != "auto":
            params["feed"] = feed
        if token:
            params["page_token"] = token
        payload = request_json(session, DATA_URL, headers=headers, params=params)
        if not isinstance(payload, dict):
            raise DownloadError("Unexpected Alpaca historical-bars response type.")
        frame = bars_to_frame(payload, asset_map)
        if not frame.empty:
            frames.append(frame)
        token = payload.get("next_page_token")
        pages += 1
        if not token:
            break
        if pages > 5000:
            raise DownloadError("Pagination safety limit exceeded for one symbol batch.")
    return pd.concat(frames, ignore_index=True) if frames else bars_to_frame({"bars": {}}, asset_map)



def download_batch_with_quarantine(
    session: StdlibSession,
    headers: dict[str, str],
    symbols: list[str],
    asset_map: dict[str, dict[str, Any]],
    start: str,
    end: str,
    feed: str,
    adjustment: str,
    limit: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Retry a batch after quarantining only symbols Alpaca explicitly rejects as invalid.

    Any other HTTP 400 remains fatal. The batch restarts after a quarantine event, so partial pages
    from a failed attempt are never mixed into the successful result.
    """
    remaining = list(dict.fromkeys(str(x).upper() for x in symbols))
    rejected: list[dict[str, Any]] = []
    while remaining:
        try:
            frame = download_batch(
                session, headers, remaining, asset_map, start, end, feed, adjustment, limit
            )
            return frame, rejected
        except AlpacaHTTPError as exc:
            bad = _invalid_symbol_from_http_error(exc)
            if not bad:
                raise
            if bad not in remaining:
                raise DownloadError(
                    f"Alpaca rejected symbol {bad}, but it was not present in the current request batch. "
                    "Refusing to guess which asset to remove."
                ) from exc
            meta = asset_map.get(bad, {})
            rejected.append({
                "symbol": bad,
                "http_status": exc.status_code,
                "reason": "alpaca_historical_bars_invalid_symbol",
                "exchange": meta.get("exchange"),
                "asset_status": meta.get("status"),
                "asset_id": meta.get("id"),
                "asset_name": meta.get("name"),
            })
            remaining = [x for x in remaining if x != bad]
            print(
                f"Quarantined Alpaca-invalid symbol {bad}; retrying batch with "
                f"{len(remaining):,} remaining symbol(s)."
            )
    return bars_to_frame({"bars": {}}, asset_map), rejected

def validate_frame(df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    if df.empty:
        raise DownloadError("Download returned zero daily bars; existing Parquet was not replaced.")
    required = {"date", "timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise DownloadError(f"Downloaded data missing required columns: {sorted(missing)}")
    if df["date"].isna().any() or df["symbol"].isna().any():
        raise DownloadError("Downloaded data contains null date/symbol keys.")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise DownloadError("Downloaded data contains unparseable dates.")
    if (dates.dt.year != YEAR).any():
        bad = sorted(set(dates[dates.dt.year != YEAR].dt.strftime("%Y-%m-%d").tolist()))[:5]
        raise DownloadError(f"Output contains dates outside {YEAR}: {bad}")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if dates.min() < start_ts or dates.max() > end_ts:
        raise DownloadError(f"Output date range {dates.min().date()}..{dates.max().date()} exceeds requested {start}..{end}")
    dup = int(df.duplicated(["date", "symbol"]).sum())
    if dup:
        raise DownloadError(f"Downloaded data contains {dup:,} duplicate date/symbol rows; existing Parquet was not replaced.")
    for col in ["open", "high", "low", "close"]:
        if df[col].isna().any():
            raise DownloadError(f"Downloaded data contains null {col} values.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DownloadError("Downloaded data contains non-positive OHLC prices.")
    if (df["volume"].fillna(0) < 0).any():
        raise DownloadError("Downloaded data contains negative volume.")
    return {
        "rows": int(len(df)),
        "symbols_with_bars": int(df["symbol"].nunique()),
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "duplicate_date_symbol_rows": dup,
        "null_key_rows": int(df[["date", "symbol"]].isna().any(axis=1).sum()),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sql_quote_path(path: Path | str) -> str:
    return str(path).replace("'", "''").replace("\\", "/")


def _duckdb_module():
    try:
        import duckdb
        return duckdb
    except ImportError as exc:
        raise DownloadError("DuckDB is required for Parquet staging/validation. Run 00_setup_env.cmd first.") from exc


def write_df_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    duckdb = _duckdb_module()
    con = duckdb.connect()
    try:
        con.register("frame_to_write", df)
        qpath = _sql_quote_path(target)
        con.execute(f"COPY (SELECT * FROM frame_to_write) TO '{qpath}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()


def combine_validate_atomic(parts_dir: Path, target: Path, start: str, end: str) -> tuple[dict[str, Any], str]:
    duckdb = _duckdb_module()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp-{uuid.uuid4().hex}")
    glob = _sql_quote_path(parts_dir / "*.parquet")
    con = duckdb.connect()
    try:
        con.execute(f"CREATE TEMP VIEW bars AS SELECT * FROM read_parquet('{glob}', union_by_name=true)")
        row = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT symbol) AS symbols_with_bars,
                MIN(date) AS first_date,
                MAX(date) AS last_date,
                COUNT(*) FILTER (WHERE date IS NULL OR symbol IS NULL) AS null_key_rows,
                COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL) AS null_ohlc_rows,
                COUNT(*) FILTER (WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0) AS nonpositive_ohlc_rows,
                COUNT(*) FILTER (WHERE volume < 0) AS negative_volume_rows,
                COUNT(*) FILTER (WHERE EXTRACT(year FROM date) <> ?) AS wrong_year_rows,
                COUNT(*) FILTER (WHERE date < ?::DATE OR date > ?::DATE) AS outside_window_rows
            FROM bars
            """,
            [YEAR, start, end],
        ).fetchone()
        keys = [d[0] for d in con.description]
        stats = dict(zip(keys, row))
        if int(stats["rows"]) <= 0:
            raise DownloadError("Download returned zero daily bars; existing Parquet was not replaced.")
        for key in ("null_key_rows", "null_ohlc_rows", "nonpositive_ohlc_rows", "negative_volume_rows", "wrong_year_rows", "outside_window_rows"):
            if int(stats[key]) != 0:
                raise DownloadError(f"Validation failed: {key}={int(stats[key]):,}; existing Parquet was not replaced.")
        dup_groups = int(con.execute(
            "SELECT COUNT(*) FROM (SELECT date, symbol FROM bars GROUP BY date, symbol HAVING COUNT(*) > 1)"
        ).fetchone()[0])
        if dup_groups:
            raise DownloadError(f"Downloaded data contains {dup_groups:,} duplicate date/symbol groups; existing Parquet was not replaced.")

        qtmp = _sql_quote_path(tmp)
        con.execute(f"COPY (SELECT * FROM bars ORDER BY date, symbol) TO '{qtmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()

    try:
        verify = duckdb.connect()
        try:
            qtmp = _sql_quote_path(tmp)
            readback_rows = int(verify.execute(f"SELECT COUNT(*) FROM read_parquet('{qtmp}')").fetchone()[0])
        finally:
            verify.close()
        if readback_rows != int(stats["rows"]):
            raise DownloadError(f"Parquet readback row mismatch: wrote {int(stats['rows']):,}, read {readback_rows:,}")
        os.replace(tmp, target)
        digest = sha256_file(target)
        out = {
            "rows": int(stats["rows"]),
            "symbols_with_bars": int(stats["symbols_with_bars"]),
            "first_date": pd.Timestamp(stats["first_date"]).strftime("%Y-%m-%d"),
            "last_date": pd.Timestamp(stats["last_date"]).strftime("%Y-%m-%d"),
            "duplicate_date_symbol_groups": 0,
            "null_key_rows": 0,
        }
        return out, digest
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_symbols_csv(assets: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp-{uuid.uuid4().hex}")
    fields = ["symbol", "id", "name", "exchange", "class", "status", "tradable", "fractionable", "marginable", "shortable", "easy_to_borrow", "asset_metadata_records"]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for a in assets:
            w.writerow(a)
    os.replace(tmp, target)



def write_rejected_symbols_csv(rows: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp-{uuid.uuid4().hex}")
    fields = ["symbol", "http_status", "reason", "exchange", "asset_status", "asset_id", "asset_name"]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, target)

def write_prefiltered_symbols_csv(rows: list[dict[str, Any]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp-{uuid.uuid4().hex}")
    fields = ["symbol", "reason", "exchange", "asset_status", "asset_id", "asset_name"]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Alpaca listed US-equity daily bars for 2026 YTD and atomically replace one Parquet file.")
    parser.add_argument("--start", default=f"{YEAR}-01-01")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD; default is latest completed US daily session with a 20-minute SIP-safety buffer")
    parser.add_argument("--feed", choices=["auto", "sip", "iex"], default="auto", help="auto explicitly prefers consolidated SIP after a completed-session safety cutoff; IEX is never substituted silently")
    parser.add_argument("--adjustment", choices=["raw", "split", "dividend", "spin-off", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--max-invalid-symbols", type=int, default=2000, help="Fail closed if Alpaca rejects more listed identifiers than this")
    parser.add_argument("--max-invalid-fraction", type=float, default=0.10, help="Fail closed if rejected listed identifiers exceed this fraction of selected symbols")
    parser.add_argument("--credentials", default=None, help="Optional credential file path. Values are never logged.")
    parser.add_argument("--output", default=str(FINAL_REL).replace("\\", "/"))
    args = parser.parse_args(argv)

    root = _root_from_file()
    log_path = root / LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, log)
        sys.stderr = Tee(old_err, log)
        try:
            print("=" * 78)
            print("ALPACA 2026 YTD DAILY LISTED US-EQUITY DOWNLOADER")
            print("- Full-year-to-date refresh; output is replaced only after validation")
            print("- Existing Parquet remains untouched if authentication/download/validation fails")
            print("- Credential values are never logged")
            print("=" * 78)

            end, end_policy = resolve_end_date(args.end, feed=args.feed)
            start = date.fromisoformat(args.start)
            if start.year != YEAR:
                raise DownloadError(f"Start date must be in {YEAR}: {args.start}")
            if date.fromisoformat(end) < start:
                raise DownloadError(f"End date {end} is before start date {start}")
            if not (1 <= args.batch_size <= 200):
                raise DownloadError("--batch-size must be between 1 and 200")
            if not (1 <= args.limit <= 10000):
                raise DownloadError("--limit must be between 1 and 10000")
            if args.max_invalid_symbols < 0:
                raise DownloadError("--max-invalid-symbols must be >= 0")
            if not (0 <= args.max_invalid_fraction <= 1):
                raise DownloadError("--max-invalid-fraction must be between 0 and 1")

            creds = load_credentials(root, args.credentials)
            print(f"Credentials: found ({creds.source}); values redacted")
            print(f"Requested range: {start} -> {end}")
            print(f"Feed request: {args.feed}; adjustment: {args.adjustment}; timeframe: 1Day")
            print(f"Daily-bar end policy: {end_policy}")

            session = StdlibSession()
            headers = _headers(creds)
            actual_feed = probe_historical_feed(
                session, headers, start=str(start), end=end, feed_request=args.feed, adjustment=args.adjustment
            )
            print(f"Historical-bars feed resolved: {actual_feed} (explicit; no silent IEX fallback)")
            print("Historical symbol mapping: disabled (asof=-) so raw rows retain the queried ticker identity")

            assets_raw, asset_endpoint = fetch_assets(session, headers)
            assets, asset_filter_stats, prefiltered_symbols = select_assets(assets_raw)
            symbols = [str(a["symbol"]).upper() for a in assets]
            asset_map = {str(a["symbol"]).upper(): a for a in assets}
            print(f"Assets endpoint: {asset_endpoint}")
            print(
                f"Listed US-equity symbols (active + inactive, OTC excluded, deduplicated): {len(symbols):,}"
            )
            print(f"OTC asset-master records excluded: {asset_filter_stats['excluded_otc_records']:,}")
            print(f"Non-ticker-like / nonlisted asset-master records prefiltered: {len(prefiltered_symbols):,}")

            stage_root = root / "data" / "alpaca" / ".staging" / uuid.uuid4().hex
            parts_dir = stage_root / "parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            part_paths: list[Path] = []
            try:
                total_batches = (len(symbols) + args.batch_size - 1) // args.batch_size
                rejected_symbols: list[dict[str, Any]] = []
                for i, batch in enumerate(chunks(symbols, args.batch_size), start=1):
                    frame, rejected = download_batch_with_quarantine(
                        session,
                        headers,
                        batch,
                        asset_map,
                        str(start),
                        end,
                        actual_feed,
                        args.adjustment,
                        args.limit,
                    )
                    rejected_symbols.extend(rejected)
                    rejected_count = len(rejected_symbols)
                    if rejected_count > args.max_invalid_symbols:
                        raise DownloadError(
                            f"Alpaca rejected {rejected_count:,} listed identifiers as invalid, exceeding "
                            f"--max-invalid-symbols={args.max_invalid_symbols:,}. Refusing to silently degrade the universe."
                        )
                    if symbols and rejected_count / len(symbols) > args.max_invalid_fraction:
                        raise DownloadError(
                            f"Alpaca rejected {rejected_count:,}/{len(symbols):,} listed identifiers "
                            f"({rejected_count / len(symbols):.2%}), exceeding --max-invalid-fraction="
                            f"{args.max_invalid_fraction:.2%}."
                        )
                    if not frame.empty:
                        p = parts_dir / f"part_{i:05d}.parquet"
                        write_df_parquet(frame, p)
                        part_paths.append(p)
                    if i == 1 or i % 10 == 0 or i == total_batches:
                        print(
                            f"Batch {i:,}/{total_batches:,}: symbols={len(batch):,}, rows={len(frame):,}, "
                            f"quarantined_total={len(rejected_symbols):,}"
                        )

                if not part_paths:
                    raise DownloadError("No Alpaca bars were downloaded; existing Parquet was not replaced.")
                print(f"Combining and validating {len(part_paths):,} staged Parquet parts with DuckDB...")
                output = Path(args.output)
                if not output.is_absolute():
                    output = root / output
                stats, digest = combine_validate_atomic(parts_dir, output, str(start), end)
                write_symbols_csv(assets, root / SYMBOLS_REL)
                write_rejected_symbols_csv(rejected_symbols, root / REJECTED_SYMBOLS_REL)
                write_prefiltered_symbols_csv(prefiltered_symbols, root / PREFILTERED_SYMBOLS_REL)

                manifest = {
                    "schema_version": 3,
                    "source": "Alpaca Market Data API v2 historical stock bars",
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "year": YEAR,
                    "requested_start": str(start),
                    "requested_end": end,
                    "timeframe": "1Day",
                    "feed_request": args.feed,
                    "feed_used": actual_feed,
                    "feed_policy": "auto explicitly resolves to SIP after completed-session cutoff; IEX requires explicit --feed iex",
                    "daily_bar_end_policy": end_policy,
                    "sip_safety_buffer_minutes": SIP_BUFFER_MINUTES,
                    "historical_symbol_mapping": "disabled with asof=-; preserves queried ticker identity for downstream CRSP mapping",
                    "adjustment": args.adjustment,
                    "asset_endpoint": asset_endpoint,
                    **asset_filter_stats,
                    "assets_requested": len(symbols),
                    "symbols_quarantined_invalid": len(rejected_symbols),
                    "symbols_queried_after_quarantine": len(symbols) - len(rejected_symbols),
                    "invalid_symbol_policy": "quarantine only exact HTTP 400 invalid-symbol responses; all other HTTP 400 errors remain fatal",
                    "rejected_symbol_snapshot": str(root / REJECTED_SYMBOLS_REL),
                    "prefiltered_symbol_snapshot": str(root / PREFILTERED_SYMBOLS_REL),
                    "asset_master_prefilter_policy": "listed exchange + ticker-like syntax; excludes CUSIP/corporate-action-style identifiers before bars API while retaining inactive ticker-like listed names",
                    "otc_policy": "excluded by default because standard Alpaca stock feeds do not include OTC without separate entitlement",
                    **stats,
                    "output": str(output),
                    "output_sha256": digest,
                    "symbol_snapshot": str(root / SYMBOLS_REL),
                    "credential_source": creds.source,
                    "credential_values_logged": False,
                    "replacement_policy": "atomic full-YTD replace after complete validation",
                    "note": "Alpaca symbol data does not provide CRSP PERMNO/security-type history by itself; downstream CRSP/Alpaca identity mapping remains a separate integration step.",
                }
                manifest_path = root / MANIFEST_REL
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
                tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                os.replace(tmp_manifest, manifest_path)

                print("=" * 78)
                print("ALPACA 2026 YTD DOWNLOAD PASS")
                print(f"Rows:              {stats['rows']:,}")
                print(f"Symbols with bars: {stats['symbols_with_bars']:,}")
                print(f"Feed used:          {actual_feed}")
                print(f"Prefiltered legacy/non-ticker identifiers: {len(prefiltered_symbols):,}")
                print(f"Quarantined invalid listed symbols: {len(rejected_symbols):,}")
                print(f"Date range:        {stats['first_date']} -> {stats['last_date']}")
                print(f"Parquet:           {output}")
                print(f"SHA-256:           {digest}")
                print(f"Manifest:          {manifest_path}")
                print("=" * 78)
                return 0
            finally:
                shutil.rmtree(stage_root, ignore_errors=True)
        except Exception as exc:
            print("=" * 78)
            print("ALPACA 2026 YTD DOWNLOAD FAILED")
            print(str(exc))
            print("Existing production Parquet was not intentionally replaced unless final validation had already passed.")
            print(f"Full log: {log_path}")
            print("=" * 78)
            return 1
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


if __name__ == "__main__":
    raise SystemExit(main())
