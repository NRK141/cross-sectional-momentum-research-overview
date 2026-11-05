from pathlib import Path

import pandas as pd
import pytest

from src.live.download_alpaca_2026_ytd import (
    Credentials,
    _parse_kv_text,
    bars_to_frame,
    validate_frame,
)


def test_parse_common_env_keys():
    m = _parse_kv_text("APCA_API_KEY_ID=abc123\nAPCA_API_SECRET_KEY=secret456\n")
    assert m["APCA_API_KEY_ID"] == "abc123"
    assert m["APCA_API_SECRET_KEY"] == "secret456"


def test_bars_to_frame_and_validation():
    payload = {
        "bars": {
            "AAPL": [
                {"t": "2026-01-02T05:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000, "n": 25, "vw": 100.5},
                {"t": "2026-01-05T05:00:00Z", "o": 101, "h": 103, "l": 100, "c": 102, "v": 1100, "n": 27, "vw": 101.5},
            ]
        }
    }
    assets = {"AAPL": {"id": "id1", "exchange": "NASDAQ", "class": "us_equity", "tradable": True}}
    df = bars_to_frame(payload, assets)
    stats = validate_frame(df, "2026-01-01", "2026-12-31")
    assert stats["rows"] == 2
    assert stats["symbols_with_bars"] == 1
    assert stats["first_date"] == "2026-01-02"


def test_duplicate_key_fails_closed():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "timestamp": pd.to_datetime(["2026-01-02T05:00:00Z", "2026-01-02T05:00:00Z"], utc=True),
        "symbol": ["AAPL", "AAPL"],
        "open": [100.0, 100.0], "high": [101.0, 101.0], "low": [99.0, 99.0], "close": [100.5, 100.5], "volume": [1000, 1000],
    })
    with pytest.raises(Exception):
        validate_frame(df, "2026-01-01", "2026-12-31")


def test_downloader_has_no_requests_dependency():
    import src.live.download_alpaca_2026_ytd as mod
    assert "requests" not in mod.__dict__
    assert hasattr(mod, "StdlibSession")


def test_stdlib_response_json_and_status():
    from src.live.download_alpaca_2026_ytd import StdlibResponse, DownloadError
    ok = StdlibResponse(200, {}, b'{"ok": true}')
    assert ok.json() == {"ok": True}
    ok.raise_for_status()
    bad = StdlibResponse(418, {}, b'{"message":"teapot"}')
    with pytest.raises(DownloadError):
        bad.raise_for_status()


def test_select_assets_excludes_otc_but_keeps_inactive_listed():
    from src.live.download_alpaca_2026_ytd import select_assets
    assets = [
        {"symbol": "GOOD", "class": "us_equity", "exchange": "NASDAQ", "status": "active"},
        {"symbol": "OLD", "class": "us_equity", "exchange": "NYSE", "status": "inactive"},
        {"symbol": "OTCX", "class": "us_equity", "exchange": "OTC", "status": "active"},
    ]
    selected, stats, prefiltered = select_assets(assets)
    assert [x["symbol"] for x in selected] == ["GOOD", "OLD"]
    assert stats["excluded_otc_records"] == 1
    assert stats["selected_unique_symbols"] == 2
    assert prefiltered == []


def test_invalid_symbol_batch_is_quarantined_and_retried():
    from src.live.download_alpaca_2026_ytd import (
        StdlibResponse,
        download_batch_with_quarantine,
    )

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, *, headers, params=None, timeout=60):
            symbols = params["symbols"]
            self.calls.append(symbols)
            if "BADID" in symbols.split(","):
                return StdlibResponse(400, {}, b'{"message":"invalid symbol: BADID"}')
            return StdlibResponse(
                200,
                {},
                b'{"bars":{"GOOD":[{"t":"2026-01-02T05:00:00Z","o":10,"h":11,"l":9,"c":10.5,"v":100,"n":5,"vw":10.2}]},"next_page_token":null}',
            )

    session = FakeSession()
    asset_map = {
        "GOOD": {"id": "g", "exchange": "NASDAQ", "status": "active"},
        "BADID": {"id": "b", "exchange": "NYSE", "status": "inactive"},
    }
    df, rejected = download_batch_with_quarantine(
        session,
        {"x": "y"},
        ["GOOD", "BADID"],
        asset_map,
        "2026-01-01",
        "2026-01-31",
        "auto",
        "all",
        10000,
    )
    assert list(df["symbol"].unique()) == ["GOOD"]
    assert [x["symbol"] for x in rejected] == ["BADID"]
    assert session.calls == ["GOOD,BADID", "GOOD"]


def test_non_invalid_400_remains_fatal():
    from src.live.download_alpaca_2026_ytd import (
        AlpacaHTTPError,
        StdlibResponse,
        download_batch_with_quarantine,
    )

    class FakeSession:
        def get(self, url, *, headers, params=None, timeout=60):
            return StdlibResponse(400, {}, b'{"message":"invalid adjustment"}')

    with pytest.raises(AlpacaHTTPError):
        download_batch_with_quarantine(
            FakeSession(), {"x": "y"}, ["GOOD"], {"GOOD": {}},
            "2026-01-01", "2026-01-31", "auto", "all", 10000,
        )



def test_asset_prefilter_removes_cusip_style_identifiers_before_bars_api():
    from src.live.download_alpaca_2026_ytd import select_assets
    assets = [
        {"symbol": "AAPL", "class": "us_equity", "exchange": "NASDAQ", "status": "active"},
        {"symbol": "OLDCO", "class": "us_equity", "exchange": "NYSE", "status": "inactive"},
        {"symbol": "0029900E0", "class": "us_equity", "exchange": "NYSE", "status": "inactive"},
        {"symbol": "003CVR016", "class": "us_equity", "exchange": "NASDAQ", "status": "inactive"},
    ]
    selected, stats, prefiltered = select_assets(assets)
    assert [x["symbol"] for x in selected] == ["AAPL", "OLDCO"]
    assert stats["excluded_non_ticker_like_records"] == 2
    assert {x["symbol"] for x in prefiltered} == {"0029900E0", "003CVR016"}


def test_default_safe_end_avoids_current_weekend_date():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.live.download_alpaca_2026_ytd import resolve_end_date
    now = datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo("America/New_York"))  # Sunday
    end, policy = resolve_end_date(None, feed="auto", now=now)
    assert end == "2026-08-28"
    assert "buffer_20m" in policy


def test_default_safe_end_includes_weekday_only_after_close_buffer():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.live.download_alpaca_2026_ytd import resolve_end_date
    tz = ZoneInfo("America/New_York")
    before, _ = resolve_end_date(None, feed="auto", now=datetime(2026, 8, 31, 16, 19, tzinfo=tz))
    after, _ = resolve_end_date(None, feed="auto", now=datetime(2026, 8, 31, 16, 21, tzinfo=tz))
    assert before == "2026-08-28"
    assert after == "2026-08-31"


def test_auto_feed_probe_explicitly_uses_sip_not_implicit_default_or_iex():
    from src.live.download_alpaca_2026_ytd import StdlibResponse, probe_historical_feed

    class FakeSession:
        def __init__(self):
            self.params = None
        def get(self, url, *, headers, params=None, timeout=60):
            self.params = dict(params or {})
            return StdlibResponse(200, {}, b'{"bars":{"SPY":[]},"next_page_token":null}')

    session = FakeSession()
    resolved = probe_historical_feed(
        session, {"x": "y"}, start="2026-01-01", end="2026-08-29",
        feed_request="auto", adjustment="all",
    )
    assert resolved == "sip"
    assert session.params["feed"] == "sip"
    assert session.params["asof"] == "-"


def test_recent_sip_entitlement_probe_fails_with_no_silent_iex_fallback():
    from src.live.download_alpaca_2026_ytd import DownloadError, StdlibResponse, probe_historical_feed

    class FakeSession:
        def get(self, url, *, headers, params=None, timeout=60):
            return StdlibResponse(
                403, {},
                b'{"message":"subscription does not permit querying recent SIP data"}',
            )

    with pytest.raises(DownloadError) as exc:
        probe_historical_feed(
            FakeSession(), {"x": "y"}, start="2026-01-01", end="2026-08-30",
            feed_request="auto", adjustment="all",
        )
    text = str(exc.value)
    assert "too recent" in text
    assert "--feed iex" in text
    assert "never silently substitutes IEX" in text



def test_download_batch_disables_symbol_mapping_with_asof_dash():
    from src.live.download_alpaca_2026_ytd import StdlibResponse, download_batch

    class FakeSession:
        def __init__(self):
            self.params = None
        def get(self, url, *, headers, params=None, timeout=60):
            self.params = dict(params or {})
            return StdlibResponse(200, {}, b'{"bars":{},"next_page_token":null}')

    session = FakeSession()
    download_batch(
        session, {"x":"y"}, ["AAPL"], {"AAPL":{}},
        "2026-01-01", "2026-08-28", "sip", "all", 10000,
    )
    assert session.params["asof"] == "-"
    assert session.params["feed"] == "sip"
