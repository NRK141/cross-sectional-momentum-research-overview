from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .identity import ROOT, LIVE_BACKTEST, LIVE_SCREENER, RELEASE

LOCK = ROOT / "data" / "manifests" / "foundation2_2_main_forward_prefix_lock.json"
REPORT = ROOT / "outputs" / "diagnostics" / "foundation2_2" / "forward_prefix_lock_report.json"
NORMALIZED = ROOT / "data" / "normalized" / "year=2026" / "data_0.parquet"
NORMALIZED_MANIFEST = ROOT / "data" / "manifests" / "alpaca_normalized_2026.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_csv_digest(path: Path, cutoff: pd.Timestamp, date_col: str, sort_cols: list[str]) -> dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if date_col not in df.columns: raise RuntimeError(f"{path} missing prefix date column {date_col}")
    if date_col == "effective_month":
        dt = pd.to_datetime(df[date_col].astype(str), format="%Y-%m", errors="raise")
    else:
        dt = pd.to_datetime(df[date_col], errors="raise")
    x = df.loc[dt <= cutoff].copy()
    usable = [c for c in sort_cols if c in x.columns]
    x = x.sort_values(usable if usable else list(x.columns), kind="mergesort")
    payload = x.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return {"sha256": _sha(payload), "rows": int(len(x)), "date_col": date_col}


def _duckdb():
    try: import duckdb  # type: ignore
    except ImportError as exc: raise RuntimeError("duckdb is required for normalized forward-prefix verification") from exc
    return duckdb


def _q(name: str) -> str: return '"' + str(name).replace('"','""') + '"'
def _s(value: Any) -> str: return "'" + str(value).replace("'","''") + "'"


def _normalized_columns() -> tuple[str,str]:
    if not NORMALIZED_MANIFEST.exists(): raise FileNotFoundError(NORMALIZED_MANIFEST)
    m = _load_json(NORMALIZED_MANIFEST); r = m.get("resolved_columns", {})
    dc, pc = r.get("date"), r.get("permno")
    if not dc or not pc: raise RuntimeError("alpaca_normalized_2026 manifest lacks resolved date/PERMNO columns")
    return str(dc), str(pc)


def normalized_prefix_digest(path: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(path)
    dc, pc = _normalized_columns()
    fd, tmp_name = tempfile.mkstemp(prefix="f22_main_prefix_", suffix=".csv"); os.close(fd)
    tmp = Path(tmp_name); con = _duckdb().connect()
    try:
        con.execute(f"COPY (SELECT * FROM read_parquet({_s(path)}, hive_partitioning=false) WHERE TRY_CAST({_q(dc)} AS DATE) <= DATE {_s(cutoff.date().isoformat())} ORDER BY TRY_CAST({_q(dc)} AS DATE), TRY_CAST({_q(pc)} AS BIGINT)) TO {_s(tmp)} (FORMAT CSV, HEADER TRUE)")
        rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({_s(path)}, hive_partitioning=false) WHERE TRY_CAST({_q(dc)} AS DATE) <= DATE {_s(cutoff.date().isoformat())}").fetchone()[0])
    finally: con.close()
    try:
        h=hashlib.sha256()
        with tmp.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        return {"sha256":h.hexdigest(),"rows":rows,"date_col":dc,"permno_col":pc}
    finally: tmp.unlink(missing_ok=True)


def _last_date(path: Path) -> pd.Timestamp:
    df = pd.read_csv(path, usecols=["date"])
    if df.empty: raise RuntimeError(f"Empty equity curve: {path}")
    return pd.to_datetime(df["date"], errors="raise").max().normalize()


def _artifacts(cutoff: pd.Timestamp, normalized_path: Path = NORMALIZED) -> dict[str, Any]:
    return {
        "normalized_2026": normalized_prefix_digest(normalized_path, cutoff),
        "production_equity": _canonical_csv_digest(LIVE_BACKTEST/"equity_curve.csv", cutoff, "date", ["date"]),
        "production_events": _canonical_csv_digest(LIVE_BACKTEST/"event_log.csv", cutoff, "date", ["date","event","permno","ticker"]),
        "production_closed_trades": _canonical_csv_digest(LIVE_BACKTEST/"trade_log.csv", cutoff, "exit_date", ["exit_date","entry_date","permno","ticker"]),
        "production_rankings": _canonical_csv_digest(LIVE_SCREENER/"monthly_rankings.csv", cutoff, "effective_month", ["effective_month","rank","permno"]),
    }


def verify_lock(normalized_path: Path = NORMALIZED) -> dict[str, Any]:
    if not LOCK.exists(): return {"status":"NOT_ESTABLISHED","lock":str(LOCK)}
    lock=_load_json(LOCK); cutoff=pd.Timestamp(lock["cutoff"]); current=_artifacts(cutoff, normalized_path); expected=lock.get("artifacts",{})
    checks={}
    for name,cur in current.items():
        exp=expected.get(name,{})
        checks[name]={"pass":cur.get("sha256")==exp.get("sha256") and cur.get("rows")==exp.get("rows"),"expected_sha256":exp.get("sha256"),"current_sha256":cur.get("sha256"),"expected_rows":exp.get("rows"),"current_rows":cur.get("rows")}
    return {"status":"PASS" if checks and all(x["pass"] for x in checks.values()) else "FAIL","cutoff":lock["cutoff"],"checks":checks,"lock":str(LOCK),"baseline_release":lock.get("baseline_release")}


def verify_normalized_staging_against_lock(staging: Path) -> dict[str, Any]:
    if not LOCK.exists(): return {"status":"NOT_ESTABLISHED","lock":str(LOCK)}
    lock=_load_json(LOCK); cutoff=pd.Timestamp(lock["cutoff"]); cur=normalized_prefix_digest(staging,cutoff); exp=lock.get("artifacts",{}).get("normalized_2026",{})
    ok=cur.get("sha256")==exp.get("sha256") and cur.get("rows")==exp.get("rows")
    return {"status":"PASS" if ok else "FAIL","cutoff":lock["cutoff"],"expected":exp,"staging":cur,"guardrail":"YTD refreshes may append after the F2.2 production lock cutoff but may not rewrite already-admitted normalized observations."}


def commit_lock() -> dict[str, Any]:
    if not NORMALIZED.exists(): raise FileNotFoundError(NORMALIZED)
    previous=verify_lock()
    if previous.get("status")=="FAIL": raise RuntimeError(f"Existing F2.2 production prefix lock failed and will not be overwritten: {previous}")
    cutoff=_last_date(LIVE_BACKTEST/"equity_curve.csv"); artifacts=_artifacts(cutoff)
    payload={"schema_version":2,"status":"PASS","baseline_release":RELEASE,"committed_at_utc":datetime.now(timezone.utc).isoformat(),"cutoff":cutoff.date().isoformat(),"previous_lock_status":previous.get("status"),"artifacts":artifacts,"policy":["Foundation 2.2 production is the sole strategy baseline protected by this lock.","Monthly refreshes may append after the cutoff but may not rewrite normalized, ranking, event, equity, or closed-trade evidence already admitted into the baseline.","Closed trades are locked by exit_date; later closure of a position opened before the cutoff may legitimately append a new closed-trade row."]}
    LOCK.parent.mkdir(parents=True, exist_ok=True); tmp=LOCK.with_name(LOCK.name+".tmp"); tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8"); os.replace(tmp,LOCK); return payload


def _write_report(r: dict[str,Any]):
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(json.dumps(r,indent=2),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(); g.add_argument("--verify",action="store_true"); g.add_argument("--commit",action="store_true"); args=ap.parse_args()
    if args.commit: result=commit_lock(); report={"action":"COMMIT",**result}
    else:
        result=verify_lock(); report={"action":"VERIFY",**result}
        if result.get("status")=="FAIL": _write_report(report); raise SystemExit(2)
    _write_report(report)
    print("="*78); print("FOUNDATION 2.2 PRODUCTION PREFIX LOCK"); print("="*78); print(f"Action: {report['action']}"); print(f"Status: {report.get('status')}"); print(f"Locked through: {report.get('cutoff','-')}"); print(f"Manifest: {LOCK}"); print(f"Report: {REPORT}"); print("="*78)

if __name__=="__main__": main()
