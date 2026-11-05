from __future__ import annotations
import json
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.config import load_config, root_path
from src.common.logging_utils import log
from src.screener.ranking import rank_month_end


def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def _required_columns(cfg: dict) -> set[str]:
    return {
        "date", "permno", "ticker", "sector", "eligible", "close", "adv20",
        "sma100", "sma200", "momentum_3_1", "momentum_6_1",
        "momentum_12_1", "trend_strength",
        *cfg["screener"]["score_weights"].keys(),
    }


def _partition_years(feat: Path, start_year: int) -> list[int]:
    years = []
    for p in feat.iterdir():
        if not p.is_dir() or "=" not in p.name:
            continue
        key, value = p.name.split("=", 1)
        if key not in {"eligible_year", "year"}:
            continue
        try:
            y = int(value)
        except ValueError:
            continue
        if y >= start_year:
            years.append(y)
    return sorted(set(years))


def _schema_columns(con, glob: str) -> set[str]:
    # LIMIT 0 avoids reading the parquet body; it only resolves the unioned schema.
    cur = con.execute(
        f"SELECT * FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true) LIMIT 0"
    )
    return {d[0] for d in cur.description}


def main():
    cfg = load_config()
    feat = root_path(cfg, cfg["data"]["feature_dir"]) / "eligible"
    out = root_path(cfg, "outputs/screener")
    out.mkdir(parents=True, exist_ok=True)
    manifests = root_path(cfg, cfg["data"]["manifest_dir"])
    manifests.mkdir(parents=True, exist_ok=True)
    temp = root_path(cfg, cfg["data"]["temp_dir"])
    temp.mkdir(parents=True, exist_ok=True)

    if not feat.exists() or not any(feat.rglob("*.parquet")):
        feature_root = feat.parent
        manifest = manifests / "features_success.json"
        base_count = len(list(feature_root.glob("year=*/*.parquet"))) if feature_root.exists() else 0
        raise FileNotFoundError(
            f"Eligible feature store not found under {feat}. "
            f"Physical state: features_success.json={manifest.exists()}, base_feature_files={base_count}, "
            f"eligible_feature_files=0. Run 01_build_data.cmd to rebuild the production feature store."
        )

    glob = str(feat).replace("\\", "/") + "/**/*.parquet"
    start = pd.Timestamp(cfg["data"]["research_start_date"])
    top_n = int(cfg["screener"]["top_n"])

    import duckdb

    con = duckdb.connect()
    comp = cfg.get("compute", {})
    con.execute(f"SET threads={int(comp.get('threads', 4))}")
    if comp.get("memory_limit"):
        con.execute(f"SET memory_limit={q(comp['memory_limit'])}")
    con.execute(f"SET temp_directory={q(str(temp))}")
    con.execute(
        "SET preserve_insertion_order=" +
        ("true" if comp.get("preserve_insertion_order", False) else "false")
    )

    log("Screener preflight: validating eligible feature-store schema...")
    cols = _schema_columns(con, glob)
    missing = sorted(_required_columns(cfg).difference(cols))
    if missing:
        raise RuntimeError(
            "Eligible feature store is incompatible with the screener. "
            f"Missing columns: {missing}. Re-run the feature stage with 01_build_data.cmd "
            "after removing data/manifests/features_success.json if this store was built by an older version."
        )

    # A duplicate key would bias cross-sectional ranks and can make month-end selection unstable.
    dup = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT date, permno, COUNT(*) c
            FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
            WHERE date >= DATE {q(start.date())}
            GROUP BY date, permno
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if dup:
        raise RuntimeError(
            f"Eligible feature store contains {dup:,} duplicate date/PERMNO keys. "
            "Do not rank this store; rebuild the feature stage cleanly with --force."
        )

    years = _partition_years(feat, start.year)
    if not years:
        # Fallback for non-hive layouts. This is a tiny distinct-year result, not a large fetch.
        years = [int(r[0]) for r in con.execute(f"""
            SELECT DISTINCT YEAR(date) AS y
            FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
            WHERE date >= DATE {q(start.date())}
            ORDER BY y
        """).fetchall()]

    log(
        f"Screener preflight PASS: {len(years)} year partitions from {years[0]} to {years[-1]}. "
        "Processing one year at a time to keep RAM bounded."
    )

    output_cols = [
        "ranking_date", "effective_month", "rank", "permno", "ticker", "sector",
        "leadership_score", "momentum_3_1", "momentum_6_1", "momentum_12_1",
        "trend_strength", "close", "adv20",
    ]
    rows: list[pd.DataFrame] = []
    diagnostics = []

    for y in years:
        # One scan per year instead of one giant 2015-present fetch. Hive partition pruning keeps
        # memory and I/O bounded on the full CRSP-derived feature store.
        year_start = max(start, pd.Timestamp(f"{y}-01-01"))
        year_end = pd.Timestamp(f"{y}-12-31")
        log(f"Screener: loading common month-end cross-sections for {y}...")
        df = con.execute(f"""
            WITH ydata AS (
                SELECT *
                FROM read_parquet({q(glob)}, hive_partitioning=true, union_by_name=true)
                WHERE date BETWEEN DATE {q(year_start.date())} AND DATE {q(year_end.date())}
            ), month_ends AS (
                SELECT YEAR(date) y, MONTH(date) m, MAX(date) month_end
                FROM ydata
                GROUP BY 1,2
            )
            SELECT f.*
            FROM ydata f
            JOIN month_ends me ON f.date = me.month_end
            ORDER BY f.date, f.permno
        """).fetchdf()

        if df.empty:
            log(f"Screener warning: no month-end feature rows found for {y}; skipping.")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="raise")
        year_ranked = 0
        months = 0
        for dt, g in df.groupby("date", sort=True):
            dt = pd.Timestamp(dt)
            ranked = rank_month_end(g, cfg)
            months += 1
            if ranked.empty:
                diagnostics.append({
                    "ranking_date": dt,
                    "eligible_month_end_rows": int(g["eligible"].fillna(False).sum()),
                    "rankable_after_filters": 0,
                    "selected": 0,
                })
                continue
            selected = ranked.head(top_n).copy()
            selected["ranking_date"] = dt
            selected["effective_month"] = str(dt.to_period("M") + 1)
            rows.append(selected[output_cols])
            year_ranked += len(selected)
            diagnostics.append({
                "ranking_date": dt,
                "eligible_month_end_rows": int(g["eligible"].fillna(False).sum()),
                "rankable_after_filters": int(len(ranked)),
                "selected": int(len(selected)),
            })

        log(
            f"Screener: {y} complete - {months} month-ends, "
            f"{year_ranked:,} selected rows."
        )
        # Explicitly release the year's DataFrame before loading the next partition.
        del df

    if not rows:
        raise RuntimeError(
            "Screener completed its scans but produced zero rankings. Check the feature-store "
            "eligibility counts and leadership filters before backtesting."
        )

    outdf = pd.concat(rows, ignore_index=True)
    outdf = outdf.sort_values(["ranking_date", "rank", "permno"], kind="mergesort")
    path = out / "monthly_rankings.csv"
    tmp = out / "monthly_rankings.csv.tmp"
    outdf.to_csv(tmp, index=False)
    tmp.replace(path)

    diag = pd.DataFrame(diagnostics).sort_values("ranking_date")
    diag.to_csv(out / "monthly_screener_diagnostics.csv", index=False)

    payload = {
        "release": cfg["project"]["release"],
        "completed_at": datetime.now().isoformat(),
        "research_start": str(start.date()),
        "first_ranking_date": str(pd.to_datetime(outdf["ranking_date"]).min().date()),
        "last_ranking_date": str(pd.to_datetime(outdf["ranking_date"]).max().date()),
        "ranking_months": int(outdf["ranking_date"].nunique()),
        "ranked_rows": int(len(outdf)),
        "top_n": top_n,
        "output": str(path),
    }
    (manifests / "screener_success.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    log(
        f"Monthly screener complete: {len(outdf):,} ranked rows across "
        f"{payload['ranking_months']} months -> {path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # The launcher tees stderr to logs/screener.log. This explicit traceback ensures an
        # actionable error is always present even when the script is started by double-click.
        traceback.print_exc()
        raise
