from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
from datetime import datetime
from src.common.config import load_config, root_path
from src.common.schema import resolve_columns, sql_ident
from src.common.logging_utils import log

def read_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        return next(csv.reader(f))

def qstr(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    cfg=load_config(); raw=root_path(cfg,cfg['data']['raw_crsp_csv']); out=root_path(cfg,cfg['data']['normalized_dir']); manifests=root_path(cfg,cfg['data']['manifest_dir']); temp=root_path(cfg,cfg['data']['temp_dir'])
    manifests.mkdir(parents=True,exist_ok=True); temp.mkdir(parents=True,exist_ok=True)
    manifest=manifests/'normalized_success.json'
    if manifest.exists() and not args.force:
        log(f"Normalized stage already complete: {manifest}"); return
    if not raw.exists(): raise FileNotFoundError(f"Raw CRSP source not found: {raw}")
    if out.exists() and any(out.iterdir()):
        if not args.force: raise RuntimeError(f"{out} contains partial data but no success manifest. Re-run with --force to rebuild safely.")
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    headers=read_headers(raw); c=resolve_columns(headers)
    import duckdb
    con=duckdb.connect()
    comp=cfg['compute']; con.execute(f"SET threads={int(comp['threads'])}"); con.execute(f"SET memory_limit={qstr(comp['memory_limit'])}"); con.execute(f"SET temp_directory={qstr(str(temp))}"); con.execute(f"SET preserve_insertion_order={'true' if comp.get('preserve_insertion_order',False) else 'false'}")
    def expr(col, cast, alias, default='NULL'):
        return f"TRY_CAST({sql_ident(col)} AS {cast}) AS {alias}" if col else f"{default}::{cast} AS {alias}"
    exch = "NULL::VARCHAR AS primary_exch"
    if c.primary_exch:
        ci=sql_ident(c.primary_exch)
        exch=f"CASE WHEN TRY_CAST({ci} AS INTEGER)=1 THEN 'N' WHEN TRY_CAST({ci} AS INTEGER)=2 THEN 'A' WHEN TRY_CAST({ci} AS INTEGER)=3 THEN 'Q' ELSE UPPER(TRIM(CAST({ci} AS VARCHAR))) END AS primary_exch"
    start=cfg['data']['source_start_date']
    raw_sql=qstr(str(raw).replace('\\','/')); out_sql=qstr(str(out).replace('\\','/'))
    cols_sql=',\n'.join([
        f"TRY_CAST({sql_ident(c.date)} AS DATE) AS date",
        expr(c.permno,'BIGINT','permno'),
        f"UPPER(TRIM(CAST({sql_ident(c.ticker)} AS VARCHAR))) AS ticker",
        f"ABS(TRY_CAST({sql_ident(c.price)} AS DOUBLE)) AS close",
        expr(c.ret,'DOUBLE','ret'), expr(c.volume,'DOUBLE','volume'), exch,
        expr(c.security_type,'VARCHAR','security_type'), expr(c.security_subtype,'VARCHAR','security_subtype'), expr(c.share_type,'VARCHAR','share_type'), expr(c.sic,'INTEGER','sic')
    ])
    log("Starting one-pass CRSP normalization. This is the longest first-build stage.")
    sql=f"""
    COPY (
      WITH src AS (
        SELECT {cols_sql}
        FROM read_csv_auto({raw_sql}, header=true, sample_size=200000, ignore_errors=false, null_padding=true)
      )
      SELECT date, permno, ticker, close, ret, volume,
             CASE WHEN volume IS NULL THEN NULL ELSE close * ABS(volume) END AS dollar_volume,
             primary_exch, security_type, security_subtype, share_type, sic,
             YEAR(date) AS year
      FROM src
      WHERE date >= DATE {qstr(start)} AND date IS NOT NULL AND permno IS NOT NULL AND close > 0
    ) TO {out_sql} (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE true, ROW_GROUP_SIZE 250000)
    """
    con.execute(sql)
    stats=con.execute(f"SELECT MIN(date),MAX(date),COUNT(*),COUNT(DISTINCT permno),COUNT(DISTINCT date) FROM read_parquet({qstr(str(out).replace('\\','/')+'/**/*.parquet')}, hive_partitioning=true)").fetchone()
    payload={"stage":"normalized","completed_at":datetime.now().isoformat(),"raw_source":str(raw),"min_date":str(stats[0]),"max_date":str(stats[1]),"rows":int(stats[2]),"permnos":int(stats[3]),"dates":int(stats[4]),"resolved_columns":c.__dict__}
    manifest.write_text(json.dumps(payload,indent=2),encoding='utf-8'); log(f"Normalized store complete: {payload['rows']:,} rows, {payload['permnos']:,} PERMNO")
if __name__=='__main__': main()
