from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from src.common.config import load_config, root_path
from src.common.logging_utils import log

def q(s): return "'"+str(s).replace("'","''")+"'"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    cfg=load_config(); norm=root_path(cfg,cfg['data']['normalized_dir']); out=root_path(cfg,cfg['data']['benchmark_dir']); manifests=root_path(cfg,cfg['data']['manifest_dir']); manifest=manifests/'benchmarks_success.json'; out.mkdir(parents=True,exist_ok=True)
    if manifest.exists() and not args.force: log(f"Benchmark stage already complete: {manifest}"); return
    import duckdb, pandas as pd
    con=duckdb.connect(); glob=str(norm).replace('\\','/')+'/**/*.parquet'
    meta={}
    for sym in [cfg['benchmarks']['primary'],*cfg['benchmarks']['comparisons']]:
        rows=con.execute(f"SELECT permno,COUNT(*) n,MIN(date) mn,MAX(date) mx FROM read_parquet({q(glob)},hive_partitioning=true) WHERE UPPER(ticker)=? GROUP BY permno ORDER BY n DESC",[sym]).fetchall()
        if not rows: raise RuntimeError(f"Benchmark {sym} not found in normalized CRSP store")
        permno=int(rows[0][0])
        df=con.execute(f"SELECT date,COALESCE(ret,close/LAG(close) OVER (PARTITION BY permno ORDER BY date)-1) AS return,close,permno,ticker FROM read_parquet({q(glob)},hive_partitioning=true) WHERE permno=? ORDER BY date",[permno]).fetchdf()
        df.to_csv(out/f'{sym.lower()}_daily.csv',index=False)
        meta[sym]={"permno":permno,"rows":len(df),"min_date":str(df.date.min()),"max_date":str(df.date.max())}
        log(f"Benchmark {sym}: PERMNO {permno}, {len(df):,} rows")
    manifest.write_text(json.dumps({"stage":"benchmarks","completed_at":datetime.now().isoformat(),"benchmarks":meta},indent=2),encoding='utf-8')
if __name__=='__main__': main()
