from __future__ import annotations
import json, os, uuid
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from .alpaca_crsp_bridge import ROOT, RAW, RAW_MANIFEST, BridgeError, sha256_file, _duckdb, _s
from .download_alpaca_2026_ytd import StdlibSession,_headers,load_credentials,download_batch_with_quarantine

BENCH=('SPY','QQQ','DIA'); MANIFEST=ROOT/'data/manifests/alpaca_benchmark_extension_2026.json'

def main():
    if not RAW.exists() or not RAW_MANIFEST.exists(): raise BridgeError('Missing Alpaca YTD raw file/manifest')
    rm=json.loads(RAW_MANIFEST.read_text(encoding='utf-8')); feed=str(rm.get('feed_used','sip')); adj=str(rm.get('adjustment','all'))
    creds=load_credentials(ROOT,None); session=StdlibSession(); headers=_headers(creds)
    asset_map={s:{'symbol':s} for s in BENCH}
    anchor,_=download_batch_with_quarantine(session,headers,list(BENCH),asset_map,'2025-12-01','2025-12-31',feed,adj,10000)
    con=_duckdb().connect()
    try: raw=con.execute(f"SELECT TRY_CAST(date AS DATE) AS date, UPPER(symbol) AS symbol, TRY_CAST(close AS DOUBLE) AS close FROM read_parquet({_s(RAW)}) WHERE UPPER(symbol) IN ('SPY','QQQ','DIA')").df()
    finally: con.close()
    raw['date']=pd.to_datetime(raw.date)
    records=[]
    for sym in BENCH:
        old=ROOT/'data/benchmarks'/f'{sym.lower()}_daily.csv'
        if not old.exists(): raise BridgeError(f'Missing existing benchmark CSV: {old}')
        hist=pd.read_csv(old); hist['date']=pd.to_datetime(hist['date']); hist=hist[hist.date<=pd.Timestamp('2025-12-31')].copy()
        a=anchor[anchor.symbol.str.upper()==sym].copy(); a['date']=pd.to_datetime(a.date); a=a.sort_values('date')
        y=raw[raw.symbol==sym][['date','close']].sort_values('date').copy()
        if a.empty or y.empty: raise BridgeError(f'Missing Alpaca benchmark data for {sym}')
        last=a.iloc[-1]; seq=pd.concat([pd.DataFrame([{'date':last['date'],'close':last['close']}]),y],ignore_index=True).sort_values('date')
        seq['return']=seq.close.astype(float).pct_change(); yret=seq[seq.date.dt.year==2026][['date','return']]
        if yret['return'].isna().any(): raise BridgeError(f'Null forward benchmark return for {sym}')
        out=pd.concat([hist[['date','return']],yret],ignore_index=True).drop_duplicates('date',keep='last').sort_values('date')
        tmp=old.with_name(old.name+'.tmp-'+uuid.uuid4().hex); out.assign(date=out.date.dt.strftime('%Y-%m-%d')).to_csv(tmp,index=False); os.replace(tmp,old)
        records.append({'symbol':sym,'rows':len(out),'last_date':out.date.max().date().isoformat(),'sha256':sha256_file(old)})
    MANIFEST.parent.mkdir(parents=True,exist_ok=True); MANIFEST.write_text(json.dumps({'schema_version':1,'status':'PASS','generated_at_utc':datetime.now(timezone.utc).isoformat(),'source_raw_sha256':sha256_file(RAW),'feed':feed,'adjustment':adj,'files':records,'guardrail':'Existing benchmark rows through 2025-12-31 are preserved; only 2026 returns are replaced from Alpaca using a December-2025 Alpaca anchor.'},indent=2),encoding='utf-8')
    print('='*78); print('ALPACA 2026 BENCHMARK EXTENSION PASS'); print('='*78)
    for r in records: print(f"{r['symbol']}: {r['last_date']} rows={r['rows']:,}")
    print(f'Manifest: {MANIFEST}'); print('='*78)
if __name__=='__main__': main()
