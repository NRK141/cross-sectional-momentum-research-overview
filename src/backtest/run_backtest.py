from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd
from src.common.config import load_config, root_path
from src.common.logging_utils import log
from src.common.metrics import portfolio_metrics, trade_metrics
from src.backtest.engine import simulate

def q(s): return "'"+str(s).replace("'","''")+"'"

def safe(v):
    if isinstance(v,float) and (math.isnan(v) or math.isinf(v)): return None
    return v

def main():
    cfg=load_config(); root=Path(cfg['_root']); feat=root_path(cfg,cfg['data']['feature_dir'])/'eligible'; rankings_path=root/'outputs/screener/monthly_rankings.csv'; out=root/'outputs/backtest'; out.mkdir(parents=True,exist_ok=True)
    if not rankings_path.exists(): raise FileNotFoundError('Run the F2.2 production cycle or src.screener.build_monthly_rankings first')
    rankings=pd.read_csv(rankings_path); rankings['ranking_date']=pd.to_datetime(rankings['ranking_date'])
    import duckdb
    con=duckdb.connect(); glob=str(feat).replace('\\','/')+'/**/*.parquet'; start=cfg['data']['score_start_date']
    daily=con.execute(f"SELECT date,permno,ticker,sector,eligible,close,ret1,ret5,rsi2,distance_20d_high,sma20,sma50,sma100,sma200 FROM read_parquet({q(glob)},hive_partitioning=true) WHERE date >= DATE {q(start)} ORDER BY date,permno").fetchdf()
    log(f"Running next-close shared-capital baseline on {len(daily):,} daily stock rows...")
    eq,trades,events=simulate(daily,rankings,cfg)
    eq.to_csv(out/'equity_curve.csv',index=False); trades.to_csv(out/'trade_log.csv',index=False); events.to_csv(out/'event_log.csv',index=False)
    pm=portfolio_metrics(eq.set_index('date')['equity'],int(cfg['execution']['annualization_days'])); tm=trade_metrics(trades)
    years=max((pd.to_datetime(eq.date).max()-pd.to_datetime(eq.date).min()).days/365.25,1e-9) if len(eq) else 0
    tm['annualized_trades']=float(len(trades)/years) if years else 0.0
    # Benchmark summary.
    bench={}
    for sym in [cfg['benchmarks']['primary'],*cfg['benchmarks']['comparisons']]:
        p=root_path(cfg,cfg['data']['benchmark_dir'])/f'{sym.lower()}_daily.csv'; b=pd.read_csv(p); b['date']=pd.to_datetime(b['date']); b=b[b.date>=pd.Timestamp(start)].copy()
        growth=(1+b['return'].fillna(0).astype(float)).cumprod(); bench[sym]=portfolio_metrics(growth,int(cfg['execution']['annualization_days']))
    summary={'release':cfg['project']['release'],'strategy_metrics':pm,'trade_diagnostics':tm,'benchmark_metrics':bench,'config_snapshot':cfg,'production_role':'SOLE_PRODUCTION_STRATEGY'}
    summary={k:({kk:safe(vv) for kk,vv in v.items()} if isinstance(v,dict) else v) for k,v in summary.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    log(f"Backtest complete. CAGR={pm['cagr']:.2%} Sharpe={pm['sharpe']:.2f} MaxDD={pm['max_drawdown']:.2%} Trades={len(trades)} PF={tm['profit_factor']}")
if __name__=='__main__': main()
