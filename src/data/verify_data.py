from __future__ import annotations
import json
from datetime import datetime
from src.common.config import load_config, root_path
from src.common.logging_utils import log

def q(s): return "'"+str(s).replace("'","''")+"'"

def main():
    cfg=load_config(); feat=root_path(cfg,cfg['data']['feature_dir'])/'eligible'; manifests=root_path(cfg,cfg['data']['manifest_dir']); glob=str(feat).replace('\\','/')+'/**/*.parquet'
    if not list(feat.rglob('*.parquet')): raise FileNotFoundError('Eligible feature store missing')
    import duckdb
    con=duckdb.connect()
    dup=con.execute(f"SELECT COUNT(*) FROM (SELECT date,permno,COUNT(*) c FROM read_parquet({q(glob)},hive_partitioning=true) GROUP BY date,permno HAVING c>1)").fetchone()[0]
    stats=con.execute(f"SELECT MIN(date),MAX(date),COUNT(*),COUNT(DISTINCT permno),COUNT(DISTINCT date),SUM(CASE WHEN eligible THEN 1 ELSE 0 END) FROM read_parquet({q(glob)},hive_partitioning=true)").fetchone()
    if dup: raise RuntimeError(f"Feature store contains {dup} duplicate date/PERMNO keys")
    result={"stage":"verify","completed_at":datetime.now().isoformat(),"min_date":str(stats[0]),"max_date":str(stats[1]),"rows":int(stats[2]),"permnos":int(stats[3]),"dates":int(stats[4]),"eligible_rows":int(stats[5]),"duplicate_keys":int(dup)}
    (manifests/'data_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); log(f"Data verification PASS: {result}")
if __name__=='__main__': main()
