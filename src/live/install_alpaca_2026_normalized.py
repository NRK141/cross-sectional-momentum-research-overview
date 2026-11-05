from __future__ import annotations
import json, os, shutil, uuid, hashlib
from datetime import datetime, timezone
from pathlib import Path
from .alpaca_crsp_bridge import ROOT, sha256_file, BridgeError
from src.production.forward_prefix_lock import verify_normalized_staging_against_lock

STAGING=ROOT/'data/alpaca/2026/bridge/normalized_year2026_staging.parquet'
SRC_MANIFEST=ROOT/'data/manifests/alpaca_normalized_2026.json'
TARGET_DIR=ROOT/'data/normalized/year=2026'; TARGET=TARGET_DIR/'data_0.parquet'
MANIFEST=ROOT/'data/manifests/alpaca_normalized_2026_install.json'

def hist_hashes():
    out={}
    for p in sorted((ROOT/'data/normalized').glob('year=*/*.parquet')):
        try: year=int(p.parent.name.split('=',1)[1])
        except Exception: continue
        if year <= 2025: out[str(p.relative_to(ROOT)).replace('\\','/')]=sha256_file(p)
    return out

def main():
    if not STAGING.exists() or not SRC_MANIFEST.exists(): raise BridgeError('Missing normalized staging/manifest; run the Foundation 2.2 monthly refresh or src.live.build_alpaca_2026_normalized first')
    m=json.loads(SRC_MANIFEST.read_text(encoding='utf-8'))
    if m.get('status')!='PASS' or sha256_file(STAGING)!=m.get('staging_sha256'): raise BridgeError('Staging identity failed')
    before=hist_hashes(); expected=m.get('historical_normalized_hashes_before',{})
    if before!=expected: raise BridgeError('Historical normalized <=2025 files changed since staging build; rebuild staging rather than installing across a moving history')
    prefix = verify_normalized_staging_against_lock(STAGING)
    if prefix.get('status') == 'FAIL':
        raise BridgeError(
            f"2026 staging rewrites observations already protected by the prospective prefix lock through {prefix.get('cutoff')}; install refused. "
            f"Inspect data/manifests/foundation2_2_main_forward_prefix_lock.json before accepting any vendor restatement."
        )
    TARGET_DIR.mkdir(parents=True,exist_ok=True)
    tmp=TARGET.with_name(TARGET.name+'.tmp-'+uuid.uuid4().hex)
    shutil.copy2(STAGING,tmp); os.replace(tmp,TARGET)
    after=hist_hashes()
    if after!=before: raise BridgeError('Historical normalized <=2025 hashes changed during 2026 install')
    payload={'schema_version':2,'status':'PASS','installed_at_utc':datetime.now(timezone.utc).isoformat(),'target':str(TARGET),'target_sha256':sha256_file(TARGET),'historical_hashes_unchanged':True,'historical_hash_count':len(after),'source_manifest_sha256':sha256_file(SRC_MANIFEST),'forward_prefix_lock_preinstall':prefix,'guardrail':'Only data/normalized/year=2026/data_0.parquet is replaced. <=2025 normalized partitions are hash-verified unchanged.'}
    MANIFEST.parent.mkdir(parents=True,exist_ok=True); MANIFEST.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('='*78); print('ALPACA 2026 NORMALIZED INSTALL PASS'); print('='*78); print(f'Target: {TARGET}'); print('Historical <=2025 partitions: UNCHANGED'); print(f'Manifest: {MANIFEST}'); print('='*78)
if __name__=='__main__': main()
