from src.common.schema import resolve_columns

def test_ciz_schema_resolution():
    c=resolve_columns(['DlyCalDt','PERMNO','Ticker','DlyPrc','DlyRet','DlyVol','PrimaryExch','SecurityType','SecuritySubType','ShareType','SICCD'])
    assert c.date=='DlyCalDt' and c.price=='DlyPrc' and c.permno=='PERMNO'

def test_minimal_schema_resolution():
    c=resolve_columns(['date','permno','ticker','close'])
    assert c.ret is None and c.volume is None
