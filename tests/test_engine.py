import pandas as pd
from src.backtest.engine import candidate_ok

def test_pullback_candidate_is_directional_not_mean_reversion_only():
    cfg={'entry':{'min_5d_pullback':-.10,'max_5d_pullback':-.02,'max_rsi2':35,'min_distance_from_20d_high':-.12,'max_distance_from_20d_high':-.02,'require_positive_1d_resumption':True,'require_close_above_sma50':True}}
    r=pd.Series({'eligible':True,'ret5':-.04,'rsi2':30,'distance_20d_high':-.05,'ret1':.01,'close':105,'sma50':100})
    assert candidate_ok(r,cfg)
    r['ret1']=-.01
    assert not candidate_ok(r,cfg)
