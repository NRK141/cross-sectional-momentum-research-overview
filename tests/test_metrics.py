import pandas as pd
from src.common.metrics import trade_metrics

def test_positive_expectancy_can_have_sub_50_win_rate():
    t=pd.DataFrame({'net_return':[.08,.07,-.02,-.02,-.02]})
    m=trade_metrics(t)
    assert m['win_rate']==0.4
    assert m['expectancy_per_trade']>0
    assert m['payoff_ratio']>3
    assert m['profit_factor']>2
