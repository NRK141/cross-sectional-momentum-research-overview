from __future__ import annotations
import numpy as np
import pandas as pd
from data_access import HistoryStore
from run_stochastic_standalone import _signal_alpha_frame, _alpha_columns_for_model
from run_development_optimizer import _panel_structures


def _hist(raw, returns, periods=6):
    dt=pd.bdate_range("2024-06-03",periods=periods)
    frame=pd.DataFrame({"date":dt,"permno":[123]*periods,"close":raw,"ret1":returns})
    return HistoryStore.from_daily(frame),dt


def test_forward_split_no_artificial_loss_on_held_position():
    h, dates=_hist([120,125,128,65,67,69],[np.nan,125/120-1,128/125-1,0.016,67/65-1,69/67-1])
    before=h.economic_price(123,dates[2]); after=h.economic_price(123,dates[3])
    assert np.isclose(after/before-1,0.016)
    assert (after/before-1)>-0.04
    assert h.corporate_action_events
    assert any(x["date"]==str(dates[3].date()) for x in h.corporate_action_events)


def test_reverse_split_no_artificial_gain():
    h, dates=_hist([10,10.2,10.1,50.5,51,50.8], [np.nan,.02,10.1/10.2-1.,0.,51/50.5-1,50.8/51-1])
    assert np.isclose(h.economic_price(123,dates[3])/h.economic_price(123,dates[2])-1,0)
    assert h.corporate_action_events


def test_signal_features_point_in_time_split_invariant():
    dates=pd.bdate_range("2023-01-02",periods=230)
    gross=1.0004
    raw=100*gross**np.arange(230,dtype=float)
    raw[210:]/=2.0
    ret=np.full(230,gross-1.0);ret[0]=np.nan
    h=HistoryStore.from_daily(pd.DataFrame({"date":dates,"permno":[5]*230,"close":raw,"ret1":ret}))
    feat=h.repaired_signal_features(5,dates[225]); assert feat is not None
    assert feat['close']/feat['sma200']>1.0
    assert abs(feat['ret5']-(gross**5-1))<1e-10
    breadth=h.economic_breadth([dates[225]])
    assert np.isclose(breadth.iloc[0]['breadth_above_sma200'],1.0)


def test_hump_alpha_penalizes_extreme_pullback():
    frame=pd.DataFrame({"close":[115]*4,"sma50":[110]*4,"sma100":[108]*4,
      "sma200":[100]*4,"ret5":[-.01,-.035,-.05,-.20],"rsi2":[55,23,15,1],
      "distance_20d_high":[-.05]*4})
    a=_signal_alpha_frame(frame)
    assert float(a.loc[1,'alpha_hump_pullback'])>float(a.loc[3,'alpha_hump_pullback'])
    assert _alpha_columns_for_model('HUMP_PULLBACK')==('alpha_hump_pullback','alpha_pct_hump_pullback')


def test_screen_panel_accepts_hump_alpha_fields():
    row=dict(date=pd.Timestamp('2020-01-31'),forecast_stop_pct=0.04,permno=1,ticker='X',sector='T',
      alpha_balanced=.6,alpha_trend_heavy=.7,alpha_pullback_heavy=.6,alpha_hump_pullback=.8,
      alpha_pct_balanced=.6,alpha_pct_trend_heavy=.7,alpha_pct_pullback_heavy=.6,alpha_pct_hump_pullback=.8,
      mc_p_stop=.2,mc_p_stop_se=.02,mc_stop_probability_ucb=.22,risk_volatility=.3,
      mc_mean_se=.001,mc_converged=True,breadth_above_sma200=.58,is_month_end=True)
    panel=_panel_structures(pd.DataFrame([row]))
    assert panel[(pd.Timestamp('2020-01-31'),0.04)]['orders']['HUMP_PULLBACK'].tolist()==[0]


def test_screen_trade_horizon_ignores_split_quote_drop():
    from run_development_optimizer import _screen_replay, _panel_structures
    dates=pd.bdate_range('2024-05-01',periods=54)
    prices=120.0*(1.002**np.arange(len(dates)))
    split_day=int(np.where(dates==pd.Timestamp('2024-06-12'))[0][0])
    prices[split_day:]/=2.0
    rets=np.full(len(dates),.002)
    rets[0]=np.nan
    h=HistoryStore.from_daily(pd.DataFrame({'date':dates,'permno':[10]*len(dates), 'close':prices,'ret1':rets}))
    date=pd.Timestamp('2024-05-31')
    row=dict(date=date,forecast_stop_pct=0.04,permno=10,ticker='SPLIT',sector='TEST',
      alpha_balanced=.8,alpha_trend_heavy=.8,alpha_pullback_heavy=.8,alpha_hump_pullback=.8,
      alpha_pct_balanced=.95,alpha_pct_trend_heavy=.95,alpha_pct_pullback_heavy=.95,alpha_pct_hump_pullback=.95,
      mc_p_stop=.2,mc_p_stop_se=.01,mc_stop_probability_ucb=.21,risk_volatility=.30,
      mc_mean_se=.0001,mc_converged=True,breadth_above_sma200=.66,is_month_end=True)
    structures=_panel_structures(pd.DataFrame([row]))
    combo={'max_active_positions':5,'daily_close_stop_pct':.04,'alpha_model':'BALANCED',
          'min_alpha_percentile':.8,'max_mc_stop_probability':.4,
          'max_risk_volatility':.5,'min_market_breadth_200':.55}
    eq, tr = _screen_replay(combo, dates, structures, h, cost=0.0, last_signal=date)
    assert len(tr)==1
    assert tr.iloc[0]['exit_reason']=='MODEL_HORIZON'
    assert tr.iloc[0]['net_return']>0
    assert eq.iloc[-1]['equity']>100000
