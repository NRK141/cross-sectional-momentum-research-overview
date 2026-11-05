from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
import pandas as pd

@dataclass
class Position:
    permno:int; ticker:str; sector:str; entry_date:pd.Timestamp; entry_price:float; shares:float; entry_cost:float; peak_price:float; held_days:int=0


def candidate_ok(r: pd.Series, cfg: dict) -> bool:
    e=cfg['entry'];
    if not bool(r.get('eligible',False)): return False
    if not (float(e['min_5d_pullback']) <= float(r['ret5']) <= float(e['max_5d_pullback'])): return False
    if float(r['rsi2']) > float(e['max_rsi2']): return False
    if not (float(e['min_distance_from_20d_high']) <= float(r['distance_20d_high']) <= float(e['max_distance_from_20d_high'])): return False
    if e.get('require_positive_1d_resumption',True) and float(r['ret1']) <= 0: return False
    if e.get('require_close_above_sma50',True) and float(r['close']) <= float(r['sma50']): return False
    return True

def simulate(daily: pd.DataFrame, rankings: pd.DataFrame, cfg: dict):
    p=cfg['portfolio']; ex=cfg['exit']; exec_cfg=cfg['execution']; cost=float(exec_cfg['transaction_cost_bps_per_side'])/10000.0
    cash=float(p['initial_capital']); positions={}; pending=[]; last_exit={}; trades=[]; events=[]; equity=[]
    ranking_map={m:set(g.permno.astype(int)) for m,g in rankings.groupby('effective_month')}
    rank_score={(m,int(r.permno)):float(r.leadership_score) for m,g in rankings.groupby('effective_month') for _,r in g.iterrows()}
    daily=daily.sort_values(['date','permno']).copy(); daily['date']=pd.to_datetime(daily['date'])
    for date,g in daily.groupby('date',sort=True):
        date=pd.Timestamp(date); g=g.set_index('permno',drop=False); month=date.to_period('M').strftime('%Y-%m')
        # Execute previously scheduled next-close orders first, respecting current portfolio state.
        for order in pending:
            perm=int(order['permno'])
            if perm in positions or perm not in g.index: continue
            if len(positions)>=int(p['max_active_positions']): continue
            row=g.loc[perm]; price=float(row['close']);
            if not math.isfinite(price) or price<=0: continue
            eq=cash+sum(pos.shares*float(g.loc[k]['close']) if k in g.index else pos.shares*pos.entry_price for k,pos in positions.items())
            sector=str(row.get('sector','Other'))
            current_sector=sum(pos.shares*float(g.loc[k]['close']) for k,pos in positions.items() if k in g.index and pos.sector==sector)
            w=min(float(p['base_position_weight']),float(p['max_position_weight']))
            target=eq*w; gross=sum(pos.shares*float(g.loc[k]['close']) for k,pos in positions.items() if k in g.index)
            room=max(0.0,eq*float(p['max_gross_exposure'])-gross); sector_room=max(0.0,eq*float(p['max_sector_weight'])-current_sector); notional=min(target,room,sector_room,cash/(1+cost))
            if notional<=0: continue
            shares=notional/price; fee=notional*cost; cash-=notional+fee
            positions[perm]=Position(perm,str(row['ticker']),sector,date,price,shares,fee,price,0)
            events.append({'date':date,'ticker':str(row['ticker']),'permno':perm,'event':'ENTRY','price':price,'leadership_score':order['score']})
        pending=[]
        # Update/exit held positions using current close.
        for perm,pos in list(positions.items()):
            if perm not in g.index: continue
            row=g.loc[perm]; price=float(row['close']); pos.held_days+=1; pos.peak_price=max(pos.peak_price,price)
            reason=None
            if price <= pos.entry_price*(1-float(ex['initial_stop_pct'])): reason='STOP'
            elif pos.peak_price >= pos.entry_price*(1+float(ex['trailing_activates_at_gain'])) and price <= pos.peak_price*(1-float(ex['trailing_stop_pct_from_peak'])): reason='TRAIL'
            elif pos.held_days>=int(ex['minimum_holding_days_before_trend_exit']) and price < float(row['sma20']): reason='TREND_BREAK'
            elif pos.held_days>=int(ex['maximum_holding_days']): reason='MAX_HOLD'
            if reason:
                proceeds=pos.shares*price; fee=proceeds*cost; cash+=proceeds-fee; gross=(price/pos.entry_price)-1; net=(proceeds-fee-(pos.shares*pos.entry_price+pos.entry_cost))/(pos.shares*pos.entry_price+pos.entry_cost)
                trades.append({'ticker':pos.ticker,'permno':perm,'sector':pos.sector,'entry_date':pos.entry_date,'exit_date':date,'entry_price':pos.entry_price,'exit_price':price,'held_days':pos.held_days,'gross_return':gross,'net_return':net,'exit_reason':reason})
                events.append({'date':date,'ticker':pos.ticker,'permno':perm,'event':'EXIT','price':price,'reason':reason}); del positions[perm]; last_exit[perm]=date
        # Create signals on today's close for execution next close.
        selected=ranking_map.get(month,set()); candidates=[]
        for perm in selected:
            if perm in positions or perm not in g.index: continue
            if perm in last_exit and (date-last_exit[perm]).days < int(cfg['entry']['candidate_cooldown_days']): continue
            row=g.loc[perm]
            try: ok=candidate_ok(row,cfg)
            except Exception: ok=False
            if ok: candidates.append((rank_score.get((month,perm),0.0),perm,str(row['ticker'])))
        candidates.sort(reverse=True)
        for score,perm,ticker in candidates[:int(cfg['entry']['max_new_entries_per_day'])]:
            pending.append({'permno':perm,'ticker':ticker,'score':score,'signal_date':date})
            events.append({'date':date,'ticker':ticker,'permno':perm,'event':'SIGNAL','leadership_score':score})
        # Mark to market.
        open_value=0.0
        for perm,pos in positions.items():
            px=float(g.loc[perm]['close']) if perm in g.index else pos.entry_price
            open_value += pos.shares*px
        equity.append({'date':date,'equity':cash+open_value,'cash':cash,'gross_exposure':open_value/(cash+open_value) if cash+open_value>0 else 0.0,'positions':len(positions)})
    # Liquidate terminal positions at last known close for clean metrics.
    if len(daily):
        last_date=pd.Timestamp(daily['date'].max()); glast=daily[daily['date']==last_date].set_index('permno',drop=False)
        for perm,pos in list(positions.items()):
            if perm not in glast.index: continue
            price=float(glast.loc[perm]['close']); proceeds=pos.shares*price; fee=proceeds*cost; cash+=proceeds-fee; net=(proceeds-fee-(pos.shares*pos.entry_price+pos.entry_cost))/(pos.shares*pos.entry_price+pos.entry_cost)
            trades.append({'ticker':pos.ticker,'permno':perm,'sector':pos.sector,'entry_date':pos.entry_date,'exit_date':last_date,'entry_price':pos.entry_price,'exit_price':price,'held_days':pos.held_days,'gross_return':price/pos.entry_price-1,'net_return':net,'exit_reason':'TERMINAL'})
    return pd.DataFrame(equity),pd.DataFrame(trades),pd.DataFrame(events)
