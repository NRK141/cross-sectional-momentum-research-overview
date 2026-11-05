import pandas as pd
from src.screener.ranking import rank_month_end


def cfg():
    return {
        "screener": {
            "min_12_1_momentum": 0.0,
            "min_6_1_momentum": 0.0,
            "require_close_above_sma100": True,
            "require_close_above_sma200": True,
            "score_weights": {
                "momentum_3_1": 0.20,
                "momentum_6_1": 0.35,
                "momentum_12_1": 0.35,
                "trend_strength": 0.10,
            },
        }
    }


def row(permno, m3, m6, m12, trend, adv=20_000_000, eligible=True):
    return {
        "permno": permno,
        "ticker": f"T{permno}",
        "eligible": eligible,
        "close": 120.0,
        "adv20": adv,
        "sma100": 100.0,
        "sma200": 90.0,
        "momentum_3_1": m3,
        "momentum_6_1": m6,
        "momentum_12_1": m12,
        "trend_strength": trend,
    }


def test_rank_month_end_orders_leaders():
    frame = pd.DataFrame([
        row(1, .10, .20, .30, .20),
        row(2, .20, .30, .40, .30),
        row(3, .05, .10, .15, .10),
    ])
    ranked = rank_month_end(frame, cfg())
    assert ranked.iloc[0].permno == 2
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_rank_month_end_drops_nonfinite_and_ineligible_rows():
    frame = pd.DataFrame([
        row(1, .10, .20, .30, .20),
        row(2, None, .30, .40, .30),
        row(3, .20, .30, .40, .30, eligible=False),
    ])
    ranked = rank_month_end(frame, cfg())
    assert ranked["permno"].tolist() == [1]


def test_rank_month_end_is_deterministic_on_tie():
    frame = pd.DataFrame([
        row(8, .10, .20, .30, .20, adv=20_000_000),
        row(3, .10, .20, .30, .20, adv=20_000_000),
    ])
    ranked = rank_month_end(frame, cfg())
    assert ranked["permno"].tolist() == [3, 8]
