from __future__ import annotations

def sic_division(sic) -> str:
    try: s=int(float(sic))
    except Exception: return "Unknown"
    if 100 <= s <= 999: return "Agriculture"
    if 1000 <= s <= 1499: return "Mining"
    if 1500 <= s <= 1799: return "Construction"
    if 2000 <= s <= 3999: return "Manufacturing"
    if 4000 <= s <= 4999: return "Transportation/Utilities"
    if 5000 <= s <= 5199: return "Wholesale"
    if 5200 <= s <= 5999: return "Retail"
    if 6000 <= s <= 6799: return "Finance"
    if 7000 <= s <= 8999: return "Services"
    if 9100 <= s <= 9729: return "Public Administration"
    return "Other"
