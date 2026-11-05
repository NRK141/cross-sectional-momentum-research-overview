from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CRSPColumns:
    date: str
    permno: str
    ticker: str
    price: str
    ret: str | None
    volume: str | None
    primary_exch: str | None
    security_type: str | None
    security_subtype: str | None
    share_type: str | None
    sic: str | None

CANDIDATES = {
    "date": ["date", "dlycaldt", "caldt"],
    "permno": ["permno"],
    "ticker": ["ticker", "htick", "symbol"],
    "price": ["dlyprc", "prc", "close", "price"],
    "ret": ["dlyret", "ret", "return"],
    "volume": ["dlyvol", "vol", "volume"],
    "primary_exch": ["primaryexch", "primary_exch", "primexch", "exchcd"],
    "security_type": ["securitytype", "security_type", "sectype"],
    "security_subtype": ["securitysubtype", "security_subtype", "secsubtype"],
    "share_type": ["sharetype", "share_type", "shrtype"],
    "sic": ["siccd", "sic", "sic_code"],
}

def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "")

def resolve_columns(headers: list[str]) -> CRSPColumns:
    by_norm = {_norm(h): h for h in headers}
    out = {}
    for key, names in CANDIDATES.items():
        found = None
        for n in names:
            nn = _norm(n)
            if nn in by_norm:
                found = by_norm[nn]
                break
        out[key] = found
    missing = [k for k in ("date","permno","ticker","price") if not out[k]]
    if missing:
        raise ValueError(f"Required CRSP columns not found: {missing}. Headers={headers[:80]}")
    return CRSPColumns(**out)

def sql_ident(name: str) -> str:
    return '"' + name.replace('"','""') + '"'
