from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


_SOBOL_BITS = 30
_DATA_PATH = Path(__file__).with_name("sobol_joe_kuo_512.npz")


@dataclass(frozen=True)
class RQMCDesign:
    n_paths: int
    dimension: int
    scramble_id: int
    seed: int


def stable_seed(base_seed: int, context_key: str, scramble_id: int) -> int:
    """Stable 32-bit seed; use forecast-date context for common random numbers."""
    payload = f"{base_seed}|{context_key}|{scramble_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


@lru_cache(maxsize=1)
def _direction_seed_data() -> tuple[np.ndarray, np.ndarray]:
    if not _DATA_PATH.exists():
        raise RuntimeError(f"bundled Sobol direction-number data missing: {_DATA_PATH}")
    with np.load(_DATA_PATH, allow_pickle=False) as z:
        return np.asarray(z["poly"], dtype=np.uint32), np.asarray(z["vinit"], dtype=np.uint32)


@lru_cache(maxsize=32)
def _direction_numbers(dimension: int, bits: int = _SOBOL_BITS) -> np.ndarray:
    poly, vinit = _direction_seed_data()
    if dimension <= 0 or dimension > len(poly):
        raise ValueError(f"Sobol dimension must be in [1,{len(poly)}], got {dimension}")
    if bits < 1 or bits > 32:
        raise ValueError("Sobol bits must be in [1,32]")
    sv = np.zeros((dimension, bits), dtype=np.uint32)
    for i in range(bits):
        sv[0, i] = np.uint32(1 << (bits - 1 - i))
    for j in range(1, dimension):
        p = int(poly[j])
        degree = p.bit_length() - 1
        a = ((p >> 1) & ((1 << (degree - 1)) - 1)) if degree > 1 else 0
        for i in range(1, min(degree, bits) + 1):
            sv[j, i - 1] = np.uint32(int(vinit[j, i - 1]) << (bits - i))
        for i in range(degree + 1, bits + 1):
            value = int(sv[j, i - degree - 1]) ^ (int(sv[j, i - degree - 1]) >> degree)
            for k in range(1, degree):
                if (a >> (degree - 1 - k)) & 1:
                    value ^= int(sv[j, i - k - 1])
            sv[j, i - 1] = np.uint32(value)
    sv.setflags(write=False)
    return sv


def _lms_shift_scramble(sv: np.ndarray, seed: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Left linear-matrix scramble plus independent digital shift over GF(2)."""
    rng = np.random.default_rng(seed)
    dimension = sv.shape[0]
    out = np.empty_like(sv)
    shifts = np.empty(dimension, dtype=np.uint32)
    weights = (np.uint32(1) << np.arange(bits - 1, -1, -1, dtype=np.uint32))

    for d in range(dimension):
        shift_bits = rng.integers(0, 2, size=bits, dtype=np.uint8).astype(np.uint32)
        shifts[d] = np.sum(shift_bits * weights, dtype=np.uint32)

        # Nonsingular lower-triangular matrix: randomized subdiagonal, unit diagonal.
        L = np.tril(rng.integers(0, 2, size=(bits, bits), dtype=np.uint8)).astype(np.uint32)
        np.fill_diagonal(L, 1)
        direction_bits = ((sv[d, :, None] & weights[None, :]) != 0).astype(np.uint32)
        scrambled_bits = (direction_bits @ L.T) & np.uint32(1)
        out[d] = np.sum(scrambled_bits * weights[None, :], axis=1, dtype=np.uint32)
    return out, shifts


def _sobol_base2(power: int, dimension: int, seed: int) -> np.ndarray:
    bits = _SOBOL_BITS
    if power < 1 or power > bits:
        raise ValueError(f"power must be in [1,{bits}]")
    base = _direction_numbers(dimension, bits)
    sv, shift = _lms_shift_scramble(base, seed, bits)
    n = 1 << power
    ints = np.empty((n, dimension), dtype=np.uint32)
    quasi = shift.copy()
    ints[0] = quasi
    for i in range(1, n):
        # Direction index = position of least-significant set bit in i.
        c = (i & -i).bit_length() - 1
        quasi = np.bitwise_xor(quasi, sv[:, c])
        ints[i] = quasi
    return ints.astype(np.float64) / float(1 << bits)


def sobol_uniforms(
    *,
    power: int,
    dimension: int,
    base_seed: int,
    context_key: str,
    scramble_id: int,
    clip_eps: float = 1e-12,
) -> tuple[np.ndarray, RQMCDesign]:
    """Generate one independently randomized Sobol replication, SciPy-free.

    Uses Joe-Kuo direction numbers with an LMS+digital-shift scramble. Power-of-two
    point counts preserve Sobol balance properties. Same context_key/scramble_id
    is deterministic, enabling common random numbers across same-date candidates.
    """
    seed = stable_seed(base_seed, context_key, scramble_id)
    u = _sobol_base2(power, dimension, seed)
    np.clip(u, clip_eps, 1.0 - clip_eps, out=u)
    return u, RQMCDesign(
        n_paths=1 << power,
        dimension=dimension,
        scramble_id=scramble_id,
        seed=seed,
    )
