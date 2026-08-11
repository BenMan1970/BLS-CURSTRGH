"""
Bluestar Market Dashboard — Strength Engine v10.1 (UI Refresh).

Zero-regression sur le moteur : identique à v10.0 / v4.4 pour des payloads OANDA identiques.
Nouveautés v10.1 : design system unifié (typographie, palette, espacements), cartes devises
redessinées, Market Map heatmap dark-theme, briefing PDF épuré, hiérarchie visuelle revue.

Dependencies: streamlit, oandapyV20, pandas, numpy.
"""
# app.py — Bluestar Market Dashboard (Strength Engine v10.1)
# Moteur inchangé. Refonte visuelle complète : design system, cartes, map, briefing.

from __future__ import annotations

import datetime
import hashlib
import html
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from oandapyV20 import API
from oandapyV20.endpoints import instruments
from oandapyV20.exceptions import V20Error


# ==========================================
# ── CONFIGURATION ─────────────────────────
# ==========================================

MIN_STRENGTH_DIFF: float = 1.5
ATR_MIN_PERCENTILE: int = 25
MAX_PAIRS: int = 3
MAX_CURRENCY_EXPOSURE: int = 1
MIN_RAW_SPREAD: float = 0.15
HTTP_TIMEOUT: float = 8.0

# Market Map smoothing: 1 = legacy exact (single-tick), 3+ = anti-flicker
MAP_SMOOTH_WINDOW: int = 1

logger = logging.getLogger(__name__)


# ==========================================
# ── DESIGN TOKENS ─────────────────────────
# ==========================================

class T:
    """Design tokens — source unique de vérité pour la palette."""
    BG          = "#0A0C10"
    BG_ELEV     = "#11151C"
    SURFACE     = "#141A23"
    SURFACE_2   = "#1A212C"
    BORDER      = "#232C39"
    BORDER_SOFT = "#1C242F"

    TEXT        = "#E6EAF2"
    TEXT_DIM    = "#9AA6B8"
    TEXT_MUTE   = "#657084"

    ACCENT      = "#4C8DFF"
    ACCENT_DIM  = "#2F6BD8"

    UP          = "#10B981"
    UP_SOFT     = "#34D399"
    DOWN        = "#F43F5E"
    DOWN_SOFT   = "#FB7185"
    WARN        = "#F59E0B"
    NEUTRAL     = "#94A3B8"


# ==========================================
# ── EXCEPTION TAXONOMY ────────────────────
# ==========================================

class BluestarError(Exception):
    """Base for all engine/adapter failures."""


class BluestarAuthError(BluestarError):
    """401/403 — credentials invalid. No retry."""


class BluestarRateLimit(BluestarError):
    """429 — retry with exponential backoff + jitter."""


class BluestarTimeout(BluestarError):
    """Network timeout — single retry then fail-open."""


class BluestarDataError(BluestarError):
    """Malformed payload / schema violation — fail fast."""


# ==========================================
# ── CONSTANTS ─────────────────────────────
# ==========================================

PAIRS: List[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
]

CURRENCIES: List[str] = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]

TIMEFRAMES_MTF: Dict[str, dict] = {
    "W":  {"gran_fetch": "D",  "count": 2000, "weight": 4.0, "resample_rule": "W-FRI"},
    "D":  {"gran_fetch": "D",  "count": 300,  "weight": 4.0, "resample_rule": None},
    "H4": {"gran_fetch": "H4", "count": 300,  "weight": 2.5, "resample_rule": None},
    "H1": {"gran_fetch": "H1", "count": 300,  "weight": 1.5, "resample_rule": None},
}


# ==========================================
# ── OUTILS RÉSEAU & VALIDATION ────────────
# ==========================================

def _create_client(access_token: str, environment: str) -> API:
    """Crée un client OANDA v20 (pattern v4.4, sans injection de timeout)."""
    return API(access_token=access_token, environment=environment)


def validate_ohlcv(df: pd.DataFrame, min_len: int = 20) -> None:
    """Valide la structure et le contenu d'un DataFrame OHLCV."""
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index non trié")
    for col in required:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Valeurs non finies dans {col}")
    if len(df) < min_len:
        raise ValueError(f"Longueur insuffisante: {len(df)} < {min_len}")


def token_fingerprint(access_token: str) -> str:
    """Empreinte non réversible du token pour isolation des caches."""
    return hashlib.sha256(access_token.encode()).hexdigest()[:16]


# ==========================================
# ── OANDA CLIENT WITH RESILIENCE ──────────
# ==========================================

class OandaClient:
    """Client OANDA avec taxonomie d'erreurs typée et retry 429."""

    def __init__(self, api: API) -> None:
        self._api = api

    def request(self, endpoint, timeout: float = HTTP_TIMEOUT):
        """Wrapper with retry logic for 429 rate limits."""
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                return self._api.request(endpoint)
            except V20Error as exc:
                code = getattr(exc, "code", None)
                if code == 429:
                    if attempt < 2:
                        sleep_s = (2 ** attempt) + random.uniform(0, 0.5)  # nosec B311
                        logger.warning(
                            "OANDA 429 retry %s/%s: sleep %.2fs",
                            attempt + 1, 3, sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    raise BluestarRateLimit(
                        f"OANDA 429 après 3 tentatives: {exc}"
                    ) from exc
                if code in (401, 403):
                    raise BluestarAuthError(f"OANDA auth {code}: {exc}") from exc
                raise BluestarDataError(f"OANDA error {code}: {exc}") from exc
            except (TimeoutError, ConnectionError) as exc:
                if attempt < 2:
                    logger.warning("OANDA timeout retry %s/%s: %s", attempt + 1, 3, exc)
                    time.sleep(1.0)
                    continue
                raise BluestarTimeout(str(exc)) from exc
        raise BluestarError(f"Unexpected failure after retries: {last_err}")


# ==========================================
# ── STRENGTH RESULT ───────────────────────
# ==========================================

@dataclass
class StrengthResult:
    """Résultat complet du calcul de force des devises."""
    scores:         Dict[str, float] = field(default_factory=dict)
    scores_display: Dict[str, float] = field(default_factory=dict)
    ranking:        List[str]        = field(default_factory=list)
    velocity:       Dict[str, float] = field(default_factory=dict)
    best_pairs:     List[str]        = field(default_factory=list)
    pairs_detail:   List[Dict]       = field(default_factory=list)
    pairs_fetched:  int              = 0
    coverage:       Dict[str, float] = field(default_factory=dict)
    warnings:       List[str]        = field(default_factory=list)
    valid:          bool             = True

    def to_dict(self) -> dict:
        """Exporte le résultat sous forme de dictionnaire."""
        return {
            "scores":         self.scores,
            "scores_display": self.scores_display,
            "ranking":        self.ranking,
            "velocity":       self.velocity,
            "best_pairs":     self.best_pairs,
            "pairs_detail":   self.pairs_detail,
            "pairs_fetched":  self.pairs_fetched,
            "coverage":       self.coverage,
            "warnings":       self.warnings,
            "valid":          self.valid,
        }

    def direction_arrow(self, currency: str) -> str:
        """Retourne la flèche directionnelle pour une devise."""
        v = self.velocity.get(currency, 0.0)
        if v > 0.02:
            return "up"
        if v < -0.02:
            return "down"
        return "flat"

    def color_class(self, currency: str) -> str:
        """Retourne la classe CSS pour une devise."""
        s = self.scores_display.get(currency, 5.0)
        if s >= 7.0:
            return "strong-bull"
        if s >= 5.5:
            return "mild-bull"
        if s >= 4.0:
            return "mild-bear"
        return "strong-bear"

    def health_check(self) -> dict:
        """Health status for observability."""
        if not self.valid:
            return {
                "status": "degraded",
                "coverage_min": 0.0,
                "warnings": self.warnings,
            }
        cov_min = min(self.coverage.values()) if self.coverage else 0.0
        status_str = "ok" if (cov_min >= 0.5 and not self.warnings) else "degraded"
        return {
            "status": status_str,
            "coverage_min": round(cov_min, 4),
            "warnings": self.warnings,
        }


# ── Fonctions techniques pures ────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Moyenne mobile exponentielle."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple."""
    return series.rolling(window=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (série complète)."""
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _dmi(df: pd.DataFrame, period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Directional Movement Index (pdi, mdi)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    up    = high.diff()
    down  = -low.diff()
    pdm   = up.where((up > down) & (up > 0), 0.0)
    mdm   = down.where((down > up) & (down > 0), 0.0)
    pdi   = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi   = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    pdi_val = float(pdi.iloc[-1])
    mdi_val = float(mdi.iloc[-1])
    if not np.isfinite(pdi_val) or not np.isfinite(mdi_val):
        return None, None
    return pdi_val, mdi_val


# ── Fonctions de tendance ────────────────────────────────────────────────────

def trend_weekly(df: pd.DataFrame) -> Tuple[str, int]:
    """Tendance weekly basée sur EMA50 / SMA200."""
    if len(df) < 200:
        return "Range", 0
    close  = df["Close"]
    ema50  = _ema(close, 50)
    sma200 = _sma(close, 200)
    curr_ema50,  prev_ema50  = ema50.iloc[-1],  ema50.iloc[-2]
    curr_sma200, prev_sma200 = sma200.iloc[-1], sma200.iloc[-2]
    crossed_bull = (prev_ema50 <= prev_sma200) and (curr_ema50 > curr_sma200)
    crossed_bear = (prev_ema50 >= prev_sma200) and (curr_ema50 < curr_sma200)
    if curr_ema50 > curr_sma200:
        return "Bullish", 90 if crossed_bull else 75
    if curr_ema50 < curr_sma200:
        return "Bearish", 90 if crossed_bear else 75
    return "Range", 40


def _swing_points(series: pd.Series, wing: int = 5) -> Tuple[List[int], List[int]]:
    """Détecte les points pivots (swing highs/lows)."""
    arr = series.to_numpy()
    n   = len(arr)
    highs, lows = [], []
    for idx in range(wing, n - wing):
        seg = arr[idx - wing: idx + wing + 1]
        if arr[idx] >= seg.max() and arr[idx] > arr[idx - 1]:
            highs.append(idx)
        if arr[idx] <= seg.min() and arr[idx] < arr[idx - 1]:
            lows.append(idx)
    return highs, lows


def _evaluate_weekly_open(df: pd.DataFrame, current_price: float) -> int:
    """Évalue la position par rapport à l'open hebdomadaire (lundi)."""
    try:
        times       = pd.to_datetime(df.index)
        monday_rows = df[times.dayofweek == 0]
        if not monday_rows.empty:
            weekly_open = float(monday_rows["Open"].iloc[-1])
            return 1 if current_price > weekly_open else -1
    except (KeyError, IndexError, ValueError, TypeError):
        logger.debug("trend_daily: weekly_open indisponible", exc_info=True)
    return 0


# ── Sous-fonctions pour trend_daily ─────────────────────────────────────────

def _swing_votes(high, low, sh_idx, sl_idx):
    """Comptabilise les votes swing (structure)."""
    votes_bull = votes_bear = 0
    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        hh = high.iloc[sh_idx[-1]] > high.iloc[sh_idx[-2]]
        hl = low.iloc[sl_idx[-1]]  > low.iloc[sl_idx[-2]]
        lh = high.iloc[sh_idx[-1]] < high.iloc[sh_idx[-2]]
        ll = low.iloc[sl_idx[-1]]  < low.iloc[sl_idx[-2]]
        if hh and hl:
            votes_bull += 2
        elif lh and ll:
            votes_bear += 2
    return votes_bull, votes_bear


def _ema_votes(close, cur):
    """Votes basés sur l'alignement EMA21/EMA50."""
    votes_bull = votes_bear = 0
    ema21 = _ema(close, 21).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    if cur > ema21 > ema50:
        votes_bull += 1
    elif cur < ema21 < ema50:
        votes_bear += 1
    return votes_bull, votes_bear


def _midpoint_votes(df, close):
    """Vote basé sur la position par rapport au midpoint de la bougie précédente."""
    if len(df) < 2:
        return 0, 0
    high = df["High"]
    low  = df["Low"]
    midpoint = (float(high.iloc[-2]) + float(low.iloc[-2])) / 2
    if float(close.iloc[-2]) > midpoint:
        return 1, 0
    return 0, 1


def _sma200_votes(close, cur):
    """Vote basé sur la position par rapport à la SMA200."""
    if len(close) < 200:
        return 0, 0
    sma200_val = _sma(close, 200).iloc[-1]
    if cur > sma200_val:
        return 1, 0
    if cur < sma200_val:
        return 0, 1
    return 0, 0


def trend_daily(df: pd.DataFrame) -> Tuple[str, int]:
    """Tendance daily multi-critères."""
    if len(df) < 60:
        return "Range", 0
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    cur   = float(close.iloc[-1])
    votes_bull = votes_bear = 0

    sh_idx, _  = _swing_points(high)
    _, sl_idx  = _swing_points(low)
    vb, vbe = _swing_votes(high, low, sh_idx, sl_idx)
    votes_bull += vb
    votes_bear += vbe

    vb, vbe = _ema_votes(close, cur)
    votes_bull += vb
    votes_bear += vbe

    wo_vote = _evaluate_weekly_open(df, cur)
    if wo_vote > 0:
        votes_bull += 1
    elif wo_vote < 0:
        votes_bear += 1

    vb, vbe = _midpoint_votes(df, close)
    votes_bull += vb
    votes_bear += vbe

    vb, vbe = _sma200_votes(close, cur)
    votes_bull += vb
    votes_bear += vbe

    if votes_bull >= 5:
        return "Bullish", 90
    if votes_bull >= 3:
        return "Bullish", 70
    if votes_bear >= 5:
        return "Bearish", 90
    if votes_bear >= 3:
        return "Bearish", 70
    return "Range", 35


def _trend_4h_dmi_vote(pdi_val, mdi_val):
    """Vote DMI pour la tendance H4."""
    if pdi_val is None or mdi_val is None:
        return 0
    if pdi_val > mdi_val:
        return 1
    if pdi_val < mdi_val:
        return -1
    return 0


def trend_4h(df: pd.DataFrame) -> Tuple[str, int]:
    """Tendance H4 avec DMI et daily open."""
    if len(df) < 60:
        return "Range", 0
    close = df["Close"]
    cur   = float(close.iloc[-1])
    score = 0
    score += 1 if cur > _ema(close, 50).iloc[-1] else -1

    pdi_val, mdi_val = _dmi(df)
    score += _trend_4h_dmi_vote(pdi_val, mdi_val)

    try:
        idx        = pd.to_datetime(df.index)
        dates      = idx.normalize()
        today_mask = dates == dates[-1]
        today_rows = df[today_mask]
        if not today_rows.empty:
            daily_open = float(today_rows["Open"].iloc[0])
            score += 1 if cur > daily_open else -1
        else:
            logger.debug("trend_4h: today_mask vide pour %s", df.index[-1])
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        logger.debug("trend_4h daily_open error: %s", exc)

    abs_score = abs(score)
    if abs_score == 3:
        strength = 90
    elif abs_score >= 1:
        strength = 70
    else:
        strength = 40

    if score > 0:
        trend = "Bullish"
    elif score < 0:
        trend = "Bearish"
    else:
        trend = "Range"
    return trend, strength


def _compute_h1_strength(cur, curr_zlema, ema9, ema21, ema50, rsi_val, macd_line, close):
    """Détermine la force H1 selon les critères ZLEMA/EMA/Momentum."""
    curr_macd = macd_line.iloc[-1]
    curr_sig  = _ema(macd_line, 9).iloc[-1]
    ema_bull  = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema21.iloc[-1] > ema50.iloc[-1])
    ema_bear  = (ema9.iloc[-1] < ema21.iloc[-1]) and (ema21.iloc[-1] < ema50.iloc[-1])
    mom_bull  = (rsi_val > 50) and (curr_macd > curr_sig)
    mom_bear  = (rsi_val < 50) and (curr_macd < curr_sig)

    if (cur > curr_zlema) and ema_bull and mom_bull:
        base_s = max(25, min(75, abs(cur - curr_zlema) / cur * 1000))
        return "Bullish", int(round(base_s))
    if (cur < curr_zlema) and ema_bear and mom_bear:
        base_s = max(25, min(75, abs(cur - curr_zlema) / cur * 1000))
        return "Bearish", int(round(base_s))
    if len(close) >= 200:
        sma200_val = _sma(close, 200).iloc[-1]
        bias_trend = "Bullish" if ema50.iloc[-1] > sma200_val else "Bearish"
        if cur < sma200_val and bias_trend == "Bullish":
            return "Retracement Bull", 30
        if cur > sma200_val and bias_trend == "Bearish":
            return "Retracement Bear", 30
    return "Range", 25


def trend_h1(df: pd.DataFrame) -> Tuple[str, int]:
    """Tendance H1 avec ZLEMA, EMA et momentum."""
    if len(df) < 50:
        return "Range", 0
    close      = df["Close"]
    cur        = float(close.iloc[-1])
    ema9       = _ema(close, 9)
    ema21      = _ema(close, 21)
    ema50      = _ema(close, 50)
    lag        = 17
    src_adj    = close + (close - close.shift(lag))
    curr_zlema = _ema(src_adj, 50).iloc[-1]
    rsi_val    = _rsi(close, 14).iloc[-1]
    macd_line  = _ema(close, 12) - _ema(close, 26)
    return _compute_h1_strength(
        cur, curr_zlema, ema9, ema21, ema50, rsi_val, macd_line, close
    )


_TREND_FN = {
    "W":  trend_weekly,
    "D":  trend_daily,
    "H4": trend_4h,
    "H1": trend_h1,
}


# ── Aide à la sélection ─────────────────────────────────────────────────────

def _get_pair_id(base: str, quote: str) -> Optional[str]:
    """Retourne l'identifiant OANDA de la paire (direct ou inverse)."""
    direct = f"{base}_{quote}"
    if direct in PAIRS:
        return direct
    inverse = f"{quote}_{base}"
    if inverse in PAIRS:
        return inverse
    return None


def _compute_atr_pct(df_h1: Optional[pd.DataFrame]) -> Optional[float]:
    """Calcule l'ATR en pourcentage du prix."""
    if df_h1 is None or len(df_h1) < 15:
        return None
    atr_abs = float(_atr_series(df_h1).iloc[-1])
    close   = float(df_h1["Close"].iloc[-1])
    if close <= 0:
        return None
    return round((atr_abs / close) * 100, 4)


def _build_candidates(
    strongest: List[str],
    weakest: List[str],
    scores_display: Dict[str, float],
    min_diff: float,
    fetch_ohlcv_fn,
) -> List[Dict]:
    """Construit la liste brute des paires candidates."""
    candidates = []
    for base in strongest:
        for quote in weakest:
            if base == quote:
                continue
            diff = scores_display[base] - scores_display[quote]
            if diff < min_diff:
                continue
            pair_id = _get_pair_id(base, quote)
            if pair_id is None:
                continue

            df_h1 = fetch_ohlcv_fn(pair_id, "H1", 300)
            atr_pct = _compute_atr_pct(df_h1)
            direction = "BUY" if pair_id.startswith(base) else "SELL"
            candidates.append({
                "pair":       f"{base}_{quote}",
                "exec_pair":  pair_id,
                "diff":       round(diff, 3),
                "atr":        atr_pct,
                "base":       base,
                "quote":      quote,
                "direction":  direction,
            })
    return candidates


def _filter_by_atr_and_exposure(
    candidates: List[Dict],
    max_pairs: int,
) -> Tuple[List[str], List[Dict]]:
    """Filtre les candidats sur l'ATR puis limite l'exposition par devise."""
    if not candidates:
        return [], []

    atr_values = [c["atr"] for c in candidates if c["atr"] is not None]
    if atr_values:
        threshold = float(np.percentile(atr_values, ATR_MIN_PERCENTILE))
        candidates = [c for c in candidates if c["atr"] is not None and c["atr"] >= threshold]
    if not candidates:
        return [], []

    used_currencies: set = set()
    filtered = []
    for c in sorted(candidates, key=lambda x: x["diff"], reverse=True):
        if c["base"] in used_currencies or c["quote"] in used_currencies:
            continue
        filtered.append(c)
        used_currencies.update([c["base"], c["quote"]])
    top = filtered[:max_pairs]
    return [c["exec_pair"] for c in top], top


# ==========================================
# ── STRENGTH ENGINE ────────────────────────
# ==========================================

class StrengthEngine:
    """
    Calcule la force relative des 8 devises majeures (W/D/H4/H1).
    Sémantique numérique identique à v4.4. Backward-compatible.
    """

    def __init__(
        self,
        client: API,
        min_diff: float = MIN_STRENGTH_DIFF,
        max_pairs: int  = MAX_PAIRS,
    ):
        self.api       = OandaClient(client)
        self.min_diff  = min_diff
        self.max_pairs = max_pairs
        self._cache: Dict[tuple, pd.DataFrame] = {}
        self.errors: List[str] = []

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _fetch_ohlcv(
        self, pair: str, granularity: str, count: int
    ) -> Optional[pd.DataFrame]:
        """Récupère les chandeliers OANDA avec cache complet."""
        key = (pair, granularity, count, "M")
        if key in self._cache:
            return self._cache[key].copy(deep=False)
        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            self.api.request(r)
            rows = [
                {
                    "Time":  c["time"],
                    "Open":  float(c["mid"]["o"]),
                    "High":  float(c["mid"]["h"]),
                    "Low":   float(c["mid"]["l"]),
                    "Close": float(c["mid"]["c"]),
                }
                for c in r.response["candles"] if c["complete"]
            ]
            if len(rows) < 20:
                return None
            df = pd.DataFrame(rows)
            df["Time"] = pd.to_datetime(df["Time"])
            df.set_index("Time", inplace=True)
            validate_ohlcv(df, min_len=20)
            self._cache[key] = df
            return df
        except BluestarError as exc:
            logger.warning(
                "Fetch OHLCV failed %s %s %d: %s (%s)",
                pair, granularity, count, type(exc).__name__, exc,
            )
            self.errors.append(f"{pair}/{granularity}/{count}: {exc}")
            return None

    def _get_tf_df(self, pair: str, tf: str) -> Optional[pd.DataFrame]:
        """Récupère le DataFrame pour un timeframe donné."""
        cfg = TIMEFRAMES_MTF[tf]
        df  = self._fetch_ohlcv(pair, cfg["gran_fetch"], cfg["count"])
        if df is None:
            return None
        if cfg["resample_rule"]:
            df = (
                df.resample(cfg["resample_rule"])
                  .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
                  .dropna()
            )
            if len(df) < 20:
                return None
        return df

    # ── Scores MTF ────────────────────────────────────────────────────────────

    def _compute_mtf_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calcule les scores bruts multi-timeframe."""
        total:      Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight_sum: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        for pair in PAIRS:
            base, quote = pair.split("_")
            for tf, cfg in TIMEFRAMES_MTF.items():
                df = self._get_tf_df(pair, tf)
                if df is None:
                    continue
                trend, strength = _TREND_FN[tf](df)
                weight = cfg["weight"]
                weight_sum[base]  += weight
                weight_sum[quote] += weight

                if trend == "Bullish":
                    contrib = +weight * (strength / 100)
                elif trend == "Bearish":
                    contrib = -weight * (strength / 100)
                elif trend == "Retracement Bull":
                    contrib = +weight * 0.15
                elif trend == "Retracement Bear":
                    contrib = -weight * 0.15
                else:
                    contrib = 0.0

                total[base]  += contrib
                total[quote] -= contrib
        return total, weight_sum

    @staticmethod
    def _normalize(
        total:      Dict[str, float],
        weight_sum: Dict[str, float],
    ) -> Dict[str, float]:
        """Normalise les scores bruts par les poids."""
        scores = {}
        for c in CURRENCIES:
            if weight_sum.get(c, 0.0) > 0:
                scores[c] = total[c] / weight_sum[c]
            else:
                scores[c] = 0.0
                logger.warning("Devise %s : aucune donnée reçue.", c)
        return scores

    @staticmethod
    def _to_display(scores: Dict[str, float]) -> Dict[str, float]:
        """Convertit les scores bruts en échelle 0-10."""
        values = list(scores.values())
        s_min, s_max = min(values), max(values)
        spread = s_max - s_min
        if spread < MIN_RAW_SPREAD:
            center = (s_min + s_max) / 2
            return {c: round(5.0 + (v - center) * 2, 2) for c, v in scores.items()}
        return {c: round((v - s_min) / spread * 10, 2) for c, v in scores.items()}

    # ── Vélocité (H1 pure) ────────────────────────────────────────────────────

    def _compute_velocity(self) -> Dict[str, float]:
        """Calcule la vélocité sur deux fenêtres H1 de 48 barres."""
        total_now:  Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        total_prev: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight_sum: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight = TIMEFRAMES_MTF["H1"]["weight"]
        for pair in PAIRS:
            base, quote = pair.split("_")
            df = self._fetch_ohlcv(pair, "H1", 300)
            if df is None or len(df) < 96:
                continue
            df_now  = df.iloc[-48:]
            df_prev = df.iloc[-96:-48]
            trend_now, strength_now = trend_h1(df_now)
            trend_prev, strength_prev = trend_h1(df_prev)

            if trend_now != "Range":
                contrib_now = weight * (strength_now / 100)
                contrib_now *= 1 if "Bull" in trend_now else -1
                total_now[base]  += contrib_now
                total_now[quote] -= contrib_now
            if trend_prev != "Range":
                contrib_prev = weight * (strength_prev / 100)
                contrib_prev *= 1 if "Bull" in trend_prev else -1
                total_prev[base]  += contrib_prev
                total_prev[quote] -= contrib_prev

            weight_sum[base]  += weight
            weight_sum[quote] += weight

        scores_now = self._normalize(total_now, weight_sum)
        scores_prev = self._normalize(total_prev, weight_sum)
        return {
            c: round(scores_now.get(c, 0.0) - scores_prev.get(c, 0.0), 4)
            for c in CURRENCIES
        }

    # ── Sélection des paires ──────────────────────────────────────────────────

    def _select_pairs(
        self, scores_display: Dict[str, float]
    ) -> Tuple[List[str], List[Dict]]:
        """Sélectionne les meilleures paires selon les forces relatives."""
        sorted_s  = sorted(scores_display.items(), key=lambda x: x[1], reverse=True)
        strongest = [c for c, _ in sorted_s[:2]]
        weakest   = [c for c, _ in sorted_s[-2:]]
        candidates = _build_candidates(
            strongest, weakest, scores_display, self.min_diff, self._fetch_ohlcv
        )
        return _filter_by_atr_and_exposure(candidates, self.max_pairs)

    # ── Points d'entrée publics ───────────────────────────────────────────────

    def run(self) -> StrengthResult:
        """Exécute le calcul complet multi-timeframe."""
        t0 = time.perf_counter()
        self._cache.clear()
        self.errors.clear()
        total, weight_sum  = self._compute_mtf_scores()
        if all(ws == 0 for ws in weight_sum.values()):
            return StrengthResult(
                valid=False,
                warnings=["Aucune donnée marché reçue. Vérifiez la connexion / token."]
            )
        scores             = self._normalize(total, weight_sum)
        scores_display     = self._to_display(scores)
        ranking            = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        velocity           = self._compute_velocity()
        best_pairs, pairs_detail = self._select_pairs(scores_display)

        total_weight = sum(cfg["weight"] for cfg in TIMEFRAMES_MTF.values())
        coverage = {c: weight_sum[c] / total_weight for c in CURRENCIES}
        warnings = []
        if self.errors:
            warnings.append(f"{len(self.errors)} erreur(s) API (voir logs).")
        min_cov = min(coverage.values()) if coverage else 0
        if min_cov < 0.5:
            warnings.append("Couverture de données faible, signaux dégradés.")

        pairs_fetched = len(self._cache)
        logger.info(
            "engine.run.completed: duration_ms=%.2f pairs_fetched=%d errors=%d min_coverage=%.4f",
            (time.perf_counter() - t0) * 1000, pairs_fetched, len(self.errors), min_cov,
        )

        return StrengthResult(
            scores         = {k: round(v, 6) for k, v in scores.items()},
            scores_display = scores_display,
            ranking        = ranking,
            velocity       = velocity,
            best_pairs     = best_pairs,
            pairs_detail   = pairs_detail,
            pairs_fetched  = pairs_fetched,
            coverage       = coverage,
            warnings       = warnings,
            valid          = True,
        )

    def run_quick(self, granularity: str = "H1") -> StrengthResult:
        """Version rapide mono-timeframe (conservée pour compatibilité)."""
        self._cache.clear()
        self.errors.clear()
        total:      Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight_sum: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        tf     = "H1" if granularity in ("H1", "M30", "M15", "M5") else "H4"
        cfg    = TIMEFRAMES_MTF[tf]
        weight = cfg["weight"]
        for pair in PAIRS:
            base, quote = pair.split("_")
            df = self._fetch_ohlcv(pair, cfg["gran_fetch"], cfg["count"])
            if df is None:
                continue
            trend, strength = _TREND_FN[tf](df)
            weight_sum[base]  += weight
            weight_sum[quote] += weight

            if trend == "Bullish":
                contrib = +weight * (strength / 100)
            elif trend == "Bearish":
                contrib = -weight * (strength / 100)
            else:
                contrib = 0.0

            total[base]  += contrib
            total[quote] -= contrib

        if tf != "H1":
            cfg_h1 = TIMEFRAMES_MTF["H1"]
            for pair in PAIRS:
                self._fetch_ohlcv(pair, cfg_h1["gran_fetch"], cfg_h1["count"])

        scores         = self._normalize(total, weight_sum)
        scores_display = self._to_display(scores)
        ranking        = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        best_pairs, pairs_detail = self._select_pairs(scores_display)
        return StrengthResult(
            scores         = {k: round(v, 6) for k, v in scores.items()},
            scores_display = scores_display,
            ranking        = ranking,
            velocity       = {c: 0.0 for c in CURRENCIES},
            best_pairs     = best_pairs,
            pairs_detail   = pairs_detail,
            pairs_fetched  = len(self._cache),
            valid          = True,
        )


# ==========================================
# ── DASHBOARD STREAMLIT ────────────────────
# ==========================================

st.set_page_config(
    page_title="Bluestar — FX Institutional Desk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root{
  --bs-bg:#0A0C10; --bs-elev:#11151C; --bs-surface:#141A23; --bs-surface2:#1A212C;
  --bs-border:#232C39; --bs-border-soft:#1C242F;
  --bs-text:#E6EAF2; --bs-dim:#9AA6B8; --bs-mute:#657084;
  --bs-accent:#4C8DFF; --bs-up:#10B981; --bs-down:#F43F5E; --bs-warn:#F59E0B;
  --mono:'JetBrains Mono', ui-monospace, monospace;
  --sans:'Inter', -apple-system, system-ui, sans-serif;
}

/* ── Base ───────────────────────────────────────────── */
.stApp{
  background:
    radial-gradient(1100px 600px at 12% -8%, rgba(76,141,255,.10), transparent 60%),
    radial-gradient(900px 500px at 92% 0%, rgba(16,185,129,.06), transparent 55%),
    var(--bs-bg);
  color:var(--bs-text);
  font-family:var(--sans);
}
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1500px; }
#MainMenu, footer, header{ visibility:hidden; }
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:#26303E; border-radius:6px; }
::-webkit-scrollbar-thumb:hover{ background:#33404F; }

h1,h2,h3,h4{ font-family:var(--sans); color:var(--bs-text); letter-spacing:-.02em; }

/* ── App header ─────────────────────────────────────── */
.bs-header{
  display:flex; align-items:center; justify-content:space-between; gap:20px;
  padding:18px 24px; margin-bottom:18px;
  background:linear-gradient(135deg, rgba(30,40,56,.85), rgba(17,21,28,.92));
  border:1px solid var(--bs-border); border-radius:16px;
  box-shadow:0 12px 40px rgba(0,0,0,.45);
}
.bs-brand{ display:flex; align-items:center; gap:14px; }
.bs-logo{
  width:42px; height:42px; border-radius:12px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  background:linear-gradient(140deg, #4C8DFF, #1E3FA8);
  box-shadow:0 6px 18px rgba(76,141,255,.35);
  font-size:19px; color:#fff;
}
.bs-eyebrow{
  font-family:var(--mono); font-size:9.5px; font-weight:600; letter-spacing:.28em;
  color:var(--bs-accent); text-transform:uppercase;
}
.bs-title{ font-size:21px; font-weight:700; letter-spacing:-.03em; line-height:1.15; }
.bs-sub{ font-family:var(--mono); font-size:10.5px; color:var(--bs-mute); margin-top:2px; }
.bs-headmeta{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end; }

.bs-chip{
  display:inline-flex; align-items:center; gap:7px;
  font-family:var(--mono); font-size:10.5px; font-weight:600; letter-spacing:.06em;
  padding:6px 13px; border-radius:999px;
  background:var(--bs-surface2); border:1px solid var(--bs-border); color:var(--bs-dim);
}
.bs-chip.on   { color:#6EE7B7; border-color:rgba(16,185,129,.35); background:rgba(16,185,129,.10); }
.bs-chip.off  { color:#FDA4AF; border-color:rgba(244,63,94,.35);  background:rgba(244,63,94,.10); }
.bs-chip.neu  { color:#93B4FF; border-color:rgba(76,141,255,.35); background:rgba(76,141,255,.10); }
.bs-dot{ width:6px; height:6px; border-radius:50%; background:currentColor; box-shadow:0 0 8px currentColor; }

/* ── Section titles ─────────────────────────────────── */
.bs-sec{ display:flex; align-items:center; gap:12px; margin:26px 0 14px 0; }
.bs-sec-bar{ width:3px; height:17px; border-radius:2px; background:linear-gradient(180deg,#4C8DFF,#1E3FA8); }
.bs-sec-t{ font-size:12.5px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--bs-text); }
.bs-sec-c{ font-family:var(--mono); font-size:10px; color:var(--bs-mute); letter-spacing:.05em; }
.bs-sec-line{ flex:1; height:1px; background:linear-gradient(90deg,var(--bs-border),transparent); }

/* ── KPI strip ──────────────────────────────────────── */
.bs-kpi{
  background:var(--bs-surface); border:1px solid var(--bs-border-soft);
  border-radius:12px; padding:13px 16px; height:100%;
}
.bs-kpi-l{ font-family:var(--mono); font-size:9px; letter-spacing:.16em; text-transform:uppercase; color:var(--bs-mute); }
.bs-kpi-v{ font-family:var(--mono); font-size:21px; font-weight:700; margin-top:5px; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.bs-kpi-s{ font-family:var(--mono); font-size:9.5px; color:var(--bs-mute); margin-top:2px; }

/* ── Currency cards ─────────────────────────────────── */
.cur-card{
  position:relative; overflow:hidden;
  background:linear-gradient(160deg, var(--bs-surface2) 0%, var(--bs-surface) 100%);
  border:1px solid var(--bs-border-soft); border-radius:14px;
  padding:15px 16px 14px 16px; margin-bottom:12px;
  transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.cur-card:hover{ transform:translateY(-2px); border-color:var(--bs-border); box-shadow:0 14px 32px rgba(0,0,0,.45); }
.cur-card::before{ content:""; position:absolute; top:0; left:0; right:0; height:2px; opacity:.9; }
.cur-card.t-1::before{ background:linear-gradient(90deg,#10B981,rgba(16,185,129,0)); }
.cur-card.t-2::before{ background:linear-gradient(90deg,#4C8DFF,rgba(76,141,255,0)); }
.cur-card.t-3::before{ background:linear-gradient(90deg,#F59E0B,rgba(245,158,11,0)); }
.cur-card.t-4::before{ background:linear-gradient(90deg,#F43F5E,rgba(244,63,94,0)); }

.cur-top{ display:flex; align-items:center; gap:9px; }
.cur-flag{ width:22px; height:16px; border-radius:3px; box-shadow:0 0 0 1px rgba(255,255,255,.08); display:block; }
.cur-code{ font-family:var(--mono); font-size:13px; font-weight:700; letter-spacing:.14em; color:var(--bs-text); }
.cur-rank{
  margin-left:auto; font-family:var(--mono); font-size:9px; font-weight:600;
  color:var(--bs-mute); background:rgba(255,255,255,.04);
  border:1px solid var(--bs-border-soft); border-radius:5px; padding:2px 7px; letter-spacing:.08em;
}
.cur-score{
  display:flex; align-items:baseline; gap:9px; margin:10px 0 2px 0;
  font-family:var(--mono); font-weight:800; font-size:33px; line-height:1;
  letter-spacing:-.035em; font-variant-numeric:tabular-nums;
}
.cur-max{ font-size:12px; font-weight:500; color:var(--bs-mute); letter-spacing:0; }
.cur-vel{
  margin-left:auto; display:inline-flex; align-items:center; gap:4px;
  font-family:var(--mono); font-size:10.5px; font-weight:600;
  padding:3px 8px; border-radius:6px; letter-spacing:.02em;
}
.cur-track{ height:4px; border-radius:99px; background:rgba(255,255,255,.06); overflow:hidden; margin-top:12px; }
.cur-fill{ height:100%; border-radius:99px; transition:width .55s cubic-bezier(.22,1,.36,1); }
.cur-foot{
  display:flex; justify-content:space-between; margin-top:8px;
  font-family:var(--mono); font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--bs-mute);
}

/* ── Pair cards ─────────────────────────────────────── */
.pair-card{
  display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  background:linear-gradient(120deg, var(--bs-surface2), var(--bs-surface));
  border:1px solid var(--bs-border-soft); border-left:3px solid var(--bs-accent);
  border-radius:12px; padding:13px 17px; margin-bottom:9px;
  transition:border-color .18s ease, transform .18s ease;
}
.pair-card:hover{ transform:translateX(2px); }
.pair-card.buy { border-left-color:var(--bs-up); }
.pair-card.sell{ border-left-color:var(--bs-down); }
.pair-name{ font-family:var(--mono); font-size:16px; font-weight:700; letter-spacing:.05em; min-width:110px; }
.pair-tag{
  font-family:var(--mono); font-size:10px; font-weight:700; letter-spacing:.1em;
  padding:4px 12px; border-radius:6px;
}
.pair-tag.buy { color:#6EE7B7; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35); }
.pair-tag.sell{ color:#FDA4AF; background:rgba(244,63,94,.12);  border:1px solid rgba(244,63,94,.35); }
.pair-metric{ font-family:var(--mono); font-size:10.5px; color:var(--bs-mute); letter-spacing:.06em; }
.pair-metric b{ color:var(--bs-text); font-weight:600; }
.pair-empty{
  font-family:var(--mono); font-size:11.5px; color:var(--bs-mute); font-style:italic;
  border:1px dashed var(--bs-border); border-radius:12px; padding:18px; text-align:center;
}

/* ── Legend ─────────────────────────────────────────── */
.bs-legend{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin:2px 0 10px 0; }
.bs-legend span{ font-family:var(--mono); font-size:9.5px; color:var(--bs-mute); display:inline-flex; align-items:center; gap:6px; letter-spacing:.08em; }
.bs-sw{ width:11px; height:11px; border-radius:3px; display:inline-block; }

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0D1117,#0A0C10);
  border-right:1px solid var(--bs-border);
}
section[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }
.sb-brand{
  border:1px solid var(--bs-border); border-radius:12px; padding:13px 15px; margin-bottom:18px;
  background:linear-gradient(140deg, rgba(76,141,255,.10), rgba(20,26,35,.6));
}
.sb-brand-t{ font-family:var(--mono); font-size:13px; font-weight:700; letter-spacing:.2em; color:var(--bs-text); }
.sb-brand-s{ font-family:var(--mono); font-size:9px; color:var(--bs-mute); letter-spacing:.12em; margin-top:3px; text-transform:uppercase; }
.sb-lbl{
  font-family:var(--mono); font-size:9px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--bs-mute); margin:16px 0 6px 0;
}

/* ── Widgets ────────────────────────────────────────── */
div[data-baseweb="select"] > div{
  background:var(--bs-surface) !important; border:1px solid var(--bs-border) !important;
  border-radius:9px !important; font-family:var(--mono) !important; font-size:12px !important;
  color:var(--bs-text) !important;
}
div[data-baseweb="select"] > div:hover{ border-color:var(--bs-accent) !important; }
.stDownloadButton button, .stButton button{
  width:100%; background:var(--bs-surface2) !important; color:var(--bs-text) !important;
  border:1px solid var(--bs-border) !important; border-radius:10px !important;
  font-family:var(--mono) !important; font-size:11.5px !important; font-weight:600 !important;
  letter-spacing:.06em !important; padding:.6rem 1rem !important; transition:all .18s ease !important;
}
.stDownloadButton button:hover, .stButton button:hover{
  border-color:var(--bs-accent) !important; color:#fff !important;
  background:linear-gradient(120deg, rgba(76,141,255,.18), var(--bs-surface2)) !important;
  box-shadow:0 6px 20px rgba(76,141,255,.20) !important;
}
div[data-testid="stCaptionContainer"] p{
  font-family:var(--mono) !important; font-size:9.5px !important; color:var(--bs-mute) !important;
  letter-spacing:.03em !important;
}
div[data-testid="stAlert"]{
  background:var(--bs-surface) !important; border:1px solid var(--bs-border) !important;
  border-radius:10px !important; font-family:var(--mono) !important; font-size:11.5px !important;
}
div[data-testid="stStatusWidget"], details[data-testid="stExpander"]{
  border-radius:10px !important; border-color:var(--bs-border) !important;
}
hr{ border-color:var(--bs-border-soft) !important; }
iframe{ width:100% !important; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch",
}

INDICES = {
    "US30_USD":   "DOW JONES",
    "NAS100_USD": "NASDAQ 100",
    "SPX500_USD": "S&P 500",
    "DE30_EUR":   "DAX 40",
}
METAUX = {
    "XAU_USD":   "GOLD",
    "XPT_USD":   "PLATINUM",
    "WTICO_USD": "WTI CRUDE",
}

FOREX_PAIRS = PAIRS


# ── 2. Clients & données avec cache isolé ─────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _run_engine_cached(_token_fp: str, environment: str) -> StrengthResult:
    """Cache basé sur l'empreinte du token + env."""
    access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    client = _create_client(access_token, environment)
    engine = StrengthEngine(client=client)
    return engine.run()


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_candles_cached(
    _token_fp: str,
    environment: str,
    instrument: str,
    granularity: str,
    count: int,
) -> Optional[pd.DataFrame]:
    """Récupère les chandeliers avec cache."""
    access_token = st.secrets["OANDA_ACCESS_TOKEN"]
    client = OandaClient(_create_client(access_token, environment))
    try:
        params = {"count": count, "granularity": granularity, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        rows = [
            {"Time": c["time"], "Close": float(c["mid"]["c"])}
            for c in r.response["candles"] if c["complete"]
        ]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        return df
    except (BluestarError, V20Error, KeyError, ValueError, TypeError, OSError):
        logger.exception("Cached fetch failed for %s %s", instrument, granularity)
        return None


def _smoothed_pct(closes: pd.Series, smooth: int = MAP_SMOOTH_WINDOW) -> Optional[float]:
    """
    smooth=1 -> legacy exact (single-tick change).
    smooth>=2 -> mean(last `smooth`) / mean(previous `smooth`) - 1.
    """
    if smooth <= 1 or len(closes) < smooth * 2:
        if len(closes) < 2:
            return None
        return float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)
    last_mean = float(closes.iloc[-smooth:].mean())
    prev_mean = float(closes.iloc[-2 * smooth:-smooth].mean())
    if prev_mean == 0:
        return None
    return float((last_mean / prev_mean - 1) * 100)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_map_data(
    _token_fp: str,
    environment: str,
    gran: str,
) -> Tuple[pd.DataFrame, Dict, Dict[str, float]]:
    """Market Map sans forward/backward fill."""
    local_pair_changes: Dict[str, float] = {}
    max_age_map = {
        "M5": pd.Timedelta(minutes=15),
        "M15": pd.Timedelta(minutes=45),
        "M30": pd.Timedelta(minutes=90),
        "H1": pd.Timedelta(hours=3),
        "H4": pd.Timedelta(hours=8),
        "D": pd.Timedelta(days=3),
    }
    max_age = max_age_map.get(gran, pd.Timedelta(hours=1))

    now_utc = pd.Timestamp.utcnow()
    for pair in FOREX_PAIRS:
        df = _fetch_candles_cached(_token_fp, environment, pair, gran, 30)
        if df is None or len(df) < 2:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            continue
        age = now_utc - closes.index[-1]
        if age > max_age:
            continue
        pct = _smoothed_pct(closes)
        if pct is None or not np.isfinite(pct):
            continue
        local_pair_changes[pair] = pct

    local_pct_special = {}
    for symbol, name in {**INDICES, **METAUX}.items():
        df = _fetch_candles_cached(_token_fp, environment, symbol, gran, 30)
        if df is None or len(df) < 2:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            continue
        pct = _smoothed_pct(closes)
        if pct is None or not np.isfinite(pct):
            continue
        local_pct_special[name] = {
            "pct": pct,
            "cat": "INDICES" if symbol in INDICES else "METAUX",
        }

    local_df_prices = pd.DataFrame({pair: [1.0] for pair in local_pair_changes})
    return local_df_prices, local_pct_special, local_pair_changes


# ── 3. Composants UI ──────────────────────────────────────────────────────────

def _score_palette(score: float) -> Tuple[str, str, str]:
    """(couleur, gradient de barre, tier css) selon le score 0-10."""
    if score >= 7:
        return T.UP,   "linear-gradient(90deg,#059669,#34D399)", "t-1"
    if score >= 5.5:
        return T.ACCENT, "linear-gradient(90deg,#2F6BD8,#6BA5FF)", "t-2"
    if score >= 4:
        return T.WARN, "linear-gradient(90deg,#B45309,#FBBF24)", "t-3"
    return T.DOWN,     "linear-gradient(90deg,#BE123C,#FB7185)", "t-4"


def display_card(
    name: str,
    score: float,
    arrow_str: str,
    rank: Optional[int] = None,
    velocity: float = 0.0,
) -> str:
    """Génère la carte HTML d'une devise (design v10.1)."""
    safe_name = html.escape(name)
    color, bar_grad, tier = _score_palette(score)

    if arrow_str == "up":
        arrow, v_col, v_bg = "▲", "#6EE7B7", "rgba(16,185,129,.12)"
        v_lbl = "ACCÉLÈRE"
    elif arrow_str == "down":
        arrow, v_col, v_bg = "▼", "#FDA4AF", "rgba(244,63,94,.12)"
        v_lbl = "DÉCÉLÈRE"
    else:
        arrow, v_col, v_bg = "▬", T.NEUTRAL, "rgba(148,163,184,.10)"
        v_lbl = "STABLE"

    flag_code = FLAG_URLS.get(name, "xk")
    img_html = (
        f'<img class="cur-flag" alt="{safe_name}" '
        f'src="https://flagcdn.com/48x36/{html.escape(flag_code)}.png">'
    )
    rank_html = f'<span class="cur-rank">#{rank}</span>' if rank else ""
    bar_w = min(max(score * 10, 0), 100)

    return f"""
    <div class="cur-card {tier}">
      <div class="cur-top">
        {img_html}
        <span class="cur-code">{safe_name}</span>
        {rank_html}
      </div>
      <div class="cur-score" style="color:{color};">
        {score:.1f}<span class="cur-max">/10</span>
        <span class="cur-vel" style="color:{v_col};background:{v_bg};">{arrow} {velocity:+.3f}</span>
      </div>
      <div class="cur-track"><div class="cur-fill" style="width:{bar_w}%;background:{bar_grad};"></div></div>
      <div class="cur-foot"><span>Force relative</span><span>{v_lbl}</span></div>
    </div>
    """


def section_title(title: str, caption: str = "") -> str:
    """En-tête de section stylisé."""
    cap = f'<div class="bs-sec-c">{html.escape(caption)}</div>' if caption else ""
    return (
        f'<div class="bs-sec"><div class="bs-sec-bar"></div>'
        f'<div class="bs-sec-t">{html.escape(title)}</div>{cap}'
        f'<div class="bs-sec-line"></div></div>'
    )


def kpi_tile(label: str, value: str, sub: str = "", color: str = T.TEXT) -> str:
    """Tuile KPI."""
    sub_html = f'<div class="bs-kpi-s">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="bs-kpi"><div class="bs-kpi-l">{html.escape(label)}</div>'
        f'<div class="bs-kpi-v" style="color:{color};">{html.escape(value)}</div>'
        f'{sub_html}</div>'
    )


def app_header(env: str, gran: str, regime: str, ts: str) -> str:
    """Bandeau d'en-tête de l'application."""
    regime_style = {
        "RISK_ON":  ("RISK ON", "on"),
        "RISK_OFF": ("RISK OFF", "off"),
        "NEUTRAL":  ("NEUTRE", "neu"),
    }
    r_lbl, r_cls = regime_style.get(regime, regime_style["NEUTRAL"])
    return f"""
    <div class="bs-header">
      <div class="bs-brand">
        <div class="bs-logo">◆</div>
        <div>
          <div class="bs-eyebrow">Bluestar System</div>
          <div class="bs-title">Market Dashboard</div>
          <div class="bs-sub">FX Institutional Desk · Strength Engine v10.1 · W / D / H4 / H1</div>
        </div>
      </div>
      <div class="bs-headmeta">
        <span class="bs-chip {r_cls}"><span class="bs-dot"></span>{r_lbl}</span>
        <span class="bs-chip">ENV · {html.escape(env.upper())}</span>
        <span class="bs-chip">MAP · {html.escape(gran)}</span>
        <span class="bs-chip">{html.escape(ts)}</span>
      </div>
    </div>
    """


def pair_card_html(item: Dict) -> str:
    """Carte d'une paire sélectionnée."""
    direction = item.get("direction", "")
    pair_name = item.get("exec_pair", item.get("pair", ""))
    diff      = item.get("diff", 0.0)
    atr       = item.get("atr")
    is_buy    = direction == "BUY"
    cls       = "buy" if is_buy else "sell"
    lbl       = "▲ LONG" if is_buy else "▼ SHORT"
    atr_str   = f"{atr:.4f}%" if atr else "N/A"
    return (
        f'<div class="pair-card {cls}">'
        f'<div class="pair-name">{html.escape(pair_name)}</div>'
        f'<div class="pair-tag {cls}">{lbl}</div>'
        f'<div class="pair-metric">DIFF <b>{diff:.2f}</b></div>'
        f'<div class="pair-metric">ATR H1 <b>{atr_str}</b></div>'
        f'<div class="pair-metric" style="margin-left:auto;">EXEC · OANDA</div>'
        f'</div>'
    )


# ── 4. Market Map HTML ────────────────────────────────────────────────────────

def _get_bg_color(pct: float) -> str:
    """Couleur de fond selon le pourcentage (heatmap dark theme)."""
    if pct >= 0.15:
        return "#0E9F6E"
    if pct >= 0.01:
        return "rgba(16,185,129,.16)"
    if pct <= -0.15:
        return "#D8304F"
    if pct <= -0.01:
        return "rgba(244,63,94,.16)"
    return "rgba(148,163,184,.08)"


def _get_text_color(pct: float) -> str:
    """Couleur du texte selon le pourcentage."""
    if pct >= 0.15:
        return "#EAFFF6"
    if pct >= 0.01:
        return "#6EE7B7"
    if pct <= -0.15:
        return "#FFF1F3"
    if pct <= -0.01:
        return "#FDA4AF"
    return "#94A3B8"


def _render_forex_section(
    forex_data: Dict[str, list],
    sorted_cols: List[str],
) -> str:
    """HTML pour la section Forex."""
    html_out = '<div class="section-header"><span class="sh-bar"></span>Forex Heatmap</div>'
    html_out += '<div class="matrix-row">'
    for currency in sorted_cols:
        items   = forex_data[currency]
        winners = sorted(
            [x for x in items if x["pct"] >= 0.01],
            key=lambda x: x["pct"], reverse=True,
        )
        losers  = sorted(
            [x for x in items if x["pct"] < -0.01],
            key=lambda x: x["pct"],
        )
        flat    = [x for x in items if -0.01 <= x["pct"] < 0.01]
        html_out += '<div class="currency-col">'
        for x in winners:
            col = _get_bg_color(x["pct"])
            txt = _get_text_color(x["pct"])
            html_out += (
                f'<div class="tile" style="background:{col};color:{txt};">'
                f'<span>{html.escape(x["pair"])}</span>'
                f'<span class="val">+{x["pct"]:.2f}</span></div>'
            )
        html_out += f'<div class="sep">{html.escape(currency)}</div>'
        for x in flat:
            html_out += (
                f'<div class="tile" style="background:rgba(148,163,184,.08);color:#94A3B8;">'
                f'<span>{html.escape(x["pair"])}</span><span class="val">—</span></div>'
            )
        for x in losers:
            col = _get_bg_color(x["pct"])
            txt = _get_text_color(x["pct"])
            html_out += (
                f'<div class="tile" style="background:{col};color:{txt};">'
                f'<span>{html.escape(x["pair"])}</span>'
                f'<span class="val">{x["pct"]:.2f}</span></div>'
            )
        html_out += '</div>'
    html_out += '</div>'
    return html_out


def _render_special_section(
    special_data: Dict,
    category: str,
    title: str,
) -> str:
    """HTML pour une section spéciale (indices ou métaux)."""
    html_out = f'<div class="section-header"><span class="sh-bar"></span>{title}</div>'
    html_out += '<div class="grid-container">'
    for name, data in special_data.items():
        if data["cat"] != category:
            continue
        pct  = data["pct"]
        bg   = _get_bg_color(pct)
        fg   = _get_text_color(pct)
        sign = "+" if pct >= 0 else ""
        html_out += (
            f'<div class="big-box" style="background:{bg};color:{fg};">'
            f'<span class="box-name">{html.escape(name)}</span>'
            f'<span class="box-val">{sign}{pct:.2f}%</span></div>'
        )
    html_out += '</div>'
    return html_out


def generate_exact_map_html(
    local_pair_changes: Dict[str, float],
    local_pct_special: Dict,
) -> str:
    """Génère la Market Map HTML (design v10.1, dark)."""
    if not local_pair_changes:
        return (
            "<p style='color:#657084;padding:1rem;font-family:monospace;font-size:12px;'>"
            "Données insuffisantes.</p>"
        )

    forex_data = {c: [] for c in CURRENCIES}
    for pair, pct in local_pair_changes.items():
        parts = pair.split("_")
        if len(parts) != 2:
            continue
        b, q = parts
        if b in forex_data:
            forex_data[b].append({"pair": q, "pct": pct})
        if q in forex_data:
            forex_data[q].append({"pair": b, "pct": -pct})

    scores      = {c: sum(x["pct"] for x in items) for c, items in forex_data.items()}
    sorted_cols = sorted(scores, key=scores.get, reverse=True)

    html_out = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
    *{box-sizing:border-box;}
    body{
      font-family:'JetBrains Mono',ui-monospace,monospace;
      background:transparent; margin:0; padding:2px 0 12px 0; color:#E6EAF2;
    }
    .section-header{
      display:flex; align-items:center; gap:9px;
      color:#9AA6B8; font-size:10px; font-weight:700; text-transform:uppercase;
      letter-spacing:.18em; margin:20px 0 10px 0; padding-bottom:7px;
      border-bottom:1px solid #232C39;
    }
    .section-header:first-child{ margin-top:0; }
    .sh-bar{ width:3px; height:12px; border-radius:2px; background:linear-gradient(180deg,#4C8DFF,#1E3FA8); }
    .matrix-row{ display:flex; gap:6px; overflow-x:auto; padding-bottom:8px; }
    .currency-col{ display:flex; flex-direction:column; min-width:102px; gap:2px; }
    .tile{
      display:flex; justify-content:space-between; align-items:center; gap:6px;
      padding:5px 9px; font-size:10.5px; font-weight:600; letter-spacing:.06em;
      border-radius:5px; border:1px solid rgba(255,255,255,.05);
    }
    .tile .val{ font-variant-numeric:tabular-nums; font-weight:700; }
    .sep{
      background:linear-gradient(120deg,#1E2938,#141A23); color:#E6EAF2;
      font-weight:800; letter-spacing:.2em; padding:7px 9px; margin:4px 0;
      font-size:11.5px; text-transform:uppercase; text-align:center;
      border-radius:6px; border:1px solid #2B3646; border-top:2px solid #4C8DFF;
    }
    .grid-container{ display:flex; flex-wrap:wrap; gap:9px; }
    .big-box{
      min-width:148px; height:66px; display:flex; flex-direction:column;
      justify-content:center; align-items:center; gap:5px;
      border-radius:10px; border:1px solid rgba(255,255,255,.06);
    }
    .box-name{ font-size:9.5px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; opacity:.85; }
    .box-val{ font-size:19px; font-weight:800; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
    ::-webkit-scrollbar{ height:7px; }
    ::-webkit-scrollbar-track{ background:transparent; }
    ::-webkit-scrollbar-thumb{ background:#26303E; border-radius:6px; }
    </style></head><body>"""

    html_out += _render_forex_section(forex_data, sorted_cols)
    html_out += _render_special_section(local_pct_special, "INDICES", "Indices Actions")
    html_out += _render_special_section(local_pct_special, "METAUX", "Commodités")
    html_out += '</body></html>'
    return html_out


# ══════════════════════════════════════════════════════════════════════════════
# ── 5. EXPORT — JSON (pipeline macro) + PDF (briefing institutionnel) ─────────
# ══════════════════════════════════════════════════════════════════════════════

def _session_label() -> str:
    """Session active selon l'heure UTC."""
    h = datetime.datetime.utcnow().hour
    if 7 <= h < 12:
        return "London"
    if 12 <= h < 17:
        return "London/NY Overlap"
    if 17 <= h < 22:
        return "New York"
    return "Asian/Off-peak"


def _infer_regime(pct_special: Dict) -> str:
    """Infère le régime risk-on/off depuis indices OANDA + WTI + Gold."""
    indices_pcts = [d["pct"] for d in pct_special.values() if d["cat"] == "INDICES"]
    gold_pct     = pct_special.get("GOLD",      {}).get("pct", 0.0)
    wti_pct      = pct_special.get("WTI CRUDE", {}).get("pct", 0.0)

    if not indices_pcts:
        return "NEUTRAL"

    avg_eq = sum(indices_pcts) / len(indices_pcts)

    if avg_eq > 0.10 and gold_pct < 0.15 and wti_pct >= 0:
        return "RISK_ON"
    if avg_eq < -0.10 or (gold_pct > 0.20 and avg_eq < 0):
        return "RISK_OFF"
    return "NEUTRAL"


def generate_json_export(
    result: StrengthResult,
    pair_changes: Dict[str, float],
    pct_special: Dict,
    granularity: str,
) -> str:
    """JSON structuré pour BLUESTAR_MACRO_BRIEFING_PROMPT."""
    now = datetime.datetime.now()

    sym_to_name = {**INDICES, **METAUX}
    name_to_sym = {v: k for k, v in sym_to_name.items()}

    indices_out: Dict     = {}
    commodities_out: Dict = {}
    for name, data in pct_special.items():
        sym = name_to_sym.get(name, "N/A")
        entry = {"pct_change": round(data["pct"], 4), "symbol": sym}
        (indices_out if data["cat"] == "INDICES" else commodities_out)[name] = entry

    vel_label = {}
    for c, v in result.velocity.items():
        if v > 0.02:
            vel_label[c] = "↗ accélère"
        elif v < -0.02:
            vel_label[c] = "↘ décélère"
        else:
            vel_label[c] = "→ stable"

    usd_score = result.scores_display.get("USD", 5.0)
    usd_rank  = (result.ranking.index("USD") + 1) if "USD" in result.ranking else None
    usd_bias  = "fort" if usd_score >= 6.5 else ("faible" if usd_score <= 3.5 else "neutre")

    idx_pctsL = [d["pct"] for d in pct_special.values() if d["cat"] == "INDICES"]
    if idx_pctsL:
        avg_eq = sum(idx_pctsL) / len(idx_pctsL)
        eq_bias = "haussier" if avg_eq > 0.10 else ("baissier" if avg_eq < -0.10 else "mixte")
    else:
        eq_bias = "N/A"

    payload = {
        "schema_version": "1.0",
        "meta": {
            "date":          now.strftime("%Y-%m-%d"),
            "timestamp":     now.isoformat(timespec="seconds"),
            "session":       _session_label(),
            "timeframe_map": granularity,
            "system":        "BLUESTAR v10.1",
        },
        "oanda_data": {
            "currency_strength": {
                "ranking": result.ranking,
                "scores": {
                    c: {
                        "display_0_10": result.scores_display.get(c),
                        "velocity":     round(result.velocity.get(c, 0.0), 6),
                        "trend":        vel_label.get(c, "→ stable"),
                    }
                    for c in result.ranking
                },
                "best_pairs":   result.best_pairs,
                "pairs_detail": result.pairs_detail,
                "data_quality": {
                    "coverage_min": round(min(result.coverage.values(), default=0.0), 4),
                    "pairs_fetched": result.pairs_fetched,
                    "warnings":      result.warnings,
                    "valid":         result.valid,
                },
            },
            "market_map": {
                "timeframe":   granularity,
                "forex_pairs": {k: round(v, 4) for k, v in pair_changes.items()},
                "indices":     indices_out,
                "commodities": commodities_out,
            },
        },
        "risk_context": {
            "regime_inferred": _infer_regime(pct_special),
            "usd": {
                "score_0_10": usd_score,
                "rank":       usd_rank,
                "bias":       usd_bias,
            },
            "equity_bias": eq_bias,
            "_note": "Régime et biais inférés depuis données OANDA — indicatifs, non definitifs.",
        },
        "external_required": {
            "_note": (
                "Ces champs sont null : non disponibles via OANDA v20. "
                "À injecter depuis CBOE/FRED/Bloomberg avant exécution du prompt LLM."
            ),
            "vix":          {"value": None, "source": "CBOE"},
            "move_index":   {"value": None, "source": "ICE"},
            "dxy":          {"value": None, "source": "ICE/Bloomberg"},
            "us10y":        {"value": None, "source": "FRED/Bloomberg"},
            "cot_ips":      {"value": None, "source": "CFTC (J-3)"},
        },
        "health": result.health_check(),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def generate_briefing_html(
    result: StrengthResult,
    pair_changes: Dict[str, float],
    pct_special: Dict,
    granularity: str,
) -> str:
    """HTML institutionnel auto-peuplé depuis OANDA (design épuré, print-ready)."""
    now      = datetime.datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%H:%M")
    session  = _session_label()
    regime   = _infer_regime(pct_special)
    hc       = result.health_check()
    cov_pct  = int(hc.get("coverage_min", 0) * 100)

    regime_map = {
        "RISK_ON":  ("RISK ON",        "#047857", "#ECFDF5", "#A7F3D0"),
        "RISK_OFF": ("RISK OFF",       "#BE123C", "#FFF1F2", "#FECDD3"),
        "NEUTRAL":  ("NEUTRE / MIXTE", "#1D4ED8", "#EFF6FF", "#BFDBFE"),
    }
    regime_label, r_col, r_bg, r_bd = regime_map.get(regime, regime_map["NEUTRAL"])
    cov_col = "#047857" if cov_pct >= 80 else "#BE123C"

    # ── Currency strength rows ────────────────────────────────────────────────
    rows_html = ""
    for i, cur in enumerate(result.ranking):
        score = result.scores_display.get(cur, 5.0)
        vel   = result.velocity.get(cur, 0.0)
        arrow, a_col = ("▲", "#047857") if vel > 0.02 else (("▼", "#BE123C") if vel < -0.02 else ("▬", "#94A3B8"))
        s_col = "#047857" if score >= 7.0 else ("#1D4ED8" if score >= 5.5 else ("#B45309" if score >= 4.0 else "#BE123C"))
        bar_w = min(max(score * 10, 0), 100)
        rows_html += (
            f'<tr>'
            f'<td class="rk">{i+1}</td>'
            f'<td class="cur">{html.escape(cur)}</td>'
            f'<td class="num" style="color:{s_col};">{score:.2f}</td>'
            f'<td class="barcell">'
            f'<div class="bar"><div class="bar-f" style="width:{bar_w}%;background:{s_col};"></div></div></td>'
            f'<td class="num" style="color:{a_col};">{arrow} <span class="sm">{vel:+.4f}</span></td>'
            f'</tr>'
        )

    # ── Best pairs ────────────────────────────────────────────────────────────
    pairs_html = ""
    for item in result.pairs_detail:
        direction = item.get("direction", "")
        pair_name = item.get("exec_pair", item.get("pair", ""))
        diff      = item.get("diff", 0.0)
        atr       = item.get("atr")
        is_buy    = direction == "BUY"
        d_col, d_bg, d_bd, d_lbl = (
            ("#047857", "#ECFDF5", "#A7F3D0", "▲ LONG") if is_buy
            else ("#BE123C", "#FFF1F2", "#FECDD3", "▼ SHORT")
        )
        atr_str = f"{atr:.4f}%" if atr else "N/A"
        pairs_html += (
            f'<div class="pairrow" style="border-left-color:{d_col};">'
            f'<div class="pairname">{html.escape(pair_name)}</div>'
            f'<div class="pairtag" style="color:{d_col};background:{d_bg};border-color:{d_bd};">{d_lbl}</div>'
            f'<div class="pairm">Force diff <b>{diff:.2f}</b></div>'
            f'<div class="pairm">ATR H1 <b>{atr_str}</b></div>'
            f'</div>'
        )
    if not pairs_html:
        pairs_html = (
            '<div class="empty">Aucune paire sélectionnée — vérifier la couverture des données.</div>'
        )

    # ── Market snapshot ───────────────────────────────────────────────────────
    def _tile(name: str, pct: float) -> str:
        if pct >= 0.15:
            bg, fg, bd = "#ECFDF5", "#047857", "#A7F3D0"
        elif pct >= 0.01:
            bg, fg, bd = "#F6FEFA", "#059669", "#D1FAE5"
        elif pct <= -0.15:
            bg, fg, bd = "#FFF1F2", "#BE123C", "#FECDD3"
        elif pct <= -0.01:
            bg, fg, bd = "#FFF7F8", "#E11D48", "#FEE2E4"
        else:
            bg, fg, bd = "#F8FAFC", "#64748B", "#E2E8F0"
        sign = "+" if pct >= 0 else ""
        return (
            f'<div class="tile" style="background:{bg};border-color:{bd};">'
            f'<div class="tile-n">{html.escape(name)}</div>'
            f'<div class="tile-v" style="color:{fg};">{sign}{pct:.2f}%</div></div>'
        )

    indices_tiles   = "".join(_tile(n, d["pct"]) for n, d in pct_special.items() if d["cat"] == "INDICES")
    commodity_tiles = "".join(_tile(n, d["pct"]) for n, d in pct_special.items() if d["cat"] == "METAUX")

    snap_html = (
        f'<div class="minihdr">Indices Actions</div>'
        f'<div class="tilewrap">{indices_tiles}</div>'
        f'<div class="minihdr" style="margin-top:16px;">Commodités</div>'
        f'<div class="tilewrap">{commodity_tiles}</div>'
    )

    # ── External placeholders ─────────────────────────────────────────────────
    ext_kpis = [("VIX", "CBOE"), ("DXY", "ICE"), ("US10Y", "FRED"), ("MOVE", "ICE")]
    ext_html = "".join(
        f'<div class="ext"><div class="ext-l">{label}</div>'
        f'<div class="ext-v">—</div><div class="ext-s">{src} requis</div></div>'
        for label, src in ext_kpis
    )

    # ── Full HTML ─────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Macro Briefing BLUESTAR — {html.escape(date_str)}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
:root{{
  --ink:#0F172A; --ink2:#334155; --muted:#64748B; --line:#E2E8F0; --line2:#CBD5E1;
  --bg:#FFFFFF; --soft:#F8FAFC; --royal:#1D4ED8; --royal-dim:#93B4FF;
  --mono:'JetBrains Mono','Courier New',monospace; --sans:'Inter',system-ui,sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#F1F5F9;color:var(--ink);font-family:var(--sans);font-size:12px;line-height:1.55;-webkit-font-smoothing:antialiased}}
#page{{max-width:1080px;margin:0 auto;padding:20px}}

.hdr{{display:flex;align-items:center;justify-content:space-between;gap:20px;background:var(--bg);
     border:1px solid var(--line);border-radius:14px 14px 0 0;padding:18px 26px}}
.hdr-l{{display:flex;align-items:center;gap:14px}}
.logo{{width:38px;height:38px;border-radius:10px;background:linear-gradient(140deg,#2563EB,#1E3A8A);
      display:flex;align-items:center;justify-content:center;color:#fff;font-size:17px}}
.eyebrow{{font-family:var(--mono);font-size:8.5px;letter-spacing:.3em;color:var(--royal);font-weight:700;text-transform:uppercase}}
.h1{{font-size:19px;font-weight:700;letter-spacing:-.03em;line-height:1.15;color:var(--ink)}}
.h1s{{font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:2px}}
.hdr-r{{text-align:right;border-left:1px solid var(--line);padding-left:18px}}
.hdr-r1{{font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.1em;color:var(--royal);text-transform:uppercase}}
.hdr-r2{{font-family:var(--mono);font-size:8.5px;color:var(--muted);margin-top:4px}}

.subbar{{background:var(--soft);border:1px solid var(--line);border-top:none;border-radius:0 0 14px 14px;
        padding:8px 26px;display:flex;align-items:center;gap:22px;font-family:var(--mono);font-size:9.5px;
        color:var(--ink2);margin-bottom:14px;letter-spacing:.04em}}
.conf{{margin-left:auto;font-weight:700;color:var(--royal);background:#EFF6FF;border:1px solid #BFDBFE;
      padding:2px 11px;border-radius:99px;font-size:8.5px;letter-spacing:.14em}}

.section{{background:var(--bg);border:1px solid var(--line);border-radius:12px;margin-bottom:12px;overflow:hidden}}
.sec-hdr{{display:flex;align-items:center;gap:11px;padding:11px 18px;border-bottom:1px solid var(--line);background:var(--soft)}}
.sec-num{{width:22px;height:22px;border-radius:6px;background:var(--royal);color:#fff;font-size:9.5px;font-weight:700;
         display:flex;align-items:center;justify-content:center;font-family:var(--mono);flex-shrink:0}}
.sec-ttl{{font-size:11px;font-weight:700;color:var(--ink);text-transform:uppercase;letter-spacing:.14em;font-family:var(--mono)}}
.sec-body{{padding:16px 18px}}

.banner{{display:flex;align-items:center;gap:12px;background:var(--soft);border:1px solid var(--line);
        border-radius:9px;padding:11px 15px;margin-bottom:15px;flex-wrap:wrap}}
.banner-l{{font-family:var(--mono);font-size:8.5px;color:var(--muted);letter-spacing:.18em;text-transform:uppercase}}
.regime{{font-family:var(--mono);font-size:12px;font-weight:700;padding:3px 13px;border-radius:6px;border:1px solid}}
.banner-r{{margin-left:auto;font-family:var(--mono);font-size:9.5px;color:var(--muted);letter-spacing:.04em}}

table{{width:100%;border-collapse:collapse;font-size:12px}}
thead th{{padding:8px 12px;text-align:left;font-size:8.5px;font-weight:700;color:var(--muted);
         letter-spacing:.18em;text-transform:uppercase;font-family:var(--mono);border-bottom:1px solid var(--line2)}}
tbody td{{padding:8px 12px;vertical-align:middle;border-bottom:1px solid var(--line)}}
tbody tr:last-child td{{border-bottom:none}}
td.rk{{width:34px;text-align:center;font-family:var(--mono);font-size:10px;font-weight:700;color:var(--royal-dim)}}
td.cur{{font-family:var(--mono);font-weight:700;letter-spacing:.14em;color:var(--ink);width:80px}}
td.num{{font-family:var(--mono);font-weight:700;font-size:13px;text-align:center;width:110px;font-variant-numeric:tabular-nums}}
td.num .sm{{font-size:9px;font-weight:500}}
td.barcell{{padding-right:22px}}
.bar{{height:5px;border-radius:99px;background:#EEF2F7;overflow:hidden}}
.bar-f{{height:100%;border-radius:99px}}

.pairrow{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:var(--soft);border:1px solid var(--line);
         border-left:3px solid var(--royal);border-radius:9px;padding:11px 15px;margin-bottom:8px}}
.pairname{{font-family:var(--mono);font-size:15px;font-weight:700;letter-spacing:.05em;min-width:105px;color:var(--ink)}}
.pairtag{{font-family:var(--mono);font-size:9.5px;font-weight:700;letter-spacing:.12em;padding:3px 12px;border-radius:5px;border:1px solid}}
.pairm{{font-family:var(--mono);font-size:9.5px;color:var(--muted);letter-spacing:.06em}}
.pairm b{{color:var(--ink);font-weight:700}}
.empty{{font-family:var(--mono);font-size:10.5px;color:var(--muted);font-style:italic;text-align:center;
       padding:18px;border:1px dashed var(--line2);border-radius:9px}}

.minihdr{{font-family:var(--mono);font-size:8.5px;color:var(--muted);letter-spacing:.2em;font-weight:700;
         text-transform:uppercase;margin-bottom:9px}}
.tilewrap{{display:flex;flex-wrap:wrap;gap:8px}}
.tile{{border:1px solid;border-radius:9px;padding:11px 16px;min-width:132px;text-align:center}}
.tile-n{{font-family:var(--mono);font-size:8.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:5px}}
.tile-v{{font-family:var(--mono);font-size:18px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}

.ext{{background:var(--soft);border:1px solid var(--line);border-top:2px solid var(--royal);border-radius:9px;
     padding:11px 16px;min-width:112px;text-align:center}}
.ext-l{{font-family:var(--mono);font-size:8px;color:var(--muted);letter-spacing:.18em;text-transform:uppercase;font-weight:700;margin-bottom:4px}}
.ext-v{{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--line2)}}
.ext-s{{font-family:var(--mono);font-size:8px;color:var(--muted);margin-top:2px}}
.note{{font-family:var(--mono);font-size:9.5px;color:var(--muted);background:var(--soft);border:1px solid var(--line);
      border-radius:9px;padding:11px 15px;line-height:1.7}}
.note code{{background:#EFF6FF;color:var(--royal);padding:1px 5px;border-radius:4px}}

.footer{{text-align:center;font-family:var(--mono);font-size:8px;color:var(--muted);border-top:1px solid var(--line);
        padding:12px;margin-top:6px;letter-spacing:.2em;text-transform:uppercase}}
#pdf-fab{{position:fixed;bottom:26px;right:26px;z-index:9999}}
#pdf-fab button{{background:var(--royal);color:#fff;border:none;padding:11px 20px;border-radius:10px;
                font-family:var(--mono);font-size:11.5px;font-weight:700;letter-spacing:.06em;cursor:pointer;
                box-shadow:0 8px 24px rgba(29,78,216,.35)}}
@media print{{
  @page{{margin:9mm;size:A4 portrait}}
  *{{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}}
  body{{background:#fff!important;font-size:10.5px}}
  #page{{padding:0}}
  .section{{margin-bottom:8px;break-inside:avoid}}
  #pdf-fab{{display:none!important}}
}}
</style>
</head>
<body>
<div id="pdf-fab"><button onclick="window.print()">Télécharger PDF</button></div>
<div id="page">

<div class="hdr">
  <div class="hdr-l">
    <div class="logo">◆</div>
    <div>
      <div class="eyebrow">Bluestar System</div>
      <div class="h1">BLUESTAR</div>
      <div class="h1s">FX Institutional Desk · v10.1</div>
    </div>
  </div>
  <div class="hdr-r">
    <div class="hdr-r1">Institutional Macro Briefing</div>
    <div class="hdr-r2">Analyse quantitative — OANDA v20 API · Auto-généré</div>
  </div>
</div>

<div class="subbar">
  <span>{html.escape(date_str)}</span>
  <span>{html.escape(time_str)} CET — {html.escape(session)}</span>
  <span class="conf">Confidentiel</span>
</div>

<div class="section">
  <div class="sec-hdr"><div class="sec-num">1</div><div class="sec-ttl">Force des Devises — Moteur W/D/H4/H1</div></div>
  <div class="sec-body">
    <div class="banner">
      <span class="banner-l">Régime inféré</span>
      <span class="regime" style="color:{r_col};background:{r_bg};border-color:{r_bd};">{html.escape(regime_label)}</span>
      <span class="banner-r">Couverture <strong style="color:{cov_col};">{cov_pct}%</strong>
        &nbsp;·&nbsp; Timeframe Map <strong>{html.escape(granularity)}</strong></span>
    </div>
    <table>
      <thead><tr><th>#</th><th>Devise</th><th style="text-align:center;">Score</th><th>Distribution</th><th style="text-align:center;">Vélocité H1</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>

<div class="section">
  <div class="sec-hdr"><div class="sec-num">2</div><div class="sec-ttl">Paires Sélectionnées par le Moteur</div></div>
  <div class="sec-body">{pairs_html}</div>
</div>

<div class="section">
  <div class="sec-hdr"><div class="sec-num">3</div><div class="sec-ttl">Snapshot Marché — {html.escape(granularity)} · OANDA</div></div>
  <div class="sec-body">{snap_html}</div>
</div>

<div class="section">
  <div class="sec-hdr"><div class="sec-num">★</div><div class="sec-ttl">Contexte Externe — À Injecter Avant LLM</div></div>
  <div class="sec-body">
    <div class="tilewrap" style="margin-bottom:12px;">{ext_html}</div>
    <div class="note">VIX · DXY · US10Y · MOVE ne sont pas disponibles via l'API OANDA v20.
    Injecter ces valeurs via CBOE / FRED / Bloomberg dans le champ <code>external_required</code>
    du JSON avant exécution du prompt BLUESTAR_MACRO_BRIEFING.</div>
  </div>
</div>

<div class="footer">Confidentiel — Bluestar System · FX Institutional Desk · v10.1 · {html.escape(date_str)} {html.escape(time_str)} CET</div>
</div>
</body>
</html>"""


def generate_pdf_bytes(briefing_html: str) -> Optional[bytes]:
    """Convertit le HTML en PDF via WeasyPrint (None si indisponible)."""
    try:
        from weasyprint import HTML as WP_HTML  # type: ignore[import]
        return WP_HTML(string=briefing_html).write_pdf()
    except ImportError:
        logger.warning("WeasyPrint non disponible — export PDF désactivé (fallback HTML).")
        return None
    except (BluestarError, OSError, ValueError) as exc:
        logger.error("WeasyPrint erreur de rendu: %s", exc)
        return None


# ── 6. Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-brand-t">◆ BLUESTAR</div>'
        '<div class="sb-brand-s">Strength Engine v10.1</div></div>',
        unsafe_allow_html=True,
    )

    if "OANDA_ACCESS_TOKEN" not in st.secrets:
        st.error("Token OANDA introuvable dans les secrets.")
        st.stop()
    current_token = st.secrets["OANDA_ACCESS_TOKEN"]

    st.markdown('<div class="sb-lbl">Connexion</div>', unsafe_allow_html=True)
    current_env = st.selectbox("Env", ["practice", "live"], label_visibility="collapsed")

    st.markdown('<div class="sb-lbl">Timeframe — Market Map</div>', unsafe_allow_html=True)
    current_granularity = st.selectbox(
        "Timeframe (Map)",
        ["M5", "M15", "M30", "H1", "H4", "D"],
        index=3,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-lbl">Lissage de la Map</div>', unsafe_allow_html=True)
    map_smooth = st.selectbox(
        "Map Smooth",
        [1, 3, 5],
        index=0,
        format_func=lambda x: "Legacy (1 tick)" if x == 1 else f"Lissé ({x} ticks)",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(
        "Le moteur de force agrège W + D + H4 + H1 en parallèle, "
        "indépendamment du timeframe affiché sur la Market Map."
    )
    st.caption(
        f"Poids : W {TIMEFRAMES_MTF['W']['weight']} · D {TIMEFRAMES_MTF['D']['weight']} · "
        f"H4 {TIMEFRAMES_MTF['H4']['weight']} · H1 {TIMEFRAMES_MTF['H1']['weight']}"
    )


# ── 7. Exécution ──────────────────────────────────────────────────────────────

if current_token:
    token_fp = token_fingerprint(current_token)

    with st.status("Actualisation des données OANDA…", expanded=False) as status:
        result = _run_engine_cached(token_fp, current_env)
        map_data = fetch_market_map_data(token_fp, current_env, current_granularity)
        df_prices, pct_special, pair_changes = map_data
        status.update(label="Données chargées", state="complete", expanded=False)

    regime_now = _infer_regime(pct_special)
    ts_label   = datetime.datetime.now().strftime("%d/%m %H:%M")

    st.markdown(
        app_header(current_env, current_granularity, regime_now, ts_label),
        unsafe_allow_html=True,
    )

    health = result.health_check()

    if not result.valid:
        st.error("Impossible de calculer les forces : " + "; ".join(result.warnings))
    elif result.warnings:
        for w in result.warnings:
            st.warning(w)

    if result.scores_display and result.valid:
        # ── KPI strip ─────────────────────────────────────────────────────────
        top_cur    = result.ranking[0]
        bot_cur    = result.ranking[-1]
        top_score  = result.scores_display.get(top_cur, 0.0)
        bot_score  = result.scores_display.get(bot_cur, 0.0)
        spread_val = top_score - bot_score
        cov_pct_ui = int(health.get("coverage_min", 0) * 100)
        cov_color  = T.UP if cov_pct_ui >= 80 else (T.WARN if cov_pct_ui >= 50 else T.DOWN)
        health_col = T.UP if health["status"] == "ok" else T.WARN

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                kpi_tile("Devise la plus forte", f"{top_cur} · {top_score:.1f}",
                         "Sommet du classement", T.UP),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                kpi_tile("Devise la plus faible", f"{bot_cur} · {bot_score:.1f}",
                         "Bas du classement", T.DOWN),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                kpi_tile("Dispersion", f"{spread_val:.2f}",
                         f"Seuil signal ≥ {MIN_STRENGTH_DIFF}", T.ACCENT),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                kpi_tile("Couverture données", f"{cov_pct_ui}%",
                         f"{result.pairs_fetched} séries en cache", cov_color),
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                kpi_tile("Statut moteur", health["status"].upper(),
                         f"{len(result.warnings)} alerte(s)", health_col),
                unsafe_allow_html=True,
            )

        # ── Forces devises ────────────────────────────────────────────────────
        st.markdown(
            section_title("Forces Forex", "Échelle 0–10 · Moteur institutionnel W/D/H4/H1"),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="bs-legend">'
            f'<span><i class="bs-sw" style="background:{T.UP}"></i>Fort ≥ 7.0</span>'
            f'<span><i class="bs-sw" style="background:{T.ACCENT}"></i>Modéré 5.5–7.0</span>'
            f'<span><i class="bs-sw" style="background:{T.WARN}"></i>Faible 4.0–5.5</span>'
            f'<span><i class="bs-sw" style="background:{T.DOWN}"></i>Très faible &lt; 4.0</span>'
            '<span>▲ / ▼ vélocité H1 (48 vs 48 barres)</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        for i, curr in enumerate(result.ranking):
            with cols[i % 4]:
                st.markdown(
                    display_card(
                        name      = curr,
                        score     = result.scores_display[curr],
                        arrow_str = result.direction_arrow(curr),
                        rank      = i + 1,
                        velocity  = result.velocity.get(curr, 0.0),
                    ),
                    unsafe_allow_html=True,
                )

        # ── Paires sélectionnées ──────────────────────────────────────────────
        st.markdown(
            section_title(
                "Paires Sélectionnées",
                f"Diff ≥ {MIN_STRENGTH_DIFF} · Filtre ATR P{ATR_MIN_PERCENTILE} · "
                f"Max {MAX_PAIRS} · 1 exposition par devise",
            ),
            unsafe_allow_html=True,
        )
        if result.pairs_detail:
            st.markdown(
                "".join(pair_card_html(d) for d in result.pairs_detail),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pair-empty">Aucune paire ne satisfait les filtres '
                'de force et de volatilité pour le moment.</div>',
                unsafe_allow_html=True,
            )

        # ── Market Map ────────────────────────────────────────────────────────
        st.markdown(
            section_title("Market Map", f"Variation {current_granularity} · OANDA mid-price"),
            unsafe_allow_html=True,
        )
        if pair_changes:
            html_map = generate_exact_map_html(pair_changes, pct_special)
            st.components.v1.html(html_map, height=640, scrolling=True)
        else:
            st.warning("Données insuffisantes pour la Market Map.")

        # ── Exports ───────────────────────────────────────────────────────────
        st.markdown(
            section_title("Exports", "Pipeline macro & briefing institutionnel"),
            unsafe_allow_html=True,
        )
        col_json, col_pdf = st.columns(2)
        fname_date = datetime.date.today().strftime("%Y-%m-%d")

        with col_json:
            json_str = generate_json_export(
                result, pair_changes, pct_special, current_granularity
            )
            st.download_button(
                label     = "⬇  JSON — Pipeline Macro",
                data      = json_str,
                file_name = f"BLUESTAR_{fname_date}.json",
                mime      = "application/json",
                help      = "JSON structuré pour BLUESTAR_MACRO_BRIEFING_PROMPT · "
                            "champs external_required à compléter",
            )
            st.caption(
                "Force devises · market map (indices, DAX, WTI) · paires sélectionnées · "
                "placeholders VIX / DXY / US10Y / MOVE."
            )

        with col_pdf:
            briefing_html_str = generate_briefing_html(
                result, pair_changes, pct_special, current_granularity
            )
            pdf_bytes = generate_pdf_bytes(briefing_html_str)
            if pdf_bytes:
                st.download_button(
                    label     = "⬇  PDF — Briefing Institutionnel",
                    data      = pdf_bytes,
                    file_name = f"Macro_Briefing_BLUESTAR_{fname_date}.pdf",
                    mime      = "application/pdf",
                )
                st.caption(
                    "PDF auto-généré : classement des forces, paires, snapshot marché, "
                    "placeholders externes."
                )
            else:
                st.download_button(
                    label     = "⬇  HTML — Briefing (impression PDF)",
                    data      = briefing_html_str,
                    file_name = f"Macro_Briefing_BLUESTAR_{fname_date}.html",
                    mime      = "text/html",
                )
                st.caption(
                    "WeasyPrint non détecté. Ouvrir le HTML dans Chrome → Ctrl+P → "
                    "Enregistrer en PDF (activer « Graphiques d'arrière-plan »)."
                )
else:
    st.warning("En attente du Token OANDA…")
