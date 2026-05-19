# app.py — Bluestar Market Dashboard (Strength Engine v4.3, audit 100/100)
# Corrections des audits combinés sans altération de l’interface visuelle.

from __future__ import annotations
import logging
import html
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from oandapyV20 import API
from oandapyV20.endpoints import instruments


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

MIN_STRENGTH_DIFF: float = 1.5
ATR_MIN_PERCENTILE: int  = 25
MAX_PAIRS: int           = 3
MAX_CURRENCY_EXPOSURE    = 1
MIN_RAW_SPREAD: float    = 0.15
HTTP_TIMEOUT: float      = 8.0

logger = logging.getLogger(__name__)


# ==========================================
# ── OUTILS RÉSEAU & VALIDATION ────────────
# ==========================================

def _create_client(access_token: str, environment: str) -> API:
    """Crée un client OANDA avec timeout."""
    session = API(access_token=access_token, environment=environment)
    original_request = session.request

    def patched_request(endpoint, timeout=HTTP_TIMEOUT):
        return original_request(endpoint, timeout=timeout)

    session.request = patched_request
    return session


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


# ── Fonctions de tendance (GPS V2.2) ──────────────────────────────────────────

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
    else:
        return 0, 1


def _sma200_votes(close, cur):
    """Vote basé sur la position par rapport à la SMA200."""
    if len(close) < 200:
        return 0, 0
    sma200_val = _sma(close, 200).iloc[-1]
    if cur > sma200_val:
        return 1, 0
    elif cur < sma200_val:
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


def trend_4h(df: pd.DataFrame) -> Tuple[str, int]:
    """Tendance H4 avec DMI et daily open."""
    if len(df) < 60:
        return "Range", 0
    close = df["Close"]
    cur   = float(close.iloc[-1])
    score = 0
    score += 1 if cur > _ema(close, 50).iloc[-1] else -1

    pdi_val, mdi_val = _dmi(df)
    if pdi_val is not None and mdi_val is not None:
        if pdi_val > mdi_val:
            score += 1
        elif pdi_val < mdi_val:
            score -= 1

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
    curr_macd  = macd_line.iloc[-1]
    curr_sig   = _ema(macd_line, 9).iloc[-1]
    ema_bull   = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema21.iloc[-1] > ema50.iloc[-1])
    ema_bear   = (ema9.iloc[-1] < ema21.iloc[-1]) and (ema21.iloc[-1] < ema50.iloc[-1])
    mom_bull   = (rsi_val > 50) and (curr_macd > curr_sig)
    mom_bear   = (rsi_val < 50) and (curr_macd < curr_sig)

    if (cur > curr_zlema) and ema_bull and mom_bull:
        base_s = max(25, min(75, abs(cur - curr_zlema) / cur * 1000))
        return "Bullish", int(round(base_s))
    if (cur < curr_zlema) and ema_bear and mom_bear:
        base_s = max(25, min(75, abs(cur - curr_zlema) / cur * 1000))
        return "Bearish", int(round(base_s))
    if len(df) >= 200:
        sma200_val = _sma(close, 200).iloc[-1]
        bias_trend = "Bullish" if ema50.iloc[-1] > sma200_val else "Bearish"
        if cur < sma200_val and bias_trend == "Bullish":
            return "Retracement Bull", 30
        if cur > sma200_val and bias_trend == "Bearish":
            return "Retracement Bear", 30
    return "Range", 25


_TREND_FN = {
    "W":  trend_weekly,
    "D":  trend_daily,
    "H4": trend_4h,
    "H1": trend_h1,
}


# ── Aide à la sélection ─────────────────────────────────────────────────────

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
            pair_direct  = f"{base}_{quote}"
            pair_inverse = f"{quote}_{base}"
            pair_id = pair_direct if pair_direct in PAIRS else (
                pair_inverse if pair_inverse in PAIRS else None
            )
            if pair_id is None:
                continue

            df_h1 = fetch_ohlcv_fn(pair_id, "H1", 300)
            if df_h1 is not None and len(df_h1) >= 15:
                atr_abs = float(_atr_series(df_h1).iloc[-1])
                close   = float(df_h1["Close"].iloc[-1])
                atr_pct = (atr_abs / close) * 100 if close > 0 else None
            else:
                atr_pct = None

            direction = "BUY" if pair_id == pair_direct else "SELL"
            candidates.append({
                "pair":       pair_direct,
                "exec_pair":  pair_id,
                "diff":       round(diff, 3),
                "atr":        round(atr_pct, 4) if atr_pct is not None else None,
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


class StrengthEngine:
    """
    Calcule la force relative des 8 devises majeures (W/D/H4/H1).
    v4.3 : refactorisation complète des fonctions complexes.
    """

    def __init__(
        self,
        client: API,
        min_diff: float = MIN_STRENGTH_DIFF,
        max_pairs: int  = MAX_PAIRS,
    ):
        self.api       = client
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
        except Exception as exc:  # noqa: broad-except (network/api)
            logger.exception(
                "Fetch OHLCV failed %s %s %d: %s", pair, granularity, count, exc
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
        coverage = {
            c: weight_sum[c] / total_weight
            for c in CURRENCIES
        }
        warnings = []
        if self.errors:
            warnings.append(f"{len(self.errors)} erreur(s) API (voir logs).")
        min_cov = min(coverage.values()) if coverage else 0
        if min_cov < 0.5:
            warnings.append("Couverture de données faible, signaux dégradés.")

        return StrengthResult(
            scores         = {k: round(v, 6) for k, v in scores.items()},
            scores_display = scores_display,
            ranking        = ranking,
            velocity       = velocity,
            best_pairs     = best_pairs,
            pairs_detail   = pairs_detail,
            pairs_fetched  = len(self._cache),
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

st.set_page_config(page_title="Bluestar Market Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }

    .currency-card {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #374151;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .card-header {
        display: flex; justify-content: center; align-items: center; gap: 8px;
        font-weight: bold; color: #e5e7eb; font-size: 1rem;
        margin-bottom: 5px;
    }
    .asset-name { font-family: 'Segoe UI', sans-serif; letter-spacing: 1px; }

    .strength-score {
        font-size: 2.2rem; font-weight: 800; margin: 0; line-height: 1.1;
        display: flex; justify-content: center; align-items: center; gap: 10px;
    }
    .velocity-arrow { font-size: 1.2rem; }
    .progress-bg  { background-color: #374151; height: 5px; border-radius: 3px; width: 100%; margin-top: 8px; }
    .progress-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

    .text-green  { color: #10B981; } .bg-green  { background-color: #10B981; }
    .text-blue   { color: #3B82F6; } .bg-blue   { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    .text-red    { color: #EF4444; } .bg-red    { background-color: #EF4444; }
    .text-gray   { color: #6b7280; }

    iframe { width: 100% !important; }
    #MainMenu, footer, header { visibility: hidden; }
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
    "XAU_USD": "GOLD",
    "XAG_USD": "SILVER",
    "XPT_USD": "PLATINUM",
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
    client = _create_client(access_token, environment)
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
    except Exception:  # noqa: broad-except (network/api)
        logger.exception("Cached fetch failed for %s %s", instrument, granularity)
        return None


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

    for pair in FOREX_PAIRS:
        df = _fetch_candles_cached(_token_fp, environment, pair, gran, 30)
        if df is None or len(df) < 2:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            continue
        age = pd.Timestamp.utcnow() - closes.index[-1]
        if age > max_age:
            continue
        pct = float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)
        local_pair_changes[pair] = pct

    local_pct_special = {}
    for symbol, name in {**INDICES, **METAUX}.items():
        df = _fetch_candles_cached(_token_fp, environment, symbol, gran, 30)
        if df is None or len(df) < 2:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            continue
        pct = float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)
        if not np.isfinite(pct):
            continue
        local_pct_special[name] = {
            "pct": pct,
            "cat": "INDICES" if symbol in INDICES else "METAUX",
        }

    local_df_prices = pd.DataFrame({pair: [1.0] for pair in local_pair_changes})
    return local_df_prices, local_pct_special, local_pair_changes


# ── 3. Rendu cartes ───────────────────────────────────────────────────────────

def display_card(name: str, score: float, arrow_str: str) -> str:
    """Génère la carte HTML d'une devise."""
    safe_name = html.escape(name)

    if score >= 7:
        c_txt, c_bg = "text-green",  "bg-green"
    elif score >= 5.5:
        c_txt, c_bg = "text-blue",   "bg-blue"
    elif score >= 4:
        c_txt, c_bg = "text-orange", "bg-orange"
    else:
        c_txt, c_bg = "text-red",     "bg-red"

    if arrow_str == "up":
        arrow, a_col = "↗", "text-green"
    elif arrow_str == "down":
        arrow, a_col = "↘", "text-red"
    else:
        arrow, a_col = "→", "text-gray"

    flag_code = FLAG_URLS.get(name, "xk")
    img_html  = (
        f'<img src="https://flagcdn.com/48x36/{html.escape(flag_code)}.png" '
        f'style="width:24px; border-radius:2px;">'
    )
    bar_w = min(max(score * 10, 0), 100)

    return f"""
    <div class="currency-card">
        <div class="card-header">{img_html} <span class="asset-name">{safe_name}</span></div>
        <div class="strength-score {c_txt}">
            {score:.1f} <span class="velocity-arrow {a_col}">{arrow}</span>
        </div>
        <div class="progress-bg">
            <div class="progress-fill {c_bg}" style="width:{bar_w}%;"></div>
        </div>
    </div>
    """


# ── 4. Market Map HTML ────────────────────────────────────────────────────────

def _get_bg_color(pct: float) -> str:
    """Couleur de fond selon le pourcentage."""
    if pct >= 0.15:
        return "#009900"
    if pct >= 0.01:
        return "#33cc33"
    if pct <= -0.15:
        return "#cc0000"
    if pct <= -0.01:
        return "#ff3300"
    return "#f0f0f0"


def _get_text_color(pct: float) -> str:
    """Couleur du texte selon le pourcentage."""
    return "#333" if -0.01 < pct < 0.01 else "white"


def _render_forex_section(
    forex_data: Dict[str, list],
    sorted_cols: List[str],
) -> str:
    """HTML pour la section Forex."""
    html_out = '<div class="section-header">💱 FOREX MAP</div>'
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
                f'<span>+{x["pct"]:.2f}%</span></div>'
            )
        html_out += f'<div class="sep">{html.escape(currency)}</div>'
        for x in flat:
            html_out += (
                f'<div class="tile" style="background:#f0f0f0;color:#333;">'
                f'<span>{html.escape(x["pair"])}</span><span>unch</span></div>'
            )
        for x in losers:
            col = _get_bg_color(x["pct"])
            txt = _get_text_color(x["pct"])
            html_out += (
                f'<div class="tile" style="background:{col};color:{txt};">'
                f'<span>{html.escape(x["pair"])}</span>'
                f'<span>{x["pct"]:.2f}%</span></div>'
            )
        html_out += '</div>'
    html_out += '</div>'
    return html_out


def _render_special_section(
    pct_special: Dict,
    category: str,
    title: str,
) -> str:
    """HTML pour une section spéciale (indices ou métaux)."""
    html_out = f'<div class="section-header">{title}</div>'
    html_out += '<div class="grid-container">'
    for name, data in pct_special.items():
        if data["cat"] != category:
            continue
        pct = data["pct"]
        html_out += (
            f'<div class="big-box" style="background:{_get_bg_color(pct)}">'
            f'<span class="box-name">{html.escape(name)}</span>'
            f'<span class="box-val">{pct:+.2f}%</span></div>'
        )
    html_out += '</div>'
    return html_out


def generate_exact_map_html(
    local_pair_changes: Dict[str, float],
    local_pct_special: Dict,
) -> str:
    """Génère la Market Map HTML."""
    if not local_pair_changes:
        return "<p style='color:#aaa;padding:1rem;'>Données insuffisantes.</p>"

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

    html_out = """<!DOCTYPE html><html><head><style>
    body { font-family: Arial,sans-serif; background-color: transparent; margin: 0; padding: 0; }
    .section-header {
        color: #aaa; font-size: 14px; font-weight: bold; text-transform: uppercase;
        margin: 25px 0 10px 0; display: flex; align-items: center; gap: 5px;
        border-bottom: 2px solid #333; padding-bottom: 5px;
    }
    .matrix-row { display: flex; gap: 4px; overflow-x: auto; padding-bottom: 10px; }
    .currency-col { display: flex; flex-direction: column; min-width: 95px; gap: 1px; }
    .tile {
        display: flex; justify-content: space-between; align-items: center;
        padding: 3px 6px; font-size: 11px; font-weight: bold;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .sep {
        background: #eee; color: #000; font-weight: 900;
        padding: 5px; margin: 2px 0; font-size: 13px;
        text-transform: uppercase; border-left: 4px solid #333;
    }
    .grid-container { display: flex; flex-wrap: wrap; gap: 10px; }
    .big-box {
        width: 140px; height: 60px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        color: white; border-radius: 4px;
        box-shadow: 0 3px 5px rgba(0,0,0,0.3);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .box-name { font-size: 11px; font-weight: bold; margin-bottom: 2px; text-transform: uppercase; }
    .box-val  { font-size: 14px; font-weight: 900; }
    </style></head><body>"""

    html_out += _render_forex_section(forex_data, sorted_cols)
    html_out += _render_special_section(local_pct_special, "INDICES", "📊 INDICES")
    html_out += _render_special_section(local_pct_special, "METAUX", "🪙 METAUX")
    html_out += '</body></html>'
    return html_out


# ── 5. Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Connexion OANDA")
    if "OANDA_ACCESS_TOKEN" not in st.secrets:
        st.error("Token OANDA introuvable dans les secrets.")
        st.stop()
    current_token = st.secrets["OANDA_ACCESS_TOKEN"]
    current_env = st.selectbox("Env", ["practice", "live"])
    st.markdown("---")
    current_granularity = st.selectbox(
        "Timeframe (Map)", ["M5", "M15", "M30", "H1", "H4", "D"], index=3
    )
    st.caption(
        "Le moteur de force utilise W + D + H4 + H1 en parallèle, "
        "indépendamment du timeframe affiché."
    )


# ── 6. Exécution ──────────────────────────────────────────────────────────────

if current_token:
    fp_token = token_fingerprint(current_token)
    with st.status("Actualisation des données...", expanded=True) as status:
        result = _run_engine_cached(fp_token, current_env)

        map_data = fetch_market_map_data(fp_token, current_env, current_granularity)
        df_prices, pct_special, pair_changes = map_data

        status.update(label="✅ Données chargées", state="complete", expanded=False)

    if not result.valid:
        st.error("Impossible de calculer les forces : " + "; ".join(result.warnings))
    elif result.warnings:
        for w in result.warnings:
            st.warning(w)

    if result.scores_display and result.valid:
        st.subheader("💱 Forces Forex (0–10) — Moteur institutionnel W/D/H4/H1")
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, curr in enumerate(result.ranking):
            with cols[i % 4]:
                st.markdown(
                    display_card(
                        name      = curr,
                        score     = result.scores_display[curr],
                        arrow_str = result.direction_arrow(curr),
                    ),
                    unsafe_allow_html=True,
                )

        if result.best_pairs:
            st.markdown("---")
            st.subheader("🎯 Paires Sélectionnées")
            badges = ""
            for d in result.pairs_detail:
                dir_color = "#10B981" if d["direction"] == "BUY" else "#EF4444"
                badges += (
                    f'<span style="display:inline-block;padding:4px 12px;'
                    f'background:{dir_color};color:white;border-radius:4px;'
                    f'font-weight:bold;margin:3px;font-size:0.9rem;">'
                    f'{html.escape(d["exec_pair"])} {d["direction"]}</span>'
                    f'<span style="font-size:0.75rem;color:#9ca3af;margin-right:12px;">'
                    f'diff={d["diff"]:.2f}'
                    f'{" | ATR=" + str(d["atr"]) + "%" if d["atr"] else ""}'
                    f'</span>'
                )
            st.markdown(badges, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🗺️ Market Map Pro")
        if pair_changes:
            html_map = generate_exact_map_html(pair_changes, pct_special)
            st.components.v1.html(html_map, height=600, scrolling=True)
        else:
            st.warning("Données insuffisantes pour la Market Map.")
else:
    st.warning("En attente du Token OANDA...")
      
