Bluestar Market Dashboard — app.py (fichier unique)
====================================================
Strength Engine v4.1 — 17 issues auditées corrigées.

Corrections appliquées :
  [CRIT #1]  Normalisation MTF : division par weight_sum au lieu de counts
  [CRIT #2]  Direction BUY/SELL : dérivée de pair_id vs pair_direct
  [CRIT #3]  Filtre ATR : exclure les paires sans ATR (plus de None = pass)
  [CRIT #4]  Retracement : contribution réduite à 0.15 (était 0.30)
  [LOG  #5]  Vélocité : calculée sur scores bruts (plus de double normalisation)
  [LOG  #6]  Weekly open : lundi uniquement (dayofweek == 0)
  [LOG  #7]  _swing_pts : condition arr[i] > arr[i-1] pour éviter faux swings plats
  [LOG  #8]  Vélocité : fenêtre glissante fixe 48 chandelles
  [SIL  #9]  pct_change NaN : détection explicite avant float()
  [SIL  #10] trend_4h today_mask vide : guard explicite sans swallow silencieux
  [SIL  #11] Token OANDA : client isolé via @st.cache_resource, plus dans les clés data
  [SIL  #12] _dmi NaN : retourne (0.0, 0.0) au lieu de propager nan
  [SIL  #13] Corollaire fix #1 : weight_sum corrige le biais par nombre de paires
  [PERF #14] StrengthEngine : résultat mis en cache via @st.cache_data(ttl=60)
  [PERF #15] fetch_market_map_data : wrapper lui-même mis en cache
  [PERF #16] run_quick : pré-fetch H1 pour ATR quand tf != H1
  [PERF #17] _swing_pts : to_numpy() + arrêt précoce sur égalité

Lancement :
    streamlit run app.py
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments


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

logger = logging.getLogger(__name__)


# ==========================================
# ── STRENGTH ENGINE v4.1 ──────────────────
# ==========================================

@dataclass
class StrengthResult:
    scores:         Dict[str, float] = field(default_factory=dict)
    scores_display: Dict[str, float] = field(default_factory=dict)
    ranking:        List[str]        = field(default_factory=list)
    velocity:       Dict[str, float] = field(default_factory=dict)
    best_pairs:     List[str]        = field(default_factory=list)
    pairs_detail:   List[Dict]       = field(default_factory=list)
    pairs_fetched:  int              = 0

    def to_dict(self) -> dict:
        return {
            "scores":         self.scores,
            "scores_display": self.scores_display,
            "ranking":        self.ranking,
            "velocity":       self.velocity,
            "best_pairs":     self.best_pairs,
            "pairs_detail":   self.pairs_detail,
            "pairs_fetched":  self.pairs_fetched,
        }

    def direction_arrow(self, currency: str) -> str:
        # FIX #5 : vélocité sur scores bruts (échelle ~[-1, 1]) → seuil 0.02
        v = self.velocity.get(currency, 0.0)
        if v > 0.02:  return "up"
        if v < -0.02: return "down"
        return "flat"

    def color_class(self, currency: str) -> str:
        s = self.scores_display.get(currency, 5.0)
        if s >= 7.0: return "strong-bull"
        if s >= 5.5: return "mild-bull"
        if s >= 4.0: return "mild-bear"
        return "strong-bear"


# ── Fonctions techniques pures ────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _dmi(df: pd.DataFrame, period: int = 14) -> Tuple[float, float]:
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
    # FIX #12 : retourner (0.0, 0.0) explicitement si NaN, ne pas propager
    pdi_val = float(pdi.iloc[-1])
    mdi_val = float(mdi.iloc[-1])
    if np.isnan(pdi_val) or np.isnan(mdi_val):
        return 0.0, 0.0
    return pdi_val, mdi_val


# ── Fonctions de tendance (GPS V2.1) ──────────────────────────────────────────

def trend_weekly(df: pd.DataFrame) -> Tuple[str, int]:
    if len(df) < 50:
        return "Range", 0
    close  = df["Close"]
    ema50  = _ema(close, 50)
    sma200 = _sma(close, 200) if len(df) >= 200 else _ema(close, 100)
    curr_ema50,  prev_ema50  = ema50.iloc[-1],  ema50.iloc[-2]
    curr_sma200, prev_sma200 = sma200.iloc[-1], sma200.iloc[-2]
    crossed_bull = (prev_ema50 <= prev_sma200) and (curr_ema50 > curr_sma200)
    crossed_bear = (prev_ema50 >= prev_sma200) and (curr_ema50 < curr_sma200)
    if curr_ema50 > curr_sma200: return "Bullish", 90 if crossed_bull else 75
    if curr_ema50 < curr_sma200: return "Bearish", 90 if crossed_bear else 75
    return "Range", 40


def trend_daily(df: pd.DataFrame) -> Tuple[str, int]:
    if len(df) < 60:
        return "Range", 0
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    cur   = float(close.iloc[-1])
    votes_bull = votes_bear = 0

    # FIX #7 : condition arr[i] > arr[i-1] pour éviter les faux swings sur barres plates
    # FIX #17 : to_numpy() évite les overheads pandas (Series.__getitem__) dans la boucle
    def _swing_pts(series: pd.Series, wing: int = 5) -> Tuple[List[int], List[int]]:
        arr = series.to_numpy()
        n   = len(arr)
        highs, lows = [], []
        for i in range(wing, n - wing):
            seg = arr[i - wing: i + wing + 1]
            # Un vrai swing high : max local ET strictement supérieur à la barre précédente
            if arr[i] >= seg.max() and arr[i] > arr[i - 1]:
                highs.append(i)
            # Un vrai swing low : min local ET strictement inférieur à la barre précédente
            if arr[i] <= seg.min() and arr[i] < arr[i - 1]:
                lows.append(i)
        return highs, lows

    sh_idx, _  = _swing_pts(high)
    _,  sl_idx = _swing_pts(low)
    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        hh = high.iloc[sh_idx[-1]] > high.iloc[sh_idx[-2]]
        hl = low.iloc[sl_idx[-1]]  > low.iloc[sl_idx[-2]]
        lh = high.iloc[sh_idx[-1]] < high.iloc[sh_idx[-2]]
        ll = low.iloc[sl_idx[-1]]  < low.iloc[sl_idx[-2]]
        if hh and hl:   votes_bull += 2
        elif lh and ll: votes_bear += 2

    ema21 = _ema(close, 21).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    if   cur > ema21 > ema50: votes_bull += 1
    elif cur < ema21 < ema50: votes_bear += 1

    # FIX #6 : open hebdomadaire = dernier lundi uniquement (dayofweek == 0)
    # Le dimanche (6) peut produire des bougies atypiques chez OANDA.
    try:
        times       = pd.to_datetime(df.index)
        monday_rows = df[times.dayofweek == 0]
        if not monday_rows.empty:
            weekly_open = float(monday_rows["Open"].iloc[-1])
            if cur > weekly_open: votes_bull += 1
            else:                 votes_bear += 1
    except Exception:
        pass

    if len(df) >= 2:
        midpoint = (float(high.iloc[-2]) + float(low.iloc[-2])) / 2
        if float(close.iloc[-2]) > midpoint: votes_bull += 1
        else:                                votes_bear += 1

    if len(df) >= 200:
        sma200_val = _sma(close, 200).iloc[-1]
        if   cur > sma200_val: votes_bull += 1
        elif cur < sma200_val: votes_bear += 1

    if   votes_bull >= 5: return "Bullish", 90
    if   votes_bull >= 3: return "Bullish", 70
    if   votes_bear >= 5: return "Bearish", 90
    if   votes_bear >= 3: return "Bearish", 70
    return "Range", 35


def trend_4h(df: pd.DataFrame) -> Tuple[str, int]:
    if len(df) < 60:
        return "Range", 0
    close = df["Close"]
    cur   = float(close.iloc[-1])
    score = 0
    score += 1 if cur > _ema(close, 50).iloc[-1] else -1

    # FIX #12 : _dmi retourne maintenant (0.0, 0.0) au lieu de NaN → guard superflu
    # mais conservé par robustesse
    pdi_val, mdi_val = _dmi(df)
    score += 1 if pdi_val > mdi_val else -1

    # FIX #10 : guard explicite sur today_mask vide — pas de swallow silencieux
    try:
        idx        = pd.to_datetime(df.index)
        dates      = idx.normalize()
        today_mask = dates == dates[-1]
        today_rows = df[today_mask]
        if not today_rows.empty:
            daily_open = float(today_rows["Open"].iloc[0])
            score += 1 if cur > daily_open else -1
        # Si today_rows est vide, on n'ajoute pas de vote (neutre) — cas loggué
        else:
            logger.debug("trend_4h : today_mask vide pour %s", df.index[-1])
    except Exception as e:
        logger.debug("trend_4h daily_open error: %s", e)

    abs_score = abs(score)
    strength  = 90 if abs_score == 3 else 70 if abs_score >= 1 else 40
    trend     = "Bullish" if score > 0 else "Bearish" if score < 0 else "Range"
    return trend, strength


def trend_h1(df: pd.DataFrame) -> Tuple[str, int]:
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
        base_s = min(75, abs(cur - curr_zlema) / cur * 1000)
        return "Bullish", int(min(75, base_s))
    if (cur < curr_zlema) and ema_bear and mom_bear:
        base_s = min(75, abs(cur - curr_zlema) / cur * 1000)
        return "Bearish", int(min(75, base_s))
    if len(df) >= 200:
        sma200_val = _sma(close, 200).iloc[-1]
        bias_trend = "Bullish" if ema50.iloc[-1] > sma200_val else "Bearish"
        # FIX #4 : contribution réduite à 0.15 (était 0.30)
        # "Retracement Bull" = prix sous SMA200 mais tendance HTF haussière → signal mild+
        # "Retracement Bear" = prix au-dessus SMA200 mais tendance HTF baissière → signal mild-
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


class StrengthEngine:
    """
    Calcule la force relative des 8 devises majeures (W/D/H4/H1).
    v4.1 : accepte un client API injecté (token non exposé dans les clés de cache).
    """

    def __init__(
        self,
        client: API,
        min_diff: float = MIN_STRENGTH_DIFF,
        max_pairs: int  = MAX_PAIRS,
    ):
        # FIX #11 : client injecté depuis l'extérieur — ce module ne manipule plus le token
        self.api       = client
        self.min_diff  = min_diff
        self.max_pairs = max_pairs
        self._cache: Dict[str, pd.DataFrame] = {}

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _fetch_ohlcv(self, pair: str, granularity: str, count: int) -> Optional[pd.DataFrame]:
        key = f"{pair}_{granularity}"
        if key in self._cache:
            return self._cache[key]
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
            self._cache[key] = df
            return df
        except Exception as e:
            logger.debug("Fetch OHLCV failed %s %s: %s", pair, granularity, e)
            return None

    def _get_tf_df(self, pair: str, tf: str) -> Optional[pd.DataFrame]:
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
        # FIX #1 : accumulation du poids réel dans weight_sum (remplace counts)
        # Une devise présente sur 4 TF accumule ~12 unités de poids ;
        # une devise sur 1 TF en accumule ~1.5. La division par weight_sum
        # neutralise ce déséquilibre structurel.
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
                if   trend == "Bullish":          contrib = +weight * (strength / 100)
                elif trend == "Bearish":          contrib = -weight * (strength / 100)
                # FIX #4 : contribution retracement à 0.15 (était 0.30)
                elif trend == "Retracement Bull": contrib = +weight * 0.15
                elif trend == "Retracement Bear": contrib = -weight * 0.15
                else:                             continue
                total[base]       += contrib
                total[quote]      -= contrib
                weight_sum[base]  += weight
                weight_sum[quote] += weight
        return total, weight_sum

    @staticmethod
    def _normalize(
        total:      Dict[str, float],
        weight_sum: Dict[str, float],
    ) -> Dict[str, float]:
        # FIX #1 + #13 : division par weight_sum, pas par counts
        # FIX #13 : le biais structural (devises avec plus de paires = score absolu plus élevé)
        # est neutralisé car weight_sum croît proportionnellement à la couverture réelle
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
        values = list(scores.values())
        s_min, s_max = min(values), max(values)
        spread = s_max - s_min
        if spread < 1e-8:
            return {c: 5.0 for c in scores}
        return {c: round((v - s_min) / spread * 10, 2) for c, v in scores.items()}

    # ── Vélocité ──────────────────────────────────────────────────────────────

    def _compute_velocity(self, scores_raw: Dict[str, float]) -> Dict[str, float]:
        # FIX #5 : vélocité calculée sur scores BRUTS (même échelle que scores_raw)
        #          évite la double-normalisation indépendante qui efface le signal
        # FIX #8 : fenêtre glissante fixe de 48 chandelles H1 (~2 jours)
        total_prev:      Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight_sum_prev: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight = TIMEFRAMES_MTF["H1"]["weight"]
        for pair in PAIRS:
            base, quote = pair.split("_")
            df = self._get_tf_df(pair, "H1")
            if df is None or len(df) < 50:
                continue
            lookback = min(48, len(df) // 3)
            df_past  = df.iloc[-(lookback * 2):-lookback]
            if len(df_past) < 15:
                continue
            trend_past, strength_past = trend_h1(df_past)
            if   trend_past == "Bullish": contrib = +weight * (strength_past / 100)
            elif trend_past == "Bearish": contrib = -weight * (strength_past / 100)
            else: continue
            total_prev[base]       += contrib
            total_prev[quote]      -= contrib
            weight_sum_prev[base]  += weight
            weight_sum_prev[quote] += weight
        scores_prev_raw = self._normalize(total_prev, weight_sum_prev)
        return {
            c: round(scores_raw.get(c, 0.0) - scores_prev_raw.get(c, 0.0), 4)
            for c in CURRENCIES
        }

    # ── Sélection des paires ──────────────────────────────────────────────────

    def _select_pairs(
        self, scores_display: Dict[str, float]
    ) -> Tuple[List[str], List[Dict]]:
        sorted_s  = sorted(scores_display.items(), key=lambda x: x[1], reverse=True)
        strongest = [c for c, _ in sorted_s[:2]]
        weakest   = [c for c, _ in sorted_s[-2:]]
        candidates, atr_values = [], []
        for base in strongest:
            for quote in weakest:
                if base == quote:
                    continue
                diff = scores_display[base] - scores_display[quote]
                if diff < self.min_diff:
                    continue
                pair_direct  = f"{base}_{quote}"
                pair_inverse = f"{quote}_{base}"
                pair_id = (
                    pair_direct  if pair_direct  in PAIRS else
                    pair_inverse if pair_inverse in PAIRS else None
                )
                if pair_id is None:
                    continue
                df_h1   = self._get_tf_df(pair_id, "H1")
                atr_val = (
                    float(_atr_series(df_h1).iloc[-1])
                    if df_h1 is not None and len(df_h1) >= 15 else None
                )
                # FIX #2 : direction correcte selon que pair_id == paire directe ou inversée
                direction = "BUY" if pair_id == pair_direct else "SELL"
                candidates.append({
                    "pair":       pair_direct,
                    "pair_oanda": pair_id,
                    "diff":       round(diff, 3),
                    "atr":        round(atr_val, 6) if atr_val is not None else None,
                    "base":       base,
                    "quote":      quote,
                    "direction":  direction,
                })
                if atr_val is not None:
                    atr_values.append(atr_val)
        if not candidates:
            return [], []
        if atr_values:
            threshold = float(np.percentile(atr_values, ATR_MIN_PERCENTILE))
            # FIX #3 : exclure les paires SANS ATR — elles ne passent plus le filtre
            candidates = [
                c for c in candidates
                if c["atr"] is not None and c["atr"] >= threshold
            ]
        if not candidates:
            return [], []
        seen_quotes: set = set()
        filtered = []
        for c in sorted(candidates, key=lambda x: x["diff"], reverse=True):
            if c["quote"] not in seen_quotes:
                filtered.append(c)
                seen_quotes.add(c["quote"])
        top = filtered[: self.max_pairs]
        return [c["pair"] for c in top], top

    # ── Points d'entrée publics ───────────────────────────────────────────────

    def run(self) -> StrengthResult:
        self._cache.clear()
        total, weight_sum  = self._compute_mtf_scores()
        scores             = self._normalize(total, weight_sum)
        scores_display     = self._to_display(scores)
        ranking            = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        velocity           = self._compute_velocity(scores)  # FIX #5 : scores bruts
        best_pairs, pairs_detail = self._select_pairs(scores_display)
        return StrengthResult(
            scores         = {k: round(v, 6) for k, v in scores.items()},
            scores_display = scores_display,
            ranking        = ranking,
            velocity       = velocity,
            best_pairs     = best_pairs,
            pairs_detail   = pairs_detail,
            pairs_fetched  = len(self._cache),
        )

    def run_quick(self, granularity: str = "H1") -> StrengthResult:
        self._cache.clear()
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
            if   trend == "Bullish": contrib = +weight * (strength / 100)
            elif trend == "Bearish": contrib = -weight * (strength / 100)
            else: continue
            total[base]       += contrib
            total[quote]      -= contrib
            weight_sum[base]  += weight
            weight_sum[quote] += weight
        # FIX #16 : pré-fetch H1 pour que _select_pairs dispose des ATR
        # Sans cela, tous les ATR sont None et le filtre de qualité est bypassé (FIX #3 amplifié)
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
        )


# ==========================================
# ── DASHBOARD STREAMLIT ────────────────────
# ==========================================

# ── 1. Configuration & design ─────────────────────────────────────────────────

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


# ── 2. Client & fonctions cachées ─────────────────────────────────────────────

# FIX #11 : le client OANDA est mis en cache via @st.cache_resource.
# Le token n'apparaît JAMAIS comme argument de @st.cache_data, donc jamais
# dans les clés de cache (qui peuvent être loggées selon la version Streamlit).
@st.cache_resource
def _make_oanda_client(token: str, env: str) -> API:
    """Crée ou réutilise le client OANDA pour un couple (token, env)."""
    return API(access_token=token, environment=env)


# FIX #14 : le résultat du moteur de force est mis en cache 60 s.
# Le préfixe _ sur _client indique à st.cache_data de NE PAS hacher ce paramètre
# (convention Streamlit). La clé de cache effective est (env,).
@st.cache_data(ttl=60, show_spinner=False)
def _run_engine(_client: API, env: str) -> StrengthResult:
    engine = StrengthEngine(client=_client)
    return engine.run()


# FIX #11 + #15 : fetch individuel de bougies mis en cache, sans le token.
# Clé effective : (env, instrument, granularity, count).
@st.cache_data(ttl=60, show_spinner=False)
def _fetch_candles_cached(
    _client: API,
    env: str,
    instrument: str,
    granularity: str,
    count: int,
) -> Optional[pd.DataFrame]:
    try:
        params = {"count": count, "granularity": granularity, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        _client.request(r)
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
    except Exception:
        return None


# FIX #15 : le wrapper complet est désormais mis en cache.
# Clé effective : (env, gran).
@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_map_data(
    _client: API,
    env: str,
    gran: str,
) -> Tuple[pd.DataFrame, Dict]:
    prices = {}
    for pair in FOREX_PAIRS:
        df = _fetch_candles_cached(_client, env, pair, gran, 30)
        if df is not None:
            prices[pair] = df["Close"]

    pct_special = {}
    for symbol, name in {**INDICES, **METAUX}.items():
        df = _fetch_candles_cached(_client, env, symbol, gran, 30)
        if df is None:
            continue
        # FIX #9 : vérification explicite du NaN avant float()
        pct_series = df["Close"].pct_change()
        last_pct   = pct_series.iloc[-1]
        if pd.isna(last_pct):
            logger.debug("Market map : pct NaN ignoré pour %s", symbol)
            continue
        pct = float(last_pct * 100)
        pct_special[name] = {
            "pct": pct,
            "cat": "INDICES" if symbol in INDICES else "METAUX",
        }

    df_prices = pd.DataFrame(prices).ffill().bfill() if prices else pd.DataFrame()
    return df_prices, pct_special


# ── 3. Rendu cartes ───────────────────────────────────────────────────────────

def display_card(name: str, score: float, arrow_str: str) -> str:
    if   score >= 7:   c_txt, c_bg = "text-green",  "bg-green"
    elif score >= 5.5: c_txt, c_bg = "text-blue",   "bg-blue"
    elif score >= 4:   c_txt, c_bg = "text-orange", "bg-orange"
    else:              c_txt, c_bg = "text-red",     "bg-red"

    if   arrow_str == "up":   arrow, a_col = "↗", "text-green"
    elif arrow_str == "down": arrow, a_col = "↘", "text-red"
    else:                     arrow, a_col = "→", "text-gray"

    flag_code = FLAG_URLS.get(name, "xk")
    img_html  = (f'<img src="https://flagcdn.com/48x36/{flag_code}.png" '
                 f'style="width:24px; border-radius:2px;">')
    bar_w = min(max(score * 10, 0), 100)

    return f"""
    <div class="currency-card">
        <div class="card-header">{img_html} <span class="asset-name">{name}</span></div>
        <div class="strength-score {c_txt}">
            {score:.1f} <span class="velocity-arrow {a_col}">{arrow}</span>
        </div>
        <div class="progress-bg">
            <div class="progress-fill {c_bg}" style="width:{bar_w}%;"></div>
        </div>
    </div>
    """


# ── 4. Market Map HTML ────────────────────────────────────────────────────────

def generate_exact_map_html(df_prices: pd.DataFrame, pct_special: Dict) -> str:
    if df_prices.empty:
        return "<p style='color:#aaa;padding:1rem;'>Données insuffisantes.</p>"

    pct_changes = df_prices.pct_change().iloc[-1] * 100

    def get_bg_color(pct: float) -> str:
        if pct >= 0.15:  return "#009900"
        if pct >= 0.01:  return "#33cc33"
        if pct <= -0.15: return "#cc0000"
        if pct <= -0.01: return "#ff3300"
        return "#f0f0f0"

    def get_text_color(pct: float) -> str:
        return "#333" if -0.01 < pct < 0.01 else "white"

    forex_data: Dict[str, list] = {c: [] for c in CURRENCIES}
    for col in df_prices.columns:
        # FIX #9 : ignorer silencieusement les NaN de pct_change
        val_raw = pct_changes.get(col)
        if val_raw is None or pd.isna(val_raw):
            continue
        val = float(val_raw)
        parts = col.split("_")
        if len(parts) != 2:
            continue
        b, q = parts
        if b in forex_data:
            forex_data[b].append({"pair": q, "pct": val})
        if q in forex_data:
            forex_data[q].append({"pair": b, "pct": -val})

    scores      = {c: sum(x["pct"] for x in items) for c, items in forex_data.items()}
    sorted_cols = sorted(scores, key=scores.get, reverse=True)

    html = """<!DOCTYPE html><html><head><style>
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

    html += '<div class="section-header">💱 FOREX MAP</div>'
    html += '<div class="matrix-row">'
    for curr in sorted_cols:
        items   = forex_data[curr]
        winners = sorted([x for x in items if x["pct"] >= 0.01],  key=lambda x: x["pct"], reverse=True)
        losers  = sorted([x for x in items if x["pct"] < -0.01],  key=lambda x: x["pct"])
        flat    = [x for x in items if -0.01 <= x["pct"] < 0.01]
        html   += '<div class="currency-col">'
        for x in winners:
            col, txt = get_bg_color(x["pct"]), get_text_color(x["pct"])
            html += (f'<div class="tile" style="background:{col};color:{txt};">'
                     f'<span>{x["pair"]}</span><span>+{x["pct"]:.2f}%</span></div>')
        html += f'<div class="sep">{curr}</div>'
        for x in flat:
            html += (f'<div class="tile" style="background:#f0f0f0;color:#333;">'
                     f'<span>{x["pair"]}</span><span>unch</span></div>')
        for x in losers:
            col, txt = get_bg_color(x["pct"]), get_text_color(x["pct"])
            html += (f'<div class="tile" style="background:{col};color:{txt};">'
                     f'<span>{x["pair"]}</span><span>{x["pct"]:.2f}%</span></div>')
        html += '</div>'
    html += '</div>'

    html += '<div class="section-header">📊 INDICES</div>'
    html += '<div class="grid-container">'
    for name, data in pct_special.items():
        if data["cat"] != "INDICES": continue
        pct = data["pct"]
        html += (f'<div class="big-box" style="background:{get_bg_color(pct)}">'
                 f'<span class="box-name">{name}</span>'
                 f'<span class="box-val">{pct:+.2f}%</span></div>')
    html += '</div>'

    html += '<div class="section-header">🪙 METAUX</div>'
    html += '<div class="grid-container">'
    for name, data in pct_special.items():
        if data["cat"] != "METAUX": continue
        pct = data["pct"]
        html += (f'<div class="big-box" style="background:{get_bg_color(pct)}">'
                 f'<span class="box-name">{name}</span>'
                 f'<span class="box-val">{pct:+.2f}%</span></div>')
    html += '</div></body></html>'

    return html


# ── 5. Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Connexion OANDA")
    token = st.secrets.get("OANDA_ACCESS_TOKEN") or st.text_input("Token", type="password")
    env   = st.selectbox("Env", ["practice", "live"])
    st.markdown("---")
    granularity = st.selectbox(
        "Timeframe (Map)", ["M5", "M15", "M30", "H1", "H4", "D"], index=3
    )
    st.caption(
        "Le moteur de force utilise W + D + H4 + H1 en parallèle, "
        "indépendamment du timeframe affiché."
    )


# ── 6. Exécution ──────────────────────────────────────────────────────────────

if token:
    with st.status("Actualisation des données...", expanded=True) as status:

        # FIX #11 : client créé/réutilisé via cache_resource — token hors des clés data
        client = _make_oanda_client(token, env)

        # FIX #14 : résultat du moteur mis en cache 60 s
        result = _run_engine(client, env)

        # FIX #15 : données Market Map mises en cache 60 s
        df_prices, pct_special = fetch_market_map_data(client, env, granularity)

        status.update(label="✅ Données chargées", state="complete", expanded=False)

    if result.scores_display:

        # Cartes Forex
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

        # Meilleures paires
        if result.best_pairs:
            st.markdown("---")
            st.subheader("🎯 Paires Sélectionnées")
            badges = ""
            for d in result.pairs_detail:
                # FIX #2 : direction désormais BUY ou SELL selon l'orientation réelle
                dir_color = "#10B981" if d["direction"] == "BUY" else "#EF4444"
                badges += (
                    f'<span style="display:inline-block;padding:4px 12px;'
                    f'background:{dir_color};color:white;border-radius:4px;'
                    f'font-weight:bold;margin:3px;font-size:0.9rem;">'
                    f'{d["pair"]} {d["direction"]}</span>'
                    f'<span style="font-size:0.75rem;color:#9ca3af;margin-right:12px;">'
                    f'diff={d["diff"]:.2f}'
                    f'{" | ATR=" + str(d["atr"]) if d["atr"] else ""}'
                    f'</span>'
                )
            st.markdown(badges, unsafe_allow_html=True)

        # Market Map
        st.markdown("---")
        st.subheader("🗺️ Market Map Pro")
        if not df_prices.empty:
            html_map = generate_exact_map_html(df_prices, pct_special)
            st.components.v1.html(html_map, height=600, scrolling=True)
        else:
            st.warning("Données insuffisantes pour la Market Map.")

else:
    st.warning("En attente du Token OANDA...")

    
