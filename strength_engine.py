"""
strength_engine.py
==================
Moteur de force de devises — VERSION 4.0

Moteur institutionnel multi-timeframe.
Zéro dépendance UI. Importable par n'importe quel bot ou dashboard.

Architecture MTF (du plus haut au plus bas, sans Monthly) :
  W  — EMA50 vs SMA200 (bias macro)          poids 4.0
  D  — Swing structure + EMA + SMA200        poids 4.0
  H4 — EMA50 + DMI + Daily Open              poids 2.5
  H1 — ZLEMA + EMA stack + RSI/MACD          poids 1.5

Usage minimal :
    from strength_engine import StrengthEngine
    engine = StrengthEngine(token="...", env="practice")
    result = engine.run()
    print(result.scores_display)  # {"USD": 7.4, "EUR": 4.8, ...}  0-10
    print(result.best_pairs)      # ["AUD_NZD", "GBP_JPY"]
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

PAIRS: List[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
]

CURRENCIES: List[str] = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]

# Timeframes institutionnels — W = plus haut TF (pas de Monthly)
# gran_fetch   : granularite OANDA a fetcher
# count        : nombre de bougies
# weight       : poids dans le score final
# resample_rule: regle pandas (None = pas de resampling)
TIMEFRAMES_MTF: Dict[str, dict] = {
    "W":  {"gran_fetch": "D",  "count": 2000, "weight": 4.0, "resample_rule": "W-FRI"},
    "D":  {"gran_fetch": "D",  "count": 300,  "weight": 4.0, "resample_rule": None},
    "H4": {"gran_fetch": "H4", "count": 300,  "weight": 2.5, "resample_rule": None},
    "H1": {"gran_fetch": "H1", "count": 300,  "weight": 1.5, "resample_rule": None},
}

# Seuil de difference 0-10 pour une paire "tradable"
MIN_STRENGTH_DIFF: float = 1.5

# Percentile ATR minimum pour le filtre volatilite
ATR_MIN_PERCENTILE: int = 25

# Nombre max de paires retournees
MAX_PAIRS: int = 3

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATACLASS DE RESULTAT
# ─────────────────────────────────────────────

@dataclass
class StrengthResult:
    """
    Resultat complet retourne par StrengthEngine.run().
    Tous les champs sont serialisables JSON.
    """
    scores: Dict[str, float] = field(default_factory=dict)
    scores_display: Dict[str, float] = field(default_factory=dict)
    ranking: List[str] = field(default_factory=list)
    velocity: Dict[str, float] = field(default_factory=dict)
    best_pairs: List[str] = field(default_factory=list)
    pairs_detail: List[Dict] = field(default_factory=list)
    pairs_fetched: int = 0

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "scores_display": self.scores_display,
            "ranking": self.ranking,
            "velocity": self.velocity,
            "best_pairs": self.best_pairs,
            "pairs_detail": self.pairs_detail,
            "pairs_fetched": self.pairs_fetched,
        }

    def direction_arrow(self, currency: str) -> str:
        v = self.velocity.get(currency, 0.0)
        if v > 0.3:  return "up"
        if v < -0.3: return "down"
        return "flat"

    def color_class(self, currency: str) -> str:
        s = self.scores_display.get(currency, 5.0)
        if s >= 7.0: return "strong-bull"
        if s >= 5.5: return "mild-bull"
        if s >= 4.0: return "mild-bear"
        return "strong-bear"


# ─────────────────────────────────────────────
# FONCTIONS TECHNIQUES PURES
# ─────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
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
    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down) & (up > 0), 0.0)
    mdm  = down.where((down > up) & (down > 0), 0.0)
    pdi  = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi  = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    return float(pdi.iloc[-1]), float(mdi.iloc[-1])


# ─────────────────────────────────────────────
# FONCTIONS DE TENDANCE INSTITUTIONNELLES
# Portees directement depuis GPS V2.1
# ─────────────────────────────────────────────

def trend_weekly(df: pd.DataFrame) -> Tuple[str, int]:
    """
    Biais Weekly — EMA50 vs SMA200.
    Croisement recent = 90 | Tendance etablie = 75 | Range = 40.
    """
    if len(df) < 50:
        return "Range", 0

    close  = df["Close"]
    ema50  = _ema(close, 50)
    sma200 = _sma(close, 200) if len(df) >= 200 else _ema(close, 100)

    curr_ema50, prev_ema50   = ema50.iloc[-1],  ema50.iloc[-2]
    curr_sma200, prev_sma200 = sma200.iloc[-1], sma200.iloc[-2]

    crossed_bull = (prev_ema50 <= prev_sma200) and (curr_ema50 > curr_sma200)
    crossed_bear = (prev_ema50 >= prev_sma200) and (curr_ema50 < curr_sma200)

    if curr_ema50 > curr_sma200:
        return "Bullish", 90 if crossed_bull else 75
    if curr_ema50 < curr_sma200:
        return "Bearish", 90 if crossed_bear else 75
    return "Range", 40


def trend_daily(df: pd.DataFrame) -> Tuple[str, int]:
    """
    Biais Daily — 5 facteurs institutionnels, 6 votes max.
    Source: calc_institutional_trend_daily() GPS V2.1

      1. Structure swing D1 (wing=5)  -> 2 votes si HH/HL ou LH/LL
      2. EMA 21/50 stack              -> 1 vote
      3. Weekly Open                  -> 1 vote
      4. Close J-1 vs midpoint J-1    -> 1 vote
      5. SMA 200                      -> 1 vote
    """
    if len(df) < 60:
        return "Range", 0

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    cur   = float(close.iloc[-1])
    votes_bull = votes_bear = 0

    # Facteur 1 : Swing structure
    def _swing_pts(series: pd.Series, wing: int = 5):
        highs, lows = [], []
        for i in range(wing, len(series) - wing):
            w = series.iloc[i - wing: i + wing + 1]
            if series.iloc[i] == w.max(): highs.append(i)
            if series.iloc[i] == w.min(): lows.append(i)
        return highs, lows

    sh_idx, _      = _swing_pts(high)
    _,      sl_idx = _swing_pts(low)

    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        hh = high.iloc[sh_idx[-1]] > high.iloc[sh_idx[-2]]
        hl = low.iloc[sl_idx[-1]]  > low.iloc[sl_idx[-2]]
        lh = high.iloc[sh_idx[-1]] < high.iloc[sh_idx[-2]]
        ll = low.iloc[sl_idx[-1]]  < low.iloc[sl_idx[-2]]
        if hh and hl:   votes_bull += 2
        elif lh and ll: votes_bear += 2

    # Facteur 2 : EMA 21/50
    ema21 = _ema(close, 21).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    if   cur > ema21 > ema50: votes_bull += 1
    elif cur < ema21 < ema50: votes_bear += 1

    # Facteur 3 : Weekly Open
    try:
        times = pd.to_datetime(df.index)
        wo_rows = df[times.dayofweek.isin([0, 6])]
        if not wo_rows.empty:
            weekly_open = float(wo_rows["Open"].iloc[-1])
            if cur > weekly_open: votes_bull += 1
            else:                 votes_bear += 1
    except Exception:
        pass

    # Facteur 4 : Close J-1 vs midpoint J-1
    if len(df) >= 2:
        midpoint = (float(high.iloc[-2]) + float(low.iloc[-2])) / 2
        if float(close.iloc[-2]) > midpoint: votes_bull += 1
        else:                                votes_bear += 1

    # Facteur 5 : SMA 200
    if len(df) >= 200:
        sma200_val = _sma(close, 200).iloc[-1]
        if   cur > sma200_val: votes_bull += 1
        elif cur < sma200_val: votes_bear += 1

    # Resolution
    if   votes_bull >= 5: return "Bullish", 90
    if   votes_bull >= 3: return "Bullish", 70
    if   votes_bear >= 5: return "Bearish", 90
    if   votes_bear >= 3: return "Bearish", 70
    return "Range", 35


def trend_4h(df: pd.DataFrame) -> Tuple[str, int]:
    """
    Biais H4 — 3 facteurs orthogonaux.
    Source: calc_institutional_trend_4h() GPS V2.1

      1. Prix vs EMA 50   -> tendance de fond
      2. DI+ vs DI-       -> momentum directionnel
      3. Daily Open       -> reference institutionnelle

    Score +-3 -> 90 | +-1/2 -> 70 | 0 -> 40
    """
    if len(df) < 60:
        return "Range", 0

    close = df["Close"]
    cur   = float(close.iloc[-1])
    score = 0

    # Facteur 1 : EMA 50
    score += 1 if cur > _ema(close, 50).iloc[-1] else -1

    # Facteur 2 : DMI
    pdi_val, mdi_val = _dmi(df)
    if not (np.isnan(pdi_val) or np.isnan(mdi_val)):
        score += 1 if pdi_val > mdi_val else -1

    # Facteur 3 : Daily Open
    try:
        idx   = pd.to_datetime(df.index)
        dates = idx.normalize()
        today_mask = dates == dates[-1]
        daily_open = float(df[today_mask]["Open"].iloc[0])
        score += 1 if cur > daily_open else -1
    except Exception:
        pass

    abs_score = abs(score)
    strength  = 90 if abs_score == 3 else 70 if abs_score >= 1 else 40
    trend     = "Bullish" if score > 0 else "Bearish" if score < 0 else "Range"
    return trend, strength


def trend_h1(df: pd.DataFrame) -> Tuple[str, int]:
    """
    Biais H1 — ZLEMA + EMA stack + RSI/MACD.
    Source: calc_institutional_trend_intraday() GPS V2.1
    """
    if len(df) < 50:
        return "Range", 0

    close = df["Close"]
    cur   = float(close.iloc[-1])

    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)

    # ZLEMA (Zero-Lag EMA, lag=17)
    lag        = 17
    src_adj    = close + (close - close.shift(lag))
    curr_zlema = _ema(src_adj, 50).iloc[-1]

    # RSI + MACD
    rsi_val    = _rsi(close, 14).iloc[-1]
    macd_line  = _ema(close, 12) - _ema(close, 26)
    curr_macd  = macd_line.iloc[-1]
    curr_sig   = _ema(macd_line, 9).iloc[-1]

    ema_bull = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema21.iloc[-1] > ema50.iloc[-1])
    ema_bear = (ema9.iloc[-1] < ema21.iloc[-1]) and (ema21.iloc[-1] < ema50.iloc[-1])
    mom_bull = (rsi_val > 50) and (curr_macd > curr_sig)
    mom_bear = (rsi_val < 50) and (curr_macd < curr_sig)

    if (cur > curr_zlema) and ema_bull and mom_bull:
        base_s = min(75, abs(cur - curr_zlema) / cur * 1000)
        return "Bullish", int(min(75, base_s))
    if (cur < curr_zlema) and ema_bear and mom_bear:
        base_s = min(75, abs(cur - curr_zlema) / cur * 1000)
        return "Bearish", int(min(75, base_s))

    # Retracement (signal partiel)
    if len(df) >= 200:
        sma200_val   = _sma(close, 200).iloc[-1]
        bias_trend   = "Bullish" if ema50.iloc[-1] > sma200_val else "Bearish"
        if cur < sma200_val and bias_trend == "Bullish":
            return "Retracement Bull", 30
        if cur > sma200_val and bias_trend == "Bearish":
            return "Retracement Bear", 30

    return "Range", 25


# Dispatch TF -> fonction
_TREND_FN = {
    "W":  trend_weekly,
    "D":  trend_daily,
    "H4": trend_4h,
    "H1": trend_h1,
}


# ─────────────────────────────────────────────
# MOTEUR PRINCIPAL
# ─────────────────────────────────────────────

class StrengthEngine:
    """
    Calcule la force relative des 8 devises majeures.

    Pour chaque paire Forex, calcule la tendance institutionnelle
    sur W/D/H4/H1 (fonctions GPS V2.1) et accumule les contributions
    ponderees par devise (base += contribution, quote -= contribution).

    Le score final est normalise 0-10 pour l'affichage et la selection.

    Parametres
    ----------
    token    : Token OANDA (practice ou live).
    env      : "practice" ou "live".
    min_diff : Seuil de difference 0-10 pour selectionner une paire.
    max_pairs: Paires max retournees.
    """

    def __init__(
        self,
        token: str,
        env: str = "practice",
        min_diff: float = MIN_STRENGTH_DIFF,
        max_pairs: int = MAX_PAIRS,
    ):
        self.api       = API(access_token=token, environment=env)
        self.min_diff  = min_diff
        self.max_pairs = max_pairs
        self._cache: Dict[str, pd.DataFrame] = {}

    # ─── FETCH OHLCV ──────────────────────────

    def _fetch_ohlcv(
        self, pair: str, granularity: str, count: int
    ) -> Optional[pd.DataFrame]:
        """
        Recupere OHLCV (bougies completes), mis en cache par (pair, gran).
        Index : DatetimeIndex UTC.
        """
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
                for c in r.response["candles"]
                if c["complete"]
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
        """
        Retourne le DataFrame pret pour la fonction de tendance du TF donne.
        Gere le resampling Weekly.
        """
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

    # ─── SCORES MTF ───────────────────────────

    def _compute_mtf_scores(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Pour chaque paire, accumule les contributions institutionnelles ponderees.

        Regles de contribution :
          Bullish          -> base += weight * (strength/100)
                             quote -= weight * (strength/100)
          Bearish          -> base -= weight * (strength/100)
                             quote += weight * (strength/100)
          Retracement Bull -> base += weight * 0.30  (signal partiel)
          Retracement Bear -> base -= weight * 0.30
          Range            -> 0 (ignoré)
        """
        total:  Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        counts: Dict[str, int]   = {c: 0   for c in CURRENCIES}

        for pair in PAIRS:
            base, quote   = pair.split("_")
            contributed   = False

            for tf, cfg in TIMEFRAMES_MTF.items():
                df = self._get_tf_df(pair, tf)
                if df is None:
                    continue

                trend, strength = _TREND_FN[tf](df)
                weight = cfg["weight"]

                if trend == "Bullish":
                    contrib = +weight * (strength / 100)
                elif trend == "Bearish":
                    contrib = -weight * (strength / 100)
                elif trend == "Retracement Bull":
                    contrib = +weight * 0.30
                elif trend == "Retracement Bear":
                    contrib = -weight * 0.30
                else:
                    continue  # Range = pas de contribution

                total[base]  += contrib
                total[quote] -= contrib
                contributed   = True

            if contributed:
                counts[base]  += 1
                counts[quote] += 1

        return total, counts

    @staticmethod
    def _normalize(
        total: Dict[str, float], counts: Dict[str, int]
    ) -> Dict[str, float]:
        scores = {}
        for c in CURRENCIES:
            scores[c] = total[c] / counts[c] if counts[c] > 0 else 0.0
            if counts[c] == 0:
                logger.warning("Devise %s : aucune donnee recue.", c)
        return scores

    @staticmethod
    def _to_display(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalisation min-max robuste -> echelle 0-10."""
        values = list(scores.values())
        s_min, s_max = min(values), max(values)
        spread = s_max - s_min
        if spread < 1e-8:
            return {c: 5.0 for c in scores}
        return {c: round((v - s_min) / spread * 10, 2) for c, v in scores.items()}

    # ─── VELOCITE ─────────────────────────────

    def _compute_velocity(
        self, scores_display: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare le score H1 actuel au score calcule sur la premiere moitie
        de la serie H1 (deja en cache).
        Delta en unites 0-10 -> indique l'acceleration recente.
        """
        total_prev:  Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        counts_prev: Dict[str, int]   = {c: 0   for c in CURRENCIES}

        weight = TIMEFRAMES_MTF["H1"]["weight"]

        for pair in PAIRS:
            base, quote = pair.split("_")
            df = self._get_tf_df(pair, "H1")
            if df is None or len(df) < 30:
                continue

            # Premiere moitie = "passe"
            split    = max(15, len(df) // 2)
            df_past  = df.iloc[:split]

            trend_past, strength_past = trend_h1(df_past)

            if trend_past == "Bullish":
                contrib = +weight * (strength_past / 100)
            elif trend_past == "Bearish":
                contrib = -weight * (strength_past / 100)
            else:
                continue

            total_prev[base]  += contrib
            total_prev[quote] -= contrib
            counts_prev[base]  += 1
            counts_prev[quote] += 1

        scores_prev_raw  = self._normalize(total_prev, counts_prev)
        scores_prev_disp = self._to_display(scores_prev_raw)

        return {
            c: round(scores_display.get(c, 5.0) - scores_prev_disp.get(c, 5.0), 3)
            for c in CURRENCIES
        }

    # ─── SELECTION DES PAIRES ─────────────────

    def _select_pairs(
        self, scores_display: Dict[str, float]
    ) -> Tuple[List[str], List[Dict]]:
        """
        Selection en 4 filtres successifs :
        1. Top 2 fortes vs Bottom 2 faibles
        2. Difference >= min_diff (unites 0-10)
        3. Volatilite ATR (percentile 25)
        4. Anti-correlation : 1 seule paire par devise cotee
        """
        sorted_s  = sorted(scores_display.items(), key=lambda x: x[1], reverse=True)
        strongest = [c for c, _ in sorted_s[:2]]
        weakest   = [c for c, _ in sorted_s[-2:]]

        candidates = []
        atr_values = []

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
                    if df_h1 is not None and len(df_h1) >= 15
                    else None
                )

                candidates.append({
                    "pair":       pair_direct,
                    "pair_oanda": pair_id,
                    "diff":       round(diff, 3),
                    "atr":        round(atr_val, 6) if atr_val else None,
                    "base":       base,
                    "quote":      quote,
                    "direction":  "BUY",
                })
                if atr_val:
                    atr_values.append(atr_val)

        if not candidates:
            return [], []

        # Filtre ATR
        if atr_values:
            threshold  = float(np.percentile(atr_values, ATR_MIN_PERCENTILE))
            candidates = [
                c for c in candidates
                if c["atr"] is None or c["atr"] >= threshold
            ]

        # Anti-correlation
        seen_quotes: set = set()
        filtered = []
        for c in sorted(candidates, key=lambda x: x["diff"], reverse=True):
            if c["quote"] not in seen_quotes:
                filtered.append(c)
                seen_quotes.add(c["quote"])

        top = filtered[: self.max_pairs]
        return [c["pair"] for c in top], top

    # ─── POINT D'ENTREE PUBLIC ────────────────

    def run(self) -> StrengthResult:
        """
        Calcul complet : W -> D -> H4 -> H1 (logique institutionnelle GPS V2.1).
        Weekly est le plus haut timeframe. Pas de Monthly.
        """
        self._cache.clear()

        total, counts  = self._compute_mtf_scores()
        scores         = self._normalize(total, counts)
        scores_display = self._to_display(scores)
        ranking        = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        velocity       = self._compute_velocity(scores_display)
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
        """
        Version allégee : 1 seul TF institutional.
        Idéale pour les bots haute frequence (M5/M15).
        """
        self._cache.clear()

        total:  Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        counts: Dict[str, int]   = {c: 0   for c in CURRENCIES}

        tf     = "H1" if granularity in ("H1", "M30", "M15", "M5") else "H4"
        cfg    = TIMEFRAMES_MTF[tf]
        weight = cfg["weight"]

        for pair in PAIRS:
            base, quote = pair.split("_")
            df = self._fetch_ohlcv(pair, cfg["gran_fetch"], cfg["count"])
            if df is None:
                continue

            trend, strength = _TREND_FN[tf](df)

            if trend == "Bullish":
                contrib = +weight * (strength / 100)
            elif trend == "Bearish":
                contrib = -weight * (strength / 100)
            else:
                continue

            total[base]  += contrib
            total[quote] -= contrib
            counts[base]  += 1
            counts[quote] += 1

        scores         = self._normalize(total, counts)
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


# ─────────────────────────────────────────────
# USAGE STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os, json
    token = os.getenv("OANDA_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Definir OANDA_ACCESS_TOKEN en variable d'environnement.")

    engine = StrengthEngine(token=token, env="practice")
    result = engine.run()
    print(json.dumps(result.to_dict(), indent=2))
