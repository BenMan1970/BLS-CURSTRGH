"""
Bluestar Market Dashboard — Strength Engine v11.0 (Production Hardened).

Le moteur de force (W / D / H4 / H1) est NUMÉRIQUEMENT IDENTIQUE à v10.1 / v10.0 / v4.4 :
les fonctions `trend_*`, `_normalize`, `_to_display`, `_compute_velocity`,
`_build_candidates` et `_filter_by_atr_and_exposure` sont inchangées et reçoivent les mêmes
séries OHLCV (mêmes granularités OANDA, mêmes compteurs de chandelles).

Durcissement v11.0 — aucune modification de la sémantique des scores :
  * Couche I/O réécrite : timeouts HTTP réels, retry/backoff sur 429 et 5xx, taxonomie
    d'erreurs typée. Plus aucune exception réseau ne peut tuer le script Streamlit.
  * Chargement parallèle des séries (ThreadPoolExecutor) via un bundle unique : ~90 requêtes
    OANDA en parallèle au lieu de ~150 séquentielles (temps de chargement ÷ 4-6).
  * Cache `st.cache_data` réellement isolé par empreinte de token (le fingerprint était
    ignoré à cause d'un préfixe `_`, deux tokens partageaient donc le même cache).
  * Résolution de configuration robuste et diagnostiquable : `st.secrets` (plusieurs noms
    acceptés, sections TOML), variables d'environnement ; l'app démarre sans config.
  * Écran de diagnostic actionnable au lieu d'un crash si token/env absent ou refusé.
  * Mémoire bornée (`max_entries`, séries stockées en colonnes) : évite les redémarrages OOM.
  * Exports JSON / briefing / PDF : fail-open, jamais bloquants.

Configuration attendue (Streamlit Cloud → Settings → Secrets) :

    OANDA_ACCESS_TOKEN = "votre_token_practice"      # ou OANDA_API_KEY / OANDA_TOKEN
    OANDA_ENVIRONMENT  = "practice"                  # practice | live (défaut : practice)
    # optionnel — token live distinct si vous basculez sur un compte réel
    OANDA_LIVE_ACCESS_TOKEN = "votre_token_live"

Dependencies: streamlit, oandapyV20, pandas, numpy, requests.
"""
# app.py — Bluestar Market Dashboard (Strength Engine v11.0)
# Moteur numérique inchangé. Couche I/O, configuration et robustesse refondues.

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import html
import json
import logging
import os
import platform
import random
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from datetime import timezone

# `tz` doit être une INSTANCE de tzinfo : la v10.1 importait la classe `timezone`, ce qui
# faisait échouer `datetime.datetime.now(tz)` (TypeError) dans _session_label(),
# generate_json_export() et generate_briefing_html() — crash systématique de la page.
tz = timezone.utc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
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

# Market Map smoothing: 1 = legacy exact (single-tick), 3+ = anti-flicker
MAP_SMOOTH_WINDOW: int = 1

# ── Réseau, cache et volumétrie ────────────────────────────────────────────────
HTTP_CONNECT_TIMEOUT_S: float = 6.0     # handshake TCP/TLS OANDA
HTTP_READ_TIMEOUT_S: float = 25.0       # lecture réponse (2000 chandelles ≈ 1 s)
HTTP_MAX_ATTEMPTS: int = 3              # 1 essai + 2 retries (429 / 5xx / réseau)
FETCH_MAX_WORKERS: int = 8              # requêtes simultanées (limite OANDA ≈ 120 req/s)
CACHE_TTL_SECONDS: int = 60             # fraîcheur des données servies par le cache
BUNDLE_CACHE_MAX_ENTRIES: int = 6       # borne mémoire du cache de séries (~5 Mo/entrée)
RESULT_CACHE_MAX_ENTRIES: int = 8       # borne mémoire des résultats moteur / map
MIN_OHLCV_ROWS: int = 20                # longueur minimale d'une série exploitable

# Compteurs canoniques par série. Un compteur unique par (instrument, granularité) permet
# à tous les consommateurs (moteur, vélocité, Market Map) de partager la même série OANDA :
# une série plus longue se contente d'être tronquée côté consommateur, valeurs inchangées.
DAILY_COUNT: int = 2000                 # W (resample) + D — identique à v10.1
TREND_D_SLICE: int = 300                # fenêtre exacte de trend_daily (identique v10.1)
H4_COUNT: int = 300
H1_COUNT: int = 300
MAP_COUNT: int = 30                     # instruments hors moteur (indices, métaux)

# ── Configuration OANDA — noms acceptés dans st.secrets / variables d'env ──────
TOKEN_KEYS: Tuple[str, ...] = (
    "OANDA_ACCESS_TOKEN",
    "OANDA_API_KEY",
    "OANDA_TOKEN",
    "OANDA_PRACTICE_ACCESS_TOKEN",
)
LIVE_TOKEN_KEYS: Tuple[str, ...] = ("OANDA_LIVE_ACCESS_TOKEN", "OANDA_LIVE_API_KEY")
ENV_KEYS: Tuple[str, ...] = ("OANDA_ENVIRONMENT", "OANDA_ENV")
SECRET_SECTIONS: Tuple[str, ...] = ("oanda", "OANDA")
VALID_ENVIRONMENTS: Tuple[str, ...] = ("practice", "live")
DEFAULT_ENVIRONMENT: str = "practice"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
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


class BluestarConfigError(BluestarError):
    """Configuration manquante ou invalide (token, environnement). No retry."""


class BluestarAuthError(BluestarError):
    """401/403 — credentials invalid. No retry."""


class BluestarRateLimit(BluestarError):
    """429 — retry with exponential backoff + jitter."""


class BluestarTimeout(BluestarError):
    """Timeout réseau (lecture/connexion) — retryable."""


class BluestarNetworkError(BluestarError):
    """Erreur réseau transitoire (DNS, TLS, connexion coupée, 5xx) — retryable."""


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
    # `tail` : tronque la série côté consommateur (None = série complète).
    # W et D partagent désormais LA MÊME série journalière OANDA (DAILY_COUNT) ; trend_daily
    # ne se voit servir que ses TREND_D_SLICE dernières bougies, ce qui reproduit exactement
    # le fetch `count=300` de la v10.1 (indicateurs causaux => valeurs finales identiques).
    "W":  {"gran_fetch": "D",  "count": DAILY_COUNT, "weight": 4.0,
           "resample_rule": "W-FRI", "tail": None},
    "D":  {"gran_fetch": "D",  "count": DAILY_COUNT, "weight": 4.0,
           "resample_rule": None,    "tail": TREND_D_SLICE},
    "H4": {"gran_fetch": "H4", "count": H4_COUNT,    "weight": 2.5,
           "resample_rule": None,    "tail": None},
    "H1": {"gran_fetch": "H1", "count": H1_COUNT,    "weight": 1.5,
           "resample_rule": None,    "tail": None},
}


# ==========================================
# ── OUTILS RÉSEAU & VALIDATION ────────────
# ==========================================

def _create_client(access_token: str, environment: str) -> API:
    """
    Crée un client OANDA v20 avec timeouts HTTP explicites.

    oandapyV20 n'expose pas de paramètre `timeout` : les `request_params` sont fusionnés dans
    les arguments passés à `requests`, on y injecte donc le couple (connexion, lecture).
    Sans cela une réponse OANDA figée bloque indéfiniment le script Streamlit.

    ATTENTION : un `API` encapsule un `requests.Session` (non thread-safe). Un client doit
    être utilisé par un seul thread — `_fetch_one_series` en instancie un par requête.
    """
    return API(
        access_token=access_token,
        environment=environment,
        request_params={"timeout": (HTTP_CONNECT_TIMEOUT_S, HTTP_READ_TIMEOUT_S)},
    )


# ==========================================
# ── RÉSOLUTION DE CONFIGURATION ───────────
# ==========================================

def _secret_lookup(key: str) -> Optional[str]:
    """
    Lecture non bloquante d'un secret Streamlit.

    `st.secrets` lève `StreamlitSecretNotFoundError` (sous-classe de FileNotFoundError)
    lorsqu'aucun secrets.toml / secret Cloud n'est configuré, et `KeyError` lorsque la clé
    est absente : les deux cas sont traités comme « non configuré ».
    """
    try:
        value = st.secrets[key]
    except Exception:  # secrets absents, illisibles ou clé manquante
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _secret_section_lookup(keys: Sequence[str]) -> Optional[str]:
    """Cherche `key` dans les sections TOML `[oanda]` / `[OANDA]` de st.secrets."""
    for section in SECRET_SECTIONS:
        try:
            block = st.secrets[section]
        except Exception:
            continue
        if not hasattr(block, "get"):
            continue
        for key in keys:
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _first_configured(
    keys: Sequence[str], *, allow_env: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Retourne (valeur, source) en priorisant st.secrets puis les variables d'environnement.
    La `source` est un libellé affichable qui ne contient jamais la valeur du secret.
    """
    for key in keys:
        value = _secret_lookup(key)
        if value:
            return value, f"secrets:{key}"
    value = _secret_section_lookup(keys)
    if value:
        return value, "secrets:[oanda]"
    if allow_env:
        for key in keys:
            value = os.environ.get(key)
            if value and value.strip():
                return value.strip(), f"env:{key}"
    return None, None


@dataclass(frozen=True)
class Settings:
    """Configuration OANDA résolue, sans jamais exposer la valeur des tokens."""

    practice_token: Optional[str] = None
    live_token: Optional[str] = None
    default_environment: str = DEFAULT_ENVIRONMENT
    token_source: str = "absent"
    problems: Tuple[str, ...] = ()

    @property
    def configured(self) -> bool:
        """Vrai dès qu'au moins un token exploitable est disponible."""
        return bool(self.practice_token or self.live_token)

    def token_for(self, environment: str) -> Optional[str]:
        """Token associé à un environnement, avec repli sur l'autre token configuré."""
        if environment == "live":
            return self.live_token or self.practice_token
        return self.practice_token or self.live_token

    @property
    def available_environments(self) -> List[str]:
        """Environnements réellement utilisables avec la configuration courante."""
        envs: List[str] = []
        if self.live_token:
            envs.append("live")
        if self.practice_token or not envs:
            envs.insert(0, "practice")
        return envs

    def diagnostics(self) -> Dict[str, Any]:
        """Informations de diagnostic affichables (aucun secret en clair)."""
        return {
            "token_source": self.token_source,
            "practice_token": bool(self.practice_token),
            "live_token": bool(self.live_token),
            "default_environment": self.default_environment,
            "problems": list(self.problems),
        }


def load_settings() -> Settings:
    """
    Résout la configuration OANDA depuis st.secrets puis os.environ.

    Ne lève jamais : une configuration absente produit un objet `Settings` vide dont les
    `problems` alimentent l'écran de configuration affiché à l'utilisateur.
    """
    problems: List[str] = []
    practice_token, practice_source = _first_configured(TOKEN_KEYS)
    live_token, live_source = _first_configured(LIVE_TOKEN_KEYS)

    raw_env, _env_source = _first_configured(ENV_KEYS)
    environment = (raw_env or DEFAULT_ENVIRONMENT).strip().lower()
    if environment not in VALID_ENVIRONMENTS:
        problems.append(
            f"Environnement « {environment} » inconnu (valeurs acceptées : "
            f"{', '.join(VALID_ENVIRONMENTS)}) — repli sur {DEFAULT_ENVIRONMENT}."
        )
        environment = DEFAULT_ENVIRONMENT

    if practice_token and live_token and practice_token == live_token and environment == "live":
        problems.append(
            "Le même token est utilisé pour practice et live : bascule live désactivée."
        )
        live_token = None

    for token, label in ((practice_token, "practice"), (live_token, "live")):
        if token and len(token) < 20:
            problems.append(
                f"Le token {label} semble tronqué ({len(token)} caractères) — "
                "vérifiez la valeur collée dans les secrets."
            )

    if not (practice_token or live_token):
        problems.append(
            "Aucun token OANDA détecté. Ajoutez OANDA_ACCESS_TOKEN dans les secrets "
            "Streamlit (ou dans les variables d'environnement)."
        )
    elif environment == "live" and not live_token:
        problems.append(
            "Environnement live demandé mais aucun token live dédié : un token practice "
            "est utilisé et provoquera un refus d'authentification (401)."
        )
        environment = "practice"

    return Settings(
        practice_token=practice_token,
        live_token=live_token,
        default_environment=environment,
        token_source=(practice_source if practice_token else live_source) or "absent",
        problems=tuple(problems),
    )


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
    """
    Client OANDA v20 : timeouts, backoff et taxonomie d'erreurs typée.

    Toute exception émise par `requests` ou `oandapyV20` est convertie en `BluestarError`,
    de sorte qu'aucune erreur réseau ne puisse remonter jusqu'au script Streamlit et
    interrompre le rendu. Retry avec backoff exponentiel + jitter sur 429, 5xx et erreurs
    réseau transitoires (connexion coupée, timeout, réponse non-JSON tronquée).
    """

    RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(self, api: API, *, max_attempts: int = HTTP_MAX_ATTEMPTS) -> None:
        self._api = api
        self.max_attempts = max(1, int(max_attempts))

    @staticmethod
    def _status_code(exc: V20Error) -> Optional[int]:
        """Code HTTP d'une V20Error, tolérant aux codes non entiers."""
        try:
            return int(getattr(exc, "code", None))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """Backoff exponentiel plafonné + jitter (évite les retries synchronisés)."""
        return min(2.0 ** attempt, 8.0) + random.uniform(0.1, 0.5)  # nosec B311

    def request(self, endpoint):
        """
        Exécute une requête OANDA. Retourne le payload JSON décodé.

        Lève `BluestarAuthError` (401/403), `BluestarRateLimit`, `BluestarTimeout`,
        `BluestarNetworkError` (transitoires) ou `BluestarDataError` (définitif).
        """
        last_error: Optional[BluestarError] = None

        for attempt in range(self.max_attempts):
            try:
                return self._api.request(endpoint)
            except V20Error as exc:
                code = self._status_code(exc)
                if code in (401, 403):
                    raise BluestarAuthError(f"OANDA a refusé les identifiants (HTTP {code}).") from exc
                if code in self.RETRYABLE_STATUS:
                    last_error = (
                        BluestarRateLimit(f"OANDA limite de débit (HTTP {code}).")
                        if code == 429
                        else BluestarNetworkError(f"OANDA indisponible (HTTP {code}).")
                    )
                    self._sleep_before_retry(attempt, last_error)
                    continue
                raise BluestarDataError(f"OANDA a renvoyé une erreur HTTP {code}.") from exc
            except requests.RequestException as exc:
                # requests.RequestException englobe Timeout / ConnectionError / SSL / Chunked :
                # ce sont toutes des erreurs transitoires côté transport.
                if isinstance(exc, requests.exceptions.Timeout):
                    last_error = BluestarTimeout(
                        f"Timeout OANDA après {HTTP_READ_TIMEOUT_S:.0f}s."
                    )
                else:
                    last_error = BluestarNetworkError(
                        f"Erreur réseau OANDA : {type(exc).__name__}."
                    )
                logger.warning(
                    "OANDA transport %s (essai %d/%d)",
                    type(exc).__name__, attempt + 1, self.max_attempts,
                )
                self._sleep_before_retry(attempt, last_error)
                continue
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # Réponse non-JSON (page d'erreur d'un proxy, corps tronqué) : transitoire.
                last_error = BluestarDataError(
                    f"Réponse OANDA illisible ({type(exc).__name__})."
                )
                self._sleep_before_retry(attempt, last_error)
                continue
            except OSError as exc:  # socket brut hors requests (DNS, SSL bas niveau)
                last_error = BluestarNetworkError(f"Erreur socket OANDA : {type(exc).__name__}.")
                self._sleep_before_retry(attempt, last_error)
                continue
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                raise BluestarDataError(
                    f"Payload OANDA inexploitable ({type(exc).__name__}: {exc})."
                ) from exc

        raise last_error or BluestarNetworkError("OANDA injoignable après plusieurs tentatives.")

    def _sleep_before_retry(self, attempt: int, error: BluestarError) -> None:
        """Attend avec backoff avant le prochain essai (sans effet au dernier essai)."""
        if attempt + 1 < self.max_attempts:
            delay = self._backoff_seconds(attempt)
            logger.warning(
                "OANDA %s — nouvelle tentative %d/%d dans %.2fs",
                type(error).__name__, attempt + 2, self.max_attempts, delay,
            )
            time.sleep(delay)


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

    def direction_arrow(self, currency: str) -> str:
        """Retourne la flèche directionnelle pour une devise."""
        v = self.velocity.get(currency, 0.0)
        if v > 0.02:
            return "up"
        if v < -0.02:
            return "down"
        return "flat"

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


# ==========================================
# ── COUCHE DONNÉES (I/O OANDA) ────────────
# ==========================================

def _spec_key(instrument: str, granularity: str, count: int) -> str:
    """Clé canonique d'une série OHLCV."""
    return f"{instrument}|{granularity}|{int(count)}"


def _empty_series(error: Optional[str] = None) -> Dict[str, Any]:
    """Série vide (colonnes normalisées) porteuse d'un éventuel message d'erreur."""
    return {"t": [], "o": [], "h": [], "l": [], "c": [], "error": error}


def frame_from_series(series: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Reconstruit le DataFrame OHLCV indexé sur le temps (None si série inexploitable).

    Index identique à la v10.1 : `pd.to_datetime()` sur les horodatages ISO-8601 UTC
    renvoyés par OANDA => DatetimeIndex tz-aware UTC.
    """
    stamps = series.get("t") or []
    if len(stamps) < MIN_OHLCV_ROWS:
        return None
    index = pd.DatetimeIndex(pd.to_datetime(list(stamps)))
    return pd.DataFrame(
        {
            "Open":  series["o"],
            "High":  series["h"],
            "Low":   series["l"],
            "Close": series["c"],
        },
        index=index,
    )


def records_from_candles(candles: Any) -> Dict[str, Any]:
    """
    Convertit les chandelles OANDA (liste de dicts) en colonnes compactes.

    Le stockage en colonnes divise par ~8 l'empreinte mémoire du cache par rapport à une
    liste de dicts par chandelle (28 paires × 2000 bougies journalières) : c'est ce qui
    évite les redémarrages OOM sur une instance Streamlit Cloud à 1 Go.
    """
    if not isinstance(candles, list):
        raise BluestarDataError("Réponse OANDA sans tableau 'candles'.")
    series = _empty_series()
    for candle in candles:
        if not isinstance(candle, dict) or not candle.get("complete"):
            continue
        mid = candle.get("mid")
        try:
            series["t"].append(str(candle["time"]))
            series["o"].append(float(mid["o"]))
            series["h"].append(float(mid["h"]))
            series["l"].append(float(mid["l"]))
            series["c"].append(float(mid["c"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise BluestarDataError(f"Chandelle OANDA malformée : {candle!r}") from exc
    return series


# ── Plan de chargement ────────────────────────────────────────────────────────

def build_engine_plan() -> Tuple[Tuple[str, str, int], ...]:
    """
    Séries nécessaires au moteur (W, D, H4, H1 + vélocité H1).

    W et D partagent la même série journalière : 3 requêtes par paire au lieu de 4.
    """
    specs = set()
    for pair in PAIRS:
        for cfg in TIMEFRAMES_MTF.values():
            specs.add((pair, cfg["gran_fetch"], int(cfg["count"])))
        specs.add((pair, "H1", H1_COUNT))
    return tuple(sorted(specs))


def map_fetch_count(instrument: str, granularity: str) -> int:
    """
    Compteur de chandelles pour la Market Map.

    On réutilise la série déjà chargée par le moteur quand elle existe (même instantané,
    donc zéro requête supplémentaire) ; pour les indices et matières premières, 30 bougies
    suffisent (la map n'affiche qu'une variation de 1 à 5 ticks).
    """
    if instrument in PAIRS:
        if granularity == "H1":
            return H1_COUNT
        if granularity == "H4":
            return H4_COUNT
        if granularity == "D":
            return DAILY_COUNT
    return MAP_COUNT


def build_map_plan(granularity: str) -> Tuple[Tuple[str, str, int], ...]:
    """Séries nécessaires à la Market Map (forex + indices + matières premières)."""
    instruments = list(FOREX_PAIRS) + list(INDICES) + list(METAUX)
    return tuple(
        sorted((i, granularity, map_fetch_count(i, granularity)) for i in instruments)
    )


# ── Chargement parallèle + cache ──────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False, max_entries=BUNDLE_CACHE_MAX_ENTRIES)
def fetch_candles_batch(
    token_fp: str,
    environment: str,
    specs: Tuple[Tuple[str, str, int], ...],
    _access_token: str,
) -> Dict[str, Any]:
    """
    Récupère plusieurs séries OHLCV en parallèle auprès d'OANDA (cache 60 s).

    `token_fp` — empreinte SHA-256 non réversible — fait PARTIE de la clé de cache : deux
    tokens ou deux environnements ne peuvent plus se contaminer mutuellement (la v10.1
    préfixait ce paramètre d'un `_`, ce qui l'excluait silencieusement de la clé de cache).
    `_access_token` est préfixé d'un underscore, donc exclu de la clé : le secret lui-même
    n'est jamais persisté par Streamlit.

    Retourne `{ "PAIR|GRAN|COUNT": {"t": [...], "o": [...], "h": [...], "l": [...],
    "c": [...], "error": None|str} }`, plus la clé `"__fatal__"` si OANDA a refusé
    l'authentification (l'UI peut alors afficher un écran de diagnostic précis).
    Aucune exception ne remonte : une série en échec est retournée vide avec son message.
    """
    bundle: Dict[str, Dict[str, Any]] = {}
    if not specs:
        return bundle

    workers = max(1, min(FETCH_MAX_WORKERS, len(specs)))
    t0 = time.perf_counter()
    logger.info("OANDA fetch start: %d séries, %d workers", len(specs), workers)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="oanda-fetch"
    ) as pool:
        futures = {
            pool.submit(_fetch_one_series, environment, _access_token, spec): spec
            for spec in specs
        }
        for future in concurrent.futures.as_completed(futures):
            instrument, granularity, count = futures[future]
            key = _spec_key(instrument, granularity, count)
            try:
                bundle[key] = future.result()
            except Exception as exc:  # garde-fou : un worker ne doit jamais casser le run
                logger.exception("Worker OANDA en échec pour %s", key)
                bundle[key] = _empty_series(f"Erreur interne de fetch ({type(exc).__name__}).")
            if str(bundle[key].get("error") or "").startswith("BluestarAuthError"):
                # Erreur fatale : inutile de laisser l'utilisateur deviner, on la remonte.
                bundle["__fatal__"] = bundle[key]["error"]  # type: ignore[assignment]

    failed = sum(1 for k, v in bundle.items() if k != "__fatal__" and v.get("error"))
    logger.info(
        "OANDA fetch done: %d séries, %d en erreur, %.2fs",
        len(specs), failed, time.perf_counter() - t0,
    )
    return bundle


def _fetch_one_series(
    environment: str, access_token: str, spec: Tuple[str, str, int]
) -> Dict[str, Any]:
    """
    Récupère UNE série OANDA.

    Thread-safe par construction : un client `API` (donc un `requests.Session`) est créé par
    appel, car `Session` n'est pas thread-safe. Le pool de connexions reste réutilisé par
    thread via le ThreadPoolExecutor, ce qui limite les handshakes TLS.
    """
    instrument, granularity, count = spec
    try:
        client = OandaClient(_create_client(access_token, environment))
        endpoint = instruments.InstrumentsCandles(
            instrument=instrument,
            params={"count": int(count), "granularity": granularity, "price": "M"},
        )
        payload = client.request(endpoint)
        candles = payload.get("candles") if hasattr(payload, "get") else None
        return records_from_candles(candles)
    except BluestarError as exc:
        logger.warning(
            "Série indisponible %s %s %d : %s", instrument, granularity, count, exc
        )
        return _empty_series(f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # ultime garde-fou (payload exotique, erreur pandas…)
        logger.exception("Erreur inattendue sur %s %s %d", instrument, granularity, count)
        return _empty_series(f"{type(exc).__name__}: {exc}")


class MarketDataProvider:
    """
    Point d'accès unique aux séries OHLCV.

    Combine un bundle pré-chargé (une seule passe réseau parallèle) et un cache de DataFrames
    valant pour la durée d'un run. Les DataFrames retournés sont mis en cache et partagés :
    ils doivent être considérés comme LECTURE SEULE (aucune fonction de tendance ne les mute).
    """

    def __init__(
        self,
        token_fp: str,
        environment: str,
        access_token: str,
        bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.token_fp = token_fp
        self.environment = environment
        self._access_token = access_token
        self._bundle: Dict[str, Any] = dict(bundle or {})
        self._frames: Dict[Tuple[str, str, int], Optional[pd.DataFrame]] = {}
        self.errors: List[str] = []
        self.bundle_hits = 0
        self.lazy_fetches = 0

    # ── Séries brutes ─────────────────────────────────────────────────────────

    def series(self, instrument: str, granularity: str, count: int) -> Dict[str, Any]:
        """Série brute (colonnes compactes), servie par le bundle ou par le cache Streamlit."""
        key = _spec_key(instrument, granularity, count)
        cached = self._bundle.get(key)
        if cached is not None:
            self.bundle_hits += 1
            return cached
        self.lazy_fetches += 1
        fetched = fetch_candles_batch(
            self.token_fp, self.environment, ((instrument, granularity, int(count)),),
            self._access_token,
        )
        series = fetched.get(key) or _empty_series("Série absente du bundle.")
        self._bundle[key] = series
        return series

    # ── DataFrames ────────────────────────────────────────────────────────────

    def frame(
        self, instrument: str, granularity: str, count: int
    ) -> Optional[pd.DataFrame]:
        """
        DataFrame OHLCV validé, ou None si la série est indisponible.

        Reproduit exactement le contrat de `StrengthEngine._fetch_ohlcv` en v10.1 :
        série trop courte ou invalide => None + message dans `errors`.
        """
        frame_key = (instrument, granularity, int(count))
        if frame_key in self._frames:
            return self._frames[frame_key]

        series = self.series(instrument, granularity, count)
        frame = frame_from_series(series)
        if frame is not None:
            try:
                validate_ohlcv(frame, min_len=MIN_OHLCV_ROWS)
            except ValueError as exc:
                self.errors.append(f"{instrument}/{granularity}/{count}: {exc}")
                logger.warning("Série invalide %s : %s", frame_key, exc)
                frame = None
        elif series.get("error"):
            self.errors.append(f"{instrument}/{granularity}/{count}: {series['error']}")

        self._frames[frame_key] = frame
        return frame

    # ── Diagnostic ────────────────────────────────────────────────────────────

    @property
    def fatal_error(self) -> Optional[str]:
        """Message d'authentification remonté par la couche réseau, le cas échéant."""
        value = self._bundle.get("__fatal__")
        return value if isinstance(value, str) else None

    def stats(self) -> Dict[str, Any]:
        """Compteurs d'accès et erreurs de collecte, pour le panneau de diagnostic."""
        return {
            "series_from_bundle": self.bundle_hits,
            "series_lazy_fetched": self.lazy_fetches,
            "series_loaded": sum(
                1 for k, v in self._bundle.items()
                if k != "__fatal__" and isinstance(v, dict) and v.get("t")
            ),
            "series_failed": len(self.errors),
            "errors": list(self.errors),
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
    max_exposure: int = MAX_CURRENCY_EXPOSURE,
) -> Tuple[List[str], List[Dict]]:
    """
    Filtre les candidats sur l'ATR puis limite l'exposition par devise.

    `max_exposure` = nombre maximum de paires partageant une même devise (1 par défaut :
    aucune devise n'apparaît deux fois dans la sélection — comportement v10.1 inchangé).
    """
    if not candidates:
        return [], []

    atr_values = [c["atr"] for c in candidates if c["atr"] is not None]
    if atr_values:
        threshold = float(np.percentile(atr_values, ATR_MIN_PERCENTILE))
        candidates = [c for c in candidates if c["atr"] is not None and c["atr"] >= threshold]
    if not candidates:
        return [], []

    limit = max(1, int(max_exposure))
    exposure: Dict[str, int] = {}
    filtered = []
    for c in sorted(candidates, key=lambda x: x["diff"], reverse=True):
        if exposure.get(c["base"], 0) >= limit or exposure.get(c["quote"], 0) >= limit:
            continue
        filtered.append(c)
        exposure[c["base"]] = exposure.get(c["base"], 0) + 1
        exposure[c["quote"]] = exposure.get(c["quote"], 0) + 1
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
        provider: MarketDataProvider,
        min_diff: float = MIN_STRENGTH_DIFF,
        max_pairs: int  = MAX_PAIRS,
        max_exposure: int = MAX_CURRENCY_EXPOSURE,
    ):
        self.provider     = provider
        self.min_diff     = min_diff
        self.max_pairs    = max_pairs
        self.max_exposure = max(1, int(max_exposure))

    @property
    def errors(self) -> List[str]:
        """Erreurs de collecte remontées par le provider (même contrat qu'en v10.1)."""
        return self.provider.errors

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _fetch_ohlcv(
        self, pair: str, granularity: str, count: int
    ) -> Optional[pd.DataFrame]:
        """
        Récupère une série OHLCV validée via le provider.

        La mise en cache (par série) et la validation sont déléguées à `MarketDataProvider` :
        le moteur ne connaît plus ni HTTP ni OANDA, ce qui le rend testable hors ligne.
        """
        return self.provider.frame(pair, granularity, count)

    def _get_tf_df(self, pair: str, tf: str) -> Optional[pd.DataFrame]:
        """Récupère le DataFrame pour un timeframe donné."""
        cfg = TIMEFRAMES_MTF[tf]
        df  = self._fetch_ohlcv(pair, cfg["gran_fetch"], cfg["count"])
        if df is None:
            return None
        tail = cfg.get("tail")
        if tail and len(df) > tail:
            # W et D partagent la série journalière complète : trend_daily ne reçoit que la
            # fenêtre exacte de la v10.1 (EMA/SMA/EWM causaux => dernières valeurs identiques).
            df = df.iloc[-tail:]
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
        """Calcule la vélocité sur deux fenêtres H1 de 60 barres."""
        total_now:  Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        total_prev: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight_sum: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
        weight = TIMEFRAMES_MTF["H1"]["weight"]
        for pair in PAIRS:
            base, quote = pair.split("_")
            df = self._fetch_ohlcv(pair, "H1", 300)
            if df is None or len(df) < 120:
                continue
            df_now  = df.iloc[-60:]
            df_prev = df.iloc[-120:-60]
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
        return _filter_by_atr_and_exposure(candidates, self.max_pairs, self.max_exposure)

    # ── Points d'entrée publics ───────────────────────────────────────────────

    def run(self) -> StrengthResult:
        """
        Exécute le calcul complet multi-timeframe.

        Sans effet de bord : les caches et compteurs sont portés par le provider, un même
        moteur peut donc être relancé (ou instancié plusieurs fois) sans état résiduel.
        """
        t0 = time.perf_counter()
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
        pair_count = {c: 0 for c in CURRENCIES}
        for pair in PAIRS:
            b, q = pair.split("_")
            pair_count[b] += 1
            pair_count[q] += 1
        coverage = {c: (weight_sum[c] / (pair_count[c] * total_weight) if pair_count[c] else 0.0) for c in CURRENCIES}
        warnings = []
        if self.errors:
            warnings.append(f"{len(self.errors)} erreur(s) API (voir logs).")
        min_cov = min(coverage.values()) if coverage else 0
        if min_cov < 0.5:
            warnings.append("Couverture de données faible, signaux dégradés.")

        pairs_fetched = int(self.provider.stats().get("series_loaded", 0))
        logger.info(
            "engine.run.completed: duration_ms=%.2f pairs_fetched=%d errors=%d "
            "min_coverage=%.4f bundle_hits=%d lazy_fetches=%d",
            (time.perf_counter() - t0) * 1000, pairs_fetched, len(self.errors), min_cov,
            self.provider.bundle_hits, self.provider.lazy_fetches,
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


# ── 2. Orchestration des données (caches isolés par token) ────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False, max_entries=RESULT_CACHE_MAX_ENTRIES)
def load_market_bundle(
    token_fp: str,
    environment: str,
    granularity: str,
    _access_token: str,
) -> Dict[str, Any]:
    """
    Charge en UNE passe parallèle toutes les séries nécessaires au run courant
    (moteur W/D/H4/H1 + vélocité H1 + Market Map).

    Le bundle est mis en cache 60 s et partagé par le moteur et la Market Map : le run ne
    déclenche plus ~150 requêtes séquentielles mais une seule vague parallèle bornée.
    """
    specs = tuple(sorted(set(build_engine_plan()) | set(build_map_plan(granularity))))
    logger.info("bundle: %d séries demandées (map=%s)", len(specs), granularity)
    return fetch_candles_batch(token_fp, environment, specs, _access_token)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False, max_entries=RESULT_CACHE_MAX_ENTRIES)
def run_engine_cached(
    token_fp: str,
    environment: str,
    _bundle: Dict[str, Any],
    _access_token: str,
) -> Dict[str, Any]:
    """
    Exécute le moteur de force et renvoie un payload sérialisable (dict).

    Seuls `token_fp` (empreinte) et `environment` entrent dans la clé de cache : le bundle
    et le token sont préfixés d'un `_` (volumineux / secret). Le payload est volontairement
    un dict — et non une instance de dataclass définie dans `__main__` — pour éliminer tout
    risque d'échec de désérialisation du cache Streamlit.
    """
    t0 = time.perf_counter()
    provider = MarketDataProvider(token_fp, environment, _access_token, bundle=_bundle)
    result = StrengthEngine(provider).run()
    if provider.fatal_error:
        raise BluestarAuthError(provider.fatal_error)

    payload = asdict(result)
    payload["computed_at"] = datetime.datetime.now(tz).isoformat(timespec="seconds")
    payload["engine_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    payload["provider"] = provider.stats()
    payload["health"] = result.health_check()
    logger.info(
        "engine.cached: valid=%s pairs=%d engine_ms=%.1f",
        result.valid, len(result.best_pairs), payload["engine_ms"],
    )
    return payload


def result_from_payload(payload: Dict[str, Any]) -> StrengthResult:
    """Reconstruit un `StrengthResult` depuis le payload mis en cache."""
    allowed = {f.name for f in fields(StrengthResult)}
    return StrengthResult(**{k: v for k, v in payload.items() if k in allowed})


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


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False, max_entries=RESULT_CACHE_MAX_ENTRIES)
def fetch_market_map_data(
    token_fp: str,
    environment: str,
    gran: str,
    smooth: int,
    _bundle: Dict[str, Any],
    _access_token: str,
) -> Tuple[Dict, Dict[str, float], Dict[str, Any]]:
    """
    Market Map sans forward/backward fill.

    Les séries proviennent du bundle partagé (aucune requête supplémentaire quand la
    granularité demandée est H1/H4/D, déjà chargée par le moteur) et les séries non
    rafraîchies au-delà de `max_age` sont ignorées, comme en v10.1.
    """
    provider = MarketDataProvider(token_fp, environment, _access_token, bundle=_bundle)
    local_pair_changes: Dict[str, float] = {}
    skipped: List[str] = []
    max_age_map = {
        "M5": pd.Timedelta(minutes=15),
        "M15": pd.Timedelta(minutes=45),
        "M30": pd.Timedelta(minutes=90),
        "H1": pd.Timedelta(hours=3),
        "H4": pd.Timedelta(hours=8),
        "D": pd.Timedelta(days=3),
    }
    max_age = max_age_map.get(gran, pd.Timedelta(hours=1))

    now_utc = pd.Timestamp.now(tz="UTC")
    for pair in FOREX_PAIRS:
        df = provider.frame(pair, gran, map_fetch_count(pair, gran))
        if df is None or len(df) < 2:
            skipped.append(pair)
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            skipped.append(pair)
            continue
        last_ts = pd.Timestamp(closes.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        age = now_utc - last_ts
        if age > max_age:
            logger.info("map: %s ignoré (dernière bougie %s, âge %s)", pair, last_ts, age)
            skipped.append(pair)
            continue
        pct = _smoothed_pct(closes, smooth)
        if pct is None or not np.isfinite(pct):
            skipped.append(pair)
            continue
        local_pair_changes[pair] = pct

    local_pct_special: Dict[str, Dict[str, Any]] = {}
    for symbol, name in {**INDICES, **METAUX}.items():
        df = provider.frame(symbol, gran, map_fetch_count(symbol, gran))
        if df is None or len(df) < 2:
            skipped.append(symbol)
            continue
        closes = df["Close"].dropna()
        if len(closes) < 2:
            skipped.append(symbol)
            continue
        pct = _smoothed_pct(closes, smooth)
        if pct is None or not np.isfinite(pct):
            skipped.append(symbol)
            continue
        local_pct_special[name] = {
            "pct": pct,
            "cat": "INDICES" if symbol in INDICES else "METAUX",
        }

    stats = provider.stats()
    stats["skipped"] = skipped
    stats["expected"] = len(FOREX_PAIRS) + len(INDICES) + len(METAUX)
    stats["received"] = len(local_pair_changes) + len(local_pct_special)
    return local_pct_special, local_pair_changes, stats


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
          <div class="bs-sub">FX Institutional Desk · Strength Engine v11.0 · W / D / H4 / H1</div>
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
    h = datetime.datetime.now(tz).hour
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
    now = datetime.datetime.now(tz)

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
            "system":        "BLUESTAR v11.0",
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
    now      = datetime.datetime.now(tz)
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
      <div class="h1s">FX Institutional Desk · v11.0</div>
    </div>
  </div>
  <div class="hdr-r">
    <div class="hdr-r1">Institutional Macro Briefing</div>
    <div class="hdr-r2">Analyse quantitative — OANDA v20 API · Auto-généré</div>
  </div>
</div>

<div class="subbar">
  <span>{html.escape(date_str)}</span>
  <span>{html.escape(time_str)} UTC — {html.escape(session)}</span>
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

<div class="footer">Confidentiel — Bluestar System · FX Institutional Desk · v11.0 · {html.escape(date_str)} {html.escape(time_str)} UTC</div>
</div>
</body>
</html>"""


def generate_pdf_bytes(briefing_html: str) -> Optional[bytes]:
    """
    Convertit le HTML en PDF via WeasyPrint (optionnel).

    WeasyPrint est absent des requirements (il exige pango/cairo, non disponibles sur
    Streamlit Cloud) : l'appel est donc enveloppé largement — ImportError, libs système
    manquantes (OSError) et erreurs de rendu XML (SyntaxError) — afin que l'export PDF ne
    puisse jamais faire échouer le rendu du tableau de bord. Repli : téléchargement HTML.
    """
    try:
        from weasyprint import HTML as WP_HTML  # type: ignore[import]
        return WP_HTML(string=briefing_html).write_pdf()
    except ImportError:
        logger.info("WeasyPrint non disponible — export PDF désactivé (repli HTML).")
        return None
    except Exception as exc:  # fail-open volontaire : l'export ne doit jamais bloquer l'UI
        logger.warning("WeasyPrint indisponible/erreur de rendu (%s) — repli HTML.", type(exc).__name__)
        return None


# ── 6. Écrans de robustesse (configuration, authentification, diagnostic) ─────

def _utc_now_label() -> str:
    """Horodatage UTC compact pour l'en-tête (jamais l'heure locale du serveur)."""
    return datetime.datetime.now(tz).strftime("%d/%m %H:%M UTC")


def render_config_screen(settings: Settings) -> None:
    """
    Écran affiché quand aucun token OANDA n'est disponible.

    La v10.1 s'arrêtait sur un `st.stop()` après un simple message d'erreur : l'utilisateur
    voyait une application vide sans savoir quoi corriger. Ici la marche à suivre est
    explicite et l'application reste utilisable (aucun crash).
    """
    st.markdown(
        app_header("—", "—", "NEUTRAL", _utc_now_label()), unsafe_allow_html=True
    )
    st.error("Configuration OANDA absente — le tableau de bord ne peut pas charger les données.")
    with st.expander("Configurer le token OANDA (2 minutes)", expanded=True):
        st.markdown(
            """
**Streamlit Community Cloud** → *Manage app* → *Settings* → *Secrets* :

```toml
OANDA_ACCESS_TOKEN = "votre_token_practice"
OANDA_ENVIRONMENT  = "practice"      # practice | live
```

**Auto-hébergement** → copiez `.streamlit/secrets.toml.example` en
`.streamlit/secrets.toml`, ou exportez `OANDA_ACCESS_TOKEN` / `OANDA_ENVIRONMENT`.

Le token d'un compte **practice** se génère sur
<https://www.oanda.com/demo-account/tpa/personal_token>.

Noms de secrets également acceptés : `OANDA_API_KEY`, `OANDA_TOKEN`,
`OANDA_PRACTICE_ACCESS_TOKEN`, ou une section `[oanda]` avec `access_token`.
"""
        )
    for problem in settings.problems:
        st.warning(problem)


def render_auth_screen(message: str) -> None:
    """Écran affiché lorsque OANDA refuse les identifiants (HTTP 401/403)."""
    st.error("OANDA a refusé l'authentification — aucune donnée de marché disponible.")
    st.markdown(
        "1. **Token expiré ou révoqué** : régénérez-le sur "
        "<https://www.oanda.com/demo-account/tpa/personal_token> puis mettez à jour les secrets.\n"
        "2. **Mauvais environnement** : un token *practice* ne fonctionne **que** avec "
        "`OANDA_ENVIRONMENT = \"practice\"` (et inversement pour un token live).\n"
        "3. **Secret mal collé** : vérifiez l'absence d'espace ou de guillemet superflu.\n"
        "4. Après mise à jour des secrets, cliquez sur **⟳ Actualiser** dans la barre latérale."
    )
    with st.expander("Détail technique"):
        st.code(message, language="text")


def render_failure_screen(title: str, message: str, detail: str = "") -> None:
    """Écran d'échec générique (le détail technique reste consultable, jamais affiché brut)."""
    st.error(f"{title} : {message}")
    st.caption(
        "Les données n'ont pas pu être chargées. Cliquez sur **⟳ Actualiser** pour relancer "
        "une collecte ; si le problème persiste, consultez le panneau *Diagnostic*."
    )
    if detail:
        with st.expander("Détail technique"):
            st.code(detail, language="text")


def render_diagnostics(
    settings: Settings,
    environment: str,
    token_fp: str,
    payload: Dict[str, Any],
    map_stats: Dict[str, Any],
) -> None:
    """Panneau de diagnostic (sidebar) : santé, couverture, volumétrie, erreurs API."""
    provider_stats = payload.get("provider") or {}
    health = payload.get("health") or {}
    errors = list(provider_stats.get("errors") or [])
    with st.expander("Diagnostic", expanded=False):
        st.caption(
            f"**Environnement** : {environment} · **Source token** : {settings.token_source} · "
            f"**Empreinte** : `{token_fp[:8]}…`"
        )
        st.caption(
            f"**Streamlit** {st.__version__} · **pandas** {pd.__version__} · "
            f"**Python** {platform.python_version()}"
        )
        st.caption(
            f"**Séries** : {provider_stats.get('series_loaded', 0)} chargées · "
            f"{provider_stats.get('series_from_bundle', 0)} servies par le bundle · "
            f"{provider_stats.get('series_lazy_fetched', 0)} fetch tardif(s)"
        )
        st.caption(
            f"**Market Map** : {map_stats.get('received', 0)}/{map_stats.get('expected', 0)} "
            f"instruments · {len(map_stats.get('skipped') or [])} ignoré(s)"
        )
        st.caption(
            f"**Moteur** : {payload.get('engine_ms', 0)} ms · "
            f"**Calculé à** : {payload.get('computed_at', 'n/a')} · "
            f"**Couverture min** : {health.get('coverage_min', 0)}"
        )
        st.caption(f"**TTL du cache** : {CACHE_TTL_SECONDS} s · **Workers** : {FETCH_MAX_WORKERS}")
        if errors:
            st.caption(f"**{len(errors)} erreur(s) API** :")
            st.code("\n".join(errors[:12]), language="text")
        else:
            st.caption("**Aucune erreur API.**")


# ── 7. Sidebar ────────────────────────────────────────────────────────────────

SESSION_LAST_RENDER_KEY: str = "bls_last_render_ts"
SESSION_INTERVAL_KEY: str = "bls_auto_refresh_s"
AUTO_REFRESH_CHOICES: Dict[str, int] = {
    "Désactivé": 0,
    "Toutes les 1 min": 60,
    "Toutes les 5 min": 300,
    "Toutes les 15 min": 900,
}
_AUTO_REFRESH_TICK_S: int = 10      # réveil du fragment (léger : aucun appel réseau)
_HAS_FRAGMENT: bool = hasattr(st, "fragment")


def _auto_refresh_ticker() -> None:
    """
    Rafraîchissement automatique opt-in (Streamlit >= 1.37).

    Réveillé toutes les 10 s sans aucun appel réseau ; déclenche un rerun complet uniquement
    lorsque l'intervalle choisi dans la barre latérale est écoulé. Désactivé par défaut :
    aucun rerun n'est déclenché si l'utilisateur n'a pas sélectionné d'intervalle.
    """
    interval = int(st.session_state.get(SESSION_INTERVAL_KEY, 0) or 0)
    if interval <= 0:
        return
    last_render = float(st.session_state.get(SESSION_LAST_RENDER_KEY, 0.0) or 0.0)
    if time.time() - last_render >= interval:
        logger.info("Auto-refresh: rerun complet (intervalle %ss)", interval)
        st.rerun()


if _HAS_FRAGMENT:
    # Décoration conditionnelle : garde l'app importable sur un Streamlit < 1.37.
    _auto_refresh_ticker = st.fragment(run_every=_AUTO_REFRESH_TICK_S)(_auto_refresh_ticker)


settings = load_settings()
current_env: str = settings.default_environment
current_granularity: str = "H1"
map_smooth: int = MAP_SMOOTH_WINDOW
auto_refresh_s: int = 0

with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-brand-t">◆ BLUESTAR</div>'
        '<div class="sb-brand-s">Strength Engine v11.0</div></div>',
        unsafe_allow_html=True,
    )

    if not settings.configured:
        st.error("Token OANDA introuvable dans les secrets.")
    else:
        available_envs = settings.available_environments
        default_index = (
            available_envs.index(settings.default_environment)
            if settings.default_environment in available_envs else 0
        )

        st.markdown('<div class="sb-lbl">Connexion</div>', unsafe_allow_html=True)
        current_env = st.selectbox(
            "Env",
            available_envs,
            index=default_index,
            label_visibility="collapsed",
            help="Environnement OANDA ciblé. Seuls les environnements disposant d'un token "
                 "configuré sont proposés (un token practice est refusé en live et inversement).",
        )

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

        st.markdown('<div class="sb-lbl">Actualisation</div>', unsafe_allow_html=True)
        if st.button(
            "⟳ Actualiser maintenant",
            help="Vide le cache de données et relance immédiatement une collecte OANDA complète.",
        ):
            st.cache_data.clear()
            st.session_state.pop(SESSION_LAST_RENDER_KEY, None)
            st.rerun()

        if _HAS_FRAGMENT:
            refresh_label = st.selectbox(
                "Auto-refresh",
                list(AUTO_REFRESH_CHOICES.keys()),
                index=0,
                label_visibility="collapsed",
                help=f"Les données sont conservées {CACHE_TTL_SECONDS} s. Le rafraîchissement "
                     "automatique relance l'application selon l'intervalle choisi.",
            )
            auto_refresh_s = AUTO_REFRESH_CHOICES[refresh_label]
            st.session_state[SESSION_INTERVAL_KEY] = auto_refresh_s
        else:
            auto_refresh_s = 0
            st.caption(
                "Rafraîchissement automatique indisponible (streamlit ≥ 1.37 requis) — "
                "utilisez le bouton ⟳."
            )

        for problem in settings.problems:
            st.warning(problem)

    st.markdown("---")
    st.caption(
        "Le moteur de force agrège W + D + H4 + H1 indépendamment du timeframe "
        "affiché sur la Market Map."
    )
    st.caption(
        f"Poids : W {TIMEFRAMES_MTF['W']['weight']} · D {TIMEFRAMES_MTF['D']['weight']} · "
        f"H4 {TIMEFRAMES_MTF['H4']['weight']} · H1 {TIMEFRAMES_MTF['H1']['weight']}"
    )


# ── 8. Exécution ──────────────────────────────────────────────────────────────

current_token = settings.token_for(current_env)

if current_token:
    token_fp = token_fingerprint(current_token)

    with st.status("Collecte des données OANDA…", expanded=False) as status:
        try:
            bundle = load_market_bundle(token_fp, current_env, current_granularity, current_token)
            fatal = bundle.get("__fatal__")
            if isinstance(fatal, str):
                raise BluestarAuthError(fatal)
            payload = run_engine_cached(token_fp, current_env, bundle, current_token)
            pct_special, pair_changes, map_stats = fetch_market_map_data(
                token_fp, current_env, current_granularity, map_smooth, bundle, current_token
            )
        except BluestarAuthError as exc:
            status.update(label="Authentification refusée par OANDA", state="error")
            render_auth_screen(str(exc))
            st.stop()
        except BluestarError as exc:
            logger.warning("Collecte interrompue : %s", exc)
            status.update(label="Collecte impossible", state="error")
            render_failure_screen("Données indisponibles", str(exc))
            st.stop()
        except Exception as exc:  # filet de sécurité : l'app ne doit jamais montrer un traceback
            logger.exception("Erreur inattendue pendant la collecte")
            status.update(label="Erreur interne", state="error")
            render_failure_screen("Erreur interne", type(exc).__name__, traceback.format_exc())
            st.stop()
        status.update(
            label=f"Données chargées · {map_stats.get('received', 0)} instruments",
            state="complete", expanded=False,
        )

    result      = result_from_payload(payload)
    regime_now  = _infer_regime(pct_special)
    ts_label    = _utc_now_label()

    st.markdown(
        app_header(current_env, current_granularity, regime_now, ts_label),
        unsafe_allow_html=True,
    )

    health = result.health_check()

    if not result.valid:
        st.error(
            "Impossible de calculer les forces : " + "; ".join(result.warnings)
            + " — vérifiez le panneau Diagnostic ci-contre."
        )
        render_failure_screen(
            "Moteur sans données",
            "aucune série OANDA exploitable n'a été reçue",
            "\n".join((payload.get("provider") or {}).get("errors") or []) or "aucun détail",
        )
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

    # ── Horodatage de rendu + auto-refresh opt-in ─────────────────────────────
    st.session_state[SESSION_LAST_RENDER_KEY] = time.time()
    if auto_refresh_s > 0 and _HAS_FRAGMENT:
        _auto_refresh_ticker()

    # ── Diagnostic (barre latérale) ───────────────────────────────────────────
    with st.sidebar:
        render_diagnostics(settings, current_env, token_fp, payload, map_stats)
else:
    render_config_screen(settings)
