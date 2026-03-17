import streamlit as st
import pandas as pd
import numpy as np
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

from strength_engine import StrengthEngine

# ==========================================
# 1. CONFIGURATION & DESIGN
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
    .progress-bg { background-color: #374151; height: 5px; border-radius: 3px; width: 100%; margin-top: 8px; }
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

FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]


# ==========================================
# 2. FETCH (uniquement pour la Market Map)
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles_generic(token, env, instrument, granularity, count):
    """Fetch OHLCV — utilisé uniquement pour la Market Map (% change visuel)."""
    try:
        client = API(access_token=token, environment=env)
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
    except Exception:
        return None


def fetch_market_map_data(token, env, gran):
    """
    Récupère les % change pour la Market Map (forex + indices + métaux).
    Séparé du moteur de force — c'est juste de l'affichage visuel.
    """
    prices = {}
    for pair in FOREX_PAIRS:
        df = fetch_candles_generic(token, env, pair, gran, 30)
        if df is not None:
            prices[pair] = df["Close"]

    pct_special = {}
    for symbol, name in {**INDICES, **METAUX}.items():
        df = fetch_candles_generic(token, env, symbol, gran, 30)
        if df is not None:
            pct = float(df["Close"].pct_change().iloc[-1] * 100)
            pct_special[name] = {
                "pct": pct,
                "cat": "INDICES" if symbol in INDICES else "METAUX",
            }

    df_prices = pd.DataFrame(prices).ffill().bfill() if prices else pd.DataFrame()
    return df_prices, pct_special


# ==========================================
# 3. RENDU CARTES (inchangé)
# ==========================================
def display_card(name, score, arrow_str):
    """
    score     : float 0–10
    arrow_str : "up" | "down" | "flat" (vient de result.direction_arrow())
    """
    if score >= 7:     c_txt, c_bg = "text-green",  "bg-green"
    elif score >= 5.5: c_txt, c_bg = "text-blue",   "bg-blue"
    elif score >= 4:   c_txt, c_bg = "text-orange", "bg-orange"
    else:              c_txt, c_bg = "text-red",     "bg-red"

    if arrow_str == "up":
        arrow, a_col = "↗", "text-green"
    elif arrow_str == "down":
        arrow, a_col = "↘", "text-red"
    else:
        arrow, a_col = "→", "text-gray"

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


# ==========================================
# 4. MARKET MAP HTML (inchangée)
# ==========================================
def generate_exact_map_html(df_prices, pct_special):
    pct_changes = df_prices.pct_change().iloc[-1] * 100

    def get_bg_color(pct):
        if pct >= 0.15:  return "#009900"
        if pct >= 0.01:  return "#33cc33"
        if pct <= -0.15: return "#cc0000"
        if pct <= -0.01: return "#ff3300"
        return "#f0f0f0"

    def get_text_color(pct):
        return "#333" if -0.01 < pct < 0.01 else "white"

    forex_data = {}
    for base in CURRENCIES:
        forex_data[base] = []
        for col in df_prices.columns:
            if base not in col:
                continue
            val = float(pct_changes[col])
            if col.startswith(base):
                quote, pct = col.split("_")[1], val
            else:
                quote, pct = col.split("_")[0], -val
            forex_data[base].append({"pair": quote, "pct": pct})

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

        html += '<div class="currency-col">'
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


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Connexion OANDA")
    token = st.secrets.get("OANDA_ACCESS_TOKEN") or st.text_input("Token", type="password")
    env   = st.selectbox("Env", ["practice", "live"])
    st.markdown("---")
    granularity = st.selectbox("Timeframe (Map)", ["M5", "M15", "M30", "H1", "H4", "D"], index=3)
    st.caption("Le moteur de force utilise W + D + H4 + H1 en parallèle, indépendamment du timeframe affiché.")


# ==========================================
# 6. EXÉCUTION
# ==========================================
if token:
    with st.status("Actualisation des données...", expanded=True) as status:

        # ── Moteur institutionnel MTF (W → D → H4 → H1) ──────────
        engine = StrengthEngine(token=token, env=env)
        result = engine.run()

        # ── Données visuelles Market Map ──────────────────────────
        df_prices, pct_special = fetch_market_map_data(token, env, granularity)

        status.update(label="✅ Données chargées", state="complete", expanded=False)

    if result.scores_display:

        # ── CARTES FOREX ──────────────────────────────────────────
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

        # ── MEILLEURES PAIRES ─────────────────────────────────────
        if result.best_pairs:
            st.markdown("---")
            st.subheader("🎯 Paires Sélectionnées")
            badges = ""
            for d in result.pairs_detail:
                badges += (
                    f'<span style="display:inline-block;padding:4px 12px;'
                    f'background:#10B981;color:white;border-radius:4px;'
                    f'font-weight:bold;margin:3px;font-size:0.9rem;">'
                    f'{d["pair"]}</span>'
                    f'<span style="font-size:0.75rem;color:#9ca3af;margin-right:12px;">'
                    f'diff={d["diff"]:.2f}</span>'
                )
            st.markdown(badges, unsafe_allow_html=True)

        # ── MARKET MAP ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🗺️ Market Map Pro")

        if not df_prices.empty:
            html_map = generate_exact_map_html(df_prices, pct_special)
            st.components.v1.html(html_map, height=600, scrolling=True)
        else:
            st.warning("Données insuffisantes pour la Market Map.")

else:
    st.warning("En attente du Token...")
        
