import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# 1. CONFIGURATION & DESIGN
# ==========================================
st.set_page_config(page_title="Bluestar Market Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* CARTES ASSETS (Devises, Or, Indices) */
    .currency-card {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #374151;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .card-header { 
        display: flex; justify-content: center; align-items: center; gap: 8px; 
        font-weight: bold; color: #e5e7eb; font-size: 1rem; 
        margin-bottom: 2px;
    }
    .asset-name { font-family: 'Segoe UI', sans-serif; letter-spacing: 1px; }
    
    .strength-score { 
        font-size: 2.2rem; 
        font-weight: 800; 
        margin: 0;
        line-height: 1.1;
        display: flex; justify-content: center; align-items: center; gap: 10px;
    }
    
    .velocity-arrow { font-size: 1.2rem; }
    
    .progress-bg { background-color: #374151; height: 5px; border-radius: 3px; width: 100%; margin-top: 8px; }
    .progress-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
    
    /* COULEURS */
    .text-green { color: #10B981; } .bg-green { background-color: #10B981; }
    .text-blue { color: #3B82F6; } .bg-blue { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    .text-red { color: #EF4444; } .bg-red { background-color: #EF4444; }
    .text-gray { color: #6b7280; }

    /* MARKET MAP */
    iframe { width: 100% !important; }
    #MainMenu, footer, header {visibility: hidden;}
    
    h3 { color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; margin-top: 20px; border-bottom: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch",
    "XAU": "xk", # Pseudo code pour Or (Kosovo flag often used as placeholder or just generic)
    "US30": "us", "NAS100": "us", "DAX": "de"
}

# ==========================================
# 2. MOTEUR DE DONNÉES (PRECISION M5)
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles_generic(token, env, instrument, granularity, count):
    try:
        client = API(access_token=token, environment=env)
        # On demande "M" (Mid) pour la précision
        params = {"count": count, "granularity": granularity, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        candles = r.response['candles']
        
        data = []
        for c in candles:
            if c['complete']:
                data.append({"Time": c['time'], "Close": float(c['mid']['c'])})
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
        return df
    except:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def normalize_score(rsi_value):
    """Convertit un RSI (0-100) en Score Bluestar (0-10)"""
    # RSI 50 = Score 5.0
    # RSI 70 = Score 7.0
    # RSI 30 = Score 3.0
    # Formule simple : Score = RSI / 10
    # Mais pour plus de dynamique aux extrêmes :
    score = (rsi_value - 50) / 50 # -1 à 1
    return (score + 1) * 5 # 0 à 10

def process_data(token, env, gran):
    # Liste complète pour Scanner
    forex_pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    # Actifs Spéciaux (Pour Harmonisation)
    special_assets = {
        'XAU_USD': 'GOLD', 
        'US30_USD': 'US30', 
        'NAS100_USD': 'NAS100', 
        'DE30_EUR': 'DAX'
    }

    prices = {}
    
    # 1. Fetch Forex
    for pair in forex_pairs:
        df = fetch_candles_generic(token, env, pair, gran, 100)
        if df is not None: prices[pair] = df['Close']
            
    # 2. Fetch Gold/Indices
    scores_special = {}
    for symbol, name in special_assets.items():
        df = fetch_candles_generic(token, env, symbol, gran, 100)
        if df is not None:
            # Calcul Force intrinsèque basé sur RSI
            rsi = calculate_rsi(df['Close'], 14)
            # On stocke (Current, Previous) pour la vélocité
            curr_score = normalize_score(rsi.iloc[-1])
            prev_score = normalize_score(rsi.iloc[-2])
            scores_special[name] = (curr_score, prev_score)

    # 3. Calcul Force Devises (Panier Relatif)
    if not prices: return {}, {}, {}
    
    df_prices = pd.DataFrame(prices).fillna(method='ffill').fillna(method='bfill')
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    scores_forex = {} # Format: {CURR: (val, prev_val)}
    
    for curr in currencies:
        total_str_curr = 0.0
        total_str_prev = 0.0
        count = 0
        opponents = [c for c in currencies if c != curr]
        
        for opp in opponents:
            pair_d = f"{curr}_{opp}"
            pair_i = f"{opp}_{curr}"
            
            rsi_series = None
            if pair_d in df_prices.columns:
                rsi_series = calculate_rsi(df_prices[pair_d])
            elif pair_i in df_prices.columns:
                rsi_series = calculate_rsi(1/df_prices[pair_i])
            
            if rsi_series is not None:
                # Normalisation
                s_curr = normalize_score(rsi_series.iloc[-1])
                s_prev = normalize_score(rsi_series.iloc[-2])
                
                total_str_curr += s_curr
                total_str_prev += s_prev
                count += 1
        
        if count > 0:
            scores_forex[curr] = (total_str_curr / count, total_str_prev / count)

    return scores_forex, scores_special, df_prices

# ==========================================
# 3. RENDU VISUEL
# ==========================================
def display_card(name, current, previous, is_crypto_or_index=False):
    # Calcul Vélocité (Pente)
    delta = current - previous
    
    # Couleur Score
    if current >= 7: c_txt, c_bg = "text-green", "bg-green"
    elif current >= 5.5: c_txt, c_bg = "text-blue", "bg-blue"
    elif current >= 4: c_txt, c_bg = "text-orange", "bg-orange"
    else: c_txt, c_bg = "text-red", "bg-red"
    
    # Flèche Vélocité
    if delta > 0.05: arrow, a_col = "↗", "text-green" # Monte
    elif delta < -0.05: arrow, a_col = "↘", "text-red" # Descend
    else: arrow, a_col = "→", "text-gray" # Stable

    # Drapeau
    flag_code = FLAG_URLS.get(name, "xk")
    if name == "GOLD": img_html = "🟡" # Emoji pour Gold
    elif name in ["US30", "NAS100", "DAX"]: img_html = "📊"
    else: img_html = f'<img src="https://flagcdn.com/48x36/{flag_code}.png" style="width:24px; border-radius:2px;">'

    width = min(max(current * 10, 0), 100)

    return f"""
    <div class="currency-card">
        <div class="card-header">
            {img_html} <span class="asset-name">{name}</span>
        </div>
        <div class="strength-score {c_txt}">
            {current:.1f} 
            <span class="velocity-arrow {a_col}" title="Vélocité: {delta:+.2f}">{arrow}</span>
        </div>
        <div class="progress-bg"><div class="progress-fill {c_bg}" style="width:{width}%;"></div></div>
    </div>
    """

def get_market_map_html(df_prices, scores_forex):
    # Logique simplifiée pour générer la map HTML (reprise de l'ancien code mais allégée)
    # On se concentre sur les % change de la dernière bougie pour la couleur
    pct_change = df_prices.pct_change().iloc[-1] * 100
    
    def get_col(val):
        if val > 0.1: return "#10b981"
        if val < -0.1: return "#ef4444"
        return "#374151"

    # Construction simple de la matrice visuelle
    html = '<div style="display:flex; gap:5px; overflow-x:auto;">'
    
    # Tri par force pour l'ordre des colonnes
    sorted_curr = sorted(scores_forex.keys(), key=lambda x: scores_forex[x][0], reverse=True)
    
    for base in sorted_curr:
        html += f'<div style="min-width:90px; display:flex; flex-direction:column; gap:2px;">'
        html += f'<div style="background:#1f2937; color:#9ca3af; text-align:center; font-weight:bold; font-size:11px; padding:2px; border:1px solid #374151;">{base}</div>'
        
        # Paires associées
        related = [col for col in df_prices.columns if base in col]
        for pair in related:
            val = pct_change[pair]
            # Si la paire est inversée (ex: on est dans col USD, mais paire EUR_USD), on inverse le %
            display_pair = pair
            if not pair.startswith(base): 
                val = -val
                display_pair = pair.replace('_', '/') # Simplification affichage
            else:
                display_pair = pair.split('_')[1]
            
            html += f'<div style="background:{get_col(val)}; color:white; font-size:10px; padding:3px; display:flex; justify-content:space-between; border-radius:2px;"><span>{display_pair}</span><span>{val:+.2f}%</span></div>'
        html += '</div>'
    html += '</div>'
    return html

# ==========================================
# 4. EXÉCUTION
# ==========================================
with st.sidebar:
    st.header("Connexion OANDA")
    token = st.secrets.get("OANDA_ACCESS_TOKEN") or st.text_input("Token", type="password")
    env = st.selectbox("Env", ["practice", "live"])
    st.markdown("---")
    granularity = st.selectbox("Timeframe", ["M5", "M15", "M30", "H1", "H4", "D"], index=3)

if token:
    with st.status("Actualisation des données...", expanded=True) as status:
        s_forex, s_special, df_prices = process_data(token, env, granularity)
        status.update(label="Données chargées", state="complete", expanded=False)

    if s_forex:
        # 1. FOREX SECTION
        st.subheader("💱 Forces Forex (0-10)")
        sorted_fx = sorted(s_forex.keys(), key=lambda x: s_forex[x][0], reverse=True)
        
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, curr in enumerate(sorted_fx):
            with cols[i % 4]:
                st.markdown(display_card(curr, s_forex[curr][0], s_forex[curr][1]), unsafe_allow_html=True)
                
        # 2. INDICES & GOLD SECTION (HARMONISÉE)
        st.markdown("---")
        st.subheader("🏆 Indices & Gold (Score Harmonisé 0-10)")
        
        # On affiche ces actifs avec la même logique 0-10
        k_spec = list(s_special.keys())
        c_spec = st.columns(len(k_spec))
        for i, name in enumerate(k_spec):
            with c_spec[i]:
                st.markdown(display_card(name, s_special[name][0], s_special[name][1]), unsafe_allow_html=True)

        # 3. MARKET MAP PRO
        st.markdown("---")
        st.subheader("🗺️ Market Matrix (Performance)")
        st.components.v1.html(get_market_map_html(df_prices, s_forex), height=400, scrolling=True)

else:
    st.warning("En attente du Token...")
