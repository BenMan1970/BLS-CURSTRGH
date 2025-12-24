import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# 1. CONFIGURATION & DESIGN (Le meilleur des 2 versions)
# ==========================================
st.set_page_config(page_title="Bluestar Market Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* --- DESIGN DES CARTES (HAUT) --- */
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

    /* --- DESIGN DE LA MARKET MATRIX (BAS - LE RETOUR DU VISUEL) --- */
    .matrix-container { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 10px; }
    .currency-col { min-width: 110px; display: flex; flex-direction: column; gap: 2px; }
    
    .tile { 
        display: flex; justify-content: space-between; 
        padding: 5px 8px; 
        font-size: 11px; font-weight: bold; 
        border-radius: 3px; 
        color: white;
    }
    
    .sep { 
        background: #1f2937; 
        color: #9ca3af; 
        font-weight: 900; 
        text-align: center; 
        padding: 4px; 
        margin: 4px 0; 
        border: 1px solid #374151; 
        border-radius: 3px; 
        font-size: 12px;
    }
    
    /* Cache les éléments natifs Streamlit */
    iframe { width: 100% !important; }
    #MainMenu, footer, header {visibility: hidden;}
    
    h3 { color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; margin-top: 30px; border-bottom: 1px solid #374151; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch",
    "XAU": "xk", "US30": "us", "NAS100": "us", "DAX": "de"
}

# ==========================================
# 2. MOTEUR DE DONNÉES
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles_generic(token, env, instrument, granularity, count):
    try:
        client = API(access_token=token, environment=env)
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
    """Convertit RSI 0-100 en Score 0-10"""
    return ((rsi_value - 50) / 50 + 1) * 5

def process_data(token, env, gran):
    # Liste complète
    forex_pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    special_assets = {'XAU_USD': 'GOLD', 'US30_USD': 'US30', 'NAS100_USD': 'NAS100', 'DE30_EUR': 'DAX'}

    prices = {}
    
    # Fetch Forex
    for pair in forex_pairs:
        df = fetch_candles_generic(token, env, pair, gran, 100)
        if df is not None: prices[pair] = df['Close']
            
    # Fetch Gold/Indices (Calcul Score 0-10)
    scores_special = {}
    for symbol, name in special_assets.items():
        df = fetch_candles_generic(token, env, symbol, gran, 100)
        if df is not None:
            rsi = calculate_rsi(df['Close'], 14)
            scores_special[name] = (normalize_score(rsi.iloc[-1]), normalize_score(rsi.iloc[-2]))

    # Calcul Force Devises
    if not prices: return {}, {}, {}
    df_prices = pd.DataFrame(prices).fillna(method='ffill').fillna(method='bfill')
    
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    scores_forex = {}
    
    for curr in currencies:
        total_curr, total_prev, count = 0.0, 0.0, 0
        opponents = [c for c in currencies if c != curr]
        for opp in opponents:
            pair_d, pair_i = f"{curr}_{opp}", f"{opp}_{curr}"
            rsi_s = None
            if pair_d in df_prices.columns: rsi_s = calculate_rsi(df_prices[pair_d])
            elif pair_i in df_prices.columns: rsi_s = calculate_rsi(1/df_prices[pair_i])
            
            if rsi_s is not None:
                total_curr += normalize_score(rsi_s.iloc[-1])
                total_prev += normalize_score(rsi_s.iloc[-2])
                count += 1
        
        if count > 0: scores_forex[curr] = (total_curr / count, total_prev / count)

    return scores_forex, scores_special, df_prices

# ==========================================
# 3. RENDU VISUEL - FONCTIONS
# ==========================================
def display_card(name, current, previous):
    delta = current - previous
    # Couleurs
    if current >= 7: c_txt, c_bg = "text-green", "bg-green"
    elif current >= 5.5: c_txt, c_bg = "text-blue", "bg-blue"
    elif current >= 4: c_txt, c_bg = "text-orange", "bg-orange"
    else: c_txt, c_bg = "text-red", "bg-red"
    
    # Flèche Vélocité
    if delta > 0.05: arrow, a_col = "↗", "text-green"
    elif delta < -0.05: arrow, a_col = "↘", "text-red"
    else: arrow, a_col = "→", "text-gray"

    flag_code = FLAG_URLS.get(name, "xk")
    img_html = "🟡" if name == "GOLD" else "📊" if name in ["US30", "NAS100", "DAX"] else f'<img src="https://flagcdn.com/48x36/{flag_code}.png" style="width:24px; border-radius:2px;">'

    return f"""
    <div class="currency-card">
        <div class="card-header">{img_html} <span class="asset-name">{name}</span></div>
        <div class="strength-score {c_txt}">
            {current:.1f} <span class="velocity-arrow {a_col}" title="Vélocité: {delta:+.2f}">{arrow}</span>
        </div>
        <div class="progress-bg"><div class="progress-fill {c_bg}" style="width:{min(max(current * 10, 0), 100)}%;"></div></div>
    </div>
    """

def generate_visual_matrix(df_prices):
    """Génère la Market Matrix VISUELLE (Vert haut / Rouge bas)"""
    
    # Calcul des variations en % sur la dernière bougie
    pct_changes = df_prices.pct_change().iloc[-1] * 100
    
    def get_color(pct):
        if pct >= 0.20: return "#064e3b" # Vert foncé
        if pct >= 0.05: return "#10b981" # Vert
        if pct >= 0.01: return "#34d399" # Vert clair
        if pct <= -0.20: return "#7f1d1d" # Rouge foncé
        if pct <= -0.05: return "#ef4444" # Rouge
        if pct <= -0.01: return "#f87171" # Rouge clair
        return "#374151" # Gris
    
    # Organisation des données
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    forex_data = {}
    
    for base in currencies:
        forex_data[base] = []
        # Trouver toutes les paires impliquant cette devise
        for col in df_prices.columns:
            if base in col:
                val = pct_changes[col]
                # Déterminer la devise opposée et le sens du %
                if col.startswith(base):
                    quote = col.split('_')[1]
                    pct = val
                else:
                    quote = col.split('_')[0]
                    pct = -val # Inversion car Base est en 2ème position
                
                forex_data[base].append({'pair': quote, 'pct': pct})

    # Algorithme de tri des colonnes (Weighted Score pour mettre les forts à gauche)
    scores = {}
    for curr, items in forex_data.items():
        score = sum(item['pct'] for item in items)
        scores[curr] = score
    
    sorted_cols = sorted(scores, key=scores.get, reverse=True)

    # Génération HTML
    html = '<div class="matrix-container">'
    for curr in sorted_cols:
        items = forex_data[curr]
        # Tri interne : Gagnants en haut, Perdants en bas
        winners = sorted([x for x in items if x['pct'] >= 0.01], key=lambda x: x['pct'], reverse=True)
        losers = sorted([x for x in items if x['pct'] < -0.01], key=lambda x: x['pct'], reverse=True)
        flat = [x for x in items if -0.01 <= x['pct'] < 0.01]
        
        html += f'<div class="currency-col">'
        
        # Piles Vertes
        for x in winners:
            html += f'<div class="tile" style="background:{get_color(x["pct"])}"><span>{x["pair"]}</span><span>+{x["pct"]:.2f}%</span></div>'
        
        # Séparateur Central
        html += f'<div class="sep">{curr}</div>'
        
        # Pile Grise
        for x in flat:
            html += f'<div class="tile" style="background:#374151"><span>{x["pair"]}</span><span>unch</span></div>'
            
        # Piles Rouges
        for x in losers:
            html += f'<div class="tile" style="background:{get_color(x["pct"])}"><span>{x["pair"]}</span><span>{x["pct"]:.2f}%</span></div>'
            
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
        status.update(label="Prêt", state="complete", expanded=False)

    if s_forex:
        # SECTION 1: DEVISES (Scores 0-10)
        st.subheader("💱 Forces Forex (0-10)")
        sorted_fx = sorted(s_forex.keys(), key=lambda x: s_forex[x][0], reverse=True)
        
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, curr in enumerate(sorted_fx):
            with cols[i % 4]:
                st.markdown(display_card(curr, s_forex[curr][0], s_forex[curr][1]), unsafe_allow_html=True)
                
        # SECTION 2: INDICES & OR (Scores harmonisés 0-10)
        st.markdown("---")
        st.subheader("🏆 Indices & Gold (Score Harmonisé 0-10)")
        k_spec = list(s_special.keys())
        c_spec = st.columns(len(k_spec))
        for i, name in enumerate(k_spec):
            with c_spec[i]:
                st.markdown(display_card(name, s_special[name][0], s_special[name][1]), unsafe_allow_html=True)

        # SECTION 3: MARKET MATRIX VISUELLE (Le retour !)
        st.markdown("---")
        st.subheader("🗺️ Market Matrix Pro")
        st.components.v1.html(generate_visual_matrix(df_prices), height=500, scrolling=True)

else:
    st.warning("En attente du Token...")
