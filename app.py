import streamlit as st
import pandas as pd
import numpy as np
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

    /* Market Map full width */
    iframe { width: 100% !important; }
    #MainMenu, footer, header {visibility: hidden;}
    
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
    
    # Actifs pour la section bas
    indices = {'US30_USD': 'DOW JONES', 'NAS100_USD': 'NASDAQ 100', 'SPX500_USD': 'S&P 500', 'DE30_EUR': 'DAX 40'}
    metaux = {'XAU_USD': 'GOLD', 'XAG_USD': 'SILVER', 'XPT_USD': 'PLATINUM'}
    
    special_assets = {**indices, **metaux}
    prices = {}
    
    # 1. Fetch Forex
    for pair in forex_pairs:
        df = fetch_candles_generic(token, env, pair, gran, 100)
        if df is not None: prices[pair] = df['Close']
            
    # 2. Fetch Gold/Indices (Pour les scores du haut + Data du bas)
    scores_special = {}
    pct_special = {} # Pour stocker le % change
    
    for symbol, name in special_assets.items():
        df = fetch_candles_generic(token, env, symbol, gran, 100)
        if df is not None:
            # Score 0-10
            rsi = calculate_rsi(df['Close'], 14)
            scores_special[name] = (normalize_score(rsi.iloc[-1]), normalize_score(rsi.iloc[-2]))
            
            # % Change pour la Map du bas (basé sur la dernière bougie complete)
            pct = df['Close'].pct_change().iloc[-1] * 100
            
            # Catégorie
            cat = "INDICES" if symbol in indices else "METAUX"
            pct_special[name] = {'pct': pct, 'cat': cat}

    # 3. Calcul Force Devises
    if not prices: return {}, {}, {}, {}
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

    return scores_forex, scores_special, df_prices, pct_special

# ==========================================
# 3. RENDU VISUEL - EXACTEMENT COMME L'IMAGE
# ==========================================
def display_card(name, current, previous):
    delta = current - previous
    if current >= 7: c_txt, c_bg = "text-green", "bg-green"
    elif current >= 5.5: c_txt, c_bg = "text-blue", "bg-blue"
    elif current >= 4: c_txt, c_bg = "text-orange", "bg-orange"
    else: c_txt, c_bg = "text-red", "bg-red"
    
    if delta > 0.05: arrow, a_col = "↗", "text-green"
    elif delta < -0.05: arrow, a_col = "↘", "text-red"
    else: arrow, a_col = "→", "text-gray"

    flag_code = FLAG_URLS.get(name, "xk")
    img_html = "🟡" if name == "GOLD" else "📊" if name in ["US30", "NAS100", "DAX"] else f'<img src="https://flagcdn.com/48x36/{flag_code}.png" style="width:24px; border-radius:2px;">'

    return f"""
    <div class="currency-card">
        <div class="card-header">{img_html} <span class="asset-name">{name}</span></div>
        <div class="strength-score {c_txt}">
            {current:.1f} <span class="velocity-arrow {a_col}">{arrow}</span>
        </div>
        <div class="progress-bg"><div class="progress-fill {c_bg}" style="width:{min(max(current * 10, 0), 100)}%;"></div></div>
    </div>
    """

def generate_exact_map_html(df_prices, pct_special):
    """Génère la Market Map avec le design exact de la photo"""
    
    pct_changes = df_prices.pct_change().iloc[-1] * 100
    
    def get_bg_color(pct):
        # Couleurs vives comme sur l'image
        if pct >= 0.15: return "#009900" # Vert Vif
        if pct >= 0.01: return "#33cc33" # Vert Clair
        if pct <= -0.15: return "#cc0000" # Rouge Vif
        if pct <= -0.01: return "#ff3300" # Rouge Orange
        return "#f0f0f0" # Gris très clair
    
    def get_text_color(pct):
        if -0.01 < pct < 0.01: return "#333" # Texte sombre pour fond gris
        return "white"

    # --- FOREX LOGIC (Colonnes triées) ---
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    forex_data = {}
    
    for base in currencies:
        forex_data[base] = []
        for col in df_prices.columns:
            if base in col:
                val = pct_changes[col]
                if col.startswith(base):
                    quote, pct = col.split('_')[1], val
                else:
                    quote, pct = col.split('_')[0], -val
                forex_data[base].append({'pair': quote, 'pct': pct})

    # Tri des colonnes par force globale (Somme des %)
    scores = {curr: sum(i['pct'] for i in items) for curr, items in forex_data.items()}
    sorted_cols = sorted(scores, key=scores.get, reverse=True)

    # --- HTML CONSTRUCTION ---
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { font-family: 'Arial', sans-serif; background-color: transparent; margin: 0; padding: 0; }
        
        /* HEADERS */
        .section-header { 
            color: #aaa; font-size: 14px; font-weight: bold; text-transform: uppercase; 
            margin: 25px 0 10px 0; display: flex; align-items: center; gap: 5px;
            border-bottom: 2px solid #333; padding-bottom: 5px;
        }
        
        /* FOREX MATRIX */
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
        
        /* INDICES & METAUX BOXES */
        .grid-container { display: flex; flex-wrap: wrap; gap: 10px; }
        .big-box { 
            width: 140px; height: 60px; 
            display: flex; flex-direction: column; justify-content: center; align-items: center; 
            color: white; border-radius: 4px; 
            box-shadow: 0 3px 5px rgba(0,0,0,0.3);
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        .box-name { font-size: 11px; font-weight: bold; margin-bottom: 2px; text-transform: uppercase; }
        .box-val { font-size: 14px; font-weight: 900; }
        
    </style>
    </head>
    <body>
    """
    
    # 1. FOREX MAP
    html += '<div class="section-header">💱 FOREX MAP</div>'
    html += '<div class="matrix-row">'
    
    for curr in sorted_cols:
        items = forex_data[curr]
        winners = sorted([x for x in items if x['pct'] >= 0.01], key=lambda x: x['pct'], reverse=True)
        losers = sorted([x for x in items if x['pct'] < -0.01], key=lambda x: x['pct'], reverse=True)
        flat = [x for x in items if -0.01 <= x['pct'] < 0.01]
        
        html += '<div class="currency-col">'
        
        # Verts (Haut)
        for x in winners:
            col, txt = get_bg_color(x['pct']), get_text_color(x['pct'])
            html += f'<div class="tile" style="background:{col}; color:{txt};"><span>{x["pair"]}</span><span>+{x["pct"]:.2f}%</span></div>'
        
        # Séparateur
        html += f'<div class="sep">{curr}</div>'
        
        # Gris (Milieu)
        for x in flat:
             html += f'<div class="tile" style="background:#f0f0f0; color:#333;"><span>{x["pair"]}</span><span>unch</span></div>'

        # Rouges (Bas)
        for x in losers:
            col, txt = get_bg_color(x['pct']), get_text_color(x['pct'])
            html += f'<div class="tile" style="background:{col}; color:{txt};"><span>{x["pair"]}</span><span>{x["pct"]:.2f}%</span></div>'
            
        html += '</div>'
    html += '</div>'
    
    # 2. INDICES
    html += '<div class="section-header">📊 INDICES</div>'
    html += '<div class="grid-container">'
    indices_data = {k: v for k, v in pct_special.items() if v['cat'] == "INDICES"}
    for name, data in indices_data.items():
        pct = data['pct']
        col = get_bg_color(pct)
        html += f'<div class="big-box" style="background:{col}"><span class="box-name">{name}</span><span class="box-val">{pct:+.2f}%</span></div>'
    html += '</div>'

    # 3. METAUX
    html += '<div class="section-header">🪙 METAUX</div>'
    html += '<div class="grid-container">'
    metaux_data = {k: v for k, v in pct_special.items() if v['cat'] == "METAUX"}
    for name, data in metaux_data.items():
        pct = data['pct']
        col = get_bg_color(pct)
        html += f'<div class="big-box" style="background:{col}"><span class="box-name">{name}</span><span class="box-val">{pct:+.2f}%</span></div>'
    html += '</div>'

    html += "</body></html>"
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
        s_forex, s_special, df_prices, pct_special = process_data(token, env, granularity)
        status.update(label="Données chargées", state="complete", expanded=False)

    if s_forex:
        # HAUT : SCORING (Votre design actuel que vous aimiez)
        st.subheader("💱 Forces Forex & Assets (0-10)")
        sorted_fx = sorted(s_forex.keys(), key=lambda x: s_forex[x][0], reverse=True)
        
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, curr in enumerate(sorted_fx):
            with cols[i % 4]:
                st.markdown(display_card(curr, s_forex[curr][0], s_forex[curr][1]), unsafe_allow_html=True)
        
        # BAS : MARKET MAP (Le design EXACT de votre image)
        st.markdown("---")
        st.subheader("🗺️ Market Map Pro")
        
        # Fusion des données pour l'affichage indices/métaux
        # On passe df_prices et pct_special à la fonction HTML
        html_map = generate_exact_map_html(df_prices, pct_special)
        st.components.v1.html(html_map, height=600, scrolling=True)

else:
    st.warning("En attente du Token...")
