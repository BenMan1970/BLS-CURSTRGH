import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# 1. CONFIGURATION ET STYLE CSS GLOBAL
# ==========================================
st.set_page_config(page_title="Bluestar Market Dashboard", layout="wide")

# CSS Unifié pour harmoniser les deux parties
st.markdown("""
<style>
    /* Fond global */
    .stApp { background-color: #0e1117; }
    
    /* --- STYLE PARTIE 1 : STRENGTH METER --- */
    .currency-card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #374151;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .card-header { display: flex; justify-content: center; align-items: center; gap: 10px; font-weight: bold; color: white; font-size: 1.2rem; }
    .flag-img { width: 28px; height: 20px; border-radius: 2px; }
    .strength-score { font-size: 2.2rem; font-weight: 800; margin: 5px 0; }
    .progress-bg { background-color: #374151; height: 6px; border-radius: 3px; width: 100%; margin-top: 10px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 3px; transition: width 0.6s; }
    
    /* Couleurs Texte/Fond */
    .text-green { color: #10B981; } .bg-green { background-color: #10B981; }
    .text-blue { color: #3B82F6; } .bg-blue { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    .text-red { color: #EF4444; } .bg-red { background-color: #EF4444; }

    /* --- STYLE PARTIE 2 : MARKET MAP --- */
    /* On adapte le CSS de la map pour qu'il s'intègre au thème sombre */
    iframe { width: 100% !important; } /* Force la largeur du composant HTML */
    
    /* Cache les éléments natifs Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Configuration drapeaux et URLs
FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch"
}

# Mapping noms d'affichage Indices/Métaux
DISPLAY_NAMES = {
    'US30_USD': 'DOW JONES', 'NAS100_USD': 'NASDAQ 100', 'SPX500_USD': 'S&P 500', 
    'DE30_EUR': 'DAX 40', 'XAU_USD': 'GOLD', 'XPT_USD': 'PLATINUM', 'XAG_USD': 'SILVER'
}

st.title("💎 Bluestar Market Dashboard")
st.markdown("---")

# ==========================================
# 2. SIDEBAR & SECRETS
# ==========================================
with st.sidebar:
    st.header("Connexion OANDA")
    secret_token = st.secrets.get("OANDA_ACCESS_TOKEN", None)
    
    if secret_token:
        st.success("✅ Connecté (Secrets)")
        access_token = secret_token
    else:
        access_token = st.text_input("Token OANDA", type="password")

    environment = st.selectbox("Environnement", ["practice", "live"], index=0)
    
    st.markdown("---")
    st.header("Paramètres")
    
    st.subheader("Strength Meter (Haut)")
    granularity = st.selectbox("Timeframe (Strength)", ["M15", "M30", "H1", "H4", "D", "W"], index=4)
    lookback_chart = st.slider("Historique Graphique", 50, 500, 100)
    
    st.markdown("---")
    st.subheader("Market Map (Bas)")
    map_lookback = st.slider("Période Performance (Jours)", 1, 5, 1)

# ==========================================
# 3. FONCTIONS DATA (Unifiées)
# ==========================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles_generic(token, env, instrument, granularity, count):
    """Fonction générique pour récupérer les bougies de n'importe quel instrument"""
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
    except Exception:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 4. LOGIQUE STRENGTH METER (Haut)
# ==========================================
def process_strength_data(token, env, gran, count):
    pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    
    # Dictionnaire pour stocker les Close prices pour le Strength Meter
    prices_dict = {}
    
    # On utilise aussi ces données pour la Map Forex si le timeframe correspond (Optionnel, ici on simplifie)
    # Pour la rapidité, on télécharge tout ici.
    
    for pair in pairs:
        df = fetch_candles_generic(token, env, pair, gran, count + 20)
        if df is not None:
            prices_dict[pair] = df['Close']
            
    if not prices_dict: return None, None
    
    df_prices = pd.DataFrame(prices_dict).fillna(method='ffill').fillna(method='bfill')
    
    # Calcul RSI et Force
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    strength_df = pd.DataFrame(index=df_prices.index)
    
    for curr in currencies:
        total_strength = pd.Series(0.0, index=df_prices.index)
        valid_pairs = 0
        opponents = [c for c in currencies if c != curr]
        
        for opp in opponents:
            pair_direct = f"{curr}_{opp}"
            pair_inverse = f"{opp}_{curr}"
            rsi_series = None
            
            if pair_direct in df_prices.columns:
                rsi_series = calculate_rsi(df_prices[pair_direct])
            elif pair_inverse in df_prices.columns:
                rsi_series = calculate_rsi(1/df_prices[pair_inverse])
            
            if rsi_series is not None:
                total_strength += (rsi_series - 50) / 50
                valid_pairs += 1
        
        if valid_pairs > 0:
            avg_strength = total_strength / valid_pairs
            # Lissage SMA 3
            strength_df[curr] = ((avg_strength + 1) * 5).rolling(window=3).mean()
            
    return strength_df.dropna(), df_prices # On retourne aussi les prix bruts pour la map

# ==========================================
# 5. LOGIQUE MARKET MAP (Bas)
# ==========================================
def get_map_data(token, env, days_back, forex_prices_df=None):
    map_results = []
    
    # 1. Données Indices & Métaux (Téléchargement Spécifique)
    extras = ['US30_USD', 'NAS100_USD', 'SPX500_USD', 'DE30_EUR', 'XAU_USD', 'XPT_USD', 'XAG_USD']
    # Catégories
    cats = {'US30_USD':'INDICES', 'NAS100_USD':'INDICES', 'SPX500_USD':'INDICES', 'DE30_EUR':'INDICES',
            'XAU_USD':'COMMODITIES', 'XPT_USD':'COMMODITIES', 'XAG_USD':'COMMODITIES'}
    
    # Pour la map, on a besoin de Daily candles généralement pour le % change J-1 ou J-N
    for symbol in extras:
        df = fetch_candles_generic(token, env, symbol, "D", days_back + 5)
        if df is not None and len(df) >= days_back:
            now = df['Close'].iloc[-1]
            past = df['Close'].shift(days_back).iloc[-1]
            pct = ((now - past) / past) * 100
            name = DISPLAY_NAMES.get(symbol, symbol)
            map_results.append({'symbol': symbol, 'name': name, 'pct': pct, 'cat': cats[symbol]})

    # 2. Données Forex (Réutilisation ou Nouveau Fetch)
    # Pour avoir un % précis sur N jours, mieux vaut refaire un fetch Daily rapide sur les paires majeures
    forex_pairs = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD'
    ]
    
    for pair in forex_pairs:
        # Fetch Daily pour la performance
        df = fetch_candles_generic(token, env, pair, "D", days_back + 5)
        if df is not None and len(df) >= days_back:
            now = df['Close'].iloc[-1]
            past = df['Close'].shift(days_back).iloc[-1]
            pct = ((now - past) / past) * 100
            map_results.append({'symbol': pair, 'name': pair.replace('_', '/'), 'pct': pct, 'cat': 'FOREX'})
            
    return pd.DataFrame(map_results)

def generate_map_html(df):
    def get_color(pct):
        # Palette ajustée pour le dark theme
        if pct >= 0.50: return "#064e3b" # Vert foncé fort
        if pct >= 0.25: return "#065f46"
        if pct >= 0.10: return "#10b981" # Vert standard
        if pct >= 0.01: return "#34d399" 
        if pct <= -0.50: return "#7f1d1d" # Rouge foncé fort
        if pct <= -0.25: return "#991b1b"
        if pct <= -0.10: return "#ef4444" # Rouge standard
        if pct <= -0.01: return "#f87171"
        return "#374151" # Gris

    # --- SECTION FOREX (MATRICE) ---
    forex_df = df[df['cat'] == 'FOREX']
    forex_data = {}
    
    if not forex_df.empty:
        for _, row in forex_df.iterrows():
            parts = row['symbol'].split('_')
            if len(parts) == 2:
                base, quote = parts[0], parts[1]
                pct = row['pct']
                
                if base not in forex_data: forex_data[base] = []
                if quote not in forex_data: forex_data[quote] = []
                
                forex_data[base].append({'pair': f"{base}/{quote}", 'pct': pct, 'other': quote})
                forex_data[quote].append({'pair': f"{quote}/{base}", 'pct': -pct, 'other': base})

    # Algorithme de tri intelligent (Weighted Score)
    scores = {}
    for curr, items in forex_data.items():
        score = 0
        weight_sum = 0
        for item in items:
            w = 2.0 if item['other'] in ['USD', 'EUR', 'JPY'] else 1.0
            score += item['pct'] * w
            weight_sum += w
        scores[curr] = score / weight_sum if weight_sum > 0 else 0
        
    sorted_currencies = sorted(scores, key=scores.get, reverse=True)

    html_forex = '<div class="matrix-container">'
    for curr in sorted_currencies:
        items = forex_data.get(curr, [])
        winners = sorted([x for x in items if x['pct'] >= 0.01], key=lambda x: x['pct'], reverse=True)
        losers = sorted([x for x in items if x['pct'] < -0.01], key=lambda x: x['pct'], reverse=True)
        flat = [x for x in items if -0.01 <= x['pct'] < 0.01]
        
        html_forex += f'<div class="currency-col">'
        for x in winners:
            html_forex += f'<div class="tile" style="background:{get_color(x["pct"])}"><span class="pair">{x["other"]}</span><span>+{x["pct"]:.2f}%</span></div>'
        
        html_forex += f'<div class="sep">{curr}</div>'
        
        for x in flat:
            html_forex += f'<div class="tile" style="background:#374151"><span class="pair">{x["other"]}</span><span>unch</span></div>'
            
        for x in losers:
            html_forex += f'<div class="tile" style="background:{get_color(x["pct"])}"><span class="pair">{x["other"]}</span><span>{x["pct"]:.2f}%</span></div>'
        html_forex += '</div>'
    html_forex += '</div>'

    # --- SECTION INDICES & METAUX ---
    def make_grid(cat):
        sub = df[df['cat'] == cat].sort_values('pct', ascending=False)
        h = '<div class="grid-container">'
        for _, r in sub.iterrows():
            h += f'''<div class="box" style="background:{get_color(r['pct'])}">
                        <div style="font-size:0.8rem; opacity:0.9;">{r['name']}</div>
                        <div style="font-size:1.1rem; font-weight:bold;">{r['pct']:+.2f}%</div>
                     </div>'''
        h += '</div>'
        return h

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; color: white; }}
        .matrix-container {{ display: flex; gap: 8px; overflow-x: auto; padding-bottom: 10px; }}
        .currency-col {{ min-width: 110px; display: flex; flex-direction: column; gap: 2px; }}
        .tile {{ display: flex; justify-content: space-between; padding: 5px 8px; font-size: 11px; font-weight: bold; border-radius: 3px; }}
        .sep {{ background: #1f2937; color: #9ca3af; font-weight: 900; text-align: center; padding: 4px; margin: 2px 0; border: 1px solid #374151; border-radius: 3px; }}
        
        h3 {{ color: #9ca3af; border-bottom: 1px solid #374151; padding-bottom: 5px; margin-top: 25px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
        
        .grid-container {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .box {{ width: 140px; height: 75px; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
    </style>
    </head>
    <body>
        <h3>Forex Performance Map</h3>
        {html_forex}
        <h3>Indices Mondiaux</h3>
        {make_grid('INDICES')}
        <h3>Matières Premières</h3>
        {make_grid('COMMODITIES')}
    </body>
    </html>
    """

def display_currency_card(curr, value, prev_value):
    change = value - prev_value
    # Couleurs
    if value >= 7: c_txt, c_bg = "text-green", "bg-green"
    elif value >= 5.5: c_txt, c_bg = "text-blue", "bg-blue"
    elif value >= 4: c_txt, c_bg = "text-orange", "bg-orange"
    else: c_txt, c_bg = "text-red", "bg-red"
    # Flèche
    if change > 0.05: arrow, a_col = "▲", "text-green"
    elif change < -0.05: arrow, a_col = "▼", "text-red"
    else: arrow, a_col = "▶", "text-gray"

    flag = f"https://flagcdn.com/48x36/{FLAG_URLS.get(curr, 'unknown')}.png"
    width = min(max(value * 10, 0), 100)

    return f"""
    <div class="currency-card">
        <div class="card-header">
            <img src="{flag}" class="flag-img"> <span>{curr}</span>
        </div>
        <div class="strength-score {c_txt}">
            {value:.1f} <span style="font-size:1.2rem; vertical-align:middle;" class="{a_col}">{arrow}</span>
        </div>
        <div class="progress-bg"><div class="progress-fill {c_bg}" style="width:{width}%;"></div></div>
    </div>
    """

# ==========================================
# 6. EXÉCUTION PRINCIPALE
# ==========================================

if not access_token:
    st.warning("⚠️ Token OANDA manquant (Sidebar ou Secrets)")
else:
    # --- CHARGEMENT DES DONNÉES (Barre de progression unique) ---
    with st.status("Analyse des marchés en cours...", expanded=True) as status:
        
        # 1. Calcul Strength Meter
        st.write("Calcul de la force des devises (RSI)...")
        df_strength, _ = process_strength_data(access_token, environment, granularity, lookback_chart)
        
        # 2. Calcul Market Map
        st.write(f"Analyse des performances (J-{map_lookback})...")
        df_map = get_map_data(access_token, environment, map_lookback)
        
        status.update(label="Chargement terminé !", state="complete", expanded=False)

    # --- AFFICHAGE PARTIE 1 : STRENGTH METER ---
    if df_strength is not None:
        latest = df_strength.iloc[-1]
        previous = df_strength.iloc[-2]
        sorted_curr = latest.sort_values(ascending=False).index.tolist()
        
        # Cartes
        cols1 = st.columns(4)
        cols2 = st.columns(4)
        for i, curr in enumerate(sorted_curr[:4]):
            cols1[i].markdown(display_currency_card(curr, latest[curr], previous[curr]), unsafe_allow_html=True)
        for i, curr in enumerate(sorted_curr[4:]):
            cols2[i].markdown(display_currency_card(curr, latest[curr], previous[curr]), unsafe_allow_html=True)

        # Graphiques Small Multiples
        st.write(""); st.subheader("Tendances")
        df_chart = df_strength.tail(lookback_chart)
        fig = make_subplots(rows=2, cols=4, subplot_titles=sorted_curr, vertical_spacing=0.15)
        
        for idx, curr in enumerate(sorted_curr):
            row, col = (idx // 4) + 1, (idx % 4) + 1
            val = latest[curr]
            col_line = "#10B981" if val >= 5.5 else "#EF4444" if val <= 4.5 else "#3B82F6"
            
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[curr], mode='lines',
                line=dict(color=col_line, width=2), fill='tozeroy',
                fillcolor=f"rgba{tuple(int(col_line.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}"
            ), row=row, col=col)
            
        fig.update_layout(template="plotly_dark", height=400, showlegend=False, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(range=[0, 10], showgrid=False, showticklabels=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- AFFICHAGE PARTIE 2 : MARKET MAP ---
    st.markdown("---")
    st.header("🗺️ Market Map Pro")
    
    if not df_map.empty:
        html_map = generate_map_html(df_map)
        st.components.v1.html(html_map, height=800, scrolling=True)
    else:
        st.error("Impossible de générer la Market Map. Vérifiez la connexion API.")
