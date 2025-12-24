import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# CONFIGURATION ET STYLE CSS (DESIGN RÉPLIQUE)
# ==========================================
st.set_page_config(page_title="Bluestar Currency Strength", layout="wide")

# Injection CSS pour imiter currencystrengthmeter.org
st.markdown("""
<style>
    /* Fond global sombre */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Style des cartes de devises */
    .currency-card {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
        transition: transform 0.2s;
        text-align: center;
    }
    .currency-card:hover {
        transform: translateY(-2px);
        border-color: #4b5563;
    }
    
    /* En-tête de la carte (Drapeau + Code) */
    .card-header {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        color: white;
    }
    
    /* Images drapeaux */
    .flag-img {
        width: 28px;
        height: 20px;
        border-radius: 2px;
        object-fit: cover;
    }
    
    /* Score central */
    .strength-score {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    
    /* Flèche de tendance */
    .trend-arrow {
        font-size: 1.2rem;
        vertical-align: middle;
        margin-left: 5px;
    }
    
    /* Barre de progression container */
    .progress-bg {
        background-color: #374151;
        height: 8px;
        border-radius: 4px;
        width: 100%;
        margin-top: 10px;
        overflow: hidden;
    }
    
    /* Barre de progression remplissage */
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Couleurs dynamiques */
    .text-green { color: #10B981; }
    .text-blue { color: #3B82F6; }
    .text-orange { color: #F59E0B; }
    .text-red { color: #EF4444; }
    
    .bg-green { background-color: #10B981; }
    .bg-blue { background-color: #3B82F6; }
    .bg-orange { background-color: #F59E0B; }
    .bg-red { background-color: #EF4444; }

    /* Cacher le menu hamburger pour un look plus "site web" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# Mapping des drapeaux (Codes ISO pays pour flagcdn)
FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch"
}

# Couleurs pour le graphique
CHART_COLORS = {
    "USD": "#2962FF", "EUR": "#00E676", "GBP": "#FF6D00", "JPY": "#AA00FF",
    "AUD": "#00B0FF", "CAD": "#FF1744", "NZD": "#FFEA00", "CHF": "#00C853"
}

st.title("💎 Bluestar Currency Strength Meter")
st.markdown("---")

# ==========================================
# GESTION DES SECRETS & SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Configuration")
    
    # Secrets
    secret_token = st.secrets.get("OANDA_ACCESS_TOKEN", None)
    
    if secret_token:
        access_token = secret_token
    else:
        st.info("Mode manuel")
        access_token = st.text_input("OANDA Token", type="password")

    environment = st.selectbox("Environnement", ["practice", "live"], index=0)
    
    st.markdown("---")
    granularity = st.selectbox("Timeframe", ["M15", "M30", "H1", "H4", "D", "W"], index=4)
    length_input = st.number_input("Période RSI", value=14)
    smoothing = st.number_input("Lissage", value=3)
    lookback = st.slider("Historique Graphique", 50, 500, 100)

# ==========================================
# FONCTIONS BACKEND (IDENTIQUES)
# ==========================================

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=60, show_spinner=False) # Cache court (1 min) pour réactivité
def fetch_oanda_data(token, env, granular, count):
    pairs_list = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    
    try:
        client = API(access_token=token, environment=env)
    except Exception as e:
        return None, f"Erreur API: {str(e)}"

    df_dict = {}
    params = {"count": count + 50, "granularity": granular, "price": "M"}

    # Pas de barre de progression visible pour garder le look propre
    try:
        for pair in pairs_list:
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            client.request(r)
            candles = r.response['candles']
            data = []
            for candle in candles:
                if candle['complete']:
                    data.append({"Time": candle['time'], pair: float(candle['mid']['c'])})
            
            temp_df = pd.DataFrame(data)
            temp_df['Time'] = pd.to_datetime(temp_df['Time'])
            temp_df.set_index('Time', inplace=True)
            df_dict[pair] = temp_df[pair]
    except Exception:
        pass 

    if not df_dict: return None, "Erreur de données"
    
    full_df = pd.DataFrame(df_dict)
    full_df = full_df.fillna(method='ffill').fillna(method='bfill')
    return full_df, None

def calculate_strength(df, length, smooth):
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    strength_df = pd.DataFrame(index=df.index)
    
    for curr in currencies:
        total_strength = pd.Series(0.0, index=df.index)
        valid_pairs = 0
        opponents = [c for c in currencies if c != curr]
        
        for opp in opponents:
            pair_direct = f"{curr}_{opp}"
            pair_inverse = f"{opp}_{curr}"
            rsi_series = None
            
            if pair_direct in df.columns:
                rsi_series = calculate_rsi(df[pair_direct], length)
            elif pair_inverse in df.columns:
                rsi_series = calculate_rsi(1/df[pair_inverse], length)
            
            if rsi_series is not None:
                total_strength += (rsi_series - 50) / 50
                valid_pairs += 1
        
        if valid_pairs > 0:
            avg_strength = total_strength / valid_pairs
            # Conversion 0-10 et Lissage
            strength_df[curr] = ((avg_strength + 1) * 5).rolling(window=smooth).mean()
            
    return strength_df.dropna()

# ==========================================
# FONCTION D'AFFICHAGE HTML (CARD)
# ==========================================
def display_currency_card(curr, value, prev_value):
    # Calcul de la tendance
    change = value - prev_value
    
    # Détermination des couleurs et icônes
    if value >= 7:
        color_class = "text-green"
        bg_class = "bg-green"
    elif value >= 5.5:
        color_class = "text-blue"
        bg_class = "bg-blue"
    elif value >= 4:
        color_class = "text-orange"
        bg_class = "bg-orange"
    else:
        color_class = "text-red"
        bg_class = "bg-red"

    # Flèche
    if change > 0.05:
        arrow = "▲"
        arrow_color = "text-green"
    elif change < -0.05:
        arrow = "▼"
        arrow_color = "text-red"
    else:
        arrow = "▶"
        arrow_color = "text-gray"

    flag_code = FLAG_URLS.get(curr, "unknown")
    flag_url = f"https://flagcdn.com/48x36/{flag_code}.png"
    
    # Largeur de la barre (x10 car valeur sur 10)
    bar_width = min(max(value * 10, 0), 100)

    html = f"""
    <div class="currency-card">
        <div class="card-header">
            <img src="{flag_url}" class="flag-img">
            <span>{curr}</span>
        </div>
        <div class="strength-score {color_class}">
            {value:.1f}
            <span class="trend-arrow {arrow_color}">{arrow}</span>
        </div>
        <div class="progress-bg">
            <div class="progress-fill {bg_class}" style="width: {bar_width}%;"></div>
        </div>
    </div>
    """
    return html

# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================

if not access_token:
    st.warning("⚠️ Token manquant")
else:
    with st.spinner('Analyse du marché en cours...'):
        df_prices, error = fetch_oanda_data(access_token, environment, granularity, lookback)
    
    if df_prices is not None:
        df_strength = calculate_strength(df_prices, length_input, smoothing)
        
        # Récupération des deux dernières valeurs pour la tendance
        latest = df_strength.iloc[-1]
        previous = df_strength.iloc[-2]
        
        # Tri par force (Le plus fort en premier, comme sur le site de ref)
        sorted_currencies = latest.sort_values(ascending=False).index.tolist()
        
        # ==========================================
        # 1. GRILLE DE CARTES (METER)
        # ==========================================
        
        # On sépare en 2 rangées de 4 colonnes
        row1 = sorted_currencies[:4]
        row2 = sorted_currencies[4:]
        
        cols1 = st.columns(4)
        for i, curr in enumerate(row1):
            with cols1[i]:
                st.markdown(display_currency_card(curr, latest[curr], previous[curr]), unsafe_allow_html=True)
                
        cols2 = st.columns(4)
        for i, curr in enumerate(row2):
            with cols2[i]:
                st.markdown(display_currency_card(curr, latest[curr], previous[curr]), unsafe_allow_html=True)

        # ==========================================
        # 2. GRAPHIQUE HISTORIQUE
        # ==========================================
        st.write("") # Spacer
        st.write("")
        st.subheader("Historique de tendance")
        
        df_display = df_strength.tail(lookback)
        
        fig = go.Figure()
        for col in df_display.columns:
            # Opacité réduite pour les lignes non survolées (optionnel, ici tout visible)
            fig.add_trace(go.Scatter(
                x=df_display.index, 
                y=df_display[col], 
                mode='lines', 
                name=col,
                line=dict(color=CHART_COLORS[col], width=2)
            ))

        fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=7, line_dash="dot", line_color="green", opacity=0.5)
        fig.add_hline(y=3, line_dash="dot", line_color="red", opacity=0.5)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', # Transparent pour fondre avec le fond CSS
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            yaxis=dict(range=[0, 10], gridcolor='#374151'),
            xaxis=dict(gridcolor='#374151'),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    elif error:
        st.error(error)
