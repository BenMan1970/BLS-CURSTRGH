import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# 1. CONFIGURATION ET STYLE CSS (DESIGN RÉPLIQUE)
# ==========================================
st.set_page_config(page_title="Bluestar Currency Strength", layout="wide")

# Injection CSS pour le design des cartes et du fond
st.markdown("""
<style>
    /* Fond global sombre */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Style des cartes de devises */
    .currency-card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
        text-align: center;
        transition: transform 0.2s;
    }
    .currency-card:hover {
        border-color: #6b7280;
        transform: translateY(-2px);
    }
    
    /* En-tête de la carte (Drapeau + Code) */
    .card-header {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Images drapeaux */
    .flag-img {
        width: 32px;
        height: 24px;
        border-radius: 3px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Score central */
    .strength-score {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 5px 0;
        letter-spacing: -1px;
    }
    
    /* Flèche de tendance */
    .trend-arrow {
        font-size: 1.5rem;
        vertical-align: middle;
        margin-left: 8px;
    }
    
    /* Barre de progression container */
    .progress-bg {
        background-color: #374151;
        height: 6px;
        border-radius: 3px;
        width: 100%;
        margin-top: 15px;
        overflow: hidden;
    }
    
    /* Barre de progression remplissage */
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Classes de couleurs dynamiques */
    .text-green { color: #10B981; }
    .text-blue { color: #3B82F6; }
    .text-orange { color: #F59E0B; }
    .text-red { color: #EF4444; }
    
    .bg-green { background-color: #10B981; }
    .bg-blue { background-color: #3B82F6; }
    .bg-orange { background-color: #F59E0B; }
    .bg-red { background-color: #EF4444; }

    /* Nettoyage interface Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# Mapping des drapeaux
FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch"
}

st.title("💎 Bluestar Currency Strength Meter")
st.markdown("---")

# ==========================================
# 2. GESTION DES SECRETS & BARRE LATÉRALE
# ==========================================
with st.sidebar:
    st.header("Configuration")
    
    # Récupération automatique des secrets
    secret_token = st.secrets.get("OANDA_ACCESS_TOKEN", None)
    
    if secret_token:
        st.success("✅ Connecté (Secrets)")
        access_token = secret_token
    else:
        st.info("Mode Manuel")
        access_token = st.text_input("Token OANDA", type="password")

    environment = st.selectbox("Environnement", ["practice", "live"], index=0, help="Practice=Demo, Live=Réel")
    
    st.markdown("---")
    st.caption("Paramètres de calcul")
    granularity = st.selectbox("Timeframe", ["M15", "M30", "H1", "H4", "D", "W"], index=4)
    length_input = st.number_input("Période RSI", value=14, min_value=1)
    smoothing = st.number_input("Lissage", value=3, min_value=1)
    lookback = st.slider("Historique (Bougies)", 50, 500, 100)

# ==========================================
# 3. FONCTIONS DE CALCUL (BACKEND)
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

@st.cache_data(ttl=60, show_spinner=False) 
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
        return None, f"Erreur Client API: {str(e)}"

    df_dict = {}
    params = {"count": count + 50, "granularity": granular, "price": "M"}

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
    except Exception as e:
        # On continue même si une paire échoue (ex: compte restreint)
        pass 

    if not df_dict: return None, "Aucune donnée reçue. Vérifiez le token et le type de compte."
    
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
            strength_df[curr] = ((avg_strength + 1) * 5).rolling(window=smooth).mean()
            
    return strength_df.dropna()

# ==========================================
# 4. COMPOSANT VISUEL (CARTE HTML)
# ==========================================
def display_currency_card(curr, value, prev_value):
    change = value - prev_value
    
    # Logique de couleur stricte
    if value >= 7:
        color_class, bg_class = "text-green", "bg-green"
    elif value >= 5.5:
        color_class, bg_class = "text-blue", "bg-blue"
    elif value >= 4:
        color_class, bg_class = "text-orange", "bg-orange"
    else:
        color_class, bg_class = "text-red", "bg-red"

    # Logique de flèche
    if change > 0.05:
        arrow, arrow_color = "▲", "text-green"
    elif change < -0.05:
        arrow, arrow_color = "▼", "text-red"
    else:
        arrow, arrow_color = "▶", "text-gray"

    flag_url = f"https://flagcdn.com/48x36/{FLAG_URLS.get(curr, 'unknown')}.png"
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
# 5. EXÉCUTION PRINCIPALE
# ==========================================

if not access_token:
    st.warning("⚠️ Veuillez configurer votre Token OANDA dans la sidebar ou les secrets.")
else:
    with st.spinner('Connexion aux marchés OANDA...'):
        df_prices, error = fetch_oanda_data(access_token, environment, granularity, lookback)
    
    if error:
        st.error(error)
    elif df_prices is not None:
        # Calculs
        df_strength = calculate_strength(df_prices, length_input, smoothing)
        
        latest = df_strength.iloc[-1]
        previous = df_strength.iloc[-2]
        
        # Tri : Forts en premier
        sorted_currencies = latest.sort_values(ascending=False).index.tolist()
        
        # --- SECTION 1 : JAUGES (CARDS) ---
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

        # --- SECTION 2 : GRAPHIQUES "SMALL MULTIPLES" (Séparés) ---
        st.write("")
        st.write("")
        st.subheader("Tendances Individuelles")
        
        # Filtrer pour l'affichage
        df_display = df_strength.tail(lookback)
        
        # Création de la grille 2x4
        fig = make_subplots(
            rows=2, cols=4, 
            subplot_titles=sorted_currencies, # Titres dans l'ordre de force
            vertical_spacing=0.15,
            horizontal_spacing=0.05
        )
        
        for idx, curr in enumerate(sorted_currencies):
            # Calcul position grille (1-4, 5-8)
            row = (idx // 4) + 1
            col = (idx % 4) + 1
            
            # Couleur dynamique du graphique
            val = latest[curr]
            if val >= 5.5: line_col = "#10B981" # Vert
            elif val <= 4.5: line_col = "#EF4444" # Rouge
            else: line_col = "#3B82F6" # Bleu
            
            # Création de la couleur de remplissage (transparente)
            # Conversion Hex -> RGB
            hex_col = line_col.lstrip('#')
            rgb = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))
            fill_col = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)"
            
            fig.add_trace(
                go.Scatter(
                    x=df_display.index, 
                    y=df_display[curr],
                    mode='lines',
                    line=dict(color=line_col, width=2),
                    fill='tozeroy',
                    fillcolor=fill_col,
                    hoverinfo='y+x'
                ),
                row=row, col=col
            )

        # Mise en page propre sans quadrillage
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Nettoyage des axes
        fig.update_yaxes(range=[0, 10], showgrid=False, zeroline=False, tickfont=dict(size=8), showticklabels=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        
        # Ajout de lignes guides discrètes (5.0)
        fig.add_hline(y=5, line_dash="dot", line_color="gray", opacity=0.3, line_width=1)

        st.plotly_chart(fig, use_container_width=True)
