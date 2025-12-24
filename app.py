import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# CONFIGURATION ET STYLE
# ==========================================
st.set_page_config(page_title="Bluestar Currency Strength", layout="wide")

# Couleurs
COLORS = {
    "USD": "#2962FF", "EUR": "#00E676", "GBP": "#FF6D00", "JPY": "#AA00FF",
    "AUD": "#00B0FF", "CAD": "#FF1744", "NZD": "#FFEA00", "CHF": "#00C853"
}

st.title("💎 Bluestar Currency Strength Meter")
st.markdown("via **OANDA API**")
st.markdown("---")

# ==========================================
# GESTION DES SECRETS (Correction ici)
# ==========================================
with st.sidebar:
    st.header("🔑 Connexion OANDA")
    
    # 1. Chargement depuis les Secrets avec vos clés spécifiques
    secret_token = st.secrets.get("OANDA_ACCESS_TOKEN", None)
    secret_account = st.secrets.get("OANDA_ACCOUNT_ID", None)
    
    if secret_token:
        st.success(f"✅ Token chargé depuis les secrets")
        if secret_account:
            st.caption(f"ID Compte : {secret_account}")
        access_token = secret_token
    else:
        st.info("Aucun secret trouvé. Entrez le token manuellement.")
        access_token = st.text_input("OANDA Access Token", type="password")

    # Choix de l'environnement (Important car le token diffère entre Practice et Live)
    environment = st.selectbox("Environnement", ["practice", "live"], index=0, 
                             help="Practice = Compte Démo, Live = Compte Réel")
    
    st.markdown("---")
    st.header("⚙️ Paramètres")
    
    granularity = st.selectbox("Unité de temps", ["M15", "M30", "H1", "H4", "D", "W"], index=4)
    length_input = st.number_input("Période RSI", min_value=1, max_value=100, value=14)
    smoothing = st.number_input("Lissage (Moyenne Mobile)", min_value=1, max_value=10, value=3)
    lookback = st.slider("Bougies affichées", 30, 500, 100)

# ==========================================
# FONCTIONS
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_oanda_data(token, env, granular, count):
    """Récupère les données OANDA. Note: L'Account ID n'est pas nécessaire pour lire les bougies."""
    
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
        return None, f"Erreur d'initialisation API : {str(e)}"

    df_dict = {}
    params = {"count": count + 50, "granularity": granular, "price": "M"}

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pair in enumerate(pairs_list):
        status_text.text(f"Téléchargement {pair}...")
        try:
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            client.request(r)
            
            candles = r.response['candles']
            data = []
            for candle in candles:
                if candle['complete']:
                    data.append({
                        "Time": candle['time'],
                        pair: float(candle['mid']['c'])
                    })
            
            temp_df = pd.DataFrame(data)
            temp_df['Time'] = pd.to_datetime(temp_df['Time'])
            temp_df.set_index('Time', inplace=True)
            df_dict[pair] = temp_df[pair]
            
        except Exception:
            pass 
        
        progress_bar.progress((idx + 1) / len(pairs_list))
    
    status_text.empty()
    progress_bar.empty()

    if not df_dict:
        return None, "Échec de récupération. Vérifiez si votre Token correspond bien à l'environnement choisi (Practice vs Live)."

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
# MAIN EXECUTION
# ==========================================

if not access_token:
    st.warning("Veuillez configurer OANDA_ACCESS_TOKEN dans les secrets ou l'entrer manuellement.")
else:
    df_prices, error_msg = fetch_oanda_data(access_token, environment, granularity, lookback+100)
    
    if error_msg:
        st.error(error_msg)
    elif df_prices is not None:
        df_strength = calculate_strength(df_prices, length_input, smoothing)
        df_display = df_strength.tail(lookback)
        
        # 1. Graphique
        fig = go.Figure()
        for col in df_display.columns:
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display[col], 
                                   mode='lines', name=col, line=dict(color=COLORS[col], width=2)))

        fig.add_hline(y=5, line_dash="dash", line_color="gray", annotation_text="Neutre")
        fig.add_hline(y=7, line_dash="dot", line_color="green", annotation_text="Fort")
        fig.add_hline(y=3, line_dash="dot", line_color="red", annotation_text="Faible")
        
        fig.update_layout(title=f"Currency Strength ({granularity})", template="plotly_dark", height=600, yaxis=dict(range=[0, 10]))
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Tableau
        st.subheader("Classement Actuel")
        last_values = df_display.iloc[-1].sort_values(ascending=False)
        rank_df = pd.DataFrame({"Devise": last_values.index, "Force": last_values.values})
        rank_df["Rang"] = range(1, len(rank_df) + 1)
        
        def color_strength(val):
            if val >= 7: return 'background-color: rgba(16, 185, 129, 0.8); color: white;' 
            if val >= 5.5: return 'background-color: rgba(59, 130, 246, 0.8); color: white;' 
            if val >= 4: return 'background-color: rgba(245, 158, 11, 0.8); color: white;' 
            return 'background-color: rgba(239, 68, 68, 0.8); color: white;' 

        st.dataframe(rank_df[["Rang", "Devise", "Force"]].style.applymap(color_strength, subset=['Force']).format({"Force": "{:.2f}/10"}), use_container_width=True, hide_index=True)
