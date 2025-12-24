import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from requests.exceptions import ConnectionError

# ==========================================
# CONFIGURATION ET STYLE
# ==========================================
st.set_page_config(page_title="Bluestar Currency Strength (OANDA)", layout="wide")

# Couleurs (Design conservé)
COLORS = {
    "USD": "#2962FF", "EUR": "#00E676", "GBP": "#FF6D00", "JPY": "#AA00FF",
    "AUD": "#00B0FF", "CAD": "#FF1744", "NZD": "#FFEA00", "CHF": "#00C853"
}

st.title("💎 Bluestar Currency Strength Meter")
st.markdown("via **OANDA API**")
st.markdown("---")

# ==========================================
# BARRE LATÉRALE (INPUTS)
# ==========================================
with st.sidebar:
    st.header("🔑 Connexion OANDA")
    
    # Inputs pour l'API
    access_token = st.text_input("Token d'accès API", type="password", help="Votre token personnel OANDA (fxTrade Practice ou Live)")
    environment = st.selectbox("Type de Compte", ["practice", "live"], index=0)
    
    st.markdown("---")
    st.header("⚙️ Paramètres Indicateur")
    
    granularity = st.selectbox("Unité de temps", ["M15", "M30", "H1", "H4", "D", "W"], index=4)
    length_input = st.number_input("Période RSI", min_value=1, max_value=100, value=14)
    smoothing = st.number_input("Lissage (Moyenne Mobile)", min_value=1, max_value=10, value=3)
    lookback = st.slider("Nombre de bougies affichées", 30, 500, 100)

# ==========================================
# FONCTIONS DE CALCUL
# ==========================================

def calculate_rsi(series, period):
    """Calcule le RSI manuellement sur une série Pandas"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=300, show_spinner=False)
def fetch_oanda_data(token, env, granular, count=500):
    """Récupère les données des 28 paires majeures via l'API OANDA"""
    
    # Liste standard des paires OANDA (Format BASE_QUOTE)
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
        return None, f"Erreur de connexion API : {str(e)}"

    df_dict = {}
    params = {"count": count + 50, "granularity": granular, "price": "M"} # M = Mid price

    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pair in enumerate(pairs_list):
        status_text.text(f"Récupération {pair}...")
        try:
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            client.request(r)
            
            # Extraction des données : Time et Close
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
            
            # Fusionner dans le dictionnaire global
            df_dict[pair] = temp_df[pair]
            
        except Exception as e:
            # Si une paire échoue, on continue (certains comptes n'ont pas toutes les paires)
            print(f"Erreur sur {pair}: {e}")
        
        progress_bar.progress((idx + 1) / len(pairs_list))
    
    status_text.empty()
    progress_bar.empty()

    if not df_dict:
        return None, "Aucune donnée récupérée. Vérifiez votre Token et vos permissions."

    # Création du DataFrame global synchronisé
    full_df = pd.DataFrame(df_dict)
    full_df = full_df.fillna(method='ffill').fillna(method='bfill') # Gérer les trous de cotation
    return full_df, None

def calculate_strength(df, length, smooth):
    """Calcule la force relative de chaque devise"""
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    strength_df = pd.DataFrame(index=df.index)
    
    for curr in currencies:
        total_strength = pd.Series(0.0, index=df.index)
        valid_pairs = 0
        
        opponents = [c for c in currencies if c != curr]
        
        for opp in opponents:
            # Construction des noms de paires possibles selon la convention OANDA
            # OANDA a des paires fixes (ex: EUR_USD existe, USD_EUR n'existe pas)
            pair_direct = f"{curr}_{opp}"
            pair_inverse = f"{opp}_{curr}"
            
            rsi_series = None
            
            if pair_direct in df.columns:
                # La paire existe (ex: on veut EUR, et EUR_USD est dispo)
                price = df[pair_direct]
                rsi_series = calculate_rsi(price, length)
                
            elif pair_inverse in df.columns:
                # La paire est inversée (ex: on veut USD, mais seul EUR_USD existe)
                # Astuce mathématique : RSI de (1/Prix) = RSI inversé
                price = df[pair_inverse]
                rsi_series = calculate_rsi(1/price, length)
            
            if rsi_series is not None:
                # Normalisation (-1 à 1)
                norm = (rsi_series - 50) / 50
                total_strength += norm
                valid_pairs += 1
        
        if valid_pairs > 0:
            avg_strength = total_strength / valid_pairs
            # Conversion vers 0-10
            final_val = (avg_strength + 1) * 5
            # Lissage
            strength_df[curr] = final_val.rolling(window=smooth).mean()
            
    return strength_df.dropna()

# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================

if not access_token:
    st.warning("👈 Veuillez entrer votre Token API OANDA dans la barre latérale pour commencer.")
    st.info("Vous pouvez obtenir un token 'Practice' dans votre compte OANDA : Manage API Access.")

else:
    # 1. Récupération des données
    df_prices, error_msg = fetch_oanda_data(access_token, environment, granularity, count=lookback+100)
    
    if error_msg:
        st.error(error_msg)
    elif df_prices is not None:
        
        # 2. Calcul des forces
        df_strength = calculate_strength(df_prices, length_input, smoothing)
        
        # Filtrage pour affichage
        df_display = df_strength.tail(lookback)
        
        # 3. GRAPHIQUE
        fig = go.Figure()
        for col in df_display.columns:
            fig.add_trace(go.Scatter(
                x=df_display.index, 
                y=df_display[col], 
                mode='lines', 
                name=col,
                line=dict(color=COLORS[col], width=2)
            ))

        # Niveaux
        fig.add_hline(y=5, line_dash="dash", line_color="gray", annotation_text="Neutre")
        fig.add_hline(y=7, line_dash="dot", line_color="green", annotation_text="Fort")
        fig.add_hline(y=3, line_dash="dot", line_color="red", annotation_text="Faible")

        fig.update_layout(
            title=f"Currency Strength Meter ({granularity})",
            xaxis_title="Temps",
            yaxis_title="Force (0-10)",
            template="plotly_dark",
            height=600,
            yaxis=dict(range=[0, 10]),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. TABLEAU DE CLASSEMENT
        st.subheader("Classement en temps réel")
        
        # Dernière valeur disponible
        last_values = df_display.iloc[-1].sort_values(ascending=False)
        
        rank_df = pd.DataFrame({
            "Devise": last_values.index,
            "Force": last_values.values
        })
        rank_df["Rang"] = range(1, len(rank_df) + 1)
        rank_df = rank_df[["Rang", "Devise", "Force"]]

        # Style conditionnel
        def color_strength(val):
            if isinstance(val, (int, float)):
                if val >= 7: return 'background-color: rgba(16, 185, 129, 0.8); color: white;' 
                if val >= 5.5: return 'background-color: rgba(59, 130, 246, 0.8); color: white;' 
                if val >= 4: return 'background-color: rgba(245, 158, 11, 0.8); color: white;' 
                return 'background-color: rgba(239, 68, 68, 0.8); color: white;' 
            return ''

        st.dataframe(
            rank_df.style.applymap(color_strength, subset=['Force'])
            .format({"Force": "{:.2f}/10"}),
            use_container_width=True,
            hide_index=True
        )

        # 5. ALERTES
        st.subheader("⚠️ Alertes Détectées")
        col1, col2 = st.columns(2)
        
        with col1:
            for curr, val in last_values.items():
                if val > 8:
                    st.success(f"🚀 **{curr}** est en surachat extrême (> 8.0)")
                elif val < 2:
                    st.error(f"🩸 **{curr}** est en survente extrême (< 2.0)")

        with col2:
            max_s = last_values.max()
            min_s = last_values.min()
            divergence = max_s - min_s
            if divergence > 6:
                st.warning(f"⚡ **Divergence massive ({divergence:.2f})** entre {last_values.index[0]} et {last_values.index[-1]}")
            else:
                st.info("Aucune divergence majeure détectée.")
