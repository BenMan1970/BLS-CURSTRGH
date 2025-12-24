import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Currency Strength Dashboard",
    page_icon="💱",
    layout="wide"
)

# Titre
st.title("💱 Currency Strength Meter (24H)")
st.markdown("---")

# Récupération des secrets
try:
    OANDA_API_KEY = st.secrets["OANDA_API_KEY"]
    OANDA_ACCOUNT_TYPE = st.secrets.get("OANDA_ACCOUNT_TYPE", "practice")
except:
    st.error("⚠️ Clé API Oanda manquante. Configurez OANDA_API_KEY dans les secrets Streamlit.")
    st.stop()

# URL de l'API Oanda
if OANDA_ACCOUNT_TYPE == "live":
    OANDA_URL = "https://api-fxtrade.oanda.com"
else:
    OANDA_URL = "https://api-fxpractice.oanda.com"

# Configuration des devises
CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'NZD', 'CHF']
CURRENCY_COLORS = {
    'USD': '#FF0000',
    'EUR': '#0000FF',
    'GBP': '#00FF00',
    'JPY': '#800080',
    'AUD': '#FFA500',
    'CAD': '#008080',
    'NZD': '#808080',
    'CHF': '#FFFF00'
}

@st.cache_data(ttl=300)  # Cache pour 5 minutes
def get_oanda_data(instrument, granularity='M1', count=1440):
    """Récupère les données historiques depuis Oanda"""
    headers = {
        'Authorization': f'Bearer {OANDA_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    params = {
        'granularity': granularity,
        'count': count
    }
    
    url = f"{OANDA_URL}/v3/instruments/{instrument}/candles"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'candles' in data:
            candles = data['candles']
            prices = [float(candle['mid']['c']) for candle in candles if candle['complete']]
            return prices
        return None
    except Exception as e:
        st.error(f"Erreur pour {instrument}: {str(e)}")
        return None

def calculate_24h_change(prices):
    """Calcule le changement sur 24H"""
    if prices and len(prices) >= 2:
        current = prices[-1]
        past = prices[0]
        if past != 0:
            return ((current - past) / past) * 100
    return 0

def get_trend_direction(prices):
    """Détermine la tendance (1H)"""
    if prices and len(prices) >= 60:
        current = prices[-1]
        one_hour_ago = prices[-60]
        if one_hour_ago != 0:
            if current > one_hour_ago:
                return 1
            elif current < one_hour_ago:
                return -1
    return 0

def get_currency_strength():
    """Calcule la force de chaque devise"""
    strength_data = {}
    trend_data = {}
    
    # Récupérer les données pour toutes les paires
    pairs_data = {}
    
    with st.spinner('Récupération des données Oanda...'):
        progress_bar = st.progress(0)
        total_pairs = len(CURRENCIES) * (len(CURRENCIES) - 1)
        count = 0
        
        for i, base in enumerate(CURRENCIES):
            for quote in CURRENCIES:
                if base != quote:
                    pair = f"{base}_{quote}"
                    prices = get_oanda_data(pair)
                    
                    if prices:
                        pairs_data[pair] = prices
                    
                    count += 1
                    progress_bar.progress(count / total_pairs)
        
        progress_bar.empty()
    
    # Calculer la force moyenne de chaque devise
    for currency in CURRENCIES:
        changes = []
        
        for other in CURRENCIES:
            if currency != other:
                pair_forward = f"{currency}_{other}"
                pair_backward = f"{other}_{currency}"
                
                if pair_forward in pairs_data:
                    change = calculate_24h_change(pairs_data[pair_forward])
                    changes.append(change)
                    
                    # Trend direction
                    if currency not in trend_data:
                        trend_data[currency] = get_trend_direction(pairs_data[pair_forward])
                
                elif pair_backward in pairs_data:
                    change = -calculate_24h_change(pairs_data[pair_backward])
                    changes.append(change)
                    
                    if currency not in trend_data:
                        trend_data[currency] = -get_trend_direction(pairs_data[pair_backward])
        
        strength_data[currency] = np.mean(changes) if changes else 0
    
    return strength_data, trend_data

# Récupération des données
strength, trends = get_currency_strength()

# Affichage du tableau des forces
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Strength Meter")
    
    # Créer le graphique
    sorted_currencies = sorted(CURRENCIES, key=lambda x: strength.get(x, 0), reverse=True)
    
    fig = go.Figure()
    
    for currency in sorted_currencies:
        value = strength.get(currency, 0)
        trend = trends.get(currency, 0)
        arrow = "▲" if trend > 0 else "▼" if trend < 0 else "→"
        color = CURRENCY_COLORS.get(currency, '#666666')
        
        fig.add_trace(go.Bar(
            y=[currency],
            x=[value],
            orientation='h',
            name=currency,
            marker=dict(color=color),
            text=f"{arrow} {value:.2f}%",
            textposition='outside',
            showlegend=False
        ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Force relative (%)",
        yaxis_title="",
        template="plotly_dark",
        margin=dict(l=100, r=100, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏆 Classement")
    
    ranking_data = []
    for i, currency in enumerate(sorted_currencies, 1):
        value = strength.get(currency, 0)
        trend = trends.get(currency, 0)
        arrow = "🟢" if trend > 0 else "🔴" if trend < 0 else "⚪"
        
        ranking_data.append({
            "Rang": i,
            "Devise": currency,
            "Force": f"{value:.2f}%",
            "Tendance": arrow
        })
    
    df_ranking = pd.DataFrame(ranking_data)
    st.dataframe(df_ranking, use_container_width=True, hide_index=True)

# Statistiques détaillées
st.markdown("---")
st.subheader("📈 Statistiques détaillées")

cols = st.columns(4)
for i, currency in enumerate(sorted_currencies):
    with cols[i % 4]:
        value = strength.get(currency, 0)
        trend = trends.get(currency, 0)
        
        delta_color = "normal" if trend == 0 else "off"
        arrow = "↗" if trend > 0 else "↘" if trend < 0 else "→"
        
        st.metric(
            label=f"{currency}",
            value=f"{value:.2f}%",
            delta=f"{arrow} 1H",
            delta_color=delta_color
        )

# Footer
st.markdown("---")
st.caption(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: Oanda API")
st.caption("💡 Les données sont mises en cache pendant 5 minutes. Rafraîchissez la page pour obtenir les dernières données.")

# Bouton de rafraîchissement
if st.button("🔄 Rafraîchir les données"):
    st.cache_data.clear()
    st.rerun()
