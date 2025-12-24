import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

# ==========================================
# 1. CONFIGURATION & DESIGN ÉPURÉ
# ==========================================
st.set_page_config(page_title="Bluestar Market Dashboard", layout="wide")

# CSS optimisé pour la lecture rapide "Fort vs Faible"
st.markdown("""
<style>
    /* Fond global */
    .stApp { background-color: #0e1117; }
    
    /* --- STYLE CARTES (Haut) --- */
    .currency-card {
        background-color: #1f2937;
        border-radius: 8px; /* Plus carré pour effet "Bloc" */
        padding: 15px 10px;
        margin-bottom: 10px;
        border: 1px solid #374151;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .card-header { 
        display: flex; justify-content: center; align-items: center; gap: 8px; 
        font-weight: bold; color: #e5e7eb; font-size: 1.1rem; 
        margin-bottom: 5px;
    }
    .flag-img { width: 24px; height: 18px; border-radius: 2px; }
    
    .strength-score { 
        font-size: 2.5rem; /* Score très gros pour lecture immédiate */
        font-weight: 800; 
        margin: 0;
        line-height: 1.2;
    }
    
    .progress-bg { background-color: #374151; height: 6px; border-radius: 3px; width: 100%; margin-top: 8px; }
    .progress-fill { height: 100%; border-radius: 3px; }
    
    /* Couleurs Statoriques */
    .text-green { color: #10B981; } .bg-green { background-color: #10B981; }
    .text-blue { color: #3B82F6; } .bg-blue { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    .text-red { color: #EF4444; } .bg-red { background-color: #EF4444; }

    /* --- STYLE MARKET MAP (Bas) --- */
    iframe { width: 100% !important; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Séparateur discret */
    hr { margin: 2em 0; border-color: #374151; }
</style>
""", unsafe_allow_html=True)

FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp",
    "AUD": "au", "CAD": "ca", "NZD": "nz", "CHF": "ch"
}

DISPLAY_NAMES = {
    'US30_USD': 'DOW JONES', 'NAS100_USD': 'NASDAQ 100', 'SPX500_USD': 'S&P 500', 
    'DE30_EUR': 'DAX 40', 'XAU_USD': 'GOLD', 'XPT_USD': 'PLATINUM', 'XAG_USD': 'SILVER'
}

st.title("💎 Bluestar Market Dashboard")

# ==========================================
# 2. DATA ENGINE (Backend)
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

def process_strength_data(token, env, gran):
    # On a besoin d'environ 50 bougies minimum pour stabiliser le RSI et le lissage
    count = 60 
    pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    
    prices_dict = {}
    for pair in pairs:
        df = fetch_candles_generic(token, env, pair, gran, count)
        if df is not None:
            prices_dict[pair] = df['Close']
            
    if not prices_dict: return None
    
    df_prices = pd.DataFrame(prices_dict).fillna(method='ffill').fillna(method='bfill')
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    strength_results = {}
    
    # Calcul optimisé uniquement sur la dernière valeur (pas besoin d'historique complet pour l'affichage)
    for curr in currencies:
        total_strength = 0.0
        valid_pairs = 0
        opponents = [c for c in currencies if c != curr]
        
        for opp in opponents:
            pair_direct = f"{curr}_{opp}"
            pair_inverse = f"{opp}_{curr}"
            
            rsi_val = 50.0
            if pair_direct in df_prices.columns:
                series = calculate_rsi(df_prices[pair_direct])
                rsi_val = series.iloc[-1]
            elif pair_inverse in df_prices.columns:
                series = calculate_rsi(1/df_prices[pair_inverse])
                rsi_val = series.iloc[-1]
            
            total_strength += (rsi_val - 50) / 50
            valid_pairs += 1
        
        if valid_pairs > 0:
            avg = total_strength / valid_pairs
            score = (avg + 1) * 5
            strength_results[curr] = score
            
    return strength_results

def get_map_data(token, env, days_back):
    map_results = []
    
    # 1. Indices & Métaux
    extras = ['US30_USD', 'NAS100_USD', 'SPX500_USD', 'DE30_EUR', 'XAU_USD', 'XPT_USD', 'XAG_USD']
    cats = {'US30_USD':'INDICES', 'NAS100_USD':'INDICES', 'SPX500_USD':'INDICES', 'DE30_EUR':'INDICES',
            'XAU_USD':'COMMODITIES', 'XPT_USD':'COMMODITIES', 'XAG_USD':'COMMODITIES'}
    
    for symbol in extras:
        df = fetch_candles_generic(token, env, symbol, "D", days_back + 5)
        if df is not None and len(df) >= days_back:
            now = df['Close'].iloc[-1]
            past = df['Close'].shift(days_back).iloc[-1]
            pct = ((now - past) / past) * 100
            map_results.append({'symbol': symbol, 'name': DISPLAY_NAMES.get(symbol, symbol), 'pct': pct, 'cat': cats[symbol]})

    # 2. Forex (Daily performance)
    forex_pairs = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD'
    ]
    for pair in forex_pairs:
        df = fetch_candles_generic(token, env, pair, "D", days_back + 5)
        if df is not None and len(df) >= days_back:
            now = df['Close'].iloc[-1]
            past = df['Close'].shift(days_back).iloc[-1]
            pct = ((now - past) / past) * 100
            map_results.append({'symbol': pair, 'name': pair.replace('_', '/'), 'pct': pct, 'cat': 'FOREX'})
            
    return pd.DataFrame(map_results)

# ==========================================
# 3. HTML GENERATORS
# ==========================================
def display_currency_card(curr, value):
    # Pas de flèche de tendance, juste l'état actuel
    if value >= 7: c_txt, c_bg = "text-green", "bg-green"
    elif value >= 5.5: c_txt, c_bg = "text-blue", "bg-blue"
    elif value >= 4: c_txt, c_bg = "text-orange", "bg-orange"
    else: c_txt, c_bg = "text-red", "bg-red"

    flag = f"https://flagcdn.com/48x36/{FLAG_URLS.get(curr, 'unknown')}.png"
    width = min(max(value * 10, 0), 100)

    return f"""
    <div class="currency-card">
        <div class="card-header">
            <img src="{flag}" class="flag-img"> <span>{curr}</span>
        </div>
        <div class="strength-score {c_txt}">{value:.1f}</div>
        <div class="progress-bg"><div class="progress-fill {c_bg}" style="width:{width}%;"></div></div>
    </div>
    """

def generate_map_html(df):
    def get_color(pct):
        if pct >= 0.50: return "#064e3b" 
        if pct >= 0.25: return "#065f46"
        if pct >= 0.10: return "#10b981" 
        if pct >= 0.01: return "#34d399" 
        if pct <= -0.50: return "#7f1d1d" 
        if pct <= -0.25: return "#991b1b"
        if pct <= -0.10: return "#ef4444" 
        if pct <= -0.01: return "#f87171"
        return "#374151"

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
                forex_data[base].append({'other': quote, 'pct': pct})
                forex_data[quote].append({'other': base, 'pct': -pct})

    # Weighted Sort
    scores = {}
    for curr, items in forex_data.items():
        score = 0
        w_sum = 0
        for item in items:
            w = 2.0 if item['other'] in ['USD', 'EUR', 'JPY'] else 1.0
            score += item['pct'] * w
            w_sum += w
        scores[curr] = score / w_sum if w_sum > 0 else 0
    sorted_currencies = sorted(scores, key=scores.get, reverse=True)

    html = '<div style="display: flex; gap: 8px; overflow-x: auto; padding-bottom: 5px;">'
    for curr in sorted_currencies:
        items = forex_data.get(curr, [])
        winners = sorted([x for x in items if x['pct'] >= 0.01], key=lambda x: x['pct'], reverse=True)
        losers = sorted([x for x in items if x['pct'] < -0.01], key=lambda x: x['pct'], reverse=True)
        flat = [x for x in items if -0.01 <= x['pct'] < 0.01]
        
        html += '<div style="min-width: 100px; display: flex; flex-direction: column; gap: 2px;">'
        for x in winners:
            html += f'<div style="background:{get_color(x["pct"])}; display:flex; justify-content:space-between; padding:4px 6px; font-size:11px; font-weight:bold; border-radius:3px; color:white;"><span>{x["other"]}</span><span>+{x["pct"]:.2f}%</span></div>'
        html += f'<div style="background:#1f2937; color:#9ca3af; text-align:center; padding:3px; font-weight:900; font-size:12px; border:1px solid #374151; border-radius:3px;">{curr}</div>'
        for x in flat:
            html += f'<div style="background:#374151; display:flex; justify-content:space-between; padding:4px 6px; font-size:11px; font-weight:bold; border-radius:3px; color:white;"><span>{x["other"]}</span><span>unch</span></div>'
        for x in losers:
            html += f'<div style="background:{get_color(x["pct"])}; display:flex; justify-content:space-between; padding:4px 6px; font-size:11px; font-weight:bold; border-radius:3px; color:white;"><span>{x["other"]}</span><span>{x["pct"]:.2f}%</span></div>'
        html += '</div>'
    html += '</div>'

    def make_grid(cat):
        sub = df[df['cat'] == cat].sort_values('pct', ascending=False)
        h = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
        for _, r in sub.iterrows():
            h += f'''<div style="background:{get_color(r['pct'])}; width:130px; height:60px; display:flex; flex-direction:column; justify-content:center; align-items:center; border-radius:5px; box-shadow:0 2px 4px rgba(0,0,0,0.2); color:white;">
                        <div style="font-size:0.75rem; opacity:0.9;">{r['name']}</div>
                        <div style="font-size:1rem; font-weight:bold;">{r['pct']:+.2f}%</div>
                     </div>'''
        h += '</div>'
        return h

    return f"""
    <!DOCTYPE html>
    <html>
    <head><style>body {{ font-family: 'Segoe UI', sans-serif; margin: 0; color: white; }} h3 {{ color: #9ca3af; border-bottom: 1px solid #374151; padding-bottom: 5px; margin-top: 20px; font-size: 13px; text-transform: uppercase; }}</style></head>
    <body>
        <h3>Forex Matrix</h3>{html}
        <h3>Indices</h3>{make_grid('INDICES')}
        <h3>Matières Premières</h3>{make_grid('COMMODITIES')}
    </body>
    </html>
    """

# ==========================================
# 4. EXÉCUTION
# ==========================================
with st.sidebar:
    st.header("Connexion OANDA")
    token = st.secrets.get("OANDA_ACCESS_TOKEN") or st.text_input("Token", type="password")
    env = st.selectbox("Type", ["practice", "live"])
    
    st.markdown("---")
    st.subheader("Réglages")
    granularity = st.selectbox("Strength Timeframe", ["M15", "M30", "H1", "H4", "D", "W"], index=4)
    map_days = st.slider("Map Lookback (Jours)", 1, 5, 1)

if not token:
    st.warning("⚠️ Token manquant")
else:
    with st.status("Actualisation des données...", expanded=True) as status:
        scores = process_strength_data(token, env, granularity)
        df_map = get_map_data(token, env, map_days)
        status.update(label="Prêt", state="complete", expanded=False)

    if scores:
        # Tri des scores
        sorted_curr = sorted(scores, key=scores.get, reverse=True)
        
        # Affichage Cartes (2 rangées de 4)
        cols1 = st.columns(4)
        cols2 = st.columns(4)
        
        for i, curr in enumerate(sorted_curr[:4]):
            cols1[i].markdown(display_currency_card(curr, scores[curr]), unsafe_allow_html=True)
        for i, curr in enumerate(sorted_curr[4:]):
            cols2[i].markdown(display_currency_card(curr, scores[curr]), unsafe_allow_html=True)
            
        # Résumé textuel rapide pour confluence
        strongest = sorted_curr[0]
        weakest = sorted_curr[-1]
        st.info(f"💡 **Idée Confluence :** Acheter **{strongest}** / Vendre **{weakest}** (Paire : {strongest}/{weakest})")

    if not df_map.empty:
        st.markdown("---")
        st.subheader("🗺️ Market Map Pro")
        st.components.v1.html(generate_map_html(df_map), height=600, scrolling=True)
