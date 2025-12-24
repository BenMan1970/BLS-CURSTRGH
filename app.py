//@version=5
indicator("💎 Bluestar Currency Strength Meter", overlay=false, max_bars_back=500)

// ==========================================
// CONFIGURATION
// ==========================================
lengthInput = input.int(14, "Période de Calcul", minval=1, maxval=100)
smoothing = input.int(3, "Lissage", minval=1, maxval=10)
showTable = input.bool(true, "Afficher Tableau", group="Affichage")
tablePosition = input.string("top_right", "Position Tableau", 
    options=["top_left", "top_center", "top_right", "middle_left", "middle_center", 
    "middle_right", "bottom_left", "bottom_center", "bottom_right"], group="Affichage")

// ==========================================
// FONCTION: RÉCUPÉRATION SÉCURISÉE DES PAIRES
// ==========================================
// ✅ CORRECTION: Utiliser uniquement les paires valides OANDA
getPairData(base, quote) =>
    // Construire le symbole dans le bon ordre (OANDA utilise BASE_QUOTE)
    pair = base + quote
    
    // Vérifier si la paire existe en format standard
    validPairs = array.from(
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD",
        "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD",
        "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD",
        "CADJPY", "CADCHF", "NZDJPY", "NZDCAD", "NZDCHF", "CHFJPY")
    
    pairExists = array.includes(validPairs, pair)
    
    // Si la paire existe, récupérer les données
    if pairExists
        [close, high, low] = request.security("OANDA:" + pair, timeframe.period, [close, high, low], 
            gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off)
        [close, high, low, true]
    else
        // Essayer la paire inversée
        inversePair = quote + base
        inverseExists = array.includes(validPairs, inversePair)
        
        if inverseExists
            [c, h, l] = request.security("OANDA:" + inversePair, timeframe.period, [close, high, low],
                gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off)
            // ✅ Inverser les valeurs pour obtenir la cotation correcte
            [1/c, 1/l, 1/h, true]
        else
            [na, na, na, false]

// ==========================================
// FONCTION: CALCUL DE LA FORCE D'UNE DEVISE
// ==========================================
calcCurrencyStrength(currency) =>
    var opponents = array.new_string(0)
    
    // Définir les devises opposées
    if currency == "USD"
        opponents := array.from("EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD")
    else if currency == "EUR"
        opponents := array.from("USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD")
    else if currency == "GBP"
        opponents := array.from("USD", "EUR", "JPY", "CHF", "AUD", "CAD", "NZD")
    else if currency == "JPY"
        opponents := array.from("USD", "EUR", "GBP", "CHF", "AUD", "CAD", "NZD")
    else if currency == "AUD"
        opponents := array.from("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "NZD")
    else if currency == "CAD"
        opponents := array.from("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD")
    else if currency == "NZD"
        opponents := array.from("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD")
    else if currency == "CHF"
        opponents := array.from("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD")
    
    strength = 0.0
    validPairs = 0
    
    // Calculer la force relative contre chaque devise
    for i = 0 to array.size(opponents) - 1
        opponent = array.get(opponents, i)
        [closePrice, highPrice, lowPrice, isValid] = getPairData(currency, opponent)
        
        if isValid and not na(closePrice)
            // Calculer le RSI pour cette paire
            change = ta.change(closePrice)
            gain = change >= 0 ? change : 0.0
            loss = change < 0 ? -change : 0.0
            
            avgGain = ta.rma(gain, lengthInput)
            avgLoss = ta.rma(loss, lengthInput)
            
            rs = avgLoss == 0 ? 100.0 : avgGain / avgLoss
            rsi = 100 - (100 / (1 + rs))
            
            // Ajouter à la force (RSI > 50 = devise forte)
            strength += (rsi - 50) / 50  // Normaliser entre -1 et 1
            validPairs += 1
    
    // Moyenne et normalisation 0-10
    avgStrength = validPairs > 0 ? strength / validPairs : 0.0
    normalized = (avgStrength + 1) * 5  // Convertir -1/1 vers 0-10
    
    // Appliquer le lissage
    ta.sma(normalized, smoothing)

// ==========================================
// CALCUL DES FORCES
// ==========================================
usdStrength = calcCurrencyStrength("USD")
eurStrength = calcCurrencyStrength("EUR")
gbpStrength = calcCurrencyStrength("GBP")
jpyStrength = calcCurrencyStrength("JPY")
audStrength = calcCurrencyStrength("AUD")
cadStrength = calcCurrencyStrength("CAD")
nzdStrength = calcCurrencyStrength("NZD")
chfStrength = calcCurrencyStrength("CHF")

// ==========================================
// AFFICHAGE GRAPHIQUE
// ==========================================
plot(usdStrength, "USD", color=color.new(#2962FF, 0), linewidth=2)
plot(eurStrength, "EUR", color=color.new(#00E676, 0), linewidth=2)
plot(gbpStrength, "GBP", color=color.new(#FF6D00, 0), linewidth=2)
plot(jpyStrength, "JPY", color=color.new(#AA00FF, 0), linewidth=2)
plot(audStrength, "AUD", color=color.new(#00B0FF, 0), linewidth=2)
plot(cadStrength, "CAD", color=color.new(#FF1744, 0), linewidth=2)
plot(nzdStrength, "NZD", color=color.new(#FFEA00, 0), linewidth=2)
plot(chfStrength, "CHF", color=color.new(#00C853, 0), linewidth=2)

// Ligne médiane
hline(5, "Neutre", color=color.gray, linestyle=hline.style_dashed)
hline(7, "Fort", color=color.green, linestyle=hline.style_dotted)
hline(3, "Faible", color=color.red, linestyle=hline.style_dotted)

// ==========================================
// TABLEAU DE CLASSEMENT
// ==========================================
if showTable and barstate.islast
    // Créer les données pour le tri
    var currencies = array.from("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF")
    var strengths = array.from(usdStrength, eurStrength, gbpStrength, jpyStrength, 
        audStrength, cadStrength, nzdStrength, chfStrength)
    
    // Trier par force décroissante (bubble sort)
    size = array.size(currencies)
    for i = 0 to size - 2
        for j = 0 to size - 2 - i
            if array.get(strengths, j) < array.get(strengths, j + 1)
                // Échanger
                tempStr = array.get(strengths, j)
                tempCur = array.get(currencies, j)
                array.set(strengths, j, array.get(strengths, j + 1))
                array.set(currencies, j, array.get(currencies, j + 1))
                array.set(strengths, j + 1, tempStr)
                array.set(currencies, j + 1, tempCur)
    
    // Créer le tableau
    var table tbl = table.new(tablePosition, 3, 9, 
        bgcolor=color.new(#000000, 10), frame_color=color.new(#3B82F6, 0), frame_width=2)
    
    // En-tête
    table.cell(tbl, 0, 0, "Rang", bgcolor=color.new(#1E3A8A, 0), text_color=color.white, 
        text_size=size.small, text_font_family=font.family_monospace)
    table.cell(tbl, 1, 0, "Devise", bgcolor=color.new(#1E3A8A, 0), text_color=color.white, 
        text_size=size.small, text_font_family=font.family_monospace)
    table.cell(tbl, 2, 0, "Force", bgcolor=color.new(#1E3A8A, 0), text_color=color.white, 
        text_size=size.small, text_font_family=font.family_monospace)
    
    // Remplir le tableau
    for i = 0 to size - 1
        curr = array.get(currencies, i)
        str = array.get(strengths, i)
        
        // Couleur selon la force
        bgColor = str >= 7 ? color.new(#10B981, 80) : 
                  str >= 5.5 ? color.new(#3B82F6, 80) : 
                  str >= 4 ? color.new(#F59E0B, 80) : 
                  color.new(#EF4444, 80)
        
        // Icône
        icon = str >= 7 ? "🟢" : str >= 5.5 ? "🔵" : str >= 4 ? "🟡" : "🔴"
        
        table.cell(tbl, 0, i + 1, str.tostring("#" + str(i + 1)), 
            bgcolor=bgColor, text_color=color.white, text_size=size.small)
        table.cell(tbl, 1, i + 1, icon + " " + curr, 
            bgcolor=bgColor, text_color=color.white, text_size=size.normal, 
            text_font_family=font.family_monospace)
        table.cell(tbl, 2, i + 1, str.tostring("0.0") + "/10", 
            bgcolor=bgColor, text_color=color.white, text_size=size.small)

// ==========================================
// ALERTES
// ==========================================
// Alerte devise très forte
alertcondition(usdStrength > 8 or eurStrength > 8 or gbpStrength > 8 or jpyStrength > 8 or 
    audStrength > 8 or cadStrength > 8 or nzdStrength > 8 or chfStrength > 8, 
    title="Devise Très Forte", 
    message="Une devise a dépassé 8.0/10")

// Alerte devise très faible
alertcondition(usdStrength < 2 or eurStrength < 2 or gbpStrength < 2 or jpyStrength < 2 or 
    audStrength < 2 or cadStrength < 2 or nzdStrength < 2 or chfStrength < 2, 
    title="Devise Très Faible", 
    message="Une devise est tombée sous 2.0/10")

// Alerte divergence forte
divergence = math.max(usdStrength, eurStrength, gbpStrength, jpyStrength, audStrength, cadStrength, nzdStrength, chfStrength) - 
             math.min(usdStrength, eurStrength, gbpStrength, jpyStrength, audStrength, cadStrength, nzdStrength, chfStrength)

alertcondition(divergence > 6, 
    title="Divergence Extrême", 
    message="Écart de force > 6.0 entre devises")
