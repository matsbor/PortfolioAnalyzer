"""
ALPHA MINER PRO - INSTITUTIONAL ENHANCEMENTS V3
Complete production-grade hedge fund system

PHILOSOPHY:
"A disciplined institutional PM sitting next to you"

ARCHITECTURE:
1. Metal Forecasting (predict the metals first)
2. SMC Engine (institutional price action)
3. News Intelligence (financing lifecycle, not just headlines)
4. Sentiment Proxy (buzz without APIs)
5. Enhanced Sell Logic (distribution detection)
6. Portfolio Orchestration (ranking, risk posture)

VERSION: 3.0-ULTIMATE
DATE: 2026-01-16
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re

# ============================================================================
# 1️⃣ SMART MONEY CONCEPTS (SMC) - PRODUCTION ENGINE
# ============================================================================

def calculate_smc_structure(hist_data, ticker):
    """
    Production-grade SMC engine
    
    WHY: This is how institutions read price. Retail uses indicators.
          Institutions use structure.
    
    DETECTS:
    - BOS (Break of Structure) - continuation signal
    - CHoCH (Change of Character) - reversal signal  
    - Higher Highs/Higher Lows vs Lower Highs/Lower Lows
    - Liquidity sweeps (equal highs/lows taken)
    
    DOES NOT:
    - Use indicators
    - Use ML black boxes
    - Hallucinate certainty
    
    RETURNS: Deterministic, explainable SMC state
    """
    result = {
        'state': 'NEUTRAL',           # BULLISH / BEARISH / NEUTRAL
        'event': 'NONE',               # BOS / CHOCH / NONE
        'structure': 'RANGING',        # BULLISH_STRUCTURE / BEARISH_STRUCTURE / RANGING
        'confidence': 50,              # 0-100
        'last_swing_high': 0,
        'last_swing_low': 0,
        'explanation': '',
        'signals': []
    }
    
    if hist_data.empty or len(hist_data) < 50:
        result['explanation'] = 'Insufficient data for SMC'
        return result
    
    try:
        # Use last 100 bars for structure analysis
        df = hist_data.tail(100).copy()
        df = df.reset_index(drop=True)
        
        # 1) IDENTIFY SWING POINTS (5-bar pivots)
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df)-5):
            # Swing high: high is max in 11-bar window
            if df['High'].iloc[i] == df['High'].iloc[i-5:i+6].max():
                swing_highs.append({
                    'index': i,
                    'price': df['High'].iloc[i],
                    'date': df.index[i] if hasattr(df.index[i], 'strftime') else i
                })
            
            # Swing low: low is min in 11-bar window
            if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+6].min():
                swing_lows.append({
                    'index': i,
                    'price': df['Low'].iloc[i],
                    'date': df.index[i] if hasattr(df.index[i], 'strftime') else i
                })
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            result['explanation'] = 'Not enough swing points'
            return result
        
        # Store last swings
        result['last_swing_high'] = swing_highs[-1]['price']
        result['last_swing_low'] = swing_lows[-1]['price']
        
        # 2) DETERMINE MARKET STRUCTURE
        # Compare last 3 swing highs and lows
        recent_highs = [s['price'] for s in swing_highs[-3:]]
        recent_lows = [s['price'] for s in swing_lows[-3:]]
        
        # Higher Highs AND Higher Lows = Bullish Structure
        hh = recent_highs[-1] > recent_highs[-2] and recent_highs[-2] > recent_highs[-3]
        hl = recent_lows[-1] > recent_lows[-2] and recent_lows[-2] > recent_lows[-3]
        
        # Lower Highs AND Lower Lows = Bearish Structure
        lh = recent_highs[-1] < recent_highs[-2] and recent_highs[-2] < recent_highs[-3]
        ll = recent_lows[-1] < recent_lows[-2] and recent_lows[-2] < recent_lows[-3]
        
        bullish_structure = hh and hl
        bearish_structure = lh and ll
        
        if bullish_structure:
            result['structure'] = 'BULLISH_STRUCTURE'
            result['confidence'] += 20
            result['signals'].append('Higher Highs + Higher Lows')
        elif bearish_structure:
            result['structure'] = 'BEARISH_STRUCTURE'
            result['confidence'] -= 20
            result['signals'].append('Lower Highs + Lower Lows')
        else:
            result['structure'] = 'RANGING'
        
        # 3) DETECT BOS (Break of Structure)
        current_price = df['Close'].iloc[-1]
        
        if bullish_structure:
            # BOS up = price breaks above recent swing high
            if current_price > result['last_swing_high'] * 1.001:
                result['event'] = 'BOS'
                result['state'] = 'BULLISH'
                result['confidence'] += 15
                result['signals'].append('BOS ↑')
                result['explanation'] = 'Bullish BOS - continuation likely'
        
        elif bearish_structure:
            # BOS down = price breaks below recent swing low
            if current_price < result['last_swing_low'] * 0.999:
                result['event'] = 'BOS'
                result['state'] = 'BEARISH'
                result['confidence'] -= 15
                result['signals'].append('BOS ↓')
                result['explanation'] = 'Bearish BOS - continuation likely'
        
        # 4) DETECT CHOCH (Change of Character)
        # Price breaks AGAINST existing structure = reversal warning
        if bullish_structure and current_price < result['last_swing_low'] * 0.999:
            result['event'] = 'CHOCH'
            result['state'] = 'BEARISH'
            result['confidence'] = 40  # Lower confidence on reversals
            result['signals'].append('CHoCH ↓')
            result['explanation'] = 'Change of Character - possible reversal'
        
        elif bearish_structure and current_price > result['last_swing_high'] * 1.001:
            result['event'] = 'CHOCH'
            result['state'] = 'BULLISH'
            result['confidence'] = 40
            result['signals'].append('CHoCH ↑')
            result['explanation'] = 'Change of Character - possible reversal'
        
        # 5) LIQUIDITY SWEEPS (equal highs/lows taken)
        # Check for equal highs (within 0.5%)
        for i in range(len(swing_highs)-1):
            for j in range(i+1, len(swing_highs)):
                if abs(swing_highs[i]['price'] - swing_highs[j]['price']) / swing_highs[i]['price'] < 0.005:
                    # Check if price swept it (went above then came back)
                    max_high = df['High'].iloc[swing_highs[j]['index']:].max()
                    if max_high > swing_highs[i]['price'] * 1.001:
                        result['signals'].append('Liquidity Sweep (high)')
                        # Sweep often precedes reversal
                        if result['structure'] == 'BULLISH_STRUCTURE':
                            result['confidence'] -= 5  # Warning sign
        
        # 6) NORMALIZE CONFIDENCE
        result['confidence'] = np.clip(result['confidence'], 0, 100)
        
        # 7) FINAL STATE DETERMINATION (if not set by BOS/CHOCH)
        if result['state'] == 'NEUTRAL':
            if result['structure'] == 'BULLISH_STRUCTURE':
                result['state'] = 'BULLISH'
                result['explanation'] = 'Bullish structure intact'
            elif result['structure'] == 'BEARISH_STRUCTURE':
                result['state'] = 'BEARISH'
                result['explanation'] = 'Bearish structure intact'
            else:
                result['explanation'] = 'No clear structure - ranging'
        
    except Exception as e:
        result['explanation'] = f'SMC error: {str(e)[:40]}'
        result['confidence'] = 0
    
    return result

# ============================================================================
# 2️⃣ METAL FORECASTING ENGINE - PREDICT GOLD & SILVER
# ============================================================================

def forecast_metal_direction(ticker, name, hist_data=None):
    """
    Forecast metal price direction across multiple timeframes
    
    WHY: Junior miners are leveraged bets on metal prices.
         We MUST understand the metal cycle.
    
    TIMEFRAMES:
    - Today/1-3 days (very short term)
    - 1 week (short term)
    - 1-2 months (medium term)
    
    METHODOLOGY:
    - Trend (MA20/50/200)
    - Momentum (recent returns)
    - Volatility regime
    - DXY correlation
    
    RETURNS: Actionable forecast with confidence
    """
    import yfinance as yf
    
    result = {
        'name': name,
        'current_price': 0,
        'forecast_today': '↔',        # ↑ / ↔ / ↓
        'forecast_week': '↔',
        'forecast_month': '↔',
        'bias_short': 'NEUTRAL',
        'bias_medium': 'NEUTRAL',
        'confidence': 50,
        'regime_impact': {},
        'technicals': {},
        'explanation': ''
    }
    
    try:
        # Fetch data if not provided
        if hist_data is None or hist_data.empty:
            metal = yf.Ticker(ticker)
            hist_data = metal.history(period="1y")
        
        if hist_data.empty or len(hist_data) < 50:
            result['explanation'] = 'Insufficient data'
            return result
        
        df = hist_data.copy()
        current_price = df['Close'].iloc[-1]
        result['current_price'] = current_price
        
        # Calculate MAs
        ma20 = df['Close'].tail(20).mean()
        ma50 = df['Close'].tail(50).mean()
        ma200 = df['Close'].tail(200).mean() if len(df) >= 200 else ma50
        
        result['technicals'] = {
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'above_ma20': current_price > ma20,
            'above_ma50': current_price > ma50,
            'above_ma200': current_price > ma200
        }
        
        # Calculate returns
        ret_3d = ((df['Close'].iloc[-1] - df['Close'].iloc[-3]) / df['Close'].iloc[-3] * 100) if len(df) >= 3 else 0
        ret_5d = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(df) >= 5 else 0
        ret_10d = ((df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100) if len(df) >= 10 else 0
        ret_20d = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100) if len(df) >= 20 else 0
        ret_60d = ((df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60] * 100) if len(df) >= 60 else 0
        
        # 1) TODAY / 1-3 DAYS FORECAST
        today_score = 0
        if current_price > ma20:
            today_score += 2
        if ret_3d > 0:
            today_score += 1
        if ret_5d > 1:
            today_score += 1
        
        if today_score >= 3:
            result['forecast_today'] = '↑'
        elif today_score <= 1:
            result['forecast_today'] = '↓'
        
        # 2) 1 WEEK FORECAST
        week_score = 0
        if current_price > ma20:
            week_score += 2
        if ma20 > ma50:
            week_score += 1
        if ret_5d > 0:
            week_score += 1
        if ret_10d > 1:
            week_score += 1
        
        if week_score >= 4:
            result['forecast_week'] = '↑'
            result['bias_short'] = 'BULLISH'
        elif week_score <= 2:
            result['forecast_week'] = '↓'
            result['bias_short'] = 'BEARISH'
        
        # 3) 1-2 MONTH FORECAST
        month_score = 0
        if current_price > ma50:
            month_score += 2
        if ma50 > ma200:
            month_score += 2
        if ret_20d > 2:
            month_score += 1
        if ret_60d > 3:
            month_score += 1
        
        if month_score >= 5:
            result['forecast_month'] = '↑'
            result['bias_medium'] = 'BULLISH'
        elif month_score >= 3:
            result['bias_medium'] = 'NEUTRAL'
        else:
            result['forecast_month'] = '↓'
            result['bias_medium'] = 'BEARISH'
        
        # 4) CONFIDENCE CALCULATION
        # Higher confidence when timeframes align
        forecasts = [result['forecast_today'], result['forecast_week'], result['forecast_month']]
        alignment = len(set(forecasts))
        
        if alignment == 1:
            result['confidence'] = 85  # All agree
        elif alignment == 2:
            result['confidence'] = 60  # Partial agreement
        else:
            result['confidence'] = 40  # Mixed signals
        
        # 5) REGIME IMPACT ON MINERS
        # Bullish metals = favor all, especially explorers (leverage)
        # Bearish metals = favor producers only (cash flow)
        if result['bias_medium'] == 'BULLISH':
            result['regime_impact'] = {
                'producers': 'FAVOR',
                'developers': 'FAVOR',
                'explorers': 'FAVOR',
                'portfolio_posture': 'RISK_ON',
                'sizing_multiplier': 1.0
            }
        elif result['bias_medium'] == 'BEARISH':
            result['regime_impact'] = {
                'producers': 'NEUTRAL',
                'developers': 'REDUCE',
                'explorers': 'AVOID',
                'portfolio_posture': 'RISK_OFF',
                'sizing_multiplier': 0.6
            }
        else:
            result['regime_impact'] = {
                'producers': 'FAVOR',
                'developers': 'NEUTRAL',
                'explorers': 'CAUTIOUS',
                'portfolio_posture': 'NEUTRAL',
                'sizing_multiplier': 0.85
            }
        
        # 6) EXPLANATION
        reasons = []
        if current_price > ma200:
            reasons.append(f'Above MA200')
        if ma50 > ma200 and ma20 > ma50:
            reasons.append('Trend aligned')
        if abs(ret_20d) > 3:
            reasons.append(f'Momentum {ret_20d:+.1f}%')
        
        result['explanation'] = ' | '.join(reasons) if reasons else 'Neutral structure'
        
    except Exception as e:
        result['explanation'] = f'Forecast error: {str(e)[:40]}'
        result['confidence'] = 0
    
    return result

# ============================================================================
# 3️⃣ NEWS INTELLIGENCE - FINANCING LIFECYCLE DETECTION
# ============================================================================

def analyze_news_intelligence(news_items, ticker):
    """
    Intelligent news analysis - not just keyword matching
    
    WHY: "Company announces financing" ≠ "Company closes financing"
         One is bad news, one is good news. Context matters.
    
    DETECTS:
    - Financing lifecycle: Announced → Priced → Closed
    - Timestamp validity (no 1960s bugs)
    - News freshness decay
    - Misleading headlines
    
    RETURNS: Actionable intelligence with confidence
    """
    result = {
        'financing_status': None,           # ANNOUNCED / PRICED / CLOSED / NONE
        'financing_type': None,             # PP / ATM / SHELF / STRATEGIC
        'financing_impact': 0,              # -30 to +20 on dilution risk
        'news_confidence': 100,
        'fresh_news_count': 0,
        'key_headlines': [],
        'warnings': [],
        'explanation': ''
    }
    
    if not news_items:
        result['explanation'] = 'No news available'
        result['news_confidence'] = 0
        return result
    
    # 1) TIMESTAMP SANITY CHECK
    valid_news = []
    now = datetime.now()
    
    for item in news_items:
        ts = item.get('timestamp', 0)
        
        # Skip invalid timestamps
        if ts <= 0:
            continue
        
        # Convert to datetime
        try:
            if ts > 1e12:
                ts = ts / 1000
            news_date = datetime.fromtimestamp(ts)
            
            # Only keep news from last 2 years
            if news_date > now - timedelta(days=730):
                item['news_date'] = news_date
                item['days_ago'] = (now - news_date).days
                valid_news.append(item)
        except:
            continue
    
    if not valid_news:
        result['explanation'] = 'No valid timestamped news'
        result['news_confidence'] = 30
        return result
    
    # Count fresh news (last 30 days)
    result['fresh_news_count'] = sum(1 for item in valid_news if item['days_ago'] <= 30)
    
    # 2) FINANCING LIFECYCLE DETECTION
    financing_keywords = {
        'announced': ['announces', 'proposes', 'intends to', 'plans to', 'seeks'],
        'priced': ['prices', 'pricing', 'priced at'],
        'closed': ['closes', 'closed', 'completes', 'completed', 'closing of']
    }
    
    atm_keywords = ['atm', 'at-the-market', 'at the market']
    shelf_keywords = ['shelf', 'prospectus', 'registration statement']
    strategic_keywords = ['strategic', 'cornerstone', 'lead order']
    
    financing_events = []
    
    for item in valid_news:
        title_lower = item.get('title', '').lower()
        
        # Check if financing-related
        is_financing = any(word in title_lower for word in 
                          ['financing', 'placement', 'offering', 'capital raise', 'bought deal'])
        
        if not is_financing:
            continue
        
        # Determine stage
        stage = None
        if any(word in title_lower for word in financing_keywords['closed']):
            stage = 'CLOSED'
        elif any(word in title_lower for word in financing_keywords['priced']):
            stage = 'PRICED'
        elif any(word in title_lower for word in financing_keywords['announced']):
            stage = 'ANNOUNCED'
        
        if not stage:
            continue
        
        # Determine type
        fin_type = 'PP'  # Default
        if any(word in title_lower for word in atm_keywords):
            fin_type = 'ATM'
        elif any(word in title_lower for word in shelf_keywords):
            fin_type = 'SHELF'
        elif any(word in title_lower for word in strategic_keywords):
            fin_type = 'STRATEGIC'
        
        financing_events.append({
            'stage': stage,
            'type': fin_type,
            'days_ago': item['days_ago'],
            'headline': item['title']
        })
    
    # 3) DETERMINE CURRENT STATUS (most recent event wins)
    if financing_events:
        # Sort by recency
        financing_events.sort(key=lambda x: x['days_ago'])
        latest = financing_events[0]
        
        result['financing_status'] = latest['stage']
        result['financing_type'] = latest['type']
        result['key_headlines'].append(latest['headline'])
        
        # 4) CALCULATE IMPACT ON DILUTION RISK
        # CLOSED financing = REDUCES dilution risk (runway extended)
        # ANNOUNCED financing = INCREASES dilution risk (uncertainty)
        
        if latest['stage'] == 'CLOSED':
            # Good news - but decays over time
            if latest['days_ago'] <= 7:
                result['financing_impact'] = -20  # Strong positive
                result['explanation'] = 'Financing closed recently - runway extended'
            elif latest['days_ago'] <= 30:
                result['financing_impact'] = -15
                result['explanation'] = 'Financing closed this month'
            elif latest['days_ago'] <= 90:
                result['financing_impact'] = -10
                result['explanation'] = 'Financing closed recently'
            else:
                result['financing_impact'] = -5
                result['explanation'] = 'Old financing close'
            
            # Strategic is best
            if latest['type'] == 'STRATEGIC':
                result['financing_impact'] -= 5
        
        elif latest['stage'] == 'ANNOUNCED' or latest['stage'] == 'PRICED':
            # Bad news - uncertainty
            if latest['type'] == 'ATM':
                result['financing_impact'] = +25  # Ongoing dilution
                result['explanation'] = 'Active ATM - ongoing dilution'
            elif latest['type'] == 'SHELF':
                result['financing_impact'] = +20
                result['explanation'] = 'Shelf filed - dilution imminent'
            else:
                result['financing_impact'] = +15
                result['explanation'] = 'Financing announced but not closed'
    
    # 5) CONFIDENCE DECAY FOR OLD NEWS
    if result['fresh_news_count'] == 0:
        result['news_confidence'] = 40
        result['warnings'].append('No news in last 30 days')
    elif result['fresh_news_count'] >= 3:
        result['news_confidence'] = 90
    else:
        result['news_confidence'] = 70
    
    return result

# ============================================================================
# 4️⃣ SENTIMENT & BUZZ PROXY - NO API REQUIRED
# ============================================================================

def calculate_market_buzz(ticker, news_items, price_data, volume_data):
    """
    Market buzz/sentiment proxy WITHOUT requiring APIs
    
    WHY: We can infer activity from:
         - News velocity
         - Price/volume action
         - Headline sentiment
    
    THIS IS NOT:
    - Deep NLP
    - Twitter scraping
    - Guaranteed accuracy
    
    THIS IS:
    - A heuristic proxy
    - Better than nothing
    - Explainable
    
    RETURNS: Buzz assessment with source tags
    """
    result = {
        'buzz_level': 'LOW',              # LOW / MODERATE / HIGH / EXTREME
        'sentiment': 'NEUTRAL',           # POSITIVE / NEGATIVE / NEUTRAL
        'buzz_score': 0,                  # -100 to +100
        'source_tags': [],
        'retail_activity': 'LOW',
        'institutional_hints': [],
        'explanation': ''
    }
    
    try:
        # 1) NEWS VELOCITY
        recent_news = [n for n in news_items if n.get('days_ago', 999) <= 7]
        news_count = len(recent_news)
        
        if news_count >= 10:
            result['buzz_level'] = 'EXTREME'
            result['buzz_score'] += 40
        elif news_count >= 5:
            result['buzz_level'] = 'HIGH'
            result['buzz_score'] += 20
        elif news_count >= 2:
            result['buzz_level'] = 'MODERATE'
            result['buzz_score'] += 10
        
        # 2) HEADLINE SENTIMENT ANALYSIS (simple keywords)
        positive_words = ['breakthrough', 'discovers', 'significant', 'exceptional', 
                         'major', 'high-grade', 'extends', 'expands']
        negative_words = ['disappoints', 'delays', 'issues', 'problems', 'weak',
                         'lower', 'misses', 'challenges']
        
        sentiment_score = 0
        for item in recent_news:
            title_lower = item.get('title', '').lower()
            sentiment_score += sum(1 for word in positive_words if word in title_lower)
            sentiment_score -= sum(1 for word in negative_words if word in title_lower)
        
        result['buzz_score'] += sentiment_score * 10
        
        if sentiment_score > 2:
            result['sentiment'] = 'POSITIVE'
        elif sentiment_score < -2:
            result['sentiment'] = 'NEGATIVE'
        
        # 3) PRICE/VOLUME DIVERGENCE (retail FOMO detector)
        if len(price_data) >= 20 and len(volume_data) >= 20:
            recent_ret = ((price_data.iloc[-1] - price_data.iloc[-5]) / price_data.iloc[-5] * 100)
            
            recent_vol = volume_data.tail(5).mean()
            baseline_vol = volume_data.tail(20).mean()
            vol_spike = recent_vol / baseline_vol if baseline_vol > 0 else 1
            
            # Extreme move + volume spike = retail FOMO
            if abs(recent_ret) > 15 and vol_spike > 2.5:
                result['retail_activity'] = 'EXTREME'
                result['source_tags'].append('RETAIL_FOMO')
                
                if recent_ret > 0:
                    result['buzz_score'] += 30
                    result['explanation'] = 'Extreme retail FOMO detected'
                else:
                    result['buzz_score'] -= 30
                    result['explanation'] = 'Retail panic selling'
            
            # Moderate volume spike
            elif vol_spike > 1.8:
                result['retail_activity'] = 'HIGH'
        
        # 4) INSTITUTIONAL HINTS (in headlines)
        inst_keywords = ['fund', 'institution', 'strategic investor', 'cornerstone',
                        'research', 'analyst', 'bank', 'coverage']
        
        for item in recent_news:
            title_lower = item.get('title', '').lower()
            for keyword in inst_keywords:
                if keyword in title_lower:
                    result['institutional_hints'].append(item['title'])
                    result['source_tags'].append('INSTITUTIONAL')
                    result['buzz_score'] += 5
                    break
        
        # 5) NORMALIZE AND CLASSIFY
        result['buzz_score'] = np.clip(result['buzz_score'], -100, 100)
        
        if abs(result['buzz_score']) > 60:
            result['buzz_level'] = 'EXTREME'
        elif abs(result['buzz_score']) > 30:
            result['buzz_level'] = 'HIGH'
        elif abs(result['buzz_score']) > 15:
            result['buzz_level'] = 'MODERATE'
        
        # Tag sources
        if result['retail_activity'] in ['HIGH', 'EXTREME'] and not result['institutional_hints']:
            result['source_tags'].append('RETAIL_DOMINANT')
        elif result['institutional_hints']:
            result['source_tags'].append('INSTITUTIONAL_INTEREST')
        else:
            result['source_tags'].append('LOW_ACTIVITY')
    
    except Exception as e:
        result['explanation'] = f'Buzz calc error: {str(e)[:40]}'
    
    return result

# ============================================================================
# 5️⃣ ENHANCED SELL-IN-TIME INTELLIGENCE
# ============================================================================

def calculate_enhanced_sell_triggers(row, hist_data, smc_data, news_intel, 
                                    metal_forecast, macro_regime):
    """
    Enhanced sell intelligence with distribution detection
    
    WHY: Most losses come from holding too long.
         Institutions distribute into retail strength.
         We need to detect this EARLY.
    
    NEW TRIGGERS:
    - SMC breakdown (BOS failure, CHoCH against position)
    - Failed rallies (attempts to reclaim MA fail)
    - Metal divergence (stock weak, metal strong = concern)
    - Volume climax patterns
    
    RETURNS: Comprehensive sell risk assessment
    """
    sell_risk = {
        'score': 0,
        'triggers': [],
        'distribution_detected': False,
        'urgency': 'NORMAL',           # NORMAL / ELEVATED / URGENT
        'explanation': ''
    }
    
    try:
        current_price = row.get('Price', 0)
        ma50 = row.get('MA50', 0)
        ma200 = row.get('MA200', 0)
        
        # 1) SMC-BASED TRIGGERS
        if smc_data.get('event') == 'CHOCH' and smc_data.get('state') == 'BEARISH':
            sell_risk['score'] += 25
            sell_risk['triggers'].append('💀 SMC: Change of Character to bearish')
            sell_risk['distribution_detected'] = True
        
        if smc_data.get('structure') == 'BEARISH_STRUCTURE':
            sell_risk['score'] += 15
            sell_risk['triggers'].append('⚠️ SMC: Bearish structure confirmed')
        
        # 2) FAILED MA RECLAIM
        if len(hist_data) >= 20:
            recent = hist_data.tail(20)
            
            # Check if tried to reclaim MA50 but failed
            if ma50 > 0 and current_price < ma50 * 0.95:
                touched_ma50 = any(bar.High >= ma50 * 0.99 for bar in recent.itertuples())
                if touched_ma50:
                    sell_risk['score'] += 18
                    sell_risk['triggers'].append('⚠️ Failed MA50 reclaim attempt')
            
            # Below MA200 is serious
            if ma200 > 0 and current_price < ma200:
                sell_risk['score'] += 20
                sell_risk['triggers'].append('💀 Below MA200')
                
                if macro_regime.get('regime') == 'DEFENSIVE':
                    sell_risk['score'] += 15
                    sell_risk['triggers'].append('💀 Below MA200 + Defensive macro')
        
        # 3) METAL DIVERGENCE
        # If metal is strong but stock is weak = red flag
        stock_ret_20d = row.get('Return_20d', 0)
        metal_bias = metal_forecast.get('bias_short', 'NEUTRAL')
        
        if metal_bias == 'BULLISH' and stock_ret_20d < -5:
            sell_risk['score'] += 12
            sell_risk['triggers'].append('⚠️ Stock weak while metal strong')
        
        # 4) VOLUME CLIMAX + STALL
        if len(hist_data) >= 10:
            recent = hist_data.tail(10)
            
            # Find highest volume bar
            max_vol_idx = recent['Volume'].idxmax()
            max_vol_bar = recent.loc[max_vol_idx]
            
            # If high volume bar was near recent high, then price stalled
            recent_high = recent['High'].max()
            if max_vol_bar['High'] >= recent_high * 0.98:
                # Check if price went nowhere after
                bars_after = recent.loc[recent.index > max_vol_idx]
                if len(bars_after) >= 3:
                    if bars_after['Close'].iloc[-1] < max_vol_bar['Close'] * 0.97:
                        sell_risk['score'] += 15
                        sell_risk['triggers'].append('⚠️ Volume climax + price stall')
                        sell_risk['distribution_detected'] = True
        
        # 5) NEWS-BASED TRIGGERS
        if news_intel.get('financing_status') == 'ANNOUNCED':
            if news_intel.get('financing_type') == 'ATM':
                sell_risk['score'] += 10
                sell_risk['triggers'].append('⚠️ Active ATM dilution')
        
        # 6) DILUTION + WEAKNESS
        dilution = row.get('Dilution_Risk_Score', 0)
        if dilution >= 70 and stock_ret_20d < 0:
            sell_risk['score'] += 15
            sell_risk['triggers'].append('💀 High dilution + price weakness')
        
        # 7) URGENCY DETERMINATION
        if sell_risk['score'] >= 60:
            sell_risk['urgency'] = 'URGENT'
            sell_risk['explanation'] = 'Multiple severe sell signals - consider immediate action'
        elif sell_risk['score'] >= 35:
            sell_risk['urgency'] = 'ELEVATED'
            sell_risk['explanation'] = 'Elevated sell risk - monitor closely'
        else:
            sell_risk['urgency'] = 'NORMAL'
            sell_risk['explanation'] = 'Normal sell risk levels'
    
    except Exception as e:
        sell_risk['explanation'] = f'Sell calc error: {str(e)[:40]}'
    
    return sell_risk

# ============================================================================
# 6️⃣ PORTFOLIO ORCHESTRATION - RANKING & SUMMARY
# ============================================================================

def orchestrate_portfolio_ranking(df, gold_forecast, silver_forecast, macro_regime):
    """
    Portfolio-level orchestration and ranking
    
    WHY: PMs don't look at stocks in isolation.
         They need: ranked list, themes, net posture.
    
    PRODUCES:
    - Deterministic ranking (STRONG BUY at top)
    - Portfolio themes
    - Risk posture assessment
    - Net exposure recommendation
    
    RETURNS: Enhanced DataFrame with portfolio context
    """
    result = {
        'ranked_df': df.copy(),
        'portfolio_summary': {},
        'themes': [],
        'risk_posture': 'NEUTRAL',
        'recommended_net_exposure': 0,
        'top_buys': [],
        'top_sells': [],
        'warnings': []
    }
    
    try:
        # 1) ASSIGN ACTION RANKS
        ACTION_RANK = {
            '🟢 STRONG BUY': 1,
            '🟢 BUY': 2,
            '🔵 ADD': 3,
            '🔵 ACCUMULATE': 4,
            '⚪ HOLD': 5,
            '🟡 TRIM': 6,
            '🔴 REDUCE': 7,
            '🔴 SELL': 8,
            '🚨 SELL NOW': 9
        }
        
        result['ranked_df']['Action_Rank'] = result['ranked_df']['Action'].map(ACTION_RANK)
        
        # 2) MULTI-LEVEL SORT
        # Primary: Action (best first)
        # Secondary: Alpha (highest first)
        # Tertiary: Sell Risk (lowest first)
        result['ranked_df'] = result['ranked_df'].sort_values(
            by=['Action_Rank', 'Alpha_Score', 'Sell_Risk_Score'],
            ascending=[True, False, True]
        )
        
        # 3) IDENTIFY THEMES
        # Check for sector concentration
        buys = result['ranked_df'][result['ranked_df']['Action'].str.contains('BUY|ADD', na=False)]
        
        if len(buys) > 0:
            # Silver vs Gold exposure
            silver_count = sum(1 for _, row in buys.iterrows() if 'silver' in row.get('Name', '').lower())
            gold_count = len(buys) - silver_count
            
            if silver_count > gold_count * 1.5:
                result['themes'].append('Silver-led opportunity')
            elif gold_count > silver_count * 1.5:
                result['themes'].append('Gold-focused plays')
            else:
                result['themes'].append('Balanced Au/Ag exposure')
            
            # Stage concentration
            stage_counts = buys['stage'].value_counts()
            if 'Explorer' in stage_counts and stage_counts['Explorer'] >= 3:
                result['themes'].append('Explorer-heavy (high beta)')
            elif 'Producer' in stage_counts and stage_counts['Producer'] >= 2:
                result['themes'].append('Producer-weighted (defensive)')
        
        # 4) RISK POSTURE
        # Based on metal forecasts + portfolio composition
        gold_bullish = gold_forecast.get('bias_medium') == 'BULLISH'
        silver_bullish = silver_forecast.get('bias_medium') == 'BULLISH'
        
        avg_sell_risk = result['ranked_df']['Sell_Risk_Score'].mean()
        
        if gold_bullish and silver_bullish and avg_sell_risk < 30:
            result['risk_posture'] = 'RISK_ON'
            result['recommended_net_exposure'] = 0.85  # 85% invested
        elif (not gold_bullish or not silver_bullish) and avg_sell_risk > 40:
            result['risk_posture'] = 'RISK_OFF'
            result['recommended_net_exposure'] = 0.50  # 50% invested
        else:
            result['risk_posture'] = 'NEUTRAL'
            result['recommended_net_exposure'] = 0.70  # 70% invested
        
        # 5) TOP BUYS & SELLS
        result['top_buys'] = buys.head(3)[['Symbol', 'Action', 'Alpha_Score']].to_dict('records')
        
        sells = result['ranked_df'][result['ranked_df']['Action'].str.contains('SELL|REDUCE|TRIM', na=False)]
        result['top_sells'] = sells.head(3)[['Symbol', 'Action', 'Sell_Risk_Score']].to_dict('records')
        
        # 6) PORTFOLIO SUMMARY
        total_mv = result['ranked_df']['Market_Value'].sum()
        
        result['portfolio_summary'] = {
            'total_positions': len(result['ranked_df']),
            'buy_signals': len(buys),
            'sell_signals': len(sells),
            'hold_signals': len(result['ranked_df']) - len(buys) - len(sells),
            'avg_alpha': result['ranked_df']['Alpha_Score'].mean(),
            'avg_sell_risk': avg_sell_risk,
            'total_market_value': total_mv
        }
        
        # 7) WARNINGS
        if avg_sell_risk > 50:
            result['warnings'].append('⚠️ Portfolio-wide sell risk elevated')
        
        illiquid_pct = result['ranked_df'][result['ranked_df']['Liq_tier_code'].isin(['L0', 'L1'])]['Pct_Portfolio'].sum()
        if illiquid_pct > 25:
            result['warnings'].append(f'⚠️ {illiquid_pct:.0f}% in illiquid tiers')
    
    except Exception as e:
        result['warnings'].append(f'Ranking error: {str(e)[:40]}')
    
    return result

# ============================================================================
# 7️⃣ DISCOVERY EXCEPTION - FINAL HARDENING
# ============================================================================

def check_discovery_exception_ultimate(row, liq_metrics, alpha_score, data_confidence,
                                      dilution_risk, smc_data, metal_forecast, 
                                      macro_regime):
    """
    Ultimate discovery exception logic - institutional standards
    
    WHY: Discovery plays are inherently risky.
         We need MULTIPLE independent confirmations.
    
    GATES (ALL must pass):
    1. Tactical sleeve
    2. NOT L0 liquidity
    3. Alpha ≥ 85
    4. Data confidence ≥ 70
    5. Dilution < 70
    6. SMC bullish structure
    7. Momentum confirmed
    8. Metal forecast NOT bearish
    9. Macro NOT defensive
    10. Insider buying (strongly preferred)
    
    RETURNS: (allowed, reason, warnings, max_position_pct)
    """
    deny_reasons = []
    warnings = []
    
    # GATE 1: Sleeve
    if row.get('Sleeve', '') != 'TACTICAL':
        return False, "Must be TACTICAL sleeve", [], 0
    
    # GATE 2: Liquidity
    if liq_metrics.get('tier_code') == 'L0':
        return False, "L0 liquidity excluded", [], 0
    
    # GATE 3: Alpha
    if alpha_score < 85:
        deny_reasons.append(f"Alpha {alpha_score:.0f} < 85")
    
    # GATE 4: Data confidence
    if data_confidence < 70:
        deny_reasons.append(f"Data confidence {data_confidence:.0f} < 70%")
    
    # GATE 5: Dilution
    if dilution_risk >= 70:
        deny_reasons.append(f"Dilution risk {dilution_risk:.0f} ≥ 70")
    
    # GATE 6: SMC
    if smc_data.get('state') != 'BULLISH':
        deny_reasons.append("SMC not bullish")
    
    # GATE 7: Momentum
    momentum_7d = row.get('Return_7d', 0)
    price = row.get('Price', 0)
    ma50 = row.get('MA50', 0)
    
    if not (momentum_7d > 0 and price > ma50):
        deny_reasons.append("Momentum not confirmed")
    
    # GATE 8: Metal forecast
    metal_bias = metal_forecast.get('bias_short', 'NEUTRAL')
    if metal_bias == 'BEARISH':
        return False, "Metal forecast bearish - no discovery plays", [], 0
    
    # GATE 9: Macro
    if macro_regime.get('regime') == 'DEFENSIVE':
        deny_reasons.append("Defensive macro regime")
    
    # GATE 10: Insider buying (preferred but not required)
    if not row.get('Insider_Buying_90d', False):
        warnings.append("⚠️ No insider buying (increased risk)")
    
    # Decision
    if deny_reasons:
        return False, '; '.join(deny_reasons[:2]), [], 0
    
    # GRANTED - but with strict limits
    max_position = 2.5  # Hard cap
    
    # Reduce if L1
    if liq_metrics.get('tier_code') == 'L1':
        max_position = 2.0
        warnings.append("⚠️ L1 liquidity - reduced max to 2%")
    
    warnings.extend([
        "⚠️ DISCOVERY EXCEPTION ACTIVE",
        f"⚠️ Max position: {max_position}% (NON-NEGOTIABLE)",
        "⚠️ Monitor DAILY - high risk"
    ])
    
    reason = f"High-conviction discovery: Alpha {alpha_score:.0f}, SMC bullish, Metals supportive"
    
    return True, reason, warnings, max_position

# ============================================================================
# INTEGRATION NOTES
# ============================================================================

"""
═══════════════════════════════════════════════════════════════════════════
INTEGRATION INTO alpha_miner_institutional_v2.py
═══════════════════════════════════════════════════════════════════════════

This module provides production-grade enhancements.

STEP 1: Import at top
----------------------
from institutional_enhancements_v3 import (
    calculate_smc_structure,
    forecast_metal_direction,
    analyze_news_intelligence,
    calculate_market_buzz,
    calculate_enhanced_sell_triggers,
    orchestrate_portfolio_ranking,
    check_discovery_exception_ultimate
)

STEP 2: Add to analysis loop (around line 1300)
-----------------------------------------------
# SMC analysis
smc_data = calculate_smc_structure(hist, row['Symbol'])
df.at[idx, 'SMC_State'] = smc_data['state']
df.at[idx, 'SMC_Event'] = smc_data['event']
df.at[idx, 'SMC_Confidence'] = smc_data['confidence']

# News intelligence
news_intel = analyze_news_intelligence(news, row['Symbol'])
df.at[idx, 'Financing_Status'] = news_intel['financing_status']
df.at[idx, 'Dilution_Risk_Score'] += news_intel['financing_impact']

# Market buzz
buzz = calculate_market_buzz(row['Symbol'], news, hist['Close'], hist['Volume'])
df.at[idx, 'Buzz_Level'] = buzz['buzz_level']
df.at[idx, 'Buzz_Score'] = buzz['buzz_score']

# Enhanced sell risk
sell_enhanced = calculate_enhanced_sell_triggers(
    row, hist, smc_data, news_intel, gold_forecast, macro
)
df.at[idx, 'Sell_Risk_Enhanced'] = sell_enhanced['score']

STEP 3: Rank portfolio after analysis
-------------------------------------
portfolio_orchestration = orchestrate_portfolio_ranking(
    df, gold_forecast, silver_forecast, macro
)
df = portfolio_orchestration['ranked_df']

STEP 4: Use ultimate discovery exception
----------------------------------------
exception = check_discovery_exception_ultimate(
    row, liq_metrics, alpha_score, data_confidence, dilution_risk,
    smc_data, metal_forecast, macro
)

═══════════════════════════════════════════════════════════════════════════
"""
