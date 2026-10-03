import yfinance as yf
import pandas as pd
import re

def calculate_rsi(data, periods=14):
    # Standard financial math to calculate 14-day RSI
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def evaluate_fomo(user_prompt, ticker_symbol, demo_mode=False):
    print(f"\n🧠 [BEHAVIORAL ENGINE] Analyzing intent for {ticker_symbol}...")
    
    score = 0
    reasons = []
    prompt_lower = user_prompt.lower()

    # 1. Linguistic Urgency (+30 points)
    urgency_words = [r"rocket", r"multibagger", r"target in", r"instant", r"urgent", r"guaranteed", r"sure shot"]
    found_urgency = [word for word in urgency_words if re.search(word, prompt_lower)]
    if found_urgency:
        score += 30
        reasons.append(f"Linguistic Urgency (Detected hype words: {found_urgency[0]})")

    # 2. Diversification Penalty (+20 points)
    concentration_words = [r"all my savings", r"all in", r"entire capital", r"100% of my", r"put everything"]
    found_concentration = [word for word in concentration_words if re.search(word, prompt_lower)]
    if found_concentration:
        score += 20
        reasons.append("Concentration Risk (Attempting to allocate all capital into a single equity)")

    # 3. Technical Stretch (+30 points)
    print("📈 [SYSTEM] Fetching technical indicators (RSI & 52-Week High)...")
    if demo_mode:
        current_price = 2800.00
        fifty_two_wk_high = 2810.00
        current_rsi = 82.5 # Extremely overbought
    else:
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="3mo") # Need enough data for RSI
            current_price = hist['Close'].iloc[-1]
            fifty_two_wk_high = ticker.info.get('fiftyTwoWeekHigh', current_price)
            
            # Calculate RSI
            hist['RSI'] = calculate_rsi(hist['Close'])
            current_rsi = hist['RSI'].iloc[-1]
        except Exception as e:
            print(f"❌ [ERROR] Could not fetch technicals: {e}")
            current_price = 0
            fifty_two_wk_high = 0
            current_rsi = 50

    # Math: Is price within 5% of 52-week high AND is RSI over 75?
    if current_price >= (fifty_two_wk_high * 0.95) and current_rsi > 75:
        score += 30
        reasons.append(f"Technical Stretch (Price near 52-wk high & RSI is {current_rsi:.1f} - Overbought)")

    # --- RENDER THE UI GAUGE IN TERMINAL ---
    print("\n==================================================")
    print("      📊 IMPULSE & FOMO GAUGE RESULT")
    print("==================================================")
    
    score = min(score, 100) # Cap at 100%
    
    # Determine Color Zone
    if score <= 35:
        zone = "🟢 GREEN (0-35%): Planned / Systematic"
        advice = "Looks like a calculated move. Proceed with your standard research."
    elif score <= 70:
        zone = "🟡 YELLOW (36-70%): Momentum Chasing"
        advice = "You might be chasing a trend. Review your time horizon before executing."
    else:
        zone = "🔴 RED (71-100%): HIGH FOMO WARNING"
        advice = "SEBI warns against speculative 'hot tips'. This trade exhibits severe impulsive characteristics."

    # Draw a visual ASCII gauge
    bar_filled = int(score / 5)
    bar_empty = 20 - bar_filled
    print(f"Gauge: [{'█' * bar_filled}{'-' * bar_empty}] {score}%")
    print(f"Zone : {zone}")
    
    if reasons:
        print("\n🚩 TRIPPED WIRES:")
        for r in reasons:
            print(f"   - {r}")
            
    print(f"\n💡 SYSTEM ADVICE: {advice}")
    print("==================================================\n")

# --- RUN THE TEST ---
if __name__ == "__main__":
    # A test prompt designed to trip all the alarms
    test_prompt = "Should I put all my savings into Reliance right now? I heard it's going to be a multibagger rocket!"
    
    # Using demo_mode=True to force the RSI and 52-wk high to trigger the technical stretch
    evaluate_fomo(test_prompt, "RELIANCE.NS", demo_mode=True)