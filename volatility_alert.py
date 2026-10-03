import yfinance as yf
import time

def check_volatility(ticker_symbol="RELIANCE.NS", drop_threshold=-2.0, demo_mode=False):#change demo move to 
    print(f"\n📡 [STREAMING] Monitoring {ticker_symbol} for sudden drops...")
    
    try:
        if demo_mode:
            # Hackathon Demo Mode: Simulates a sudden market crash
            print("   [INFO] Demo Mode Active - Simulating live market hours.")
            prev_close = 2800.00
            current_price = 2715.00 # A simulated drop of ~3%
        else:
            # Production Mode: Fetches actual live data
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d")
            
            if len(hist) < 2:
                print("   [ERROR] Not enough market data available.")
                return
                
            prev_close = hist['Close'].iloc[-2]
            current_price = hist['Close'].iloc[-1]
            
        # Calculate percentage change
        pct_change = ((current_price - prev_close) / prev_close) * 100
        
        print(f"📊 Live Status: Current: ₹{current_price:.2f} | Prev Close: ₹{prev_close:.2f} | Change: {pct_change:.2f}%")
        
        # Trigger the Behavioral Intervention if the drop exceeds the threshold
        if pct_change <= drop_threshold:
            print("\n🚨 [VOLATILITY ALERT TRIGGERED] 🚨")
            print(f"⚠️ {ticker_symbol} dropped {abs(pct_change):.2f}% intraday!")
            print("\n💡 EDUCATIONAL INTERVENTION (Sent to User's Phone):")
            print("   \"Market fluctuations are normal; do not panic sell based on short-term noise.")
            print("   Remember SEBI Rule #1: Invest based on fundamentals, not emotions.")
            print("   Read this quick NCFE guide on 'Rupee Cost Averaging' to understand how dips can be opportunities.\"")
        else:
            print("✅ Market is stable. No panic detected. Continuing background monitoring...")
            
    except Exception as e:
        print(f"❌ [SYSTEM ERROR] Failed to fetch stream: {e}")

# Run the test
# Change demo_mode=False during actual weekday market hours!
if __name__ == "__main__":
    check_volatility(ticker_symbol="RELIANCE.NS", drop_threshold=-2.0, demo_mode=True)#change demo mode to false