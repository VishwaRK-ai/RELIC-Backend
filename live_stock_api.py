import yfinance as yf

def get_live_stock_data(ticker_symbol):
    try:
        # .NS is the suffix for National Stock Exchange of India (NSE)
        if not ticker_symbol.endswith(".NS"):
            ticker_symbol += ".NS"
            
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        price = info.get("currentPrice", "Unknown")
        beta = info.get("beta", "Unknown")
        high_52 = info.get("fiftyTwoWeekHigh", "Unknown")
        low_52 = info.get("fiftyTwoWeekLow", "Unknown")
        
        # Determine Risk Profile based on Beta
        # Beta > 1 = High Volatility/Risk, Beta < 1 = Lower Volatility
        if beta != "Unknown":
            risk_level = "High Risk (More volatile than the market)" if float(beta) > 1.2 else \
                         "Moderate Risk (Moves with the market)" if float(beta) > 0.8 else \
                         "Low Risk (Less volatile than the market)"
        else:
            risk_level = "Risk unknown"

        # Format this as a clean text string for the LLM to read
        live_context = (
            f"\n[LIVE SYSTEM DATA for {ticker_symbol}]: "
            f"Current Price: ₹{price}. 52-Week High: ₹{high_52}. 52-Week Low: ₹{low_52}. "
            f"Risk Profile (Beta {beta}): {risk_level}.\n"
        )
        return live_context
        
    except Exception as e:
        return "" # Fail silently if the ticker isn't found