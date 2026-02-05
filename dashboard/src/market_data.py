import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

# Suppress yfinance error logs (they're noisy for PSX stocks on weekends/holidays)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

def get_ticker_symbol(symbol):
    """
    Helper to format symbol for Yahoo Finance (PSX stocks usually need .KA suffix)
    Examples: ENGRO -> ENGRO.KA, TRG -> TRG.KA
    """
    symbol = symbol.upper().strip()
    if not symbol.endswith('.KA'):
        return f"{symbol}.KA"
    return symbol

def get_historical_price(symbol, date):
    """
    Get the closing price of a stock on a specific date.
    Returns: Price (float) or None if not found
    """
    ticker = get_ticker_symbol(symbol)
    
    try:
        start_date = pd.to_datetime(date)
        end_date = start_date + timedelta(days=1)
        
        # Suppress progress bar and use single thread
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            progress=False,
            threads=False
        )
        
        if df is not None and not df.empty:
            # Handle MultiIndex columns (recent yfinance change)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    price = df['Close'][ticker].iloc[0]
                except KeyError:
                    price = df['Close'].iloc[0]
            else:
                price = df['Close'].iloc[0]
                 
            return float(price)
        else:
            return None
    except Exception:
        return None

def get_current_price(symbol):
    """
    Get the latest realtime/delayed price.
    Uses history(period="5d") to ensure we get the latest valid close even if 1d fails.
    Avoids fast_info due to reported inaccuracies for .KA tickers.
    Silently returns 0.0 if no data found (errors suppressed).
    """
    ticker_sym = get_ticker_symbol(symbol)
    try:
        ticker = yf.Ticker(ticker_sym)
        
        # Fetch last 5 days to ensure we get data even after weekends/holidays
        # Use raise_errors=False to suppress error messages
        df = ticker.history(period="5d", raise_errors=False) 
        
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
            
        # Fallback to download for last 7 days if ticker.history fails
        try:
            end_date = datetime.today() + timedelta(days=1)
            start_date = end_date - timedelta(days=7)
            # progress=False suppresses the progress bar
            df_down = yf.download(
                ticker_sym, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True,
                threads=False  # Single-threaded to reduce error spam
            )
            if df_down is not None and not df_down.empty:
                # Handle MultiIndex if present
                if isinstance(df_down.columns, pd.MultiIndex):
                    try:
                        return float(df_down['Close'][ticker_sym].iloc[-1])
                    except:
                        return float(df_down['Close'].iloc[-1])
                return float(df_down['Close'].iloc[-1])
        except:
            pass

        return 0.0
    except Exception:
        return 0.0

def get_stock_history_bulk(symbols, period="1y"):
    """
    Fetch history for multiple stocks for charting.
    """
    tickers = [get_ticker_symbol(s) for s in symbols]
    if not tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(tickers, period=period, group_by='ticker', progress=False, threads=False)
        return data
    except Exception:
        return pd.DataFrame()

def get_stock_chart_data(symbol, period="6mo"):
    """
    Fetch OHLCV data for a single stock for charting.
    Returns a clean DataFrame with Date, Open, High, Low, Close, Volume.
    
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    """
    ticker_sym = get_ticker_symbol(symbol)
    
    try:
        # Use download for more reliable data (suppress errors)
        df = yf.download(ticker_sym, period=period, progress=False, threads=False)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten: take just the price type level
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure we have the columns we need
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Date' and 'index' in df.columns:
                    df['Date'] = df['index']
                else:
                    df[col] = 0
        
        return df[required_cols]
        
    except Exception:
        return pd.DataFrame()

# ================= TECHNICAL INDICATORS =================
def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns: macd_line, signal_line, histogram
    """
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    Returns: middle_band (SMA), upper_band, lower_band
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return middle_band, upper_band, lower_band

def add_technical_indicators(df):
    """
    Add all technical indicators to a chart DataFrame.
    Expects df to have 'Close' column.
    Returns df with added indicator columns.
    """
    if df.empty or 'Close' not in df.columns:
        return df
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    return df
