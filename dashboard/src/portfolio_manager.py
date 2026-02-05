import pandas as pd
from datetime import datetime, timedelta
from database import get_transactions, get_dividends, get_realized_pnl, add_realized_pnl
from market_data import get_current_price, get_historical_price
from data_loader import load_data_from_file

class PortfolioManager:
    def __init__(self):
        self.scraped_prices = self._load_scraped_prices()

    def _load_scraped_prices(self):
        """Load prices from the local scraped files for KSE100 and KMI30."""
        prices = {}
        try:
            kse_df, kmi_df = load_data_from_file()
            
            # Helper to populate dict
            def add_prices(df, price_col):
                if df is not None and not df.empty and price_col in df.columns:
                    for _, row in df.iterrows():
                        stock = str(row['Stock']).strip().upper()
                        # Ensure we get a valid float
                        try:
                            val = float(row[price_col])
                            if val > 0:
                                prices[stock] = val
                        except:
                            pass

            add_prices(kse_df, 'Price_KSE100')
            add_prices(kmi_df, 'Price_KMI30')
            
        except Exception as e:
            print(f"Error loading scraped prices: {e}")
        
        return prices

    def get_realtime_price(self, symbol):
        """Get price from scraped data first, then fallback to Yahoo Finance."""
        symbol_upper = symbol.upper().strip()
        
        # 1. Try Scraped Data
        if symbol_upper in self.scraped_prices:
            return self.scraped_prices[symbol_upper]
            
        # 2. Fallback to Yahoo Finance
        return get_current_price(symbol)

    def get_portfolio_summary(self):
        transactions = get_transactions()
        
        if transactions.empty:
            return pd.DataFrame(), {}

        # Group by symbol to calculate holdings
        portfolio = []
        unique_symbols = transactions['symbol'].unique()
        
        total_investment = 0
        total_current_value = 0
        
        for symbol in unique_symbols:
            # Filter transactions for this symbol
            stock_tx = transactions[transactions['symbol'] == symbol]
            
            # Simple average cost calculation
            # Logic: Weighted Average for Buys. Sells reduce quantity but keep avg cost (FIFO/Avg Cost logic simplified)
            
            quantity = 0
            total_cost = 0
            
            for _, row in stock_tx.iterrows():
                if row['type'] == 'BUY':
                    quantity += row['quantity']
                    total_cost += row['quantity'] * row['price'] + row['fees']
                elif row['type'] == 'SELL':
                    # Reduce quantity, reduce cost proportionally
                    if quantity > 0:
                        avg_cost = total_cost / quantity
                        quantity -= row['quantity']
                        total_cost -= row['quantity'] * avg_cost 
                        # Note: Realized P&L is (Sell Price - Avg Cost) * Sell Qty, but we are just tracking open positions here for the table
            
            if quantity > 0.0001: # Filter out closed positions
                avg_cost = total_cost / quantity
                
                # USE PROXY FUNCTION
                current_price = self.get_realtime_price(symbol)
                
                current_value = quantity * current_price
                
                total_investment += total_cost
                total_current_value += current_value
                
                unrealized_pnl = current_value - total_cost
                pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
                
                portfolio.append({
                    'Stock': symbol,
                    'Quantity': quantity,
                    'Avg Cost': avg_cost,
                    'Total Cost': total_cost,
                    'Current Price': current_price,
                    'Current Value': current_value,
                    'Gain/Loss': unrealized_pnl,
                    'Return %': pnl_pct,
                    'Weight %': 0 # To be calculated
                })
        
        df = pd.DataFrame(portfolio)
        
        if not df.empty:
            df['Weight %'] = (df['Current Value'] / df['Current Value'].sum() * 100)
            
        metrics = {
            'Total Investment': total_investment,
            'Current Value': total_current_value,
            'Total P&L': total_current_value - total_investment,
            'Total Return %': ((total_current_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
        }
        
        return df, metrics

    def get_holdings_pie_chart_data(self, df):
        if df.empty:
            return None
        return df[['Stock', 'Current Value']]

    # ================= DIVIDEND TRACKING =================
    def get_dividend_summary(self):
        """Get dividend summary by stock."""
        dividends = get_dividends()
        
        if dividends.empty:
            return pd.DataFrame(), 0
        
        # Group by symbol
        summary = dividends.groupby('symbol').agg({
            'total_amount': 'sum',
            'id': 'count'
        }).reset_index()
        summary.columns = ['Stock', 'Total Dividends', 'Payments']
        
        # Get current holdings for yield calculation
        df_holdings, _ = self.get_portfolio_summary()
        
        if not df_holdings.empty:
            # Merge with holdings to get total cost for yield
            summary = summary.merge(
                df_holdings[['Stock', 'Total Cost']], 
                on='Stock', 
                how='left'
            )
            summary['Total Cost'] = summary['Total Cost'].fillna(0)
            summary['Yield %'] = (summary['Total Dividends'] / summary['Total Cost'] * 100).where(summary['Total Cost'] > 0, 0)
        else:
            summary['Total Cost'] = 0
            summary['Yield %'] = 0
        
        total_dividends = dividends['total_amount'].sum()
        
        return summary, total_dividends

    # ================= REALIZED P&L =================
    def calculate_realized_pnl_for_sell(self, symbol, sell_quantity, sell_price, sell_date, fees=0):
        """
        Calculate and record realized P&L when selling.
        Uses weighted average cost basis.
        """
        transactions = get_transactions()
        stock_tx = transactions[transactions['symbol'] == symbol]
        
        # Calculate average cost
        quantity = 0
        total_cost = 0
        
        for _, row in stock_tx.iterrows():
            if row['type'] == 'BUY':
                quantity += row['quantity']
                total_cost += row['quantity'] * row['price'] + row['fees']
            elif row['type'] == 'SELL':
                if quantity > 0:
                    avg_cost = total_cost / quantity
                    quantity -= row['quantity']
                    total_cost -= row['quantity'] * avg_cost
        
        if quantity > 0:
            avg_cost = total_cost / quantity
            # Record the realized P&L
            add_realized_pnl(symbol, sell_date, sell_quantity, avg_cost, sell_price, fees)
            return (sell_price - avg_cost) * sell_quantity - fees
        
        return 0

    def get_realized_pnl_summary(self):
        """Get realized P&L summary."""
        realized = get_realized_pnl()
        
        if realized.empty:
            return pd.DataFrame(), 0, 0, 0
        
        # Summary by stock
        summary = realized.groupby('symbol').agg({
            'realized_pnl': 'sum',
            'quantity': 'sum',
            'id': 'count'
        }).reset_index()
        summary.columns = ['Stock', 'Realized P&L', 'Shares Sold', 'Trades']
        
        total_realized = realized['realized_pnl'].sum()
        total_winners = realized[realized['realized_pnl'] > 0]['realized_pnl'].sum()
        total_losers = realized[realized['realized_pnl'] < 0]['realized_pnl'].sum()
        
        return summary, total_realized, total_winners, total_losers
