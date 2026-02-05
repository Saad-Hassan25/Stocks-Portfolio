import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portfolio.db')

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Transactions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            type TEXT NOT NULL, -- 'BUY' or 'SELL'
            fees REAL DEFAULT 0
        )
    ''')
    
    # Dividends table
    c.execute('''
        CREATE TABLE IF NOT EXISTS dividends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            amount_per_share REAL NOT NULL,
            total_shares REAL NOT NULL,
            total_amount REAL NOT NULL,
            dividend_type TEXT DEFAULT 'CASH' -- 'CASH' or 'STOCK'
        )
    ''')
    
    # Realized P&L table (tracks closed positions)
    c.execute('''
        CREATE TABLE IF NOT EXISTS realized_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            sell_date TEXT NOT NULL,
            quantity REAL NOT NULL,
            buy_avg_price REAL NOT NULL,
            sell_price REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            fees REAL DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def add_transaction(symbol, date, quantity, price, type='BUY', fees=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO transactions (symbol, date, quantity, price, type, fees)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (symbol, date, quantity, price, type, fees))
    conn.commit()
    conn.close()

def get_transactions():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC", conn)
    conn.close()
    return df

def delete_transaction(transaction_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM transactions WHERE id=?", (transaction_id,))
    conn.commit()
    conn.close()

# ================= DIVIDEND FUNCTIONS =================
def add_dividend(symbol, date, amount_per_share, total_shares, dividend_type='CASH'):
    """Record a dividend payment."""
    total_amount = amount_per_share * total_shares
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO dividends (symbol, date, amount_per_share, total_shares, total_amount, dividend_type)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (symbol, date, amount_per_share, total_shares, total_amount, dividend_type))
    conn.commit()
    conn.close()

def get_dividends():
    """Get all dividend records."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM dividends ORDER BY date DESC", conn)
    conn.close()
    return df

def get_dividends_by_symbol(symbol):
    """Get dividends for a specific stock."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM dividends WHERE symbol=? ORDER BY date DESC", conn, params=(symbol,))
    conn.close()
    return df

def delete_dividend(dividend_id):
    """Delete a dividend record."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM dividends WHERE id=?", (dividend_id,))
    conn.commit()
    conn.close()

def get_total_dividends():
    """Get total dividends received."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT SUM(total_amount) FROM dividends")
    result = c.fetchone()[0]
    conn.close()
    return result if result else 0

# ================= REALIZED P&L FUNCTIONS =================
def add_realized_pnl(symbol, sell_date, quantity, buy_avg_price, sell_price, fees=0):
    """Record a realized P&L entry when a position is closed."""
    realized_pnl = (sell_price - buy_avg_price) * quantity - fees
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO realized_pnl (symbol, sell_date, quantity, buy_avg_price, sell_price, realized_pnl, fees)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, sell_date, quantity, buy_avg_price, sell_price, realized_pnl, fees))
    conn.commit()
    conn.close()

def get_realized_pnl():
    """Get all realized P&L records."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM realized_pnl ORDER BY sell_date DESC", conn)
    conn.close()
    return df

def get_total_realized_pnl():
    """Get total realized P&L."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT SUM(realized_pnl) FROM realized_pnl")
    result = c.fetchone()[0]
    conn.close()
    return result if result else 0

def clear_all_portfolio_data():
    """
    Clear all portfolio data including transactions, dividends, and realized P&L.
    WARNING: This is irreversible!
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM transactions")
    c.execute("DELETE FROM dividends")
    c.execute("DELETE FROM realized_pnl")
    conn.commit()
    conn.close()
    return True

def get_portfolio_stats():
    """Get counts of all portfolio data for confirmation display."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM transactions")
    trans_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM dividends")
    div_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM realized_pnl")
    realized_count = c.fetchone()[0]
    
    conn.close()
    return {
        'transactions': trans_count,
        'dividends': div_count,
        'realized_pnl': realized_count,
        'total': trans_count + div_count + realized_count
    }

# Initialize DB on import
init_db()
