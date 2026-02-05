import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portfolio.db')

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Transactions table (with user_id for multi-user support)
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 0,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            type TEXT NOT NULL,
            fees REAL DEFAULT 0
        )
    ''')
    
    # Add user_id column if it doesn't exist (migration for existing databases)
    try:
        c.execute('ALTER TABLE transactions ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Dividends table (with user_id)
    c.execute('''
        CREATE TABLE IF NOT EXISTS dividends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 0,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            amount_per_share REAL NOT NULL,
            total_shares REAL NOT NULL,
            total_amount REAL NOT NULL,
            dividend_type TEXT DEFAULT 'CASH'
        )
    ''')
    
    # Add user_id column if it doesn't exist (migration)
    try:
        c.execute('ALTER TABLE dividends ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    
    # Realized P&L table (with user_id)
    c.execute('''
        CREATE TABLE IF NOT EXISTS realized_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 0,
            symbol TEXT NOT NULL,
            sell_date TEXT NOT NULL,
            quantity REAL NOT NULL,
            buy_avg_price REAL NOT NULL,
            sell_price REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            fees REAL DEFAULT 0
        )
    ''')
    
    # Add user_id column if it doesn't exist (migration)
    try:
        c.execute('ALTER TABLE realized_pnl ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()

def add_transaction(symbol, date, quantity, price, type='BUY', fees=0, user_id=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO transactions (user_id, symbol, date, quantity, price, type, fees)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, symbol, date, quantity, price, type, fees))
    conn.commit()
    conn.close()

def get_transactions(user_id=0):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC", conn, params=(user_id,))
    conn.close()
    return df

def delete_transaction(transaction_id, user_id=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM transactions WHERE id=? AND user_id=?", (transaction_id, user_id))
    conn.commit()
    conn.close()

# ================= DIVIDEND FUNCTIONS =================
def add_dividend(symbol, date, amount_per_share, total_shares, dividend_type='CASH', user_id=0):
    """Record a dividend payment."""
    total_amount = amount_per_share * total_shares
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO dividends (user_id, symbol, date, amount_per_share, total_shares, total_amount, dividend_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, symbol, date, amount_per_share, total_shares, total_amount, dividend_type))
    conn.commit()
    conn.close()

def get_dividends(user_id=0):
    """Get all dividend records for a user."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM dividends WHERE user_id = ? ORDER BY date DESC", conn, params=(user_id,))
    conn.close()
    return df

def get_dividends_by_symbol(symbol, user_id=0):
    """Get dividends for a specific stock."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM dividends WHERE symbol=? AND user_id=? ORDER BY date DESC", conn, params=(symbol, user_id))
    conn.close()
    return df

def delete_dividend(dividend_id, user_id=0):
    """Delete a dividend record."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM dividends WHERE id=? AND user_id=?", (dividend_id, user_id))
    conn.commit()
    conn.close()

def get_total_dividends(user_id=0):
    """Get total dividends received for a user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT SUM(total_amount) FROM dividends WHERE user_id = ?", (user_id,))
    result = c.fetchone()[0]
    conn.close()
    return result if result else 0

# ================= REALIZED P&L FUNCTIONS =================
def add_realized_pnl(symbol, sell_date, quantity, buy_avg_price, sell_price, fees=0, user_id=0):
    """Record a realized P&L entry when a position is closed."""
    realized_pnl = (sell_price - buy_avg_price) * quantity - fees
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO realized_pnl (user_id, symbol, sell_date, quantity, buy_avg_price, sell_price, realized_pnl, fees)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, symbol, sell_date, quantity, buy_avg_price, sell_price, realized_pnl, fees))
    conn.commit()
    conn.close()

def get_realized_pnl(user_id=0):
    """Get all realized P&L records for a user."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM realized_pnl WHERE user_id = ? ORDER BY sell_date DESC", conn, params=(user_id,))
    conn.close()
    return df

def get_total_realized_pnl(user_id=0):
    """Get total realized P&L for a user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT SUM(realized_pnl) FROM realized_pnl WHERE user_id = ?", (user_id,))
    result = c.fetchone()[0]
    conn.close()
    return result if result else 0

def clear_all_portfolio_data(user_id=0):
    """
    Clear all portfolio data for a specific user.
    WARNING: This is irreversible!
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM dividends WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM realized_pnl WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True

def get_portfolio_stats(user_id=0):
    """Get counts of all portfolio data for a user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM transactions WHERE user_id = ?", (user_id,))
    trans_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM dividends WHERE user_id = ?", (user_id,))
    div_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM realized_pnl WHERE user_id = ?", (user_id,))
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
