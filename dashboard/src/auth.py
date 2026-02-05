"""
Authentication module for PSX Portfolio Manager.
Handles user registration, login, and session management.
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portfolio.db')


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    # Using SHA-256 with a fixed salt for simplicity
    # For production, consider using bcrypt
    salt = "psx_portfolio_salt_2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def init_users_table():
    """Create users table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def register_user(username: str, email: str, password: str) -> dict:
    """
    Register a new user.
    Returns: {'success': bool, 'message': str, 'user_id': int or None}
    """
    if len(password) < 6:
        return {'success': False, 'message': 'Password must be at least 6 characters', 'user_id': None}
    
    if len(username) < 3:
        return {'success': False, 'message': 'Username must be at least 3 characters', 'user_id': None}
    
    password_hash = hash_password(password)
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username.lower(), email.lower(), password_hash, created_at))
        
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Registration successful!', 'user_id': user_id}
    
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return {'success': False, 'message': 'Username already exists', 'user_id': None}
        elif 'email' in str(e):
            return {'success': False, 'message': 'Email already registered', 'user_id': None}
        else:
            return {'success': False, 'message': 'Registration failed', 'user_id': None}
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}', 'user_id': None}


def login_user(username_or_email: str, password: str) -> dict:
    """
    Authenticate a user.
    Returns: {'success': bool, 'message': str, 'user_id': int or None, 'username': str or None}
    """
    password_hash = hash_password(password)
    
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Check by username or email
        c.execute('''
            SELECT id, username FROM users 
            WHERE (username = ? OR email = ?) AND password_hash = ?
        ''', (username_or_email.lower(), username_or_email.lower(), password_hash))
        
        result = c.fetchone()
        
        if result:
            user_id, username = result
            
            # Update last login
            c.execute('''
                UPDATE users SET last_login = ? WHERE id = ?
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_id))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True, 
                'message': 'Login successful!', 
                'user_id': user_id,
                'username': username
            }
        else:
            conn.close()
            return {
                'success': False, 
                'message': 'Invalid username/email or password', 
                'user_id': None,
                'username': None
            }
    
    except Exception as e:
        return {
            'success': False, 
            'message': f'Login error: {str(e)}', 
            'user_id': None,
            'username': None
        }


def get_user_by_id(user_id: int) -> dict:
    """Get user details by ID."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, username, email, created_at, last_login 
            FROM users WHERE id = ?
        ''', (user_id,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'email': result[2],
                'created_at': result[3],
                'last_login': result[4]
            }
        return None
    except:
        return None


def change_password(user_id: int, old_password: str, new_password: str) -> dict:
    """Change user password."""
    if len(new_password) < 6:
        return {'success': False, 'message': 'New password must be at least 6 characters'}
    
    old_hash = hash_password(old_password)
    new_hash = hash_password(new_password)
    
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Verify old password
        c.execute('SELECT id FROM users WHERE id = ? AND password_hash = ?', (user_id, old_hash))
        if not c.fetchone():
            conn.close()
            return {'success': False, 'message': 'Current password is incorrect'}
        
        # Update password
        c.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Password changed successfully'}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}


# Initialize users table on import
init_users_table()
