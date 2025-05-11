import streamlit as st
import hmac
import sqlite3
import time
import re
from dataclasses import dataclass

@dataclass
class User:
    email: str
    is_active: bool
    is_authenticated: bool
    
# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users
    (email TEXT PRIMARY KEY, password TEXT, active INTEGER, created_at TEXT)
    ''')
    conn.commit()
    conn.close()

# User functions
def create_user(email, password):
    password_hash = hash_password(password)
    created_at = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, 1, ?)", (email, password_hash, created_at))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password, active FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    
    if result and result[1] == 1:
        stored_password = result[0]
        if check_password(password, stored_password):
            return User(email=email, is_active=True, is_authenticated=True)
    return None

def hash_password(password):
    return hmac.new(b"dataanalysis-salt", password.encode(), 'sha256').hexdigest()
    
def check_password(password, hash):
    return hmac.new(b"dataanalysis-salt", password.encode(), 'sha256').hexdigest() == hash

# Email validation function
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Login page
def show_login_page():
    st.title("Login to Data Analysis Tool")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                user = login_user(email, password)
                if user:
                    st.session_state.user = user
                    st.session_state.is_authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
    
    with tab2:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Sign Up", key="signup_button"):
            if not new_email or not new_password or not confirm_password:
                st.error("Please fill out all fields.")
            elif not is_valid_email(new_email):
                st.error("Please enter a valid email address.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                if create_user(new_email, new_password):
                    st.success("Account created! Please log in.")
                else:
                    st.error("An account with this email already exists.")