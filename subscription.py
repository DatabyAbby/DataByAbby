import sqlite3
import time
from datetime import datetime, timedelta

def init_subscription_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS subscriptions
    (email TEXT PRIMARY KEY, 
     plan TEXT,
     start_date TEXT,
     end_date TEXT,
     FOREIGN KEY(email) REFERENCES users(email))
    ''')
    conn.commit()
    conn.close()

def create_free_subscription(email):
    # Free trial for 14 days
    start_date = time.strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO subscriptions VALUES (?, ?, ?, ?)", 
              (email, 'free_trial', start_date, end_date))
    conn.commit()
    conn.close()

def upgrade_subscription(email, plan):
    # Plans: 'basic', 'premium', 'enterprise'
    start_date = time.strftime('%Y-%m-%d')
    
    # Set end date based on plan
    if plan == 'basic':
        end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    elif plan == 'premium':
        end_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
    elif plan == 'enterprise':
        end_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        return False
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO subscriptions VALUES (?, ?, ?, ?)", 
              (email, plan, start_date, end_date))
    conn.commit()
    conn.close()
    return True

def check_subscription(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT plan, end_date FROM subscriptions WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return {'active': False, 'plan': None, 'days_left': 0}
    
    plan, end_date = result
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    today = datetime.now()
    days_left = (end_date - today).days
    
    return {
        'active': days_left > 0,
        'plan': plan,
        'days_left': max(0, days_left)
    }

# Function to get the usage limits based on plan
def get_plan_limits(plan):
    limits = {
        'free_trial': {
            'max_file_size': 5,        # MB
            'max_rows': 10000,         # Max rows in dataset
            'models_per_day': 3,       # Number of models allowed per day
            'visualizations': True,    # Access to visualizations
            'export': True,            # Ability to export results
            'api_access': False,       # API access
            'support': 'community'     # Support level
        },
        'basic': {
            'max_file_size': 25,
            'max_rows': 50000,
            'models_per_day': 10,
            'visualizations': True,
            'export': True,
            'api_access': False,
            'support': 'email'
        },
        'premium': {
            'max_file_size': 100,
            'max_rows': 500000,
            'models_per_day': 50,
            'visualizations': True,
            'export': True,
            'api_access': True,
            'support': 'priority'
        },
        'enterprise': {
            'max_file_size': 1000,
            'max_rows': 5000000,
            'models_per_day': 1000,
            'visualizations': True,
            'export': True,
            'api_access': True,
            'support': 'dedicated'
        }
    }
    
    return limits.get(plan, limits['free_trial'])