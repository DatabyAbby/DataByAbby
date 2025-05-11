import sqlite3
import time
import json

def init_analytics_db():
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS usage_logs
    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
     user_email TEXT,
     action TEXT,
     details TEXT,
     timestamp TEXT)
    ''')
    conn.commit()
    conn.close()

def log_action(user_email, action, details=None):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    details_json = json.dumps(details) if details else '{}'
    
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute("INSERT INTO usage_logs (user_email, action, details, timestamp) VALUES (?, ?, ?, ?)",
              (user_email, action, details_json, timestamp))
    conn.commit()
    conn.close()
    
    # Also update daily stats for quota calculations
    update_daily_stats(user_email, action)

def update_daily_stats(user_email, action):
    today = time.strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    
    # Check if we have a daily record already
    c.execute("""
    SELECT * FROM daily_stats 
    WHERE user_email = ? AND date = ? AND action = ?
    """, (user_email, today, action))
    
    result = c.fetchone()
    
    if result:
        # Update existing record
        c.execute("""
        UPDATE daily_stats 
        SET count = count + 1 
        WHERE user_email = ? AND date = ? AND action = ?
        """, (user_email, today, action))
    else:
        # Create new record
        c.execute("""
        INSERT INTO daily_stats (user_email, date, action, count)
        VALUES (?, ?, ?, 1)
        """, (user_email, today, action))
    
    conn.commit()
    conn.close()

def init_daily_stats():
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS daily_stats
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     user_email TEXT,
     date TEXT,
     action TEXT,
     count INTEGER,
     UNIQUE(user_email, date, action))
    ''')
    conn.commit()
    conn.close()

def get_daily_usage(user_email, action):
    today = time.strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute("""
    SELECT count FROM daily_stats
    WHERE user_email = ? AND date = ? AND action = ?
    """, (user_email, today, action))
    
    result = c.fetchone()
    conn.close()
    
    return result[0] if result else 0

def get_user_stats(user_email):
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute("""
    SELECT 
        COUNT(*) as total_actions,
        COUNT(DISTINCT CASE WHEN action = 'data_upload' THEN id END) as uploads,
        COUNT(DISTINCT CASE WHEN action = 'model_build' THEN id END) as models,
        COUNT(DISTINCT CASE WHEN action = 'visualization' THEN id END) as visualizations,
        COUNT(DISTINCT CASE WHEN action = 'export' THEN id END) as exports
    FROM usage_logs
    WHERE user_email = ?
    """, (user_email,))
    stats = c.fetchone()
    
    # Get last 5 actions
    c.execute("""
    SELECT action, details, timestamp
    FROM usage_logs
    WHERE user_email = ?
    ORDER BY timestamp DESC
    LIMIT 5
    """, (user_email,))
    recent = c.fetchall()
    
    conn.close()
    
    return {
        "total_actions": stats[0],
        "uploads": stats[1], 
        "models": stats[2],
        "visualizations": stats[3],
        "exports": stats[4],
        "recent_activity": [
            {
                "action": r[0],
                "details": json.loads(r[1]),
                "timestamp": r[2]
            } for r in recent
        ]
    }