import os
import psycopg2
import yaml
from passlib.hash import pbkdf2_sha256

# --- CONFIGURATION LOADING ---
def load_config():
    # Only load if the file exists (prevents errors in Cloud Run)
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

# --- CONNECTION LOGIC ---
def get_connection():
    """
    Directly manages connections for Cloud Run (Unix Socket) 
    and Local Development (TCP).
    """
    # K_SERVICE is the signal that we are running on Google Cloud
    if os.getenv("K_SERVICE"):
        return psycopg2.connect(
            database="wellbeing_db",
            user="postgres",
            password="Acakdoir1!", 
            # This is the dedicated pipe for Cloud Run to Cloud SQL
            host="/cloudsql/bgita-teacher:us-central1:bgita-teacher"
        )
    else:
        # LOCAL: Use your local Postgres settings
        return psycopg2.connect(
            database="wellbeing_db",
            user="postgres",
            password="acakdoir",
            host="localhost",
            port="5432" # Change to 5433 if your netstat showed 5433
        )

# --- DATABASE INITIALIZATION ---
def init_db():
    """Creates tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Users Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            name TEXT,
            roles TEXT[] DEFAULT '{user}'
        );
    """)
    
    # Logs Table (for CBT activity/mood)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            username TEXT REFERENCES users(username),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mood TEXT,
            activity TEXT,
            reflection TEXT
        );
    """)

    # Progress Table (for the 11-chapter curriculum)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            username TEXT REFERENCES users(username),
            chapter_id INT,
            completed BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (username, chapter_id)
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")

# --- AUTHENTICATION FUNCTIONS ---
def get_credentials():
    """Fetches user data formatted for streamlit-authenticator."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, password, name, roles FROM users")
    rows = cur.fetchall()
    
    credentials = {"usernames": {}}
    for row in rows:
        credentials["usernames"][row[0]] = {
            "password": row[1],
            "name": row[2],
            "roles": row[3]
        }
    
    cur.close()
    conn.close()
    return credentials

def add_user(username, password, name, roles=['user']):
    """Hashes password and saves a new user."""
    hashed_pw = pbkdf2_sha256.hash(password)
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password, name, roles) VALUES (%s, %s, %s, %s)",
            (username, hashed_pw, name, roles)
        )
        conn.commit()
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()
    return True

# --- LOGGING FUNCTIONS ---
def save_log(username, mood, activity, reflection):
    """Saves a CBT entry to the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (username, mood, activity, reflection) VALUES (%s, %s, %s, %s)",
        (username, mood, activity, reflection)
    )
    conn.commit()
    cur.close()
    conn.close()
