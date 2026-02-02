import psycopg2
from psycopg2 import extras
import streamlit as st
from passlib.hash import pbkdf2_sha256
import streamlit_authenticator as stauth

import psycopg2
import yaml
from passlib.hash import pbkdf2_sha256

import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


raw_host = os.getenv("DB_HOST") or config['networks']['db_host']
raw_port = os.getenv("DB_PORT") or config['networks']['db_port']


clean_host = raw_host.replace("tcp://", "").split(":")[0].strip()


print(f"DEBUG: Attempting to connect to host: '{clean_host}' on port: '{raw_port}'")

DB_CONFIG = {
    "host": clean_host,
    "port": str(raw_port),
    "database": config['networks'].get('db_name', 'wellbeing_db'),
    "user": "postgres",
    "password": "acakdoir" 
}
'''
DB_CONFIG = {
    "host": "192.168.2.221",
    "port": "5433",
    "database": "wellbeing_db",
    "user": "postgres",
    "password": "acakdoir"
}
'''


def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """Initializes the PostgreSQL tables for RBAC and Lesson Tracking."""
    conn = get_connection()
    cur = conn.cursor()
    
    # 1. Users table for Admin and Patient roles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient'
        );
    """)
    
    # 2. Logs table for Emoji Moods and Chat History
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            username TEXT REFERENCES users(username),
            mood TEXT,
            prompt TEXT,
            response TEXT,
            chapter TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 3. Progress table for the 11-Hour Course requirement
    cur.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            username TEXT REFERENCES users(username),
            chapter_name TEXT,
            minutes_spent INT DEFAULT 0,
            is_completed BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (username, chapter_name)
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def add_user(username, password, name, role='patient'):
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Use the authenticator's own hasher so the "salt" is always valid
        hashed_pw = stauth.Hasher.hash(password)
        
        cur.execute(
            "INSERT INTO users (username, password, name, role) VALUES (%s, %s, %s, %s)",
            (username, hashed_pw, name, role)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def get_credentials():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cur.execute("SELECT username, password, name, role FROM users")
        rows = cur.fetchall()
        
        creds = {'usernames': {}}
        for row in rows:
            if row['password']:
                creds['usernames'][row['username']] = {
                    'name': row['name'],
                    'password': row['password'],
                    'roles': [row['role']]
                }
        return creds
    except Exception as e:
        return {'usernames': {}}
    finally:
        cur.close()
        conn.close()

def save_log(username, mood, prompt, response, chapter):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (username, mood, prompt, response, chapter) VALUES (%s, %s, %s, %s, %s)",
        (username, mood, prompt, response, chapter)
    )
    conn.commit()
    cur.close()
    conn.close()