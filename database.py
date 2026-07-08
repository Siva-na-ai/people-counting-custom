import os
import psycopg2
from dotenv import load_dotenv

# Load env variables on startup
load_dotenv()

def get_db_connection():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_SSLMODE = os.getenv("DB_SSLMODE", "require")
    
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        sslmode=DB_SSLMODE
    )
    return conn
