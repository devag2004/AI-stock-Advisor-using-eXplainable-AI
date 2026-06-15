import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise SystemExit("DATABASE_URL not found in .env")

print("Connecting to Render Postgres...")
conn = psycopg2.connect(DATABASE_URL, sslmode="require")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

try:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='age'")
    exists = cur.fetchone()
    if not exists:
        print("Adding 'age' column to users table...")
        cur.execute("ALTER TABLE users ADD COLUMN age INTEGER")
        conn.commit()
        print("'age' column added.")
    else:
        print("'age' column already exists.")
except Exception as e:
    print("Could not ensure 'age' column:", e)

cur.close()
conn.close()
print("DB ready.")
