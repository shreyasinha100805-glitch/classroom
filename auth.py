import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join("/tmp" if os.environ.get("VERCEL") else ".", "database.db")
)


def get_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database and creates the users table if not existing."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def create_user(username: str, email: str, password: str) -> tuple[dict | None, str | None]:
    """
    Creates a new user with hashed password.
    Returns (user_dict, error_message).
    """
    username = (username or "").strip()
    email = (email or "").strip().lower()

    if not username:
        return None, "Username is required."
    if len(username) < 3:
        return None, "Username must be at least 3 characters."
    if not email or "@" not in email:
        return None, "A valid email address is required."
    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters."

    password_hash = generate_password_hash(password)

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        user = {
            "id": user_id,
            "username": username,
            "email": email
        }
        return user, None
    except sqlite3.IntegrityError as err:
        err_msg = str(err).lower()
        if "users.username" in err_msg or "unique constraint failed: users.username" in err_msg:
            return None, f"Username '{username}' is already taken."
        elif "users.email" in err_msg or "unique constraint failed: users.email" in err_msg:
            return None, f"An account with email '{email}' already exists."
        else:
            return None, "Username or email is already registered."
    finally:
        conn.close()


def authenticate_user(login_identifier: str, password: str) -> tuple[dict | None, str | None]:
    """
    Authenticates a user via username OR email and password.
    Returns (user_dict, error_message).
    """
    login_identifier = (login_identifier or "").strip()
    if not login_identifier or not password:
        return None, "Please provide your username/email and password."

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
        (login_identifier, login_identifier.lower())
    )
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return None, "No account found with that username or email."

    if not check_password_hash(user_row["password_hash"], password):
        return None, "Incorrect password. Please try again."

    user = {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"]
    }
    return user, None


def get_user_by_id(user_id: int) -> dict | None:
    """Fetches user details by user ID."""
    if not user_id:
        return None

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()
    conn.close()

    if user_row:
        return {
            "id": user_row["id"],
            "username": user_row["username"],
            "email": user_row["email"],
            "created_at": user_row["created_at"]
        }
    return None
