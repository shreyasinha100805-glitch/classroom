import sqlite3
import os
import re
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join("/tmp" if os.environ.get("VERCEL") else ".", "database.db")
)


def get_db():
    """
    Creates a thread-safe connection to the SQLite database.
    Configures row factory and safe timeout to prevent locking conflicts.
    """
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row

    # Performance and concurrency optimizations for SQLite
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass

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


def validate_username(username: str) -> tuple[bool, str | None]:
    """Validates username format and length."""
    if not username:
        return False, "Username is required."
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 30:
        return False, "Username must not exceed 30 characters."
    if not re.match(r"^[a-zA-Z0-9_-]+$", username):
        return False, "Username may only contain letters, numbers, underscores, and hyphens."
    return True, None


def validate_email(email: str) -> tuple[bool, str | None]:
    """Validates email format."""
    if not email:
        return False, "Email address is required."
    email = email.strip()
    email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(email_regex, email):
        return False, "Please provide a valid email address."
    return True, None


def validate_password(password: str) -> tuple[bool, str | None]:
    """Validates password strength."""
    if not password:
        return False, "Password is required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if len(password) > 128:
        return False, "Password must not exceed 128 characters."
    return True, None


def create_user(username: str, email: str, password: str) -> tuple[dict | None, str | None]:
    """
    Creates a new user with securely hashed password.
    Returns (user_dict, error_message).
    """
    username = (username or "").strip()
    email = (email or "").strip().lower()

    valid_u, err_u = validate_username(username)
    if not valid_u:
        return None, err_u

    valid_e, err_e = validate_email(email)
    if not valid_e:
        return None, err_e

    valid_p, err_p = validate_password(password)
    if not valid_p:
        return None, err_p

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
        return {
            "id": user_id,
            "username": username,
            "email": email
        }, None
    except sqlite3.IntegrityError as err:
        err_msg = str(err).lower()
        if "users.username" in err_msg or "unique constraint failed: users.username" in err_msg:
            return None, f"Username '{username}' is already taken."
        elif "users.email" in err_msg or "unique constraint failed: users.email" in err_msg:
            return None, f"An account with email '{email}' already exists."
        else:
            return None, "Username or email is already registered."
    except Exception as exc:
        return None, "Unable to create account due to a database error."
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

    try:
        cursor.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
            (login_identifier, login_identifier.lower())
        )
        user_row = cursor.fetchone()
    except Exception:
        return None, "Authentication service temporarily unavailable."
    finally:
        conn.close()

    if not user_row:
        return None, "Invalid username or password."

    if not check_password_hash(user_row["password_hash"], password):
        return None, "Invalid username or password."

    return {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"]
    }, None


def get_user_by_id(user_id: int) -> dict | None:
    """Fetches user details by user ID."""
    if not user_id:
        return None

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
        user_row = cursor.fetchone()
    except Exception:
        user_row = None
    finally:
        conn.close()

    if user_row:
        return {
            "id": user_row["id"],
            "username": user_row["username"],
            "email": user_row["email"],
            "created_at": user_row["created_at"]
        }
    return None
