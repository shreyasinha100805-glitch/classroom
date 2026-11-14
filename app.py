import os
import logging

# ---------------------------------------------------------------------------
# CPU / ONNX thread limits — set BEFORE any imports.
# Limits OpenCV and ONNX Runtime CPU thread pools on Render's single-core
# free-tier instance. Reduces peak RSS and prevents thread over-subscription.
# OMP/MKL/OPENBLAS: also constrain RapidOCR's ONNX Runtime thread pools.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")         # prevent CUDA scanning
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # OpenMP compat on Linux

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass

# Configure structured logging before any module uses the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)

import time
import secrets
import re
from uuid import uuid4
from PIL import Image
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
    Response,
    session,
    redirect,
    url_for,
    flash
)

# Optional CORS support for cross-origin frontend
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

import auth
from pipeline import (
    PIPELINE_OUTPUT_FOLDER,
    PIPELINE_PROCESSES,
    run_pipeline_data,
    validate_image,
)

# ==============================================================================
# INFERENCE CONCURRENCY GUARD
# ==============================================================================
# TF model.predict() and EasyOCR readtext() are NOT thread-safe when called
# concurrently from multiple gunicorn threads. A semaphore with value=1
# serialises all analysis requests so only one inference runs at a time.
# Non-inference requests (health, upload, file list) are unaffected.
import threading
_inference_semaphore = threading.Semaphore(1)

# ==============================================================================
# FLASK APPLICATION CONFIGURATION & SECURITY HARDENING
# ==============================================================================

app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)

# Detect runtime environment
is_production = (
    os.environ.get("FLASK_ENV") == "production"
    or bool(os.environ.get("RENDER"))
    or bool(os.environ.get("VERCEL"))
)

# Item 1: Secure Secret Key Configuration (Zero hardcoded credentials)
env_secret = os.environ.get("SECRET_KEY")
if env_secret and env_secret.strip():
    app.secret_key = env_secret.strip()
elif is_production:
    print("[SECURITY WARNING] 'SECRET_KEY' environment variable not set in production.")
    print("[SECURITY WARNING] Generating ephemeral cryptographic key for session isolation.")
    app.secret_key = secrets.token_hex(32)
else:
    app.secret_key = os.environ.get("DEV_SECRET_KEY", "classmind-dev-session-key-" + secrets.token_hex(16))

# Session and cookie security parameters
app.config.update(
    TEMPLATES_AUTO_RELOAD=True,
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32 MB max upload limit
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=is_production,
    PERMANENT_SESSION_LIFETIME=86400 * 7  # 7-day session lifetime
)

# Item 2: Explicit Trusted CORS Origins (No wildcard + credential conflicts)
DEFAULT_TRUSTED_ORIGINS = [
    "https://classroom-tan-iota.vercel.app",
    "https://classroom-swart-one.vercel.app",
    "https://classroom-theta-silk.vercel.app",
    "https://classroom-backend-cp6o.onrender.com",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
]

env_origins = os.environ.get("TRUSTED_ORIGINS", "")
if env_origins:
    TRUSTED_ORIGINS = [o.strip() for o in env_origins.split(",") if o.strip()]
else:
    TRUSTED_ORIGINS = DEFAULT_TRUSTED_ORIGINS


def is_origin_allowed(origin: str | None) -> bool:
    """Validates origin against trusted origins and canonical project domains."""
    if not origin:
        return False
    if origin in TRUSTED_ORIGINS:
        return True
    # Match any classroom-* Vercel preview/production deployment
    if re.match(r"^https://classroom-[a-zA-Z0-9_-]+\.vercel\.app$", origin):
        return True
    # Match local development hosts on any valid port
    if re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$", origin):
        return True
    return False


if HAS_CORS:
    # Pass the validator function directly so flask-cors uses the same regex logic
    # as our @after_request handler (fixes preflight rejection for new Vercel URLs)
    CORS(
        app,
        origins=is_origin_allowed,
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
        methods=["GET", "POST", "OPTIONS", "HEAD", "PUT", "DELETE"]
    )


@app.before_request
def handle_options_preflight():
    """Handle CORS preflight OPTIONS requests with explicit origin validation."""
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        origin = request.headers.get("Origin")
        if origin and is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD, PUT, DELETE"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-Requested-With, Accept, Origin"
            )
            response.headers["Vary"] = "Origin"
        return response


@app.after_request
def add_cors_headers(response):
    """Enforce explicit origin matching on all responses (eliminates wildcard credentials)."""
    origin = request.headers.get("Origin")
    if origin and is_origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-Requested-With, Accept, Origin"
        )
        response.headers["Vary"] = "Origin"
    return response


# Initialize user database (schema preserved)
auth.init_db()

# Upload storage directory
UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER",
    os.path.join("/tmp" if os.environ.get("VERCEL") else ".", "uploads")
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PIPELINE_OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# Item 5: Safe Rate-Limited File Cleanup (Protects In-Flight Processing)
_last_cleanup_timestamp = 0.0


def cleanup_old_files(max_age_hours: int = 24, min_protection_hours: int = 2):
    """
    Purges temporary uploads older than max_age_hours to prevent disk overflow.
    Guarantees that files modified less than min_protection_hours ago are NEVER deleted,
    protecting active uploads and pipeline runs.
    Throttled to run at most once per hour.
    """
    global _last_cleanup_timestamp
    now = time.time()
    if now - _last_cleanup_timestamp < 3600:
        return

    _last_cleanup_timestamp = now
    max_age_sec = max_age_hours * 3600
    min_protection_sec = min_protection_hours * 3600

    try:
        for folder in [UPLOAD_FOLDER, PIPELINE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                continue
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    file_age = now - os.path.getmtime(item_path)
                    if file_age > max_age_sec and file_age > min_protection_sec:
                        try:
                            os.remove(item_path)
                        except OSError:
                            pass
    except Exception as exc:
        print(f"Notice: Periodic storage cleanup notice: {exc}")


# ==============================================================================
# CONTEXT PROCESSORS & GLOBAL HELPERS
# ==============================================================================

@app.context_processor
def inject_global_template_context():
    user_id = session.get("user_id")
    user = auth.get_user_by_id(user_id) if user_id else None
    if user_id and not user:
        # Clear stale session if user no longer exists in database
        session.clear()

    backend_url = (os.environ.get("BACKEND_URL") or "").strip().rstrip("/")
    if not backend_url:
        try:
            backend_url = request.url_root.rstrip("/")
        except RuntimeError:
            backend_url = ""
    return {
        "current_user": user,
        "backend_url": backend_url,
        "pipeline_processes": PIPELINE_PROCESSES
    }


# ==============================================================================
# HEALTH CHECK & SYSTEM STATUS
# ==============================================================================

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Fast health check route for Render, Vercel, and uptime probes.

    Reports the actual active CNN model (classroom_cnn.onnx, not .keras).
    OCR status is reported via three distinct fields:
      - ocr_configured: rapidocr-onnxruntime package is importable
      - ocr_initialized: RapidOCR singleton is loaded in memory (warm)
      - ocr_ready: same as ocr_initialized (alias for frontend compatibility)
    Does NOT trigger OCR model loading — health check is always fast.
    """
    # CNN: production inference uses classroom_cnn.onnx via onnxruntime.
    _default_onnx = os.path.join(
        os.path.dirname(__file__), "cnn", "model", "classroom_cnn.onnx"
    )
    cnn_model_file = os.environ.get("CNN_MODEL_PATH", _default_onnx)
    cnn_model_ready = os.path.exists(cnn_model_file)

    # OCR: use extractor helpers — no model loading triggered here.
    try:
        from ocr.extractor import ocr_is_configured, ocr_is_initialized
        ocr_configured = ocr_is_configured()
        ocr_initialized = ocr_is_initialized()
    except Exception:
        ocr_configured = False
        ocr_initialized = False

    # Current process RSS (for memory monitoring via health probe).
    rss_mb = None
    try:
        import psutil
        rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        pass

    payload = {
        "status": "healthy",
        "service": "AI Classroom Board Content Extractor",
        "version": "2.0.0",
        "timestamp": int(time.time()),
        "cnn_model_file": os.path.basename(cnn_model_file),
        "cnn_model_ready": cnn_model_ready,
        "ocr_engine": "RapidOCR (ONNX Runtime)",
        "ocr_configured": ocr_configured,
        "ocr_initialized": ocr_initialized,
        "ocr_engine_ready": ocr_initialized,  # kept for frontend compatibility
    }
    if rss_mb is not None:
        payload["rss_mb"] = rss_mb

    return jsonify(payload), 200


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ==============================================================================
# WEB PAGES
# ==============================================================================

@app.route("/")
def home():
    cleanup_old_files()
    return render_template("index.html")


@app.route("/results")
def results_page():
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        names = sorted(
            os.listdir(UPLOAD_FOLDER),
            key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
            reverse=True
        )
        for f in names[:40]:
            if allowed_file(f):
                fp = os.path.join(UPLOAD_FOLDER, f)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                ext = f.rsplit(".", 1)[1].upper() if "." in f else "IMG"
                if sz < 1024:
                    s_str = f"{sz} B"
                elif sz < 1024 * 1024:
                    s_str = f"{sz / 1024:.1f} KB"
                else:
                    s_str = f"{sz / (1024 * 1024):.1f} MB"

                dim_str = "Verified"
                try:
                    with Image.open(fp) as im:
                        dim_str = f"{im.width}x{im.height}"
                except Exception:
                    pass

                files.append({
                    "original_name": f,
                    "saved_name": f,
                    "file_size": s_str,
                    "extension": ext,
                    "dimensions": dim_str
                })

    return render_template("result.html", uploaded_files=files)


# ==============================================================================
# USER AUTHENTICATION (WEB & JSON API)
# ==============================================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            return render_template(
                "signup.html",
                error="Passwords do not match.",
                username=username,
                email=email
            )

        user, err = auth.create_user(username, email, password)
        if err:
            return render_template(
                "signup.html",
                error=err,
                username=username,
                email=email
            )

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["email"] = user["email"]
        flash(f"Welcome to BoardAI, {user['username']}! Account created successfully.", "success")
        return redirect(url_for("home"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        identifier = request.form.get("login_identifier", "").strip()
        password = request.form.get("password", "")

        user, err = auth.authenticate_user(identifier, password)
        if err:
            return render_template(
                "login.html",
                error=err,
                login_identifier=identifier
            )

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["email"] = user["email"]
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("home"))


@app.route("/api/auth/signup", methods=["POST"])
def api_signup():
    data = request.get_json(silent=True) or request.form
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", password)

    if password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match."}), 400

    user, err = auth.create_user(username, email, password)
    if err:
        return jsonify({"success": False, "error": err}), 400

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["email"] = user["email"]
    return jsonify({"success": True, "user": user}), 201


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or request.form
    identifier = data.get("login_identifier", "").strip()
    password = data.get("password", "")

    user, err = auth.authenticate_user(identifier, password)
    if err:
        return jsonify({"success": False, "error": err}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["email"] = user["email"]
    return jsonify({"success": True, "user": user}), 200


@app.route("/api/auth/me", methods=["GET"])
def api_me():
    user_id = session.get("user_id")
    user = auth.get_user_by_id(user_id) if user_id else None
    return jsonify({"authenticated": bool(user), "user": user}), 200


@app.route("/api/auth/logout", methods=["POST", "GET"])
def api_logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."}), 200


# ==============================================================================
# IMAGE UPLOAD & INGESTION
# ==============================================================================

@app.route("/upload", methods=["POST"])
@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    if not files or all(file.filename == "" for file in files):
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "No images provided in upload request."}), 400
        flash("Please select at least one valid image to upload.", "error")
        return redirect(url_for("home"))

    uploaded_files = []

    for file in files:
        if file.filename == "" or not allowed_file(file.filename):
            continue

        orig_name = secure_filename(file.filename)
        ext = orig_name.rsplit(".", 1)[1].lower() if "." in orig_name else "jpg"
        unique_name = f"{uuid4().hex}.{ext}"
        target_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

        file.save(target_path)

        # Ingestion & Validation
        val = validate_image(target_path)
        if not val["valid"]:
            try:
                os.remove(target_path)
            except OSError:
                pass
            continue

        size_bytes = val.get("size_bytes", os.path.getsize(target_path))
        if size_bytes < 1024:
            s_fmt = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            s_fmt = f"{size_bytes / 1024:.1f} KB"
        else:
            s_fmt = f"{size_bytes / (1024 * 1024):.1f} MB"

        uploaded_files.append({
            "original_name": orig_name,
            "saved_name": unique_name,
            "file_size": s_fmt,
            "extension": ext.upper(),
            "dimensions": f"{val.get('width', 0)}x{val.get('height', 0)}"
        })

    if not uploaded_files:
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "No valid classroom-board images could be verified."}), 400
        flash("No valid images could be verified. Please check file format and size.", "error")
        return redirect(url_for("home"))

    if request.is_json or request.path.startswith("/api/") or request.headers.get("Accept") == "application/json":
        return jsonify({
            "success": True,
            "count": len(uploaded_files),
            "files": uploaded_files
        }), 200

    return render_template("result.html", uploaded_files=uploaded_files)


@app.route("/api/files", methods=["GET"])
@app.route("/api/uploads", methods=["GET"])
def list_uploaded_files():
    """Returns the list of uploaded images as JSON."""
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        names = sorted(
            os.listdir(UPLOAD_FOLDER),
            key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
            reverse=True
        )
        for f in names[:50]:
            if allowed_file(f):
                fp = os.path.join(UPLOAD_FOLDER, f)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                ext = f.rsplit(".", 1)[1].upper() if "." in f else "IMG"
                if sz < 1024:
                    s_str = f"{sz} B"
                elif sz < 1024 * 1024:
                    s_str = f"{sz / 1024:.1f} KB"
                else:
                    s_str = f"{sz / (1024 * 1024):.1f} MB"

                dim_str = "Verified"
                try:
                    with Image.open(fp) as im:
                        dim_str = f"{im.width}x{im.height}"
                except Exception:
                    pass

                files.append({
                    "original_name": f,
                    "saved_name": f,
                    "file_size": s_str,
                    "extension": ext,
                    "dimensions": dim_str
                })

    return jsonify({"success": True, "count": len(files), "files": files}), 200


# ==============================================================================
# STATIC MEDIA SERVERS
# ==============================================================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/pipeline_output/<path:filename>")
def pipeline_output_file(filename):
    return send_from_directory(PIPELINE_OUTPUT_FOLDER, filename)


# ==============================================================================
# API: COMPLETE 6-STAGE AI PIPELINE
# ==============================================================================

@app.route("/api/analyze/<filename>", methods=["GET", "POST"])
def api_analyze_image(filename):
    safe_name = secure_filename(filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(image_path):
        return jsonify({
            "success": False,
            "error": f"Image file '{safe_name}' not found on server. It may have expired or not yet uploaded."
        }), 404

    # Serialise inference: TF model.predict() and EasyOCR readtext() are not
    # thread-safe under concurrent gunicorn threads. The semaphore ensures only
    # one analysis runs at a time; a second concurrent request waits rather than
    # doubling the memory spike. acquire(blocking=False) returns False immediately
    # if inference is already running so we can return a 503 instead of queuing.
    acquired = _inference_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        logger.warning("[Analyze] Inference semaphore timeout for %s — server busy", safe_name)
        return jsonify({
            "success": False,
            "error": "The server is currently processing another image. Please retry in a moment."
        }), 503

    logger.info("[Analyze] Starting pipeline for: %s", safe_name)
    try:
        import psutil
        rss_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        rss_before = None

    try:
        result = run_pipeline_data(image_path)

        try:
            import psutil
            rss_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.info("[Analyze] Complete. RSS before=%.1f MB after=%.1f MB delta=%.1f MB",
                        rss_before, rss_after, rss_after - rss_before)
        except ImportError:
            pass

        return jsonify(result), 200
    except Exception as exc:
        import traceback
        app.logger.error(
            "[Analyze] Pipeline FAILED for %s:\n%s", safe_name, traceback.format_exc()
        )
        return jsonify({
            "success": False,
            "error": "The AI pipeline encountered an error processing the image. Please try another capture.",
            "detail": str(exc)
        }), 500
    finally:
        _inference_semaphore.release()


@app.route("/api/export-notes/<filename>", methods=["GET"])
def export_notes(filename):
    format_type = request.args.get("format", "md").lower()
    safe_name = secure_filename(filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(image_path):
        return jsonify({"error": "Image file not found"}), 404

    acquired = _inference_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        return jsonify({"error": "Server is busy processing another image. Please retry."}), 503

    try:
        result = run_pipeline_data(image_path)
        notes = result.get("structured_notes", {})

        if format_type == "txt":
            content = notes.get("plain_text", "No content extracted.")
            mimetype = "text/plain"
            dl_name = f"notes_{safe_name}.txt"
        else:
            content = notes.get("markdown", "# No content extracted.")
            mimetype = "text/markdown"
            dl_name = f"notes_{safe_name}.md"

        return Response(
            content,
            mimetype=mimetype,
            headers={"Content-Disposition": f"attachment; filename={dl_name}"}
        )
    except Exception as exc:
        app.logger.error(f"Export notes error on {safe_name}: {exc}", exc_info=True)
        return jsonify({"error": "Unable to export notes."}), 500
    finally:
        _inference_semaphore.release()


# ==============================================================================
# SANITIZED GLOBAL ERROR HANDLERS (ITEM 4 & 7)
# ==============================================================================

@app.errorhandler(400)
def handle_bad_request(err):
    message = getattr(err, "description", "Bad request.")
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"success": False, "error": str(message)}), 400
    return render_template("index.html", error=str(message)), 400


@app.errorhandler(404)
def handle_not_found(err):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"success": False, "error": "Requested resource was not found."}), 404
    return render_template("index.html", error="Resource not found."), 404


@app.errorhandler(413)
def handle_file_too_large(err):
    msg = "Uploaded file exceeds the maximum allowed limit of 32 MB."
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"success": False, "error": msg}), 413
    flash(msg, "error")
    return redirect(url_for("home")), 413


@app.errorhandler(500)
def handle_internal_server_error(err):
    app.logger.error(f"Internal 500 error on {request.path}: {err}", exc_info=True)
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({
            "success": False,
            "error": "An internal server error occurred. Please try again later."
        }), 500
    return render_template("index.html", error="An internal server error occurred."), 500


@app.errorhandler(Exception)
def handle_unhandled_exception(exc):
    app.logger.error(f"Unhandled exception on {request.path}: {exc}", exc_info=True)
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while processing your request."
        }), 500
    return render_template("index.html", error="An unexpected error occurred."), 500


# ==============================================================================
# SERVER STARTUP (LOCAL / PRODUCTION WSGI)
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print(f"Starting Classroom AI Server on http://0.0.0.0:{port} (Debug: {debug_mode})")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
