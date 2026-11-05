import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Early import of torch to ensure C10 registry stability with TensorFlow
try:
    import torch
except Exception:
    pass

import time
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

# Optional CORS support for cross-origin frontend (e.g. Vercel -> Render)
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

# ============================================
# FLASK APPLICATION CONFIGURATION
# ============================================

app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)

app.secret_key = os.environ.get("SECRET_KEY", "boardai-neural-studio-secret-key-2026")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload limit

# Enable CORS for universal device and cross-origin deployment support
if HAS_CORS:
    CORS(
        app,
        resources={r"/*": {"origins": "*"}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
        methods=["GET", "POST", "OPTIONS", "HEAD"]
    )


@app.after_request
def add_cors_headers(response):
    """Ensure universal browser compatibility with dynamic origin and credential support."""
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    return response


# Initialize user database
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


def cleanup_old_files(max_age_hours: int = 48):
    """Purge temporary uploads older than max_age_hours to prevent disk overflow."""
    try:
        now = time.time()
        max_age_sec = max_age_hours * 3600
        for folder in [UPLOAD_FOLDER, PIPELINE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                continue
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    if now - os.path.getmtime(item_path) > max_age_sec:
                        try:
                            os.remove(item_path)
                        except OSError:
                            pass
    except Exception as exc:
        print(f"Notice: Periodic storage cleanup warning: {exc}")


# ============================================
# CONTEXT PROCESSORS & INJECTORS
# ============================================

@app.context_processor
def inject_global_template_context():
    user_id = session.get("user_id")
    user = auth.get_user_by_id(user_id) if user_id else None
    backend_url = os.environ.get("BACKEND_URL", "").rstrip("/")
    return {
        "current_user": user,
        "backend_url": backend_url,
        "pipeline_processes": PIPELINE_PROCESSES
    }


# ============================================
# HEALTH CHECK & SYSTEM STATUS
# ============================================

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check route for Render, Railway, Vercel, and monitoring systems.
    Designed to return quickly and reliably without triggering heavy inference or model downloads.
    """
    cnn_model_file = os.environ.get(
        "CNN_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "cnn", "model", "classroom_cnn.keras")
    )
    model_exists = os.path.exists(cnn_model_file)

    return jsonify({
        "status": "healthy",
        "service": "AI Classroom Board Content Extractor",
        "version": "2.0.0",
        "timestamp": int(time.time()),
        "cnn_model_file": os.path.basename(cnn_model_file),
        "cnn_model_ready": model_exists,
        "ocr_engine_ready": True
    }), 200


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ============================================
# WEB PAGES
# ============================================

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


# ============================================
# USER AUTHENTICATION (WEB & JSON API)
# ============================================

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


# ============================================
# IMAGE UPLOAD & INGESTION (WEB & JSON API)
# ============================================

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

        # 1.0 Image Ingestion & Validation
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
            return jsonify({"success": False, "error": "No valid classroom-board images could be processed."}), 400
        flash("No valid images could be verified. Please check file format and size.", "error")
        return redirect(url_for("home"))

    # Return JSON for API consumers (mobile apps, decoupled frontends)
    if request.is_json or request.path.startswith("/api/") or request.headers.get("Accept") == "application/json":
        return jsonify({
            "success": True,
            "count": len(uploaded_files),
            "files": uploaded_files
        }), 200

    # Otherwise render the visual gallery
    return render_template("result.html", uploaded_files=uploaded_files)


@app.route("/api/files", methods=["GET"])
@app.route("/api/uploads", methods=["GET"])
def list_uploaded_files():
    """Returns the list of uploaded images as JSON for decoupled frontend."""
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


# ============================================
# STATIC MEDIA SERVERS
# ============================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/pipeline_output/<path:filename>")
def pipeline_output_file(filename):
    return send_from_directory(PIPELINE_OUTPUT_FOLDER, filename)


# ============================================
# API: COMPLETE 6-STAGE AI PIPELINE
# ============================================

@app.route("/api/analyze/<filename>", methods=["GET", "POST"])
def api_analyze_image(filename):
    safe_name = secure_filename(filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(image_path):
        return jsonify({
            "success": False,
            "error": f"Image file '{safe_name}' not found on server. It may have expired or not yet uploaded."
        }), 404

    try:
        result = run_pipeline_data(image_path)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"AI pipeline execution failed: {str(exc)}"
        }), 500


@app.route("/api/export-notes/<filename>", methods=["GET"])
def export_notes(filename):
    format_type = request.args.get("format", "md").lower()
    safe_name = secure_filename(filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(image_path):
        return jsonify({"error": "Image file not found"}), 404

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


# ============================================
# SERVER STARTUP (LOCAL / PRODUCTION WSGI)
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    print(f"🚀 Starting Classroom AI Server on http://0.0.0.0:{port} (Debug: {debug_mode})")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
