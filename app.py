import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from werkzeug.utils import secure_filename
from uuid import uuid4
from PIL import Image

import auth

from pipeline import (
    PIPELINE_OUTPUT_FOLDER,
    PIPELINE_PROCESSES,
    run_pipeline_data,
    validate_image,
)


# ============================================
# FLASK APP
# ============================================

app = Flask(__name__, static_folder="public", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "boardai-neural-studio-secret-key-2026")

# Initialize authentication database
auth.init_db()


# ============================================
# USER SESSION INJECTOR
# ============================================

@app.context_processor
def inject_current_user():
    user_id = session.get("user_id")
    if user_id:
        user = auth.get_user_by_id(user_id)
        if user:
            return {"current_user": user}
    return {"current_user": None}


# ============================================
# UPLOAD CONFIGURATION
# ============================================

UPLOAD_FOLDER = os.path.join(
    "/tmp" if os.environ.get("VERCEL") else ".",
    "uploads"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================
# CHECK FILE TYPE
# ============================================

def allowed_file(filename):
    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================
# HOME PAGE
# ============================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        pipeline_processes=PIPELINE_PROCESSES
    )


# ============================================
# FAVICON (CLEAN 204 NO CONTENT)
# ============================================

@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ============================================
# AUTHENTICATION ROUTES (LOGIN / SIGNUP / LOGOUT)
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
        flash(f"Welcome to BoardAI, {user['username']}! Your account was successfully created.", "success")
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
    flash("You have been signed out.", "info")
    return redirect(url_for("home"))


# ============================================
# API: PIPELINE ARCHITECTURE SPECIFICATION
# ============================================

@app.route("/api/pipeline-processes")
def get_pipeline_processes():
    return jsonify({
        "success": True,
        "count": len(PIPELINE_PROCESSES),
        "processes": PIPELINE_PROCESSES
    })


# ============================================
# 1.0 IMAGE UPLOAD & VALIDATION ROUTE
# ============================================

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")

    if not files or all(file.filename == "" for file in files):
        return "No images selected."

    uploaded_files = []

    for file in files:
        if file.filename == "":
            continue

        if not allowed_file(file.filename):
            continue

        original_name = secure_filename(file.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid4().hex}.{extension}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        file.save(file_path)

        # 1.0 Image Upload & Validation check
        val = validate_image(file_path)
        if not val["valid"]:
            # If invalid or corrupted image, remove and skip
            try:
                os.remove(file_path)
            except OSError:
                pass
            continue

        file_size_bytes = val.get("size_bytes", os.path.getsize(file_path))
        if file_size_bytes < 1024:
            file_size_formatted = f"{file_size_bytes} B"
        elif file_size_bytes < 1024 * 1024:
            file_size_formatted = f"{file_size_bytes / 1024:.1f} KB"
        else:
            file_size_formatted = f"{file_size_bytes / (1024 * 1024):.1f} MB"

        uploaded_files.append({
            "original_name": original_name,
            "saved_name": unique_filename,
            "file_size": file_size_formatted,
            "extension": extension.upper(),
            "dimensions": f"{val.get('width', 0)}x{val.get('height', 0)}"
        })

    if not uploaded_files:
        return "No valid classroom-board images were uploaded."

    return render_template(
        "result.html",
        uploaded_files=uploaded_files,
        pipeline_processes=PIPELINE_PROCESSES
    )


# ============================================
# RESULTS PAGE (DIRECT ACCESS / RECENT)
# ============================================

@app.route("/results")
def results_page():
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        names = sorted(
            os.listdir(UPLOAD_FOLDER),
            key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
            reverse=True
        )
        for f in names[:25]:
            if allowed_file(f):
                fp = os.path.join(UPLOAD_FOLDER, f)
                sz = os.path.getsize(fp)
                ext = f.rsplit(".", 1)[1].upper() if "." in f else "IMG"
                if sz < 1024:
                    s_str = f"{sz} B"
                elif sz < 1024 * 1024:
                    s_str = f"{sz / 1024:.1f} KB"
                else:
                    s_str = f"{sz / (1024 * 1024):.1f} MB"

                dim_str = ""
                try:
                    with Image.open(fp) as im:
                        dim_str = f"{im.width}x{im.height}"
                except Exception:
                    dim_str = "Auto"

                files.append({
                    "original_name": f,
                    "saved_name": f,
                    "file_size": s_str,
                    "extension": ext,
                    "dimensions": dim_str
                })

    return render_template(
        "result.html",
        uploaded_files=files,
        pipeline_processes=PIPELINE_PROCESSES
    )


# ============================================
# SERVE UPLOADED IMAGES
# ============================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================
# SERVE PROCESSED PIPELINE IMAGES
# ============================================

@app.route("/pipeline_output/<filename>")
def pipeline_output_file(filename):
    return send_from_directory(
        PIPELINE_OUTPUT_FOLDER,
        filename
    )


# ============================================
# API: RUN AI PIPELINE ON IMAGE (STAGES 1.0 - 6.0)
# ============================================

@app.route("/api/analyze/<filename>", methods=["GET", "POST"])
def api_analyze_image(filename):
    safe_name = secure_filename(filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": "Image file not found on server."}), 404

    try:
        result = run_pipeline_data(image_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ============================================
# API: EXPORT STRUCTURED NOTES (STEP 6.0)
# ============================================

@app.route("/api/export-notes/<filename>", methods=["GET"])
def export_notes(filename):
    format_type = request.args.get("format", "md").lower()
    safe_name = secure_filename(filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.exists(image_path):
        return {"error": "Image file not found"}, 404

    result = run_pipeline_data(image_path)
    notes = result.get("structured_notes", {})

    if format_type == "txt":
        content = notes.get("plain_text", "No content available.")
        mimetype = "text/plain"
        dl_name = f"notes_{safe_name}.txt"
    else:
        content = notes.get("markdown", "# No content available.")
        mimetype = "text/markdown"
        dl_name = f"notes_{safe_name}.md"

    return Response(
        content,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment; filename={dl_name}"}
    )


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)