import os
import sys
import unittest
import json

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Use a test database to protect production/development data
os.environ["DATABASE_PATH"] = os.path.join(os.path.dirname(__file__), "test_phase1.db")
os.environ["FLASK_ENV"] = "testing"

import auth
import app as flask_module


class Phase1BackendTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        auth.init_db()
        cls.app = flask_module.app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Cleanup test database
        if os.path.exists(auth.DB_PATH):
            try:
                os.remove(auth.DB_PATH)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. Startup & Health Checks
    # --------------------------------------------------------------------------
    def test_01_health_endpoints(self):
        """Test health check endpoints for fast monitoring response."""
        for endpoint in ["/health", "/api/health"]:
            res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "AI Classroom Board Content Extractor")
            self.assertIn("cnn_model_ready", data)

    # --------------------------------------------------------------------------
    # 2. Existing Routes Availability
    # --------------------------------------------------------------------------
    def test_02_web_pages_load(self):
        """Verify web pages load without unhandled template errors."""
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)

        res_results = self.client.get("/results")
        self.assertEqual(res_results.status_code, 200)

        res_files = self.client.get("/api/files")
        self.assertEqual(res_files.status_code, 200)
        self.assertTrue(res_files.get_json()["success"])

    # --------------------------------------------------------------------------
    # 3. CORS & Origin Validation (No Wildcard + Credentials)
    # --------------------------------------------------------------------------
    def test_03_cors_trusted_origin_preflight(self):
        """Verify trusted origin receives explicit CORS headers."""
        trusted = "https://classroom-theta-silk.vercel.app"
        res = self.client.options(
            "/api/health",
            headers={
                "Origin": trusted,
                "Access-Control-Request-Method": "GET"
            }
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), trusted)
        self.assertEqual(res.headers.get("Access-Control-Allow-Credentials"), "true")
        self.assertNotEqual(res.headers.get("Access-Control-Allow-Origin"), "*")

    def test_04_cors_untrusted_origin_rejected(self):
        """Verify untrusted origin does NOT receive allow-origin header."""
        untrusted = "https://malicious-attacker-site.com"
        res = self.client.options(
            "/api/health",
            headers={
                "Origin": untrusted,
                "Access-Control-Request-Method": "GET"
            }
        )
        self.assertIsNone(res.headers.get("Access-Control-Allow-Origin"))

    def test_05_cors_preview_vercel_origin_allowed(self):
        """Verify canonical preview Vercel branch deployments are allowed."""
        preview_origin = "https://classroom-branch-preview.vercel.app"
        res = self.client.get(
            "/api/health",
            headers={"Origin": preview_origin}
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), preview_origin)

    # --------------------------------------------------------------------------
    # 4. Authentication & Session Handling
    # --------------------------------------------------------------------------
    def test_06_signup_and_login_flow(self):
        """Test user signup, session persistence, me endpoint, and logout."""
        test_username = "prof_tester"
        test_email = "prof.tester@university.edu"
        test_password = "SecurePassword123"

        # 1. API Signup
        res_signup = self.client.post(
            "/api/auth/signup",
            json={
                "username": test_username,
                "email": test_email,
                "password": test_password,
                "confirm_password": test_password
            }
        )
        self.assertEqual(res_signup.status_code, 201)
        data = res_signup.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["username"], test_username)

        # 2. Check /api/auth/me with session
        res_me = self.client.get("/api/auth/me")
        self.assertEqual(res_me.status_code, 200)
        self.assertTrue(res_me.get_json()["authenticated"])
        self.assertEqual(res_me.get_json()["user"]["email"], test_email)

        # 3. Logout
        res_logout = self.client.post("/api/auth/logout")
        self.assertEqual(res_logout.status_code, 200)

        # Verify logged out
        res_me_after = self.client.get("/api/auth/me")
        self.assertFalse(res_me_after.get_json()["authenticated"])

        # 4. Login with email
        res_login = self.client.post(
            "/api/auth/login",
            json={
                "login_identifier": test_email,
                "password": test_password
            }
        )
        self.assertEqual(res_login.status_code, 200)
        self.assertTrue(res_login.get_json()["success"])

        # 5. Invalid credentials rejection
        res_invalid = self.client.post(
            "/api/auth/login",
            json={
                "login_identifier": test_email,
                "password": "WrongPassword999"
            }
        )
        self.assertEqual(res_invalid.status_code, 401)
        self.assertFalse(res_invalid.get_json()["success"])

    def test_07_validation_rules(self):
        """Test username and password strength validations."""
        u_short, err1 = auth.create_user("a", "test@domain.com", "validpass123")
        self.assertIsNone(u_short)
        self.assertIn("at least 3 characters", err1)

        p_short, err2 = auth.create_user("validuser", "test2@domain.com", "123")
        self.assertIsNone(p_short)
        self.assertIn("at least 6 characters", err2)

        bad_email, err3 = auth.create_user("validuser", "not-an-email", "validpass123")
        self.assertIsNone(bad_email)
        self.assertIn("valid email", err3)

    # --------------------------------------------------------------------------
    # 5. Error Handling & Sanitization
    # --------------------------------------------------------------------------
    def test_08_sanitized_404_error(self):
        """Verify 404 returns structured JSON on API routes without stack traces."""
        res = self.client.get("/api/nonexistent-endpoint-1234")
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Requested resource was not found.")

    def test_09_sanitized_400_bad_upload(self):
        """Verify empty upload returns structured 400 error."""
        res = self.client.post("/api/upload")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No images provided", data["error"])

    # --------------------------------------------------------------------------
    # 6. File Cleanup Protection Test
    # --------------------------------------------------------------------------
    def test_10_cleanup_protects_recent_files(self):
        """Verify files created recently are strictly preserved during cleanup."""
        upload_dir = self.app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        test_file = os.path.join(upload_dir, "test_recent_active_upload.jpg")
        with open(test_file, "w") as f:
            f.write("dummy-data")

        # Run cleanup
        flask_module.cleanup_old_files(max_age_hours=0, min_protection_hours=2)

        # File MUST still exist because it was modified < 2 hours ago
        self.assertTrue(os.path.exists(test_file))

        # Cleanup test file
        try:
            os.remove(test_file)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
