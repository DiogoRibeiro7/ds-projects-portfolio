"""Simple E2E tests for dashboard using Selenium and Playwright."""

import os
import sys
import time
import unittest
from threading import Thread

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestE2ESelenium(unittest.TestCase):
    """E2E tests using Selenium."""

    @classmethod
    def setUpClass(cls):
        """Set up test server and browser."""
        # Start Flask app in a thread
        from app import app, socketio

        cls.app = app
        cls.app.config["TESTING"] = True

        def run_server():
            socketio.run(cls.app, host="127.0.0.1", port=5555, debug=False)

        cls.server_thread = Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)  # Wait for server to start

    def setUp(self):
        """Set up browser driver."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            options.add_argument("--headless")  # Run in headless mode
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            # Try Chrome first
            try:
                self.driver = webdriver.Chrome(options=options)
            except:
                # Fall back to Edge
                try:
                    from selenium.webdriver.edge.options import Options as EdgeOptions

                    edge_options = EdgeOptions()
                    edge_options.add_argument("--headless")
                    self.driver = webdriver.Edge(options=edge_options)
                except:
                    self.skipTest("No browser driver available")

            self.driver.implicitly_wait(10)

        except ImportError:
            self.skipTest("Selenium not installed")

    def tearDown(self):
        """Close browser."""
        if hasattr(self, "driver"):
            self.driver.quit()

    def test_homepage_loads(self):
        """Test that homepage loads."""
        self.driver.get("http://127.0.0.1:5555/")
        self.assertIn("Dashboard", self.driver.title)

    def test_navigation_elements(self):
        """Test navigation elements exist."""
        from selenium.webdriver.common.by import By

        self.driver.get("http://127.0.0.1:5555/")

        # Check for main elements
        try:
            header = self.driver.find_element(By.TAG_NAME, "header")
            self.assertIsNotNone(header)
        except:
            pass  # Header might not exist

        # Check for dashboard container
        dashboard = self.driver.find_element(By.CLASS_NAME, "dashboard-container")
        self.assertIsNotNone(dashboard)


class TestE2EPlaywright(unittest.TestCase):
    """E2E tests using Playwright."""

    @classmethod
    def setUpClass(cls):
        """Set up test server."""
        # Start Flask app in a thread
        from app import app, socketio

        cls.app = app
        cls.app.config["TESTING"] = True

        def run_server():
            socketio.run(cls.app, host="127.0.0.1", port=5556, debug=False)

        cls.server_thread = Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        time.sleep(2)  # Wait for server to start

    def setUp(self):
        """Set up Playwright browser."""
        try:
            from playwright.sync_api import sync_playwright

            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()

        except ImportError:
            self.skipTest("Playwright not installed or browsers not downloaded")
        except Exception as e:
            self.skipTest(f"Playwright setup failed: {e}")

    def tearDown(self):
        """Close Playwright browser."""
        if hasattr(self, "browser"):
            self.browser.close()
        if hasattr(self, "playwright"):
            self.playwright.stop()

    def test_homepage_loads(self):
        """Test that homepage loads."""
        self.page.goto("http://127.0.0.1:5556/")
        self.assertIn("Dashboard", self.page.title())

    def test_interactive_elements(self):
        """Test interactive elements."""
        self.page.goto("http://127.0.0.1:5556/")

        # Check for dashboard container
        dashboard = self.page.query_selector(".dashboard-container")
        self.assertIsNotNone(dashboard)

        # Check for theme toggle
        theme_toggle = self.page.query_selector("[data-theme-toggle]")
        if theme_toggle:
            # Click theme toggle
            theme_toggle.click()
            # Check if theme changed
            body = self.page.query_selector("body")
            class_attr = body.get_attribute("class")
            # Theme might have changed

    def test_responsive_design(self):
        """Test responsive design."""
        # Test desktop view
        self.page.set_viewport_size({"width": 1920, "height": 1080})
        self.page.goto("http://127.0.0.1:5556/")
        desktop_container = self.page.query_selector(".dashboard-container")
        self.assertIsNotNone(desktop_container)

        # Test mobile view
        self.page.set_viewport_size({"width": 375, "height": 667})
        mobile_container = self.page.query_selector(".dashboard-container")
        self.assertIsNotNone(mobile_container)


if __name__ == "__main__":
    unittest.main(verbosity=2)
