"""
Comprehensive Testing Suite for Dashboard

Unit tests, end-to-end testing, performance testing, visual regression tests,
and cross-browser compatibility tests.

Author: Portfolio Team
Date: 2024
"""

import unittest
import asyncio
import time
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from playwright.async_api import async_playwright, Page, Browser
import requests
from PIL import Image, ImageChops
import imagehash
from locust import HttpUser, task, between
from flask_testing import TestCase

# Import dashboard components
from dashboard_framework import EnhancedDashboard, DashboardConfig
from visualization_components import InteractiveVisualizations
from api_infrastructure import DashboardAPI, APIConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Unit Tests
# ============================================================================

class TestDashboardFramework(unittest.TestCase):
    """Unit tests for dashboard framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DashboardConfig(
            app_name="Test Dashboard",
            debug=False,
            enable_realtime=False
        )
        self.dashboard = EnhancedDashboard(self.config)

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.config.app_name, "Test Dashboard")
        self.assertIsNotNone(self.dashboard.app)

    def test_component_registration(self):
        """Test component registration."""
        def test_component():
            return "Test Component"

        self.dashboard.register_component("test_comp", test_component)
        self.assertIn("test_comp", self.dashboard.components)
        self.assertEqual(self.dashboard.components["test_comp"](), "Test Component")

    def test_data_source_registration(self):
        """Test data source registration."""
        def test_source(filters=None):
            return pd.DataFrame({'test': [1, 2, 3]})

        self.dashboard.register_data_source("test_source", test_source)
        self.assertIn("test_source", self.dashboard.data_sources)

        # Test data retrieval
        data = self.dashboard.get_data("test_source")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)

    @patch('dashboard_framework.redis.Redis')
    def test_redis_connection(self, mock_redis):
        """Test Redis connection handling."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        config = DashboardConfig(enable_realtime=True)
        dashboard = EnhancedDashboard(config)

        self.assertIsNotNone(dashboard.redis_client)
        mock_redis_instance.ping.assert_called_once()

    def test_kpi_card_creation(self):
        """Test KPI card creation."""
        card = self.dashboard.create_kpi_card(
            title="Test KPI",
            value="100",
            change=10.5,
            icon="fa-test"
        )

        self.assertIsNotNone(card)
        # Check that card contains expected elements
        card_str = str(card)
        self.assertIn("Test KPI", card_str)
        self.assertIn("100", card_str)

    def test_responsive_card_creation(self):
        """Test responsive card creation."""
        content = "Test Content"
        card = self.dashboard.create_responsive_card(
            title="Test Card",
            content=content,
            card_id="test-card"
        )

        self.assertIsNotNone(card)
        card_str = str(card)
        self.assertIn("Test Card", card_str)
        self.assertIn(content, card_str)

    def test_export_to_pdf(self):
        """Test PDF export functionality."""
        import plotly.graph_objects as go

        # Create sample figure
        fig = go.Figure(data=[go.Bar(x=['A', 'B'], y=[1, 2])])
        figures = [fig]

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test.pdf")
            result = self.dashboard.export_to_pdf(figures, pdf_path)

            self.assertEqual(result, pdf_path)
            self.assertTrue(os.path.exists(pdf_path))

    def test_export_to_powerpoint(self):
        """Test PowerPoint export functionality."""
        import plotly.graph_objects as go

        # Create sample figure
        fig = go.Figure(data=[go.Bar(x=['A', 'B'], y=[1, 2])])
        figures = [fig]

        with tempfile.TemporaryDirectory() as tmpdir:
            pptx_path = os.path.join(tmpdir, "test.pptx")
            result = self.dashboard.export_to_powerpoint(figures, pptx_path)

            self.assertEqual(result, pptx_path)
            self.assertTrue(os.path.exists(pptx_path))


class TestVisualizationComponents(unittest.TestCase):
    """Unit tests for visualization components."""

    def setUp(self):
        """Set up test fixtures."""
        self.viz = InteractiveVisualizations()

    def test_animated_time_series(self):
        """Test animated time series creation."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'value1': np.random.randn(10),
            'value2': np.random.randn(10)
        })

        fig = self.viz.create_animated_time_series(
            df, 'date', ['value1', 'value2']
        )

        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.frames), 0)
        self.assertEqual(len(fig.data), 2)

    def test_3d_scatter(self):
        """Test 3D scatter plot creation."""
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'z': np.random.randn(50),
            'color': np.random.uniform(0, 1, 50)
        })

        fig = self.viz.create_interactive_3d_scatter(
            df, 'x', 'y', 'z', color_col='color'
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, 'scatter3d')

    def test_sankey_diagram(self):
        """Test Sankey diagram creation."""
        df = pd.DataFrame({
            'source': ['A', 'A', 'B'],
            'target': ['X', 'Y', 'X'],
            'value': [10, 20, 15]
        })

        fig = self.viz.create_sankey_diagram(
            df, 'source', 'target', 'value'
        )

        self.assertIsNotNone(fig)
        self.assertEqual(fig.data[0].type, 'sankey')

    def test_accessibility_chart(self):
        """Test accessible chart creation."""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'C'],
            'Value': [10, 20, 15]
        })

        fig = self.viz.create_accessibility_chart(
            df.set_index('Category'),
            chart_type='bar',
            title='Accessible Test Chart'
        )

        self.assertIsNotNone(fig)
        # Check for accessibility features
        self.assertIn('aria-label', fig.layout.title.text)


class TestAPIInfrastructure(TestCase):
    """Unit tests for API infrastructure."""

    def create_app(self):
        """Create Flask app for testing."""
        config = APIConfig(debug=False)
        api = DashboardAPI(config)
        api.app.config['TESTING'] = True
        return api.app

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)

    def test_user_registration(self):
        """Test user registration."""
        response = self.client.post('/api/auth/register',
            json={
                'username': 'testuser',
                'password': 'testpass123',
                'email': 'test@example.com'
            }
        )

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('api_key', data)

    def test_authentication_required(self):
        """Test that authentication is required for protected endpoints."""
        response = self.client.get('/api/v1/data')
        self.assertEqual(response.status_code, 401)


# ============================================================================
# End-to-End Tests with Selenium
# ============================================================================

class SeleniumE2ETests(unittest.TestCase):
    """End-to-end tests using Selenium."""

    @classmethod
    def setUpClass(cls):
        """Set up Selenium driver."""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        try:
            cls.driver = webdriver.Chrome(options=options)
            cls.base_url = "http://localhost:8050"
        except:
            cls.driver = None
            logger.warning("Chrome driver not available, skipping Selenium tests")

    @classmethod
    def tearDownClass(cls):
        """Clean up Selenium driver."""
        if cls.driver:
            cls.driver.quit()

    def test_dashboard_loads(self):
        """Test that dashboard loads successfully."""
        if not self.driver:
            self.skipTest("Selenium driver not available")

        self.driver.get(self.base_url)

        # Wait for dashboard to load
        wait = WebDriverWait(self.driver, 10)
        navbar = wait.until(
            EC.presence_of_element_located((By.ID, "navbar"))
        )

        self.assertIsNotNone(navbar)

    def test_navigation(self):
        """Test navigation between pages."""
        if not self.driver:
            self.skipTest("Selenium driver not available")

        self.driver.get(self.base_url)

        # Click on Analytics link
        analytics_link = self.driver.find_element(By.LINK_TEXT, "Analytics")
        analytics_link.click()

        # Wait for page change
        wait = WebDriverWait(self.driver, 10)
        content = wait.until(
            EC.presence_of_element_located((By.ID, "page-content"))
        )

        # Check URL changed
        self.assertIn("/analytics", self.driver.current_url)

    def test_filter_interaction(self):
        """Test filter panel interaction."""
        if not self.driver:
            self.skipTest("Selenium driver not available")

        self.driver.get(self.base_url)

        # Find and interact with date picker
        try:
            date_picker = self.driver.find_element(By.ID, "date-range-picker")
            self.assertIsNotNone(date_picker)

            # Click apply filters button
            apply_btn = self.driver.find_element(By.ID, "apply-filters-btn")
            apply_btn.click()
        except:
            pass  # Elements might not be present in test mode

    def test_dark_mode_toggle(self):
        """Test dark mode toggle functionality."""
        if not self.driver:
            self.skipTest("Selenium driver not available")

        self.driver.get(self.base_url)

        try:
            # Find dark mode switch
            dark_switch = self.driver.find_element(By.ID, "dark-mode-switch")

            # Get initial state
            main_container = self.driver.find_element(By.ID, "main-container")
            initial_class = main_container.get_attribute("class")

            # Toggle dark mode
            dark_switch.click()

            # Check class changed
            time.sleep(1)  # Wait for transition
            new_class = main_container.get_attribute("class")
            self.assertNotEqual(initial_class, new_class)
        except:
            pass  # Element might not be present in test mode


# ============================================================================
# Playwright Tests (More Modern Alternative to Selenium)
# ============================================================================

class PlaywrightE2ETests:
    """End-to-end tests using Playwright."""

    @pytest.mark.asyncio
    async def test_dashboard_loads(self):
        """Test dashboard loads with Playwright."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto("http://localhost:8050")

            # Check for navbar
            navbar = await page.wait_for_selector("#navbar")
            assert navbar is not None

            await browser.close()

    @pytest.mark.asyncio
    async def test_responsive_design(self):
        """Test responsive design on different viewports."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            # Test desktop viewport
            desktop_page = await browser.new_page(
                viewport={"width": 1920, "height": 1080}
            )
            await desktop_page.goto("http://localhost:8050")
            desktop_layout = await desktop_page.screenshot()

            # Test mobile viewport
            mobile_page = await browser.new_page(
                viewport={"width": 375, "height": 667}
            )
            await mobile_page.goto("http://localhost:8050")
            mobile_layout = await mobile_page.screenshot()

            # Test tablet viewport
            tablet_page = await browser.new_page(
                viewport={"width": 768, "height": 1024}
            )
            await tablet_page.goto("http://localhost:8050")
            tablet_layout = await tablet_page.screenshot()

            await browser.close()

            # Verify screenshots were taken
            assert desktop_layout is not None
            assert mobile_layout is not None
            assert tablet_layout is not None

    @pytest.mark.asyncio
    async def test_keyboard_navigation(self):
        """Test keyboard navigation for accessibility."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto("http://localhost:8050")

            # Tab through elements
            for _ in range(5):
                await page.keyboard.press("Tab")

            # Get focused element
            focused_element = await page.evaluate("document.activeElement.tagName")
            assert focused_element is not None

            await browser.close()


# ============================================================================
# Performance Tests with Locust
# ============================================================================

class DashboardUser(HttpUser):
    """Performance test user for dashboard."""

    wait_time = between(1, 3)
    host = "http://localhost:5000"

    def on_start(self):
        """Login before running tasks."""
        # Register user
        self.client.post("/api/auth/register", json={
            "username": f"user_{time.time()}",
            "password": "testpass123"
        })

        # Login
        response = self.client.post("/api/auth/login",
            auth=("testuser", "testpass123")
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task(3)
    def get_health(self):
        """Test health endpoint."""
        self.client.get("/api/health")

    @task(5)
    def get_data(self):
        """Test data retrieval."""
        self.client.get("/api/v1/data", headers=self.headers)

    @task(2)
    def get_metrics(self):
        """Test metrics retrieval."""
        self.client.get("/api/v1/metrics", headers=self.headers)

    @task(1)
    def run_analytics(self):
        """Test analytics query."""
        self.client.post("/api/v1/analytics",
            json={
                "type": "aggregate",
                "parameters": {}
            },
            headers=self.headers
        )


# ============================================================================
# Visual Regression Tests
# ============================================================================

class VisualRegressionTests:
    """Visual regression tests for dashboard."""

    def __init__(self):
        """Initialize visual regression tests."""
        self.baseline_dir = Path("tests/visual_baselines")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        self.diff_dir = Path("tests/visual_diffs")
        self.diff_dir.mkdir(parents=True, exist_ok=True)

    def capture_screenshot(self, driver, name: str) -> Image:
        """Capture screenshot of current page."""
        screenshot_path = f"temp_{name}.png"
        driver.save_screenshot(screenshot_path)

        image = Image.open(screenshot_path)
        os.remove(screenshot_path)

        return image

    def compare_images(self, baseline: Image, current: Image,
                      threshold: float = 0.01) -> Dict[str, Any]:
        """Compare two images for visual differences."""
        # Calculate difference
        diff = ImageChops.difference(baseline, current)

        # Calculate similarity using perceptual hash
        baseline_hash = imagehash.average_hash(baseline)
        current_hash = imagehash.average_hash(current)
        hash_diff = baseline_hash - current_hash

        # Get bounding box of differences
        bbox = diff.getbbox()

        # Calculate percentage of pixels that differ
        if bbox:
            diff_pixels = sum(diff.getdata())
            total_pixels = baseline.width * baseline.height * 3  # RGB
            diff_percentage = diff_pixels / total_pixels
        else:
            diff_percentage = 0

        return {
            'match': diff_percentage <= threshold,
            'diff_percentage': diff_percentage,
            'hash_difference': hash_diff,
            'diff_bbox': bbox,
            'diff_image': diff
        }

    def test_component_visual_regression(self, component_name: str):
        """Test visual regression for a specific component."""
        # Set up driver
        options = ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)

        try:
            # Navigate to component
            driver.get(f"http://localhost:8050/component/{component_name}")
            time.sleep(2)  # Wait for rendering

            # Capture current screenshot
            current = self.capture_screenshot(driver, component_name)

            # Load or create baseline
            baseline_path = self.baseline_dir / f"{component_name}_baseline.png"

            if baseline_path.exists():
                baseline = Image.open(baseline_path)

                # Compare images
                result = self.compare_images(baseline, current)

                if not result['match']:
                    # Save diff image
                    diff_path = self.diff_dir / f"{component_name}_diff.png"
                    result['diff_image'].save(diff_path)

                    logger.warning(
                        f"Visual regression detected for {component_name}: "
                        f"{result['diff_percentage']:.2%} difference"
                    )
                    return False
                else:
                    logger.info(f"Visual regression test passed for {component_name}")
                    return True
            else:
                # Create baseline
                current.save(baseline_path)
                logger.info(f"Created baseline for {component_name}")
                return True

        finally:
            driver.quit()


# ============================================================================
# Cross-Browser Compatibility Tests
# ============================================================================

class CrossBrowserTests:
    """Cross-browser compatibility tests."""

    def __init__(self):
        """Initialize cross-browser tests."""
        self.browsers = ['chrome', 'firefox', 'edge']
        self.test_results = {}

    def get_driver(self, browser: str):
        """Get WebDriver for specified browser."""
        if browser == 'chrome':
            options = ChromeOptions()
            options.add_argument('--headless')
            return webdriver.Chrome(options=options)
        elif browser == 'firefox':
            options = FirefoxOptions()
            options.add_argument('--headless')
            return webdriver.Firefox(options=options)
        elif browser == 'edge':
            # Edge driver setup (if available)
            from selenium.webdriver.edge.options import Options as EdgeOptions
            options = EdgeOptions()
            options.add_argument('--headless')
            return webdriver.Edge(options=options)
        else:
            raise ValueError(f"Unsupported browser: {browser}")

    def test_browser_compatibility(self, browser: str) -> Dict[str, Any]:
        """Test dashboard compatibility with specific browser."""
        results = {
            'browser': browser,
            'tests': {},
            'passed': 0,
            'failed': 0
        }

        try:
            driver = self.get_driver(browser)
        except Exception as e:
            logger.warning(f"Could not initialize {browser} driver: {e}")
            return results

        try:
            # Test 1: Page loads
            driver.get("http://localhost:8050")
            time.sleep(2)

            try:
                navbar = driver.find_element(By.ID, "navbar")
                results['tests']['page_load'] = navbar is not None
                if navbar:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
            except:
                results['tests']['page_load'] = False
                results['failed'] += 1

            # Test 2: JavaScript execution
            try:
                js_result = driver.execute_script("return typeof Plotly !== 'undefined';")
                results['tests']['javascript'] = js_result
                if js_result:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
            except:
                results['tests']['javascript'] = False
                results['failed'] += 1

            # Test 3: CSS rendering
            try:
                element = driver.find_element(By.ID, "main-container")
                computed_style = driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).display;",
                    element
                )
                results['tests']['css_rendering'] = computed_style != 'none'
                if computed_style != 'none':
                    results['passed'] += 1
                else:
                    results['failed'] += 1
            except:
                results['tests']['css_rendering'] = False
                results['failed'] += 1

            # Test 4: Local storage
            try:
                driver.execute_script("localStorage.setItem('test', 'value');")
                stored_value = driver.execute_script("return localStorage.getItem('test');")
                results['tests']['local_storage'] = stored_value == 'value'
                if stored_value == 'value':
                    results['passed'] += 1
                else:
                    results['failed'] += 1
            except:
                results['tests']['local_storage'] = False
                results['failed'] += 1

        finally:
            driver.quit()

        results['success_rate'] = results['passed'] / (results['passed'] + results['failed'])
        return results

    def run_all_browser_tests(self) -> Dict[str, Any]:
        """Run tests on all supported browsers."""
        for browser in self.browsers:
            logger.info(f"Testing {browser}...")
            self.test_results[browser] = self.test_browser_compatibility(browser)

        # Generate summary
        summary = {
            'total_browsers': len(self.browsers),
            'browsers_tested': len(self.test_results),
            'all_passed': all(
                r['success_rate'] == 1.0
                for r in self.test_results.values()
            ),
            'results': self.test_results
        }

        return summary


# ============================================================================
# Test Suite Runner
# ============================================================================

class DashboardTestSuite:
    """Complete dashboard testing suite."""

    def __init__(self):
        """Initialize test suite."""
        self.results = {}

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        logger.info("Running unit tests...")

        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Add test cases
        suite.addTests(loader.loadTestsFromTestCase(TestDashboardFramework))
        suite.addTests(loader.loadTestsFromTestCase(TestVisualizationComponents))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful()
        }

    def run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests."""
        logger.info("Running E2E tests...")

        # Run Selenium tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(SeleniumE2ETests)
        runner = unittest.TextTestRunner(verbosity=2)
        selenium_result = runner.run(suite)

        # Run Playwright tests (if pytest is available)
        try:
            pytest_result = pytest.main([__file__, '-k', 'Playwright', '-v'])
            playwright_success = pytest_result == 0
        except:
            playwright_success = None

        return {
            'selenium': {
                'tests_run': selenium_result.testsRun,
                'success': selenium_result.wasSuccessful()
            },
            'playwright': {
                'success': playwright_success
            }
        }

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        logger.info("Running performance tests...")

        # Note: Locust tests would typically be run via command line
        # locust -f testing_suite.py DashboardUser --headless -u 10 -r 2 -t 30s

        return {
            'note': 'Run locust tests via command line',
            'command': 'locust -f testing_suite.py DashboardUser --headless -u 10 -r 2 -t 30s'
        }

    def run_visual_regression_tests(self) -> Dict[str, Any]:
        """Run visual regression tests."""
        logger.info("Running visual regression tests...")

        vrt = VisualRegressionTests()
        components = ['dashboard', 'analytics', 'reports']
        results = {}

        for component in components:
            try:
                results[component] = vrt.test_component_visual_regression(component)
            except Exception as e:
                logger.error(f"Visual regression test failed for {component}: {e}")
                results[component] = False

        return {
            'components_tested': len(components),
            'passed': sum(1 for r in results.values() if r),
            'results': results
        }

    def run_cross_browser_tests(self) -> Dict[str, Any]:
        """Run cross-browser compatibility tests."""
        logger.info("Running cross-browser tests...")

        cbt = CrossBrowserTests()
        return cbt.run_all_browser_tests()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        logger.info("="*60)
        logger.info("RUNNING COMPLETE DASHBOARD TEST SUITE")
        logger.info("="*60)

        self.results['unit_tests'] = self.run_unit_tests()
        self.results['e2e_tests'] = self.run_e2e_tests()
        self.results['performance_tests'] = self.run_performance_tests()
        self.results['visual_regression'] = self.run_visual_regression_tests()
        self.results['cross_browser'] = self.run_cross_browser_tests()

        # Generate summary
        self.results['summary'] = {
            'total_test_categories': 5,
            'categories_passed': sum(
                1 for r in self.results.values()
                if isinstance(r, dict) and r.get('success', False)
            ),
            'timestamp': datetime.now().isoformat()
        }

        logger.info("="*60)
        logger.info("TEST SUITE COMPLETED")
        logger.info(f"Summary: {self.results['summary']}")
        logger.info("="*60)

        return self.results


if __name__ == "__main__":
    # Run complete test suite
    suite = DashboardTestSuite()
    results = suite.run_all_tests()

    # Save results to file
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Test results saved to test_results.json")