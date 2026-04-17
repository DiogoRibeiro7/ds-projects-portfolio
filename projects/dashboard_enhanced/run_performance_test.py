"""Run a quick performance test of the dashboard."""

import os
import subprocess
import sys
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def start_server():
    """Start the Flask server."""
    from app import app, socketio

    app.config["TESTING"] = True
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)


def run_performance_test():
    """Run Locust performance test."""
    print("Starting dashboard server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(3)  # Wait for server to start

    print("\nRunning performance test...")
    print("=" * 50)

    # Run Locust in headless mode
    # 10 users, spawn rate 2/sec, run for 30 seconds
    cmd = [
        "locust",
        "-f",
        "test_performance.py",
        "--host",
        "http://127.0.0.1:5000",
        "--headless",
        "-u",
        "10",  # 10 users
        "-r",
        "2",  # Spawn 2 users per second
        "-t",
        "30s",  # Run for 30 seconds
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.TimeoutExpired:
        print("Test timed out after 60 seconds")
    except FileNotFoundError:
        print("Locust not found. Running simple performance test instead...")
        run_simple_performance_test()
    except Exception as e:
        print(f"Error running Locust: {e}")
        print("Running simple performance test instead...")
        run_simple_performance_test()


def run_simple_performance_test():
    """Run a simple performance test without Locust."""
    import concurrent.futures
    from statistics import mean, stdev

    import requests

    base_url = "http://127.0.0.1:5000"
    endpoints = ["/", "/api/config", "/api/data/default"]
    num_requests = 50
    num_workers = 5

    print("\nSimple Performance Test")
    print("=" * 50)

    def make_request(endpoint):
        """Make a single request and measure time."""
        url = base_url + endpoint
        start = time.time()
        try:
            response = requests.get(url, timeout=5)
            elapsed = time.time() - start
            return {
                "endpoint": endpoint,
                "status": response.status_code,
                "time": elapsed,
                "success": response.status_code < 400,
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                "endpoint": endpoint,
                "status": 0,
                "time": elapsed,
                "success": False,
                "error": str(e),
            }

    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")

        # Warm up
        make_request(endpoint)

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(make_request, endpoint) for _ in range(num_requests)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        times = [r["time"] for r in successful]

        if times:
            print(f"  Requests: {num_requests}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {num_requests - len(successful)}")
            print(f"  Avg response time: {mean(times):.3f}s")
            if len(times) > 1:
                print(f"  Std deviation: {stdev(times):.3f}s")
            print(f"  Min time: {min(times):.3f}s")
            print(f"  Max time: {max(times):.3f}s")
            print(f"  Requests/sec: {len(successful) / sum(times):.1f}")
        else:
            print("  All requests failed!")

    print("\n" + "=" * 50)
    print("Performance test complete!")


if __name__ == "__main__":
    run_performance_test()
