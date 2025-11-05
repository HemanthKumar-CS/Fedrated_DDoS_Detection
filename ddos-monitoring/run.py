#!/usr/bin/env python3
"""
Simple startup script to run FastAPI backend and serve frontend
"""
import subprocess
import time
import sys
from pathlib import Path


def main():
    print("🚀 Starting DDoS Detection Monitoring Dashboard...")
    print("=" * 60)

    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Start FastAPI backend on port 8000
    print("\n📡 Starting FastAPI Backend (port 8000)...")
    backend_proc = subprocess.Popen(
        [sys.executable, "backend.py"],
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(2)  # Give backend time to start

    # Start HTTP server for frontend on port 8080
    print("🌐 Starting HTTP Server for Frontend (port 8080)...")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080"],
        cwd=script_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(1)

    print("\n✅ Both servers started successfully!")
    print("=" * 60)
    print("\n📊 Dashboard ready at: http://localhost:8080")
    print("📡 API available at: http://localhost:8000")
    print("\n🔌 WebSocket Endpoints:")
    print("  - ws://localhost:8000/ws/live-detection")
    print("  - ws://localhost:8000/ws/alerts")
    print("  - ws://localhost:8000/ws/metrics")
    print("\nPress CTRL+C to stop both servers...")
    print("=" * 60)

    try:
        # Wait for both processes
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait(timeout=5)
        frontend_proc.wait(timeout=5)
        print("✅ Servers stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
