#!/usr/bin/env python3
"""
Emergency process cleanup for stuck federated learning
"""

import subprocess
import sys
import time


def cleanup_hanging_processes():
    """Kill any hanging Python processes from federated learning"""
    print("🧹 Emergency cleanup of hanging processes...")

    try:
        # Kill all Python processes (this will terminate the stuck demo)
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/f", "/im", "python.exe"],
                           capture_output=True, text=True)
            print("✅ Killed hanging Python processes")
        else:
            subprocess.run(["pkill", "-f", "python"],
                           capture_output=True, text=True)
            print("✅ Killed hanging Python processes")

        time.sleep(2)

        # Check if any processes are still running
        if sys.platform == "win32":
            result = subprocess.run(["tasklist", "/fi", "imagename eq python.exe"],
                                    capture_output=True, text=True)
            if "python.exe" in result.stdout:
                print("⚠️ Some Python processes may still be running")
            else:
                print("✅ All Python processes cleaned up")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    cleanup_hanging_processes()
    print("🎯 You can now run the demo again!")
